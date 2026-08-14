"""평가 지표.

■ 선행모델 대응
  FinSight 의 `evaluation/metrics.py` 에서 4개 지표를 가져왔다.
    Correctness / Faithfulness / Response Relevancy / Context Precision

■ 우리가 유리한 점 — 검색 지표를 **결정론적으로** 잴 수 있다
  선행모델은 PDF 를 파싱해 청크를 만들었기 때문에 "정답 청크"를 특정하기 어려웠고,
  레벤슈타인 거리로 근사했다. 우리 데이터는 **chunk_id 가 이미 확정**돼 있으므로
  골드셋에 정답 chunk_id 를 적어 두면 Recall@k·Precision@k·MRR 를 정확히 잰다.
  → LLM 없이, 매 실행 같은 값. 이게 성능 측정 단계에서 가장 믿을 만한 신호다.

■ 지표 3층 구조
  1층 (LLM 불필요, 항상 신뢰):  검색 지표, 라우팅 정확도, 소스 정확도, 비용/지연
  2층 (LLM 심판, 비용 발생):    Correctness, Faithfulness
  3층 (임베딩 필요):            Response Relevancy
  → 키가 없거나 예산이 부족하면 1층만으로도 의미 있는 비교가 가능하게 설계했다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

import config
from llm import llm


# ==========================================================================
# 1층: 결정론적 지표
# ==========================================================================
def retrieval_metrics(retrieved_ids: Sequence[str],
                      gold_ids: Sequence[str]) -> Dict[str, float]:
    """Recall@k / Precision@k / MRR / Hit.

    retrieved_ids : 파이프라인이 실제로 인용하거나 검색한 chunk_id (순서 중요)
    gold_ids      : 골드셋이 지정한 정답 chunk_id
    """
    if not gold_ids:
        return {}
    gold = set(gold_ids)
    hits = [i for i, cid in enumerate(retrieved_ids) if cid in gold]

    recall = len({retrieved_ids[i] for i in hits}) / len(gold)
    precision = (len(hits) / len(retrieved_ids)) if retrieved_ids else 0.0
    mrr = (1.0 / (hits[0] + 1)) if hits else 0.0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "mrr": round(mrr, 4),
        "hit": 1.0 if hits else 0.0,
    }


def source_accuracy(predicted: Sequence[str], expected: Sequence[str]) -> Dict[str, float]:
    """질의 분석이 고른 소스가 정답 소스를 포함하는가.

    소스 필터가 틀리면 그 뒤 검색이 아무리 좋아도 정답을 못 찾는다.
    파이프라인 초반 실패를 잡아내는 지표다.
    """
    if not expected:
        return {}
    p, e = set(predicted or []), set(expected)
    return {
        "source_hit": 1.0 if (p & e) else 0.0,           # 정답 소스를 하나라도 포함
        "source_exact": 1.0 if p == e else 0.0,
        "source_extra": len(p - e),                       # 불필요하게 넓힌 개수
    }


def routing_accuracy(pred_type: str, gold_type: Optional[str]) -> Dict[str, float]:
    """질문유형 분류가 맞았나. 유형이 틀리면 검색 폭·라우팅이 통째로 어긋난다."""
    if not gold_type:
        return {}
    return {"질문유형_correct": 1.0 if pred_type == gold_type else 0.0}


def cost_metrics(result: Dict[str, Any], elapsed_sec: float) -> Dict[str, float]:
    u = result.get("usage") or {}
    return {
        "latency_sec": round(elapsed_sec, 2),
        "llm_calls": u.get("calls", result.get("llm_calls", 0)),
        "in_tokens": u.get("in_tokens", 0),
        "out_tokens": u.get("out_tokens", 0),
    }


def pipeline_health(result: Dict[str, Any]) -> Dict[str, float]:
    """파이프라인이 정상 경로로 갔는지. 실패 경로 비율을 본다."""
    pairs = result.get("qa_pairs", []) or []
    n = len(pairs) or 1
    return {
        "n_subquestions": len(pairs),
        "evidence_rate": round(sum(1 for p in pairs if p.get("evidence_ok")) / n, 3),
        "verify_pass_rate": round(sum(1 for p in pairs if p.get("verified")) / n, 3),
        "web_fallback_rate": round(sum(1 for p in pairs if p.get("web_used")) / n, 3),
        "n_citations": len(result.get("citations", []) or []),
        "rounds": (result.get("round_no", 0) or 0) + 1,
    }


# ==========================================================================
# 2층: LLM 심판
# ==========================================================================
class FactCheck(BaseModel):
    total_facts: int = Field(description="기준 문장에서 뽑은 사실 개수")
    matched_facts: int = Field(description="그중 대상 텍스트에 존재하는 개수")
    missing: List[str] = Field(default_factory=list, description="빠진 사실")


_CORRECTNESS_SYS = """너는 답변 정확도 채점기다.

[기준답변] 을 사실 단위로 쪼갠 뒤, 각 사실이 [생성답변] 에 들어 있는지 확인한다.
- 표현이 달라도 의미가 같으면 존재하는 것으로 본다.
- 숫자·비율·기한은 값이 정확히 같아야 존재로 인정한다.
- total_facts 는 기준답변에서 뽑은 사실의 총 개수,
  matched_facts 는 그중 생성답변에 있는 개수다."""

_FAITHFULNESS_SYS = """너는 답변 충실도(환각) 채점기다.

[생성답변] 을 사실 단위로 쪼갠 뒤, 각 사실이 [근거] 에 뒷받침되는지 확인한다.
- total_facts 는 생성답변에서 뽑은 사실의 총 개수,
  matched_facts 는 그중 근거로 뒷받침되는 개수다.
- "확인되지 않습니다" 같은 유보 표현은 사실로 세지 않는다.
- 일반 상식이라도 근거에 없으면 뒷받침되지 않는 것으로 본다."""


def llm_correctness(generated: str, reference: str) -> Dict[str, float]:
    """기준답변의 사실이 생성답변에 얼마나 담겼나 (재현율 성격)."""
    if not reference:
        return {}
    try:
        r = llm.structured(
            _CORRECTNESS_SYS,
            f"[기준답변]\n{reference}\n\n[생성답변]\n{generated}",
            FactCheck, model=config.MODEL_VERIFY, max_tokens=800, stage="eval:correct")
        total = max(r.total_facts, 1)
        return {"correctness": round(r.matched_facts / total, 4),
                "correctness_missing": len(r.missing)}
    except Exception as e:
        return {"correctness_error": str(e)[:80]}


def llm_faithfulness(generated: str, contexts: List[str]) -> Dict[str, float]:
    """생성답변의 사실이 근거로 뒷받침되나 (환각의 역수)."""
    if not contexts or not generated:
        return {}
    ctx = "\n\n".join(contexts[:12])
    try:
        r = llm.structured(
            _FAITHFULNESS_SYS,
            f"[근거]\n{ctx}\n\n[생성답변]\n{generated}",
            FactCheck, model=config.MODEL_VERIFY, max_tokens=800, stage="eval:faith")
        total = max(r.total_facts, 1)
        return {"faithfulness": round(r.matched_facts / total, 4)}
    except Exception as e:
        return {"faithfulness_error": str(e)[:80]}


# ==========================================================================
# 3층: 임베딩 기반
# ==========================================================================
class GeneratedQuestions(BaseModel):
    questions: List[str] = Field(description="이 답변이 답이 될 만한 질문 3개")


def response_relevancy(question: str, generated: str) -> Dict[str, float]:
    """답변에서 역으로 질문을 만들어 원 질문과의 유사도를 잰다.

    답변이 주제만 맞고 딴소리를 하면 역질문이 원 질문과 멀어진다.
    """
    from llm import embedder

    emb = embedder()
    if not emb.enabled or not generated:
        return {}
    try:
        r = llm.structured(
            "주어진 답변을 보고, 이 답변이 정확히 답이 될 만한 질문 3개를 한국어로 만든다.",
            f"[답변]\n{generated[:2000]}",
            GeneratedQuestions, model=config.MODEL_VERIFY,
            max_tokens=400, stage="eval:relevancy")

        qv = emb.embed_one(question)
        if qv is None:
            return {}
        sims = []
        for q in r.questions[:3]:
            v = emb.embed_one(q)
            if v:
                sims.append(_cosine(qv, v))
        return {"response_relevancy": round(sum(sims) / len(sims), 4)} if sims else {}
    except Exception as e:
        return {"relevancy_error": str(e)[:80]}


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ==========================================================================
def aggregate_scores(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """케이스별 지표를 평균낸다. 숫자가 아닌 키는 건너뛴다."""
    if not rows:
        return {}
    keys: List[str] = []
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and k not in keys:
                keys.append(k)
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
    out["n_cases"] = len(rows)
    return out
