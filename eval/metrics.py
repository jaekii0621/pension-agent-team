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

import re
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


def gold_ids_for(case: Dict[str, Any]) -> List[str]:
    """골드셋 1건에서 **정답 chunk_id 목록**을 꺼낸다.

    ■ 확정 형식 (골드셋_형식_확정_0820 §1) — evidence 는 **딕셔너리 하나**다

        "evidence": {
            "tier": "doc",
            "chunk_type": ["tax"],
            "doc_id": ["doc52"],
            "chunk_id": ["doc52#p4#c003"]
        }

    ■ 예전 코드가 왜 항상 빈 리스트를 돌려줬나 (2026-08-20 수정)

      `for e in case["evidence"]` 로 **리스트처럼 순회**하고 있었다.
      딕셔너리를 for 로 돌면 나오는 것은 **키 문자열**이라
      ("tier", "chunk_type", "doc_id", "chunk_id"), `isinstance(e, dict)` 가
      전부 False 가 되어 결과가 언제나 `[]` 였다.

      그 결과 **evidence 를 아무리 정확히 채워도 근거 축이 통째로 죽었다.**
      retrieval_metrics 가 gold 를 못 받아 첫 줄에서 빈 dict 를 돌려주기 때문이다.
      (재희님 재현: gold_ids_for(D-D1 레코드) → [])

    ■ 세 가지 모양을 모두 받는다
        1) evidence 딕셔너리          ← 확정 형식
        2) evidence 리스트[{chunk_id}] ← 초안 시절 형태 (호환)
        3) gold_chunk_ids 평면 리스트  ← 그 이전 형태 (호환)

      `evidence: null` (clarify·refuse_ok) 는 빈 리스트를 돌려준다 — 근거 축을
      채점하지 않는다는 뜻이고, 그건 형식이 의도한 동작이다.
    """
    direct = case.get("gold_chunk_ids") or []
    if direct:
        return [str(x) for x in direct if x]

    ev = case.get("evidence")
    if isinstance(ev, dict):                       # ① 확정 형식
        return [str(x) for x in (ev.get("chunk_id") or []) if x]
    if isinstance(ev, list):                       # ② 초안 형태
        return [str(e.get("chunk_id")) for e in ev
                if isinstance(e, dict) and e.get("chunk_id")]
    return []


def gold_doc_ids_for(case: Dict[str, Any]) -> List[str]:
    """정답 **문서** id 목록. 명시가 없으면 chunk_id 앞부분에서 유추한다."""
    direct = case.get("gold_doc_ids") or []
    if direct:
        return [str(x) for x in direct if x]

    ev = case.get("evidence")
    if isinstance(ev, dict) and ev.get("doc_id"):  # 확정 형식은 doc_id 를 직접 준다
        return [str(x) for x in ev["doc_id"] if x]

    return [d for d in (_doc_of(cid) for cid in gold_ids_for(case)) if d]


def gold_chunk_types_for(case: Dict[str, Any]) -> List[str]:
    """정답 chunk_type 목록 (근거 축 1점 판정용)."""
    ev = case.get("evidence")
    if isinstance(ev, dict):
        return [str(x) for x in (ev.get("chunk_type") or []) if x]
    return []


def gold_tier_for(case: Dict[str, Any]) -> str:
    """정답 tier — prospectus | doc | both | "" (근거 축 0/1점 판정용)."""
    ev = case.get("evidence")
    return str(ev.get("tier") or "") if isinstance(ev, dict) else ""


# chunk_id 구분자. 우리 규약은 '#' 이다 — doc52#p4#c003
#   예전 코드는 '-' 로 잘랐다. 그러면 'doc52#p4#c003' 이 통째로 남아서
#   doc 수준 폴백(청크가 안 맞을 때 문서 단위로 봐 주는 경로)이 같이 죽었다.
#   과거 형식('doc57-p3-s2')도 남아 있을 수 있어 둘 다 받는다.
_DOC_SEP = ("#", "-")


def _doc_of(chunk_id: str) -> str:
    """chunk_id → doc_id.

        'doc52#p4#c003'          → 'doc52'
        'R2_KR5118420006#p30#c2' → 'R2_KR5118420006'
        'doc57-p3-s2'            → 'doc57'      (과거 형식 호환)

    ⚠️ '-' 를 먼저 자르면 안 된다. 투자설명서 doc_id 에는 '_' 는 있어도 '-' 가
       없지만, 앞으로 하이픈이 들어간 id 가 생기면 '#' 이 우선이어야 한다.
    """
    cid = (chunk_id or "").strip()
    if not cid:
        return ""
    for sep in _DOC_SEP:
        if sep in cid:
            return cid.split(sep, 1)[0]
    return cid


def retrieval_metrics_doc(retrieved_ids: Sequence[str],
                          gold_doc_ids: Sequence[str]) -> Dict[str, float]:
    """**문서 수준** Recall/Precision/MRR — chunk_id 폴백용.

    ■ 왜 필요한가

      청크 단위 정답은 만들기가 까다롭다. 같은 내용이 인접한 두 청크에 걸쳐 있으면
      어느 쪽을 정답으로 적어도 맞는데, chunk_id 하나만 적으면 다른 쪽을 가져온
      실행이 **틀린 것으로 채점된다.** 검색은 사실 성공했는데도 그렇다.

      문서 수준으로 한 단계 올려 보면 그 잡음이 사라진다.
        · 청크 recall 은 낮은데 문서 recall 이 높다 → 청킹/랭킹 미세 문제
        · 둘 다 낮다                               → 진짜 검색 실패

      둘을 **같이** 보라고 별도 지표로 둔다. 접미사 `_doc` 로 구분한다.
    """
    if not gold_doc_ids:
        return {}
    gold = set(gold_doc_ids)
    docs = [_doc_of(cid) for cid in retrieved_ids]
    hits = [i for i, d in enumerate(docs) if d in gold]

    return {
        "recall_doc": round(len({docs[i] for i in hits}) / len(gold), 4),
        "precision_doc": round((len(hits) / len(docs)) if docs else 0.0, 4),
        "mrr_doc": round((1.0 / (hits[0] + 1)) if hits else 0.0, 4),
        "hit_doc": 1.0 if hits else 0.0,
    }


_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def citation_integrity(answer: str, citations: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """본문의 [n] 이 전부 출처 목록 안에 있는가 — **LLM 없이** 재는 근거표시 지표.

    ■ 왜 별도 지표인가

      대회 평가에 "근거문서 표시 필수" 가 있다. 답이 맞아도 [3] 이 엉뚱한 문서를
      가리키면 감점이다. 그런데 기존 지표(recall/precision)는 **어떤 청크를
      가져왔는가**만 보고, **본문 번호가 그 청크를 제대로 가리키는가**는 안 본다.

      실측(78문항): 인용이 달린 문항의 28% 에서 본문 번호가 목록에 없었다.
      recall 은 멀쩡한데 이 지표만 무너지는 구간이다 — 그래서 따로 잰다.

    반환
        cite_dangling   : 목록에 없는 번호의 개수 (0 이어야 정상)
        cite_integrity  : 본문 번호 중 유효한 것의 비율 (1.0 이어야 정상)
        cite_unused     : 목록에 있는데 본문이 한 번도 안 쓴 출처 수
    """
    body = (answer or "").split("### 출처")[0]
    used = {int(x) for x in _CITE_RE.findall(body)}
    listed = {c.get("n") for c in (citations or []) if c.get("n") is not None}
    if not used and not listed:
        return {}
    dangling = used - listed
    return {
        "cite_dangling": float(len(dangling)),
        "cite_integrity": round(len(used - dangling) / len(used), 4) if used else 0.0,
        "cite_unused": float(len(listed - used)),
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
# 축별 점수 — 자동 2축 / 사람 2축 / 차단 게이트
# ==========================================================================
def axis_scores(row: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """채점 결과를 **축별로** 정리한다.

    ■ 왜 축을 나누나

      한 문항의 점수를 숫자 하나로 뭉치면 어디를 고쳐야 할지 알 수 없다.
      검색이 실패한 것과, 검색은 됐는데 인용을 잘못 단 것과, 답이 무례한 것은
      전혀 다른 문제인데 평균 하나에 섞여 버린다.

      그래서 4축 + 게이트로 나눈다.

        auto.retrieval   자동. 정답 청크를 가져왔는가 (recall/mrr)
        auto.grounding   자동. 본문 번호가 실제 출처를 가리키는가 (cite_integrity)
        human.usefulness 사람. 질문에 실제로 답이 됐는가        ← 빈칸으로 둔다
        human.safety     사람. 단정·권유·어조가 적절한가        ← 빈칸으로 둔다
        gate.blocked     게이트. 차단됐는가 / 어느 범주로

      **자동 2축은 반드시 채워지고, 사람 2축은 None 으로 비워 둔다.**
      비워 두는 것이 중요하다 — LLM 심판 점수를 사람 점수 자리에 넣으면
      "사람이 확인했다" 와 "모델이 추측했다" 가 구분되지 않는다.

      gate 는 점수가 아니라 **사실 기록**이다. 정당한 거절(개인정보)과
      오차단(위법조력 오판)은 같은 '차단' 이지만 채점이 정반대여야 하므로,
      범주를 그대로 남겨 두고 판정은 채점자에게 맡긴다.
    """
    def _pick(*keys):
        vals = [row[k] for k in keys if isinstance(row.get(k), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "auto": {
            "retrieval": _pick("recall", "mrr"),
            "retrieval_doc": _pick("recall_doc", "mrr_doc"),
            "grounding": row.get("cite_integrity"),
        },
        "human": {
            "usefulness": None,      # 사람이 채운다 (0~1)
            "safety": None,          # 사람이 채운다 (0~1)
        },
        "gate": {
            "blocked": not result.get("safe", True),
            "category": result.get("safety_category", ""),
            "caution": result.get("safety_caution", ""),
            "clarified": bool(result.get("needs_clarification")),
        },
    }


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
