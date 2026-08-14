"""답변 생성 노드 — 도메인 전문 에이전트 4종이 여기서 갈린다.

■ 선행모델 대응
  FinSight 의 `nodes/answer_generator.py` 자리다. 핵심 기법 두 개를 그대로 가져왔다.

  1) **답변과 인용을 한 번의 호출로 동시에 받는다.**
     답을 만든 뒤 나중에 출처를 역추적하면 정확도가 크게 떨어진다.
     구조화 출력으로 {main_answer, citations[]} 를 함께 받는다.

  2) **근거를 번호로 넘긴다.** [1] [2] 형태로 본문에 인용 번호를 박게 하고,
     번호 ↔ 청크 매핑을 코드가 들고 있다가 최종 출처 목록을 만든다.

■ 우리 도메인의 추가 설계: 전문가별 프롬프트
  라우터/분해기가 정한 `expert`(제도/세제/절차/상품)에 따라 시스템 프롬프트를 바꾼다.
  같은 근거를 줘도 전문가마다 답변 형태가 달라야 하기 때문이다.
    · 절차 전문가 → 번호 매긴 순서 + 필요서류 목록
    · 세제 전문가 → 과세 시점 구분, 연금수령 vs 연금외수령 세율 차이
    · 상품 전문가 → 표 형식 + 투자권유 아님 고지 + 낮은 신뢰도 명시

■ 소스 신뢰도 반영 (요구사항)
  근거를 프롬프트에 넣을 때 doc 문서(제도/세제/절차)를 **먼저** 배치한다.
  LLM 은 앞쪽 컨텍스트에 더 큰 가중을 두는 경향이 있어서, 순서 자체가 신뢰도 신호가 된다.
  프롬프트 규칙(GROUNDING_RULES 4항)과 이중으로 건다.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

import config
from index import format_chunk_for_llm, to_citation
from llm import llm
from prompt import prompts
from state import RAGState
from utils import Timer, step, truncate


class CitationRef(BaseModel):
    n: int = Field(description="근거 번호 (프롬프트에 표시된 [n])")
    quote: str = Field(default="", description="인용한 문장 일부")


class GeneratedAnswer(BaseModel):
    main_answer: str = Field(description="마크다운 형식 답변 본문")
    citations: List[CitationRef] = Field(default_factory=list)
    evidence_sufficient: bool = Field(
        default=True, description="근거만으로 답할 수 있었으면 true")


# --------------------------------------------------------------------------
def _order_by_trust(docs: List[dict]) -> List[dict]:
    """규정문서를 앞에, 투자설명서를 뒤에 배치한다.

    LLM 은 앞쪽 컨텍스트에 더 큰 가중을 둔다. 프롬프트 규칙(GROUNDING_RULES 3번)과
    배치 순서, 두 가지로 신뢰도 우선순위를 심는다.
    """
    return sorted(docs, key=lambda d: config.TIER_TRUST.get(d.get("tier", "doc"), 1.0),
                  reverse=True)


def _build_context(docs: List[dict]) -> tuple[str, List[dict]]:
    """근거 블록 문자열과 번호↔청크 매핑을 만든다."""
    ordered = _order_by_trust(docs)
    lines = [format_chunk_for_llm(c, idx=i + 1) for i, c in enumerate(ordered)]
    return "\n\n".join(lines), ordered


def generate_answer(state: RAGState) -> dict:
    """서브질문 하나에 대한 답을 만든다."""
    question = state["question"]
    original = state.get("original_question") or question
    docs: List[dict] = state.get("documents", []) or []
    qid = state.get("qid", "")

    # 답변 프롬프트는 **근거의 성격**이 고른다. 고정 페르소나는 없다.
    # 서브질문이 추천이거나 검색 결과가 투자설명서 위주면 상품 안내 프롬프트를 쓴다.
    prospectus = sum(1 for d in docs if d.get("tier") == "prospectus")
    is_rec = bool(state.get("is_recommendation")) or (
        bool(docs) and prospectus > len(docs) / 2)
    mode = "recommend" if is_rec else "general"

    if not docs:
        return {
            "answer": "제공된 자료로는 확인되지 않습니다.",
            "citations": [],
            "evidence_ok": False,
            "trace": step("generate_answer", qid=qid, summary="근거 0건 → 미확인 응답"),
        }

    context, ordered = _build_context(docs)
    system = (prompts.answer_recommend_system if is_rec
              else prompts.answer_system)

    user = (
        f"[사용자 원 질문]\n{original}\n\n"
        f"[지금 답해야 할 질문]\n{question}\n\n"
        f"[근거]\n{context}"
    )

    with Timer() as t:
        res = llm.structured(system, user, GeneratedAnswer,
                             model=config.MODEL_ANSWER, max_tokens=2000,
                             stage=f"answer:{mode}")

    # 인용 번호 → 실제 청크
    cites: List[dict] = []
    for ref in res.citations:
        i = ref.n - 1
        if 0 <= i < len(ordered):
            c = to_citation(ordered[i])
            c["n"] = ref.n
            c["quote"] = truncate(ref.quote, 120)
            cites.append(c)

    # 인용을 하나도 안 달았는데 답은 했다면 → 상위 3건을 근거로 붙인다.
    # (모델이 인용 표기를 빠뜨리는 일이 잦은데, 출처 없는 답변을 내보내면 안 된다)
    if not cites and res.evidence_sufficient:
        cites = [dict(to_citation(c), n=i + 1) for i, c in enumerate(ordered[:3])]

    return {
        "answer": res.main_answer,
        "citations": cites,
        "evidence_ok": bool(res.evidence_sufficient and cites),
        "llm_calls": 1,
        "trace": step("generate_answer", qid=qid,
                      summary=f"[{mode}] {len(res.main_answer)}자, 인용 {len(cites)}건, "
                              f"근거충분={res.evidence_sufficient}",
                      ms=t.ms),
    }


# ==========================================================================
# 웹 검색 결과로 답변 (폴백 경로)
# ==========================================================================
class WebAnswer(BaseModel):
    main_answer: str
    used_sources: List[int] = Field(default_factory=list, description="사용한 결과 번호")


def generate_web_answer(state: RAGState) -> dict:
    """내부 문서 실패 후, 네이버 웹 검색 결과로 답한다."""
    question = state["question"]
    results = state.get("documents", []) or []      # web_searcher 가 채워 둔다
    qid = state.get("qid", "")

    if not results:
        return {
            "answer": "제공된 자료와 웹 검색 모두에서 근거를 찾지 못했습니다.",
            "citations": [], "evidence_ok": False,
            "trace": step("generate_web_answer", qid=qid, summary="웹 결과 0건"),
        }

    block = "\n\n".join(
        f"[{i + 1}] {r.get('section') or ''}\n{r['text']}\n(출처: {r.get('url', '')})"
        for i, r in enumerate(results)
    )

    with Timer() as t:
        res = llm.structured(prompts.web_answer_system,
                             f"질문: {question}\n\n[웹 검색 결과]\n{block}",
                             WebAnswer, model=config.MODEL_ANSWER,
                             max_tokens=1500, stage="answer:web")

    cites = [
        {"n": i + 1, "chunk_id": f"web-{i+1}", "source": "웹검색",
         "url": r.get("url", ""), "snippet": truncate(r["text"], 160),
         "doc_id": r.get("title", "")}
        for i, r in enumerate(results)
    ]

    return {
        "answer": res.main_answer,
        "citations": cites,
        "evidence_ok": False,           # 웹 근거는 '내부 문서 근거 있음' 으로 치지 않는다
        "web_used": True,
        "llm_calls": 1,
        "trace": step("generate_web_answer", qid=qid,
                      summary=f"웹 {len(results)}건으로 생성", ms=t.ms),
    }
