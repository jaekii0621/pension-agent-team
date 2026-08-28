"""최상위 그래프 — 질문 하나가 지나가는 전체 경로.

■ 선행모델 대응
  FinSight 의 `workflows/e2e.py` + `workflows/repeater.py`(RRR) 를 합친 자리다.
  선행모델은 RRR 라운드를 for 문 대신 `decomposer_node_1/2/3`, `rag_1/2/3_time`,
  `aggregate1/2/3` 로 **손으로 3벌 복제**했다(LangGraph 가 정적 그래프라서).

  우리는 **순환 엣지**로 처리한다.
      decompose → rag_worker(병렬) → aggregate → [부족?] → decompose_residual → ┐
        ↑───────────────────────────────────────────────────────────────────────┘
  노드 수가 1/3 로 줄고 라운드 수를 config 로 바꿀 수 있다.
  (LangGraph 는 순환을 허용한다. 무한루프는 should_repeat 의 라운드/예산 한도로 막는다.)

■ 전체 흐름
    START
     → check_safety            안전·범위 검사
     → [safety_gate]           blocked → 안내문 → END
     → analyze_query           질문유형·펀드명·상품관여도 추출
     → resolve_slots           답이 갈리는 값(사유) 확정 시도
     → [clarify_gate]          확정 실패 → 역질문 → END (정상 종료)
     → route_query             경로 + 도메인 전문가 배정
     → [route_after_router]    out_of_scope → 안내문 / calc → 계산기 / rag → 분해
     → (calc_agent)            계산이 필요하면 먼저 실행
     → decompose               질문 분리 에이전트
     → [send_sub_questions]    ★ Send 로 서브질문 병렬 fan-out
     → rag_worker × N          각자 rag_core 서브그래프 실행
     → aggregate               결과 취합 (리듀서가 이미 합쳐 둔 것을 정리)
     → check_sufficiency       원 질문에 충분히 답했나
     → [should_repeat]         부족 → decompose_residual → 다시 fan-out (최대 MAX_ROUNDS)
     → strategy_agent          필요 시 전략 수립
     → combine_answer          최종 합성 + 인용 번호 재매핑 + 출처
     → END
"""

from __future__ import annotations

from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

import config
import edges
import nodes
import state
from utils import step, truncate

from .rag_core import rag_core


# ==========================================================================
# 서브그래프 어댑터
#   선행모델의 `map_fields_in_node` 와 같은 역할이다.
#   rag_core 는 RAGState(키 이름: answer, citations)를 쓰고
#   최상위는 OverallState(키 이름: qa_pairs, citations)를 쓴다.
#   State 스키마가 다른 두 그래프를 잇는 **어댑터**가 반드시 필요하다.
# ==========================================================================
def rag_worker(payload: Dict[str, Any]) -> dict:
    """서브질문 1건을 rag_core 로 처리하고 결과를 OverallState 형태로 변환한다."""
    qid = payload.get("qid", "")
    sub_q = payload.get("question", "")

    res = rag_core.invoke(payload, config={"recursion_limit": 40})

    verdict = res.get("verdict", {}) or {}
    qa = {
        "qid": qid,
        "question": sub_q,
        "is_recommendation": payload.get("is_recommendation", False),
        "product_affinity": (payload.get("filters") or {}).get("product_affinity"),
        "사유": (payload.get("filters") or {}).get("사유") or [],
        "answer": res.get("answer", ""),
        "citations": res.get("citations", []),
        "evidence_sufficient": res.get("evidence_sufficient"),
        # 답변이 말했는데 근거에 없는 값 — 환각 후보 (2026-08-23)
        "unsupported_values": res.get("unsupported_values") or [],
        # 이 서브질문이 실제로 본 근거 청크. combine_answer 가 **최종 답변**을
        # 같은 근거 집합과 대조하려면 필요하다 (2026-08-28).
        # 결과 파일에는 안 나간다 — run_queryset 이 qa_pairs 에서 고른 키만 쓴다.
        "documents": res.get("documents") or [],
        "evidence_ok": res.get("evidence_ok", False),
        "verified": verdict.get("pass", True),
        "score": verdict.get("score"),
    }

    # ⚠️ citations 를 여기서 올려보내지 않는다 (2026-08-18 수정)
    #
    #   예전 코드:  "citations": res.get("citations", []),
    #
    #   OverallState.citations 는 **리듀서**(state.dedup_citations)가 붙은 필드다.
    #   리듀서는 덮어쓰지 않고 **합친다.** 그래서 이 줄이 있으면
    #
    #       ① rag_worker 가 서브답변의 *지역 번호* 인용을 올려보내고
    #       ② 나중에 combine_answer 가 *전역 번호* 인용(global_cites)을 올려보내는데
    #       ③ dedup 은 chunk_id 기준으로 **먼저 온 것을 남기므로**
    #          최종 state["citations"] 에 지역 번호가 살아남는다
    #
    #   실측(78문항)에서 번호가 중복된 문항이 8건 나왔다.
    #       P-A3  citations n = [1, 2, 3, 1, 2, 3]   ← 서브질문 2개의 지역번호
    #       D-E4  citations n = [1, 2, 3, 4, 1, 2]
    #       D-F3  citations n = [1, 1]
    #
    #   사용자가 보는 답변 하단 출처 목록은 combiner 가 global_cites 로 따로 만들어서
    #   멀쩡했지만, **채점자가 읽는 산출물의 citations 필드가 오염됐다.**
    #   자동 채점을 이 필드로 하면 인용 정확도가 통째로 틀어진다.
    #
    #   서브답변의 인용은 qa_pairs[*]["citations"] 안에 이미 들어 있고
    #   combiner 의 _renumber 가 그것을 읽는다. 여기서 또 올려보낼 이유가 없다.
    return {
        "qa_pairs": [qa],
        "llm_calls": res.get("llm_calls", 0),
        "trace": res.get("trace", []),
    }


def aggregate(state_: state.OverallState) -> dict:
    """라운드 결과 정리.

    실제 병합은 이미 리듀서가 했다(qa_pairs 의 operator.add).
    이 노드는 '라운드 경계'를 만들고 요약을 남기는 역할이다.
    LangGraph 에서 fan-in 지점에 노드가 하나 있어야 다음 분기를 걸 수 있다.
    """
    pairs = state_.get("qa_pairs", []) or []
    ok = sum(1 for p in pairs if p.get("evidence_ok"))
    return {
        "trace": step("aggregate",
                      summary=f"라운드{state_.get('round_no', 0)} 취합 — "
                              f"QA {len(pairs)}건 (근거확보 {ok})"),
    }


# ==========================================================================
# 그래프 조립
# ==========================================================================
# fmt: off
graph = StateGraph(state.OverallState)

graph.add_node("analyze_query", nodes.analyze_query)
graph.add_node("route_query", nodes.route_query)
graph.add_node("decompose", nodes.decompose_question)
graph.add_node("decompose_residual", nodes.decompose_residual)
graph.add_node("rag_worker", rag_worker)
graph.add_node("aggregate", aggregate)
graph.add_node("check_sufficiency", nodes.check_sufficiency)
graph.add_node("combine_answer", nodes.combine_answer)
graph.add_node("out_of_scope", nodes.out_of_scope_answer)
graph.add_node("clarify", nodes.clarify_answer)

if config.WORKFLOW_SETTINGS["use_calculator"]:
    graph.add_node("calc_agent", nodes.calc_agent)
if config.WORKFLOW_SETTINGS["use_strategy"]:
    graph.add_node("strategy_agent", nodes.strategy_agent)

# ── 1. 진입: 안전성 검사 ──────────────────────────────────────────────
if config.WORKFLOW_SETTINGS["check_safety"]:
    graph.add_node("check_safety", nodes.check_safety)
    graph.add_edge(START, "check_safety")
    graph.add_conditional_edges(
        "check_safety",
        edges.safety_gate,
        {"proceed": "analyze_query", "blocked": "out_of_scope"},
    )
else:
    graph.add_edge(START, "analyze_query")

graph.add_edge("out_of_scope", END)
graph.add_edge("clarify", END)

# ── 2. 질의 분석 → 슬롯 해소 → 라우팅 ─────────────────────────────────
#
# resolve_slots 는 "사유처럼 답이 갈리는 값" 을 확정한다. 확정하지 못했는데 그 값이
# 꼭 필요하면 검색으로 내려보내지 않고 **되묻고 끝낸다**(정상 종료).
# 추측해서 진행하면 엉뚱한 사유의 서류를 근거로 그럴듯한 오답이 나가는데, 그건
# 사용자가 틀린 줄 모르는 종류의 실패라 가장 비싸다.
if config.WORKFLOW_SETTINGS.get("resolve_slots", True):
    graph.add_node("resolve_slots", nodes.resolve_slots)
    graph.add_edge("analyze_query", "resolve_slots")
    graph.add_conditional_edges(
        "resolve_slots",
        edges.clarify_gate,
        {"proceed": "route_query", "clarify": "clarify"},
    )
else:
    graph.add_edge("analyze_query", "route_query")

_calc_target = "calc_agent" if config.WORKFLOW_SETTINGS["use_calculator"] else "decompose"
graph.add_conditional_edges(
    "route_query",
    edges.route_after_router,
    {"rag": "decompose", "calc": _calc_target, "out_of_scope": "out_of_scope"},
)
if config.WORKFLOW_SETTINGS["use_calculator"]:
    # 계산 후에도 문서 검색을 한다 — 계산 결과에는 문서 근거가 없기 때문이다.
    graph.add_edge("calc_agent", "decompose")

# ── 3. 분해 → 병렬 실행 → 취합 (RRR 순환) ─────────────────────────────
graph.add_conditional_edges("decompose", edges.send_sub_questions,
                            ["rag_worker", "aggregate"])
graph.add_conditional_edges("decompose_residual", edges.send_sub_questions,
                            ["rag_worker", "aggregate"])
graph.add_edge("rag_worker", "aggregate")
graph.add_edge("aggregate", "check_sufficiency")

_after_rounds = "strategy_agent" if config.WORKFLOW_SETTINGS["use_strategy"] else "combine_answer"
graph.add_conditional_edges(
    "check_sufficiency",
    edges.should_repeat,
    {"repeat": "decompose_residual", "finish": _after_rounds},
)

# ── 4. 전략 → 합성 ────────────────────────────────────────────────────
if config.WORKFLOW_SETTINGS["use_strategy"]:
    graph.add_edge("strategy_agent", "combine_answer")
graph.add_edge("combine_answer", END)
# fmt: on

pension_agent = graph.compile()


# ==========================================================================
# 실행 헬퍼
# ==========================================================================
def run(question: str, *, user_profile: Dict[str, Any] | None = None,
        session_id: str = "cli", qkey: str | None = None,
        reset_usage: bool = True, follow_up: bool = True) -> dict:
    """질문 하나를 끝까지 처리한다.

    반환: OverallState (final_answer, citations, trace, llm_calls, usage …)

    ■ 토큰 계측
      `qkey` 로 이 질문에 속한 LLM 호출을 표시해 둔다(`llm.mark_question`).
      그러면 `result["usage"]` 가 **이 질문 1건이 쓴 토큰**만 담는다.
      여러 질문을 연속 실행할 때(eval)는 `reset_usage=False` 로 두고
      `llm.usage_by_question()` 으로 질문별 내역을 한 번에 꺼낼 수 있다.

    ■ 되묻기 이어받기
      직전 턴이 되묻기로 끝난 **같은 session_id** 에서 다음 입력이 그 답으로
      보이면, 원 질문에 붙여 다시 태운다(`workflows/followup.py`).
      끄려면 `follow_up=False` 또는 `PENSION_CLARIFY_FOLLOWUP=false`.
    """
    from llm import llm

    from . import followup

    key = qkey or session_id
    if reset_usage:
        llm.reset_usage()
    llm.mark_question(key)

    # ── 되묻기 이어받기 (2026-08-27) ────────────────────────────────
    #
    # 직전 턴이 되묻기로 끝났고 이번 입력이 그 답으로 보이면, 원 질문에 붙여
    # 하나의 자족적인 문장으로 만든다. 파이프라인은 손대지 않는다.
    #
    #   1턴 "의료비가 연봉의 12.5%가 안 되는데 중도인출 되나요?"
    #       → "DB / DC / 개인형IRP / 연금저축 중 어느 쪽인가요?"
    #   2턴 "IRP요"
    #       → "IRP 기준으로 답해 주세요. 의료비가 연봉의 12.5%가 안 되는데…"
    #
    # 답이 아니라 새 질문이면 대기 중이던 되묻기는 버린다(주제가 바뀐 것).
    seed_trace: list = []
    if config.CLARIFY_FOLLOWUP and follow_up:
        pend = followup.take(session_id)
        if pend and followup.looks_like_reply(question):
            merged = followup.merge(pend["question"], question)
            seed_trace = step("clarify_followup",
                              summary=f"되묻기 답변을 원 질문에 합침 — {truncate(merged, 60)}")
            question = merged

    initial: Dict[str, Any] = {
        "session_id": session_id,
        "question": question,
        "trace": seed_trace,
        "user_profile": user_profile or {},
        "round_no": 0,
        "qa_pairs": [],
        "citations": [],
        "llm_calls": 0,
        "needs_clarification": False,
        "slots": {},
    }
    result = pension_agent.invoke(initial, config={"recursion_limit": 60})
    result["usage"] = llm.usage_for(key)

    # 이번 턴이 되묻기로 끝났으면 다음 입력을 기다린다.
    if config.CLARIFY_FOLLOWUP and follow_up and result.get("needs_clarification"):
        followup.remember(session_id, question, result.get("clarification") or "")

    return result
