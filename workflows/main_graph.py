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
     → analyze_query           소스·주제·질문유형 추출 (검색 필터)
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
from utils import step

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
        "scope": (payload.get("filters") or {}).get("scope"),
        "answer": res.get("answer", ""),
        "citations": res.get("citations", []),
        "evidence_ok": res.get("evidence_ok", False),
        "verified": verdict.get("pass", True),
        "score": verdict.get("score"),
        "web_used": res.get("web_used", False),
    }

    return {
        "qa_pairs": [qa],
        "citations": res.get("citations", []),
        "web_used": res.get("web_used", False),
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
    web = sum(1 for p in pairs if p.get("web_used"))
    return {
        "trace": step("aggregate",
                      summary=f"라운드{state_.get('round_no', 0)} 취합 — "
                              f"QA {len(pairs)}건 (근거확보 {ok}, 웹폴백 {web})"),
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

# ── 2. 질의 분석 → 라우팅 ─────────────────────────────────────────────
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
        reset_usage: bool = True) -> dict:
    """질문 하나를 끝까지 처리한다.

    반환: OverallState (final_answer, citations, trace, llm_calls, usage …)

    ■ 토큰 계측
      `qkey` 로 이 질문에 속한 LLM 호출을 표시해 둔다(`llm.mark_question`).
      그러면 `result["usage"]` 가 **이 질문 1건이 쓴 토큰**만 담는다.
      여러 질문을 연속 실행할 때(eval)는 `reset_usage=False` 로 두고
      `llm.usage_by_question()` 으로 질문별 내역을 한 번에 꺼낼 수 있다.
    """
    from llm import llm

    key = qkey or session_id
    if reset_usage:
        llm.reset_usage()
    llm.mark_question(key)

    initial: Dict[str, Any] = {
        "session_id": session_id,
        "question": question,
        "user_profile": user_profile or {},
        "round_no": 0,
        "qa_pairs": [],
        "citations": [],
        "trace": [],
        "llm_calls": 0,
    }
    result = pension_agent.invoke(initial, config={"recursion_limit": 60})
    result["usage"] = llm.usage_for(key)
    return result
