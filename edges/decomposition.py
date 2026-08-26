"""분기 함수 — 서브질문 병렬 실행(fan-out)과 라운드 반복 제어.

■ 여기가 멀티에이전트의 심장이다: `Send`
  LangGraph 의 `Send("노드이름", 상태dict)` 를 **리스트로 반환**하면
  그 개수만큼 같은 노드가 **동시에 복제 실행**된다.

      서브질문 3개 → Send 3개 → rag_worker 3개가 병렬로 돈다

  그리고 병렬 결과를 다시 하나로 모으는 것이 state.py 의 **리듀서**다.
      qa_pairs: Annotated[List[dict], operator.add]

  `Send`(흩뿌리기) + 리듀서(모으기) — 이 한 쌍이 없으면 멀티에이전트가 성립하지 않는다.
  선행모델의 `edges/decomposed_questions.py` 가 정확히 같은 일을 한다.

■ 라운드 반복 (RRR)
  aggregate → 충분? → 아니면 잔여 재분해 → 다시 fan-out
  최대 config.MAX_ROUNDS 회. 한도는 무한루프 방지 장치다.
"""

from __future__ import annotations

from typing import List, Union

from langgraph.types import Send

import config
from state import OverallState
from utils import log_message


def send_sub_questions(state: OverallState) -> Union[List[Send], str]:
    """서브질문들을 rag_worker 로 흩뿌린다."""
    subs = state.get("sub_questions", []) or []
    if not subs:
        return "aggregate"

    log_message(f"서브질문 {len(subs)}건 병렬 실행")
    return [
        Send("rag_worker", {
            "qid": s["qid"],
            "question": s["question"],
            "original_question": state["question"],
            "질문유형": state.get("질문유형", "사실조회"),
            "filters": s.get("filters") or state.get("filters", {}),
            "is_recommendation": bool(s.get("is_recommendation", False)),
            # 안전검사가 넘긴 주의 신호. 차단은 안 했지만 답변 단계가 알아야 한다.
            "safety_caution": state.get("safety_caution", ""),
            "hyde_doc": "",          # 재작성 단계에서 채워진다
            "filter_retries": 0,
            "grading_retries": 0,
            "verify_retries": 0,
        })
        for s in subs
    ]


def should_repeat(state: OverallState) -> str:
    """RRR 라운드를 한 번 더 돌지 결정한다."""
    if state.get("sufficient", True):
        return "finish"
    if state.get("round_no", 0) >= config.MAX_ROUNDS - 1:
        log_message(f"라운드 한도({config.MAX_ROUNDS}) 도달 → 종료")
        return "finish"
    # 라운드 경계 검사 — **이것만으로는 부족했다** (2026-08-18 주석 보강)
    #
    # 여기는 "라운드를 한 번 더 돌까?" 를 정하는 지점이라, 이미 한 라운드를
    # 다 태운 뒤에야 예산을 본다. 서브질문 fan-out 안쪽 루프는 못 막는다.
    # 그래서 실제 호출 게이트는 llm.over_budget() 으로 옮겼고
    # (edges/retrieval_check.py · edges/verification_check.py),
    # 여기는 라운드 단위의 2차 방어선으로 남긴다.
    if (state.get("llm_calls", 0) or 0) >= config.LLM_CALL_BUDGET:
        log_message(f"LLM 호출 예산({config.LLM_CALL_BUDGET}) 소진 → 종료")
        return "finish"
    return "repeat"
