"""전략 에이전트 — 수집된 사실을 사용자의 '실행 계획'으로 바꾼다.

■ 선행모델 대응
  FinSight 의 persona 워크플로우 + `combine_answer_analysis` 가 하던 역할이다.
  다만 그쪽은 "여러 관점(트레이더/애널리스트/투자자)에서 각각 질문을 만들어
  답하게" 했다. 우리는 관점을 늘리는 대신 **판단을 한 층 얹는다.**

■ 왜 별도 에이전트인가 (검색 답변과 분리하는 이유)
  검색 답변은 "문서에 뭐라고 쓰여 있나"에 답한다 — 사실 재현.
  전략은 "그래서 나는 뭘 해야 하나"에 답한다 — 비교·판단.
  두 가지를 한 프롬프트에 섞으면 모델이 사실과 의견을 뒤섞어
  **환각 검사가 무력해진다.** 사실 답변을 먼저 검증으로 통과시킨 뒤,
  그 검증된 사실만 입력으로 전략을 만든다.

■ 안전장치
  · 새로운 사실 생성 금지 (프롬프트 + 입력에 QA 쌍만 제공)
  · 개별 상품 매수·매도 권유 금지
  · 전제 조건 명시 강제 ("총급여 5,500만원 이하라면 …")
  · 근거 부족 시 "전략을 제시할 수 없다"고 말하게 함

■ 언제 도는가
  라우터가 needs_strategy=True 로 판정했을 때만 (추천/혼합/비교 질문).
  단순 사실 조회에 전략을 붙이면 답이 장황해지고 비용만 는다.
"""

from __future__ import annotations

from typing import List

import config
from llm import llm
from prompt import prompts
from state import OverallState
from utils import Timer, step, truncate


def strategy_agent(state: OverallState) -> dict:
    qa_pairs: List[dict] = state.get("qa_pairs", []) or []
    calc = state.get("calc_result") or {}
    question = state["question"]
    profile = state.get("user_profile") or {}

    if not config.WORKFLOW_SETTINGS["use_strategy"]:
        return {"trace": step("strategy_agent", summary="비활성(설정)")}

    needs = (state.get("verification") or {}).get("needs_strategy", False)
    if not needs:
        return {"trace": step("strategy_agent", summary="불필요(라우터 판정) → 건너뜀")}

    if not qa_pairs:
        return {"trace": step("strategy_agent", summary="근거 QA 없음 → 건너뜀")}

    # 검증을 통과한 답변만 전략의 입력으로 쓴다 (환각이 전략으로 번지는 것을 막는다)
    verified = [p for p in qa_pairs if p.get("verified", True)]
    facts = "\n\n".join(
        f"[{i + 1}] Q: {p['question']}\n    A: {truncate(p.get('answer') or '', 500)}"
        for i, p in enumerate(verified)
    )

    calc_block = ""
    if calc and not calc.get("skipped") and not calc.get("error"):
        nums = {k: v for k, v in calc.items()
                if k not in ("steps", "assumptions", "explanation", "args", "tool")}
        calc_block = ("\n\n[계산 결과]\n"
                      + "\n".join(f"· {k}: {v}" for k, v in nums.items()))

    with Timer() as t:
        note = llm.chat(
            prompts.strategy_system,
            f"[사용자 질문]\n{question}\n\n"
            f"[사용자 정보]\n{profile or '제공되지 않음'}\n\n"
            f"[확인된 사실]\n{facts}{calc_block}",
            model=config.MODEL_ANSWER, max_tokens=1800, stage="strategy")

    return {
        "strategy_note": note,
        "llm_calls": 1,
        "trace": step("strategy_agent",
                      summary=f"사실 {len(verified)}건 기반 전략 {len(note)}자 생성",
                      ms=t.ms),
    }
