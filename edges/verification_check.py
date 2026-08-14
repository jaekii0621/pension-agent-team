"""분기 함수 — 검증 결과 평가.

검증 실패에는 두 종류가 있고, **고쳐야 할 곳이 다르다.**

  ① 환각 (grounded=False)
     → 근거는 있는데 답변이 근거를 벗어났다. **답변만 다시 만든다.**
       검색을 다시 할 이유가 없다(오히려 근거가 바뀌면 문제가 커진다).

  ② 품질 미달 (score < 0.7)
     → 답변이 질문에 못 답했다. 근거 자체가 부족했을 가능성이 크다.
       **질문을 재작성해 다시 검색**한다.

  ③ 재시도 소진
     → 웹 검색으로 탈출. 그것도 막혀 있으면 현재 답변을 그대로 내보내되
       검증 실패 사실을 기록에 남긴다(성능 측정 시 이 비율이 중요한 지표다).
"""

from __future__ import annotations

import config
from state import RAGState
from utils import log_message


def assess_verdict(state: RAGState) -> str:
    verdict = state.get("verdict", {}) or {}
    retries = state.get("verify_retries", 0)

    if verdict.get("pass", True):
        return "ok"

    if retries > config.MAX_VERIFY_RETRIES:
        log_message(f"검증 재시도 소진({retries}회) → 현재 답변으로 종료")
        return "give_up"

    # 환각이면 답변만 재생성, 품질 문제면 검색부터 다시
    if verdict.get("grounded") is False:
        log_message("환각 감지 → 같은 근거로 답변 재생성")
        return "regenerate"

    log_message(f"품질 미달(score={verdict.get('score')}) → 질문 재작성 후 재검색")
    return "rewrite"
