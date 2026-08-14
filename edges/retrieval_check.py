"""분기 함수 — 검색·채점 결과 평가.

■ 선행모델 대응
  FinSight 의 `edges/docs_relevance.py` 와 같은 구조다.

      관련 문서 충분?  → 답변 생성
      부족 + 재시도 여유 → 질문 재작성 후 재검색
      부족 + 재시도 소진 → 웹 검색 폴백

■ 이 3분기가 왜 중요한가
  에이전트가 무한루프에 빠지는 가장 흔한 지점이 "검색 실패 → 재검색"이다.
  카운터(grading_retries)를 state 에 두고 한도를 넘으면 **다른 경로로 탈출**시킨다.
  카운터 없이 조건만 보면 같은 실패를 영원히 반복한다.
"""

from __future__ import annotations

import config
from state import RAGState
from utils import log_message


def assess_documents(state: RAGState) -> str:
    """채점을 통과한 문서 수를 보고 다음 행동을 정한다."""
    docs = state.get("documents", []) or []
    retries = state.get("grading_retries", 0)

    if len(docs) >= config.DOCS_RELEVANCE_THRESHOLD:
        return "enough"

    if retries > config.MAX_GRADING_RETRIES:
        log_message(f"문서 확보 실패({retries}회) → 웹 검색으로 전환")
        return "web" if config.WORKFLOW_SETTINGS["web_search_fallback"] else "give_up"

    log_message(f"관련 문서 {len(docs)}건 → 질문 재작성 후 재검색 ({retries}회차)")
    return "rewrite"
