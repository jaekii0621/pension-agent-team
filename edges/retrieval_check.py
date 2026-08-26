"""분기 함수 — 검색·채점 결과 평가.

■ 선행모델 대응
  FinSight 의 `edges/docs_relevance.py` 와 같은 구조다.

      관련 문서 충분?  → 답변 생성
      부족 + 재시도 여유 → 질문 재작성 후 재검색
      부족 + 재시도 소진 → 근거 없이 '확인되지 않습니다' 응답

■ 이 3분기가 왜 중요한가
  에이전트가 무한루프에 빠지는 가장 흔한 지점이 "검색 실패 → 재검색"이다.
  카운터(grading_retries)를 state 에 두고 한도를 넘으면 **다른 경로로 탈출**시킨다.
  카운터 없이 조건만 보면 같은 실패를 영원히 반복한다.
"""

from __future__ import annotations

import config
from llm import llm
from state import RAGState
from utils import log_message


def assess_documents(state: RAGState) -> str:
    """채점을 통과한 문서 수를 보고 다음 행동을 정한다."""
    docs = state.get("documents", []) or []
    retries = state.get("grading_retries", 0)

    if len(docs) >= config.DOCS_RELEVANCE_THRESHOLD:
        return "enough"

    # ── LLM 예산 게이트 (2026-08-18 신설) ──────────────────────────────
    #
    # 재작성→재검색은 한 번 돌 때마다 LLM 을 2~3회 태운다. 예산을 넘긴 상태에서
    # 더 도는 것은 비용만 늘리고 답변 품질은 별로 올리지 못한다.
    # (예산 검사가 라운드 경계에만 있어서 문항당 최대 95회까지 나왔던 원인)
    #
    # 재시도를 포기해도 "give_up" 으로 답변은 나온다 — 근거 없이도
    # "제공된 자료로는 확인되지 않습니다" 를 낼 수 있게 되어 있다.
    if llm.over_budget():
        log_message(f"LLM 예산 초과({llm.budget_note()}) → 재검색 포기")
        return "give_up"

    if retries > config.MAX_GRADING_RETRIES:
        log_message(f"문서 확보 실패({retries}회) → 근거 없이 응답")
        return "give_up"

    log_message(f"관련 문서 {len(docs)}건 → 질문 재작성 후 재검색 ({retries}회차)")
    return "rewrite"
