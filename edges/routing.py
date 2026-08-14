"""분기 함수 — 안전성/라우팅.

엣지는 **문자열(또는 문자열 리스트)을 반환하는 함수**다. 그게 전부다.
그래프에서 `add_conditional_edges(노드, 이 함수, {반환값: 다음노드})` 로 연결한다.
"""

from __future__ import annotations

from state import OverallState
from utils import log_message


def safety_gate(state: OverallState) -> str:
    """안전/범위 검사 결과에 따라 진행 여부를 정한다."""
    if state.get("safe", True):
        return "proceed"
    log_message(f"질문 차단: {state.get('safety_reason')}")
    return "blocked"


def route_after_router(state: OverallState) -> str:
    """라우터가 정한 경로로 보낸다.

    · calc      : 계산이 핵심 → 계산기를 먼저 태우고, 그 뒤 문서 검색도 한다
                  (계산 결과만 주면 근거가 없으므로 세제 문서를 함께 붙여야 한다)
    · recommend : 상품 추천 → 검색 폭을 넓힌 RAG 로 (retriever 가 top_k 를 키운다)
    · rag       : 일반 문서 질의
    · out_of_scope : 고정 안내문
    """
    route = state.get("route", "rag")
    if route == "out_of_scope":
        return "out_of_scope"
    if route == "calc":
        return "calc"
    return "rag"          # recommend 도 RAG 경로를 타되 검색 파라미터가 달라진다
