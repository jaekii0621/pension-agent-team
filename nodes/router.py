"""라우터 에이전트 — 질문을 어느 처리 경로로 보낼지 정한다.

■ 선행모델 대응
  FinSight 의 `nodes/path_decision.py` 자리다. 그쪽은 2단 라우터였다.
    split_path_decider_1 : web / general / rag
    split_path_decider_2 : rag / decomposed_rag / persona
  우리는 1단으로 합쳤다. 이유: 우리 코퍼스는 도메인이 하나로 고정되어 있어
  "일반 잡담 vs 문서질문" 판정은 safety_checker 의 out_of_scope 가 이미 처리한다.

■ 전문가(experts) 배정을 뺐다 (2026-08)
  예전에는 소스 5종에 전문가 프롬프트를 1:1로 짝지어 두고 라우터가 배정했다.
  소스 필터를 없애면서 그 짝이 성립하지 않게 됐다.

  선행모델도 전문가를 고정하지 않았다. `create_persona` 가 질문마다 페르소나를
  새로 만들되(자유 문자열, Literal 아님), 만들어진 페르소나는 **검색에 관여하지
  않았다** — rag_e2e.invoke 에 넘어가는 건 question 하나뿐이다. 페르소나는
  "어떤 질문을 던질까"만 정하고 답변 프롬프트는 공용이었다.

  우리도 같은 구조다. 질문을 나누는 일은 분해기가 하고, 답변 프롬프트는
  **검색된 근거의 성격**(규정문서/투자설명서)이 고른다.
  → 라우터에 남은 판단은 처리 방식(rag/calc/recommend/out_of_scope)뿐이다.

■ CLOVA Router API 로 대체 가능한 지점
  네이버 CLOVA Studio 에는 라우터 기능이 따로 있다(사전에 콘솔에서 도메인을 등록).
  아래 `route_query()` 의 LLM 호출을 그 API 로 바꾸면 지연·비용이 줄 수 있다.
  지금은 콘솔 설정 없이 바로 돌아가야 하므로 LLM 분류로 구현했다.
  → 교체 시 이 함수 하나만 고치면 된다.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

import config
from llm import llm
from prompt import prompts
from state import OverallState
from utils import Timer, step


class RouteDecision(BaseModel):
    route: str = Field(description="rag | calc | recommend | out_of_scope")
    needs_calc: bool = Field(default=False, description="숫자 계산이 필요한가")
    needs_strategy: bool = Field(default=False, description="비교·판단이 필요한가")
    reason: str = Field(default="")


VALID_ROUTES = {"rag", "calc", "recommend", "out_of_scope"}


def route_query(state: OverallState) -> dict:
    question = state["question"]
    질문유형 = state.get("질문유형", "사실조회")
    filters = state.get("filters", {})

    with Timer() as t:
        res = llm.structured(
            prompts.router_system,
            f"질문: {question}\n"
            f"질의 분석 결과 - 질문유형: {질문유형}, 검색범위: {filters.get('scope')}",
            RouteDecision,
            model=config.MODEL_PLAN,
            stage="route",
        )

    route = res.route if res.route in VALID_ROUTES else "rag"

    # ── 결정론적 보정 ─────────────────────────────────────────────────
    # LLM 라우팅은 흔들린다. 질의 분석 결과와 어긋나면 분석 결과를 우선한다.
    # (성능 측정 시 같은 질문이 실행마다 다른 경로를 타면 측정이 무의미해진다)
    if 질문유형 == "추천":
        route = "recommend"
    elif 질문유형 == "계산" and route == "rag":
        route = "calc"

    needs_calc = res.needs_calc or 질문유형 in ("계산", "혼합")
    needs_strategy = res.needs_strategy or 질문유형 in ("추천", "혼합")

    return {
        "route": route,
        "verification": {"needs_calc": needs_calc, "needs_strategy": needs_strategy},
        "llm_calls": 1,
        "trace": step("route_query",
                      summary=f"route={route}, 계산={needs_calc}, 전략={needs_strategy}",
                      ms=t.ms),
    }
