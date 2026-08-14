"""계산기 에이전트 — 연금 특화.

■ 구조: LLM 2회 + 결정론적 계산 1회
      ┌ ① 도구 선택 + 인자 추출  (LLM)  ← 자연어 → 구조화된 숫자
      │  ② 계산                  (코드)  ← pension_tools.py, 재현 가능
      └ ③ 결과 설명              (LLM)  ← 숫자 → 자연어
  숫자가 만들어지는 지점에 LLM 이 없다. 이게 핵심이다.

■ 입력이 부족할 때
  "퇴직금 세금 얼마예요?" 처럼 금액·근속연수가 없으면 계산할 수 없다.
  이때 지어내지 않고 **무엇이 필요한지 되묻는 안내**를 답으로 낸다.
  (선행모델의 HITL(Human-In-The-Loop) 되묻기를 성능 측정 단계에 맞게
   '되물을 항목을 답변에 명시'하는 형태로 단순화한 것)

■ 문서 근거와의 관계
  계산 결과는 세율표(data/tax_tables.json)에서 나온다. 이건 문서 근거가 아니다.
  그래서 계산 답변에는 반드시 세제 소스의 문서 근거를 함께 붙여
  "표 값과 문서가 일치하는지" 사용자가 확인할 수 있게 한다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

import config
from llm import llm
from prompt import prompts
from state import OverallState
from utils import Timer, step

from .pension_tools import REQUIRED_ARGS, run_tool


class ToolArgs(BaseModel):
    """LLM 이 뽑아야 하는 계산 입력."""
    tool_name: str = Field(description="tax_credit_limit | retirement_income_tax | "
                                       "pension_income_tax | early_withdrawal_tax | "
                                       "withdrawal_limit | future_value | none")
    total_salary: Optional[float] = Field(default=None, description="총급여(원)")
    age: Optional[int] = Field(default=None, description="나이(만)")
    pension_savings_paid: Optional[float] = Field(default=None, description="연금저축 납입액(원)")
    irp_paid: Optional[float] = Field(default=None, description="IRP 납입액(원)")
    severance_pay: Optional[float] = Field(default=None, description="퇴직급여(원)")
    service_years: Optional[float] = Field(default=None, description="근속연수(년)")
    annual_pension: Optional[float] = Field(default=None, description="연간 연금수령액(원)")
    amount: Optional[float] = Field(default=None, description="인출액(원)")
    balance: Optional[float] = Field(default=None, description="연금계좌 평가액(원)")
    year_index: Optional[int] = Field(default=None, description="연금수령연차")
    monthly: Optional[float] = Field(default=None, description="월 납입액(원)")
    years: Optional[float] = Field(default=None, description="적립기간(년)")
    annual_return: Optional[float] = Field(default=None, description="연 수익률(%)")
    reason: Optional[str] = Field(default=None, description="부득이한사유 | 일반")
    source_type: Optional[str] = Field(default=None,
                                       description="세액공제분 | 운용수익 | 퇴직금")
    missing_args: List[str] = Field(default_factory=list,
                                    description="질문에 없어서 못 채운 필수 인자")


ARG_LABELS = {
    "severance_pay": "퇴직급여 금액",
    "service_years": "근속연수",
    "annual_pension": "연간 연금수령액",
    "age": "수령 시 나이(만)",
    "amount": "인출하려는 금액",
    "balance": "연금계좌 평가액",
    "year_index": "연금수령연차",
    "monthly": "월 납입액",
    "years": "적립 기간",
    "total_salary": "총급여(세액공제율 판정용)",
}


def calc_agent(state: OverallState) -> dict:
    """계산 도구를 골라 실행하고 결과를 설명한다."""
    question = state["question"]
    profile = state.get("user_profile") or {}

    if not config.WORKFLOW_SETTINGS["use_calculator"]:
        return {"trace": step("calc_agent", summary="비활성(설정)")}

    # ── ① 도구 선택 + 인자 추출 (LLM) ────────────────────────────────
    with Timer() as t1:
        extracted = llm.structured(
            prompts.calculator_select_system,
            f"질문: {question}\n"
            f"알려진 사용자 정보: {profile or '없음'}",
            ToolArgs, model=config.MODEL_PLAN, max_tokens=800, stage="calc:extract")

    tool = extracted.tool_name
    if tool in ("none", "", None) or tool not in REQUIRED_ARGS:
        return {
            "calc_result": {"skipped": True, "reason": "적합한 계산 도구 없음"},
            "llm_calls": 1,
            "trace": step("calc_agent", summary=f"도구 없음(tool={tool}) → 건너뜀", ms=t1.ms),
        }

    # LLM 추출값 + 사용자 프로필 병합 (프로필이 더 신뢰도 높다)
    args: Dict[str, Any] = {
        k: v for k, v in extracted.model_dump().items()
        if k not in ("tool_name", "missing_args") and v is not None
    }
    args.update({k: v for k, v in profile.items() if v is not None})

    # ── ② 계산 (코드) ────────────────────────────────────────────────
    result = run_tool(tool, args)

    if result.get("error") == "입력 부족":
        need = result.get("missing_args", [])
        labels = [ARG_LABELS.get(a, a) for a in need]
        msg = ("계산에 필요한 정보가 부족합니다. 다음 값을 알려주시면 계산해 드리겠습니다:\n"
               + "\n".join(f"· {x}" for x in labels))
        return {
            "calc_result": {"tool": tool, "needs_input": need, "message": msg},
            "llm_calls": 1,
            "trace": step("calc_agent", summary=f"{tool}: 입력 부족 {need}", ms=t1.ms),
        }

    if "error" in result:
        return {
            "calc_result": {"tool": tool, "error": result["error"]},
            "llm_calls": 1,
            "trace": step("calc_agent", summary=f"{tool}: 실패 - {result['error']}", ms=t1.ms),
        }

    # ── ③ 결과 설명 (LLM) ────────────────────────────────────────────
    numbers = {k: v for k, v in result.items()
               if k not in ("steps", "assumptions", "tool")}
    body = (
        f"[사용자 질문]\n{question}\n\n"
        f"[사용한 계산 도구] {tool}\n"
        f"[계산 결과]\n"
        + "\n".join(f"· {k}: {v}" for k, v in numbers.items())
        + "\n\n[계산 과정]\n" + "\n".join(result.get("steps", []))
        + "\n\n[적용한 가정]\n" + "\n".join(f"· {a}" for a in result.get("assumptions", []))
    )

    with Timer() as t2:
        explanation = llm.chat(prompts.calculator_explain_system, body,
                               model=config.MODEL_ANSWER, max_tokens=1500,
                               stage="calc:explain")

    return {
        "calc_result": {**result, "explanation": explanation, "args": args},
        "llm_calls": 2,
        "trace": step("calc_agent",
                      summary=f"{tool} 계산 완료 — "
                              + ", ".join(f"{k}={v}" for k, v in list(numbers.items())[:3]),
                      detail={"args": args, "steps": result.get("steps")},
                      ms=t1.ms + t2.ms),
    }
