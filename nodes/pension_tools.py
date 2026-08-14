"""연금 계산 도구 모음 — **순수 파이썬 함수**. LLM이 개입하지 않는다.

■ 선행모델과의 중요한 설계 차이
  FinSight 의 `nodes/calculator.py` 는 LLM 에게 **파이썬 코드를 생성**시켜 실행했다.
  범용 재무 지표(KPI 21종)를 다뤄야 해서 미리 함수를 다 만들 수 없었기 때문이다.

  우리는 반대로 간다. 이유:
    1) 연금 계산은 **법으로 공식이 고정**돼 있다. 매번 코드를 생성할 이유가 없다.
    2) 세금 계산을 LLM 이 즉석에서 짠 코드로 하면 검증이 불가능하다.
       같은 질문에 매번 다른 코드가 나오면 성능 측정도 안 된다.
    3) 세율이 바뀌면 `data/tax_tables.json` 만 고치면 된다. 코드도 프롬프트도 안 바뀐다.

  → **LLM 은 "어떤 도구에 어떤 숫자를 넣을지"만 정하고, 계산은 코드가 한다.**
    이게 금융 도메인에서 에이전트를 안전하게 쓰는 표준 패턴이다.

■ 모든 함수는 계산 결과와 함께 `steps`(단계별 내역)를 돌려준다.
  설명 에이전트가 이걸 보고 과정을 서술한다 → 숫자를 지어낼 여지가 없다.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

import config


# ==========================================================================
@lru_cache(maxsize=1)
def tables() -> dict:
    path = config.TABLES_DIR / "tax_tables.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _won(v: float) -> int:
    return int(round(v))


def _progressive_tax(base: float) -> tuple[int, float, int]:
    """종합소득세율 적용 → (산출세액, 적용세율, 누진공제)."""
    for row in tables()["종합소득세율"]["구간"]:
        cap = row["상한"]
        if cap is None or base <= cap:
            return _won(base * row["세율"] - row["누진공제"]), row["세율"], row["누진공제"]
    return 0, 0.0, 0


# ==========================================================================
# 1. 세액공제 한도와 절세액
# ==========================================================================
def tax_credit_limit(*, total_salary: Optional[float] = None,
                     pension_savings_paid: float = 0,
                     irp_paid: float = 0,
                     age: Optional[int] = None,
                     **_) -> Dict[str, Any]:
    """연금계좌 세액공제 한도와 예상 절세액."""
    t = tables()["세액공제"]
    limit_ps = t["한도"]["연금저축_단독"]
    limit_all = t["한도"]["연금저축_IRP_합산"]

    # 총급여 미상이면 낮은 공제율(보수적)로 계산한다 — 과대 안내를 피한다
    rate_row = t["공제율"][1]
    if total_salary is not None and total_salary <= t["공제율"][0]["총급여_상한"]:
        rate_row = t["공제율"][0]
    rate = rate_row["율"]

    ps_eligible = min(pension_savings_paid, limit_ps)
    total_eligible = min(ps_eligible + irp_paid, limit_all)

    steps: List[str] = [
        f"연금저축 납입 {pension_savings_paid:,.0f}원 중 공제대상 {ps_eligible:,.0f}원 "
        f"(연금저축 단독 한도 {limit_ps:,.0f}원)",
        f"IRP 납입 {irp_paid:,.0f}원 합산 → 공제대상 {total_eligible:,.0f}원 "
        f"(합산 한도 {limit_all:,.0f}원)",
        f"공제율 {rate * 100:.1f}% 적용 ({rate_row['비고']}"
        + (f", 총급여 {total_salary:,.0f}원 기준)" if total_salary is not None
           else ", 총급여 미제공 → 낮은 율로 보수적 계산)"),
    ]
    saving = _won(total_eligible * rate)
    steps.append(f"예상 절세액 = {total_eligible:,.0f} × {rate:.3f} = {saving:,.0f}원")

    return {
        "tool": "tax_credit_limit",
        "공제대상금액": _won(total_eligible),
        "적용공제율": rate,
        "예상절세액": saving,
        "남은한도": _won(max(0, limit_all - total_eligible)),
        "steps": steps,
        "assumptions": [
            f"기준: {tables()['_meta']['기준시점']}",
            "총급여 5,500만원(종합소득금액 4,500만원) 이하는 16.5%, 초과는 13.2%",
            "ISA 만기자금 이체에 따른 추가 한도는 반영하지 않음",
        ],
    }


# ==========================================================================
# 2. 퇴직소득세
# ==========================================================================
def retirement_income_tax(*, severance_pay: float, service_years: float, **_) -> Dict[str, Any]:
    """퇴직소득세 (환산급여 방식 5단계)."""
    if service_years <= 0:
        return {"error": "근속연수가 필요합니다 (0보다 커야 함)"}

    steps: List[str] = [f"① 퇴직급여 = {severance_pay:,.0f}원, 근속연수 {service_years}년"]

    # 1) 근속연수공제
    ded_y = 0.0
    for row in tables()["근속연수공제"]["구간"]:
        cap = row["년수상한"]
        if cap is None or service_years <= cap:
            ded_y = row["기본"] + row["초과년당"] * (service_years - row["기준년수"])
            break
    steps.append(f"② 근속연수공제 = {ded_y:,.0f}원")

    # 2) 환산급여
    converted = max(0.0, (severance_pay - ded_y)) / service_years * 12
    steps.append(f"③ 환산급여 = ({severance_pay:,.0f} - {ded_y:,.0f}) ÷ {service_years} × 12 "
                 f"= {converted:,.0f}원")

    # 3) 환산급여공제
    ded_c = 0.0
    for row in tables()["환산급여공제"]["구간"]:
        cap = row["상한"]
        if cap is None or converted <= cap:
            ded_c = row["기본"] + (converted - row["기준액"]) * row["공제율"]
            break
    steps.append(f"④ 환산급여공제 = {ded_c:,.0f}원")

    # 4) 과세표준 → 산출세액
    base = max(0.0, converted - ded_c)
    calc_tax, rate, prog = _progressive_tax(base)
    steps.append(f"⑤ 과세표준 = {converted:,.0f} - {ded_c:,.0f} = {base:,.0f}원")
    steps.append(f"⑥ 환산산출세액 = {base:,.0f} × {rate * 100:.0f}% - {prog:,.0f} "
                 f"= {calc_tax:,.0f}원")

    # 5) 환산 되돌리기
    tax = _won(calc_tax / 12 * service_years)
    local = _won(tax * 0.1)
    steps.append(f"⑦ 퇴직소득세 = {calc_tax:,.0f} ÷ 12 × {service_years} = {tax:,.0f}원")
    steps.append(f"⑧ 지방소득세(10%) = {local:,.0f}원 → 합계 {tax + local:,.0f}원")

    return {
        "tool": "retirement_income_tax",
        "퇴직소득세": tax,
        "지방소득세": local,
        "총부담세액": tax + local,
        "실수령액": _won(severance_pay - tax - local),
        "실효세율": round((tax + local) / severance_pay, 4) if severance_pay else 0,
        "steps": steps,
        "assumptions": [
            f"기준: {tables()['_meta']['기준시점']}",
            "근속연수는 1년 미만 절상 없이 입력값 그대로 계산",
            "비과세 퇴직급여·기납부세액·세액정산 특례는 반영하지 않음",
        ],
    }


# ==========================================================================
# 3. 연금소득세 (연금으로 나눠 받을 때)
# ==========================================================================
def pension_income_tax(*, annual_pension: float, age: int,
                       source_type: str = "세액공제분", **_) -> Dict[str, Any]:
    """연금 수령 시 연금소득세(분리과세)."""
    t = tables()["연금소득세율"]
    rate = t["연령별"][-1]["세율"]
    note = t["연령별"][-1]["비고"]
    for row in t["연령별"]:
        if row["나이상한"] is not None and age <= row["나이상한"]:
            rate, note = row["세율"], row["비고"]
            break

    steps = [f"① 연간 연금수령액 {annual_pension:,.0f}원, 수령 시 나이 만 {age}세",
             f"② 적용 연금소득세율 {rate * 100:.1f}% ({note})"]

    if source_type == "퇴직금":
        cut = t["퇴직금재원_감면율"]["10년이내"]
        steps.append(f"③ 재원이 퇴직금이므로 연금소득세가 아니라 "
                     f"**퇴직소득세의 {cut * 100:.0f}%** 가 적용된다 "
                     f"(11년차부터는 {t['퇴직금재원_감면율']['11년차이상'] * 100:.0f}%)")
        return {
            "tool": "pension_income_tax",
            "적용방식": "퇴직소득세 감면",
            "감면율": cut,
            "안내": "정확한 세액은 원 퇴직소득세를 알아야 계산 가능합니다. "
                    "retirement_income_tax 로 퇴직소득세를 먼저 구한 뒤 곱하세요.",
            "steps": steps,
            "assumptions": [f"기준: {tables()['_meta']['기준시점']}"],
        }

    tax = _won(annual_pension * rate)
    steps.append(f"③ 연금소득세 = {annual_pension:,.0f} × {rate:.3f} = {tax:,.0f}원")

    threshold = t["종합과세_기준금액"]
    over = annual_pension > threshold
    if over:
        steps.append(f"④ ⚠️ 연간 {threshold:,.0f}원 초과 → {t['종합과세_설명']}")

    return {
        "tool": "pension_income_tax",
        "적용세율": rate,
        "연금소득세": tax,
        "실수령액": _won(annual_pension - tax),
        "종합과세_대상": over,
        "종합과세_기준금액": threshold,
        "steps": steps,
        "assumptions": [
            f"기준: {tables()['_meta']['기준시점']}",
            "지방소득세 포함 세율",
            "연금수령 요건(가입 5년·만55세·10년 분할) 충족을 전제",
        ],
    }


# ==========================================================================
# 4. 중도인출(연금외수령) 세금
# ==========================================================================
def early_withdrawal_tax(*, amount: float, reason: str = "일반",
                         source_type: str = "세액공제분",
                         age: Optional[int] = None, **_) -> Dict[str, Any]:
    """중도인출 시 세금. 부득이한 사유 여부가 세율을 가른다."""
    t = tables()["중도인출_세율"]
    steps = [f"① 인출액 {amount:,.0f}원, 재원={source_type}, 사유={reason}"]

    if source_type == "퇴직금" and reason != "부득이한사유":
        steps.append("② 퇴직금 재원을 일반 중도인출하면 퇴직소득세가 감면 없이 부과된다")
        return {
            "tool": "early_withdrawal_tax",
            "적용방식": "퇴직소득세 100%",
            "안내": "retirement_income_tax 로 퇴직소득세를 먼저 계산해야 합니다.",
            "steps": steps,
            "assumptions": [f"기준: {tables()['_meta']['기준시점']}"],
        }

    if reason == "부득이한사유":
        pt = tables()["연금소득세율"]
        rate = pt["연령별"][-1]["세율"]
        if age is not None:
            for row in pt["연령별"]:
                if row["나이상한"] is not None and age <= row["나이상한"]:
                    rate = row["세율"]
                    break
        else:
            rate = pt["연령별"][0]["세율"]      # 나이 미상 → 가장 높은 5.5% (보수적)
        steps.append(f"② 부득이한 사유 → 기타소득세 대신 연금소득세율 "
                     f"{rate * 100:.1f}% 분리과세")
        steps.append(f"   ({t['부득이한사유']['비고']})")
    else:
        rate = t["일반"]["세액공제분"]
        steps.append(f"② 일반 중도인출 → 기타소득세 {rate * 100:.1f}% 분리과세")

    tax = _won(amount * rate)
    steps.append(f"③ 세금 = {amount:,.0f} × {rate:.3f} = {tax:,.0f}원")
    steps.append(f"④ 실수령액 = {amount - tax:,.0f}원")

    return {
        "tool": "early_withdrawal_tax",
        "적용세율": rate,
        "세금": tax,
        "실수령액": _won(amount - tax),
        "사유구분": reason,
        "steps": steps,
        "assumptions": [
            f"기준: {tables()['_meta']['기준시점']}",
            "부득이한 사유는 사유 발생일로부터 6개월 이내 증빙 제출이 필요",
            "IRP와 연금저축의 인출 가능 사유가 다르므로 절차 문서를 함께 확인할 것",
        ],
    }


# ==========================================================================
# 5. 연금수령한도
# ==========================================================================
def withdrawal_limit(*, balance: float, year_index: int, **_) -> Dict[str, Any]:
    """연금수령한도 = 평가액 ÷ (11 - 연금수령연차) × 120%"""
    t = tables()["연금수령한도"]
    if year_index >= t["분모기준"]:
        return {
            "tool": "withdrawal_limit",
            "한도": None,
            "무제한": True,
            "steps": [f"연금수령연차 {year_index}년차 → 11년차 이상이므로 한도 없음"],
            "assumptions": [f"기준: {tables()['_meta']['기준시점']}"],
        }

    denom = t["분모기준"] - year_index
    limit = balance / denom * t["배수"]
    return {
        "tool": "withdrawal_limit",
        "한도": _won(limit),
        "월환산": _won(limit / 12),
        "무제한": False,
        "steps": [
            f"① 연금계좌 평가액 {balance:,.0f}원, 연금수령연차 {year_index}년차",
            f"② 한도 = {balance:,.0f} ÷ (11 - {year_index}) × 120%",
            f"③ = {balance:,.0f} ÷ {denom} × 1.2 = {limit:,.0f}원",
            f"④ 월 환산 약 {limit / 12:,.0f}원",
        ],
        "assumptions": [
            f"기준: {tables()['_meta']['기준시점']}",
            "한도 초과분은 연금외수령으로 과세된다",
        ],
    }


# ==========================================================================
# 6. 적립 예상액
# ==========================================================================
def future_value(*, monthly: float, years: float, annual_return: float = 3.0,
                 **_) -> Dict[str, Any]:
    """월 적립 복리 미래가치."""
    n = int(round(years * 12))
    r = annual_return / 100 / 12
    principal = monthly * n
    fv = monthly * n if r == 0 else monthly * ((1 + r) ** n - 1) / r * (1 + r)

    return {
        "tool": "future_value",
        "총납입원금": _won(principal),
        "예상평가액": _won(fv),
        "운용수익": _won(fv - principal),
        "steps": [
            f"① 월 {monthly:,.0f}원 × {n}개월 = 원금 {principal:,.0f}원",
            f"② 연 {annual_return}% (월 {r * 100:.4f}%) 복리 적립 가정",
            f"③ 예상 평가액 {fv:,.0f}원 (운용수익 {fv - principal:,.0f}원)",
        ],
        "assumptions": [
            "수익률이 매달 일정하다는 단순 가정 — 실제 펀드 수익률은 변동한다",
            "수수료·보수를 차감하지 않은 값",
            "수령 시 과세는 별도로 계산해야 한다",
        ],
    }


# ==========================================================================
TOOLS = {
    "tax_credit_limit": tax_credit_limit,
    "retirement_income_tax": retirement_income_tax,
    "pension_income_tax": pension_income_tax,
    "early_withdrawal_tax": early_withdrawal_tax,
    "withdrawal_limit": withdrawal_limit,
    "future_value": future_value,
}

REQUIRED_ARGS = {
    "tax_credit_limit": [],                                   # 전부 선택 (없으면 보수적 계산)
    "retirement_income_tax": ["severance_pay", "service_years"],
    "pension_income_tax": ["annual_pension", "age"],
    "early_withdrawal_tax": ["amount"],
    "withdrawal_limit": ["balance", "year_index"],
    "future_value": ["monthly", "years"],
}


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """도구 실행. 인자가 부족하면 무엇이 없는지 돌려준다."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"알 수 없는 도구: {name}"}

    missing = [a for a in REQUIRED_ARGS.get(name, []) if args.get(a) in (None, "")]
    if missing:
        return {"error": "입력 부족", "missing_args": missing}

    try:
        return fn(**args)
    except Exception as e:
        return {"error": f"계산 실패: {e}"}
