"""드라이런(오프라인) LLM — API 키 없이 파이프라인 전체를 돌려 보는 용도.

■ 무엇을 위한 것인가
  에이전트 개발에서 가장 흔한 시간 낭비가 "그래프 배선이 틀렸는데 LLM 응답을
  기다리느라 늦게 발견"하는 것이다. 이 모드는 LLM 이 할 일을 **규칙 기반**으로
  대신해서, 키 없이 · 즉시 · 무료로 다음을 검증한다.

    ✓ 노드 연결과 분기가 의도대로 도는가
    ✓ Send 병렬 fan-out 과 리듀서 병합이 맞는가
    ✓ 검색 필터·소스 가중치·인용 번호 재매핑이 맞는가
    ✓ 재시도 한도가 무한루프를 막는가

  ✗ 검증할 수 없는 것: **답변 품질**. 그건 실제 CLOVA 키로만 측정한다.

■ 켜는 법
    .env 에  PENSION_DRY_RUN=true      또는  CLOVA_API_KEY 를 비워 두면 자동 적용

선행모델에는 이 장치가 없었다(항상 실제 API 를 호출했다). 넣어 두면
성능 측정 단계에서 실험 세팅을 바꿀 때마다 배선을 공짜로 확인할 수 있다.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Type, TypeVar

from pydantic import BaseModel

import config

from .clova_client import CallLog

T = TypeVar("T", bound=BaseModel)


class OfflineLLM:
    """ClovaLLM 과 같은 인터페이스. 규칙으로 응답을 합성한다."""

    def __init__(self) -> None:
        self.calls: List[CallLog] = []
        self.enabled = True
        self._qkey = ""

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, *, model: str = "offline",
             max_tokens: int = 1200, temperature: float = 0.0,
             stage: str = "") -> str:
        self.calls.append(
            CallLog(stage, "offline", 0, 0, 0, note="dry-run", qkey=self._qkey))
        if "합치는" in system or "편집자" in system:
            # 합성 노드: 서브답변을 그냥 이어붙인다(인용 번호는 그대로 유지)
            return _join_blocks(user)
        if "전략" in system:
            return "[드라이런] 전략 제안 자리입니다. 실제 키를 넣으면 생성됩니다."
        if "계산" in system:
            return "[드라이런] 계산 결과 설명 자리입니다."
        return "[드라이런] LLM 응답 자리입니다."

    # ------------------------------------------------------------------
    def structured(self, system: str, user: str, schema: Type[T], *,
                   model: str = "offline", max_tokens: int = 1600,
                   stage: str = "") -> T:
        self.calls.append(
            CallLog(stage, "offline", 0, 0, 0, note="dry-run", qkey=self._qkey))
        name = schema.__name__
        builder = _BUILDERS.get(name)
        data = builder(user) if builder else {}
        try:
            return schema.model_validate(data)
        except Exception:
            # 스키마가 바뀌어도 드라이런이 죽지 않도록 최소 인스턴스를 만든다
            return schema.model_construct(**data)      # type: ignore[return-value]

    # ------------------------------------------------------------------
    def count_tokens(self, messages, *, model: str = "offline"):
        """드라이런에서는 토크나이저 API 를 부르지 않는다."""
        return None

    def mark_question(self, qkey: str) -> None:
        """이후의 호출을 이 질문에 귀속시킨다. **이미 쌓인 기록은 건드리지 않는다.**"""
        self._qkey = qkey

    def usage_summary(self) -> dict:
        return {"calls": len(self.calls), "in_tokens": 0, "out_tokens": 0,
                "total_tokens": 0, "total_ms": 0, "by_stage": {}, "by_model": {},
                "mode": "dry-run"}

    def usage_for(self, qkey: str) -> dict:
        n = sum(1 for c in self.calls if c.qkey == qkey)
        return {"calls": n, "in_tokens": 0, "out_tokens": 0, "total_tokens": 0,
                "precounted_in_tokens": 0, "ms": 0, "by_stage": {}, "by_model": {},
                "mode": "dry-run"}

    def usage_by_question(self) -> dict:
        keys = {c.qkey or "-" for c in self.calls}
        return {k: self.usage_for(k) for k in keys}

    def reset_usage(self) -> None:
        self.calls.clear()
        self._qkey = ""


# ==========================================================================
# 규칙 기반 응답 합성기
# ==========================================================================
def _q(user: str) -> str:
    """프롬프트에서 질문 부분만 뽑는다."""
    m = re.search(r"질문:\s*(.+)", user)
    if m:
        return m.group(1).split("\n")[0]
    m = re.search(r"\[사용자 원 질문\]\n(.+)", user)
    return m.group(1) if m else user[:120]


def _is_recommend(text: str) -> bool:
    """상품 추천 질문인가 (LLM scope 판단의 규칙 버전)."""
    return bool(re.search(
        r"추천|어떤 (펀드|상품)|골라|담을|편입할|투자할 (만한|상품)|"
        r"수익률|총보수|보수는|위험등급|채권형|주식형|클래스", text))


def _detect_type(text: str) -> str:
    if _is_recommend(text):
        return "추천"
    if re.search(r"얼마|계산|한도가|세금이 (얼마|어떻게)|절세액", text):
        return "계산"
    if re.search(r"서류|어떻게 (하|신청)|절차|제출", text):
        return "절차"
    return "사실조회"


# 도메인 어휘 — 드라이런 범위 판정용.
#
# ⚠️ 예전에는 `연금|퇴직|IRP|DC|DB|세액공제|펀드|중도인출` 8개뿐이라 오타 하나에
#    통째로 뚫렸다. "퇴집급여를 은행에서 직접 받으려면?" → '퇴직'이 아니라 '퇴집'
#    이라 매칭 실패 → out_of_scope → 1번 노드에서 차단, 복구 경로 없음.
#    실제로 doc57 에 '퇴직급여 직접청구 프로세스' 가 있는데도 답을 못 냈다.
#
# 그래서 두 가지를 바꿨다.
#   ① 어휘를 도메인 전반으로 넓혔다
#   ② **2-gram 으로도 본다** — tokenize_ko 와 같은 발상이다.
#      "퇴집급여"의 2-gram(퇴집·집급·급여)이 "퇴직급여"(퇴직·직급·급여)와
#      '급여'를 공유하므로 오타가 나도 걸린다.
_DOMAIN_WORDS = [
    # 제도
    "연금", "퇴직", "퇴직급여", "퇴직금", "IRP", "DC", "DB", "확정급여", "확정기여",
    "규약", "가입", "근로자", "사용자", "적립금", "부담금", "근퇴법",
    # 세제
    "세액공제", "과세", "소득세", "세율", "절세", "환급", "공제", "연말정산",
    "수령한도", "종합과세", "분리과세", "과세이연",
    # 절차
    "중도인출", "중간정산", "실물이전", "계약이전", "이체", "해지", "청구", "지급",
    "신청", "서류", "증빙", "내점", "사업자", "ISA",
    # 운용·상품
    "운용", "투자한도", "위험자산", "ETF", "펀드", "수익률", "보수", "위험등급",
    "원리금보장", "TDF", "디폴트옵션", "클래스",
]


def _domain_grams() -> set:
    """도메인 어휘의 한글 2-gram 집합 (오타 내성용)."""
    out = set()
    for w in _DOMAIN_WORDS:
        if len(w) >= 2 and "가" <= w[0] <= "힣":
            out.update(w[i:i + 2] for i in range(len(w) - 1))
    return out


_DOMAIN_GRAMS = _domain_grams()


def _in_domain(q: str) -> bool:
    """연금 도메인 질문인가.

    **판정을 관대하게 한다.** 오탐(정상 질문을 막음)은 1번 노드에서 파이프라인
    전체가 죽고 복구 경로가 없다. 미탐(엉뚱한 질문을 통과시킴)은 LLM 몇 회를
    낭비하고 "제공된 자료로는 확인되지 않습니다" 로 우아하게 끝난다.
    비용이 극단적으로 비대칭이므로 **의심스러우면 통과**시킨다.
    """
    if any(w.lower() in q.lower() for w in _DOMAIN_WORDS):
        return True
    grams = {q[i:i + 2] for i in range(len(q) - 1)}
    return len(grams & _DOMAIN_GRAMS) >= 2      # 2-gram 2개 이상 겹치면 통과


def _build_safety(user: str) -> Dict[str, Any]:
    q = _q(user)
    return {"safe": True, "out_of_scope": not _in_domain(q),
            "reason": "드라이런 규칙 판정", "safe_question": q}


def _build_analysis(user: str) -> Dict[str, Any]:
    """질의 분석의 드라이런 버전.

    필터가 scope 하나로 줄었으므로 여기서 뽑을 것도 질문유형 + 펀드명뿐이다.
    """
    q = _q(user)
    qt = _detect_type(q)
    # 프롬프트에 주입된 펀드명 목록에서 질문에 등장하는 것을 찾는다
    names: List[str] = []
    for line in re.findall(r"^\s*-\s*(.+)$", user, flags=re.M):
        nm = line.strip()
        if len(nm) > 6 and nm in q:
            names.append(nm)
    return {"질문유형": qt, "fund_name": names[:3]}


def _build_route(user: str) -> Dict[str, Any]:
    q = _q(user)
    qt = _detect_type(q)
    route = {"추천": "recommend", "계산": "calc"}.get(qt, "rag")
    return {"route": route,
            "needs_calc": qt == "계산", "needs_strategy": qt == "추천",
            "reason": "드라이런 규칙 판정"}


def _build_decomposition(user: str) -> Dict[str, Any]:
    q = _q(user)
    # '그리고/와/및' 로 이어진 두 축이 있으면 쪼갠다 (아주 단순한 규칙)
    parts = [p.strip() for p in re.split(r"\s*(?:그리고|,|및)\s*", q) if len(p.strip()) > 6]
    if len(parts) < 2:
        parts = [q]
    return {"sub_questions": [
        {"question": p, "is_recommendation": _is_recommend(p)}
        for p in parts[:config.MAX_SUB_QUESTIONS]
    ]}


def _build_sufficiency(_: str) -> Dict[str, Any]:
    # 드라이런에서는 항상 충분으로 본다 (라운드 반복을 보려면 아래를 False 로)
    return {"sufficient": True, "missing": ""}


def _build_doc_grade(user: str) -> Dict[str, Any]:
    # 질문 어휘가 문서에 하나라도 겹치면 통과
    m = re.search(r"질문:\s*(.+?)\n", user)
    q = m.group(1) if m else ""
    words = {w for w in re.findall(r"[가-힣A-Za-z]{2,}", q)}
    hit = any(w in user for w in words)
    return {"relevant": "yes" if hit else "no",
            "reason": "드라이런: 어휘 중복 " + ("있음" if hit else "없음")}


def _build_answer(user: str) -> Dict[str, Any]:
    """근거 블록 앞부분을 그대로 옮겨 답변을 흉내낸다.

    실제 문장을 쓰므로 인용 번호 매핑·출처 목록 생성을 그대로 검증할 수 있다.
    """
    body = user.split("[근거]", 1)[-1]
    lines = [ln for ln in body.split("\n") if ln.strip()]
    snippets: List[str] = []
    n = 0
    for ln in lines:
        if re.match(r"^\[\d+\]", ln):
            n += 1
            continue
        if n and len(ln) > 15:
            snippets.append(f"{ln.strip()[:120]} [{n}]")
        if len(snippets) >= 3:
            break
    text = "[드라이런 답변]\n" + "\n".join(f"- {s}" for s in snippets)
    return {"main_answer": text,
            "citations": [{"n": i + 1, "quote": ""} for i in range(min(n, 3))],
            "evidence_sufficient": bool(snippets)}


def _build_web_answer(_: str) -> Dict[str, Any]:
    return {"main_answer": "[드라이런] 웹 검색 답변 자리입니다.", "used_sources": [1]}


def _build_halluc(_: str) -> Dict[str, Any]:
    return {"grounded": True, "unsupported": [], "reason": "드라이런: 검사 생략"}


def _build_quality(_: str) -> Dict[str, Any]:
    return {"addresses_question": True, "complete": True, "actionable": True,
            "score": 0.85, "missing": ""}


def _build_rewrite(user: str) -> Dict[str, Any]:
    q = _q(user).replace("원 질문:", "").strip()
    syn = {"돈 미리": "중도인출", "세금 깎": "세액공제", "빼기": "인출"}
    for k, v in syn.items():
        q = q.replace(k, v)
    return {"rewritten": q, "expanded": q + " 사유 요건 서류", "changed": "드라이런 확장"}


def _build_hyde(user: str) -> Dict[str, Any]:
    """HyDE 가상 문서의 드라이런 버전.

    실제 LLM 처럼 그럴듯한 문서를 만들 수는 없으므로, 질문 어휘에 도메인 용어를
    덧붙인 문자열을 돌려준다. **HyDE 경로가 배선대로 도는지**만 확인하는 용도다.
    """
    q = _q(user).replace("질문:", "").strip()
    return {"passage": (
        f"{q} 관련하여 근로자퇴직급여보장법 및 시행령이 정한 요건은 다음과 같다. "
        f"가입자는 사유별 증빙서류를 갖추어 퇴직연금사업자에게 제출하여야 하며, "
        f"세법상 부득이한 사유에 해당하지 않는 경우 기타소득세가 부과된다.")}


_UNIT = {"억": 100_000_000, "천만": 10_000_000, "백만": 1_000_000, "만": 10_000}


def _amounts(text: str) -> List[tuple[str, int]]:
    """'연금저축 400만원', '총급여 6000만원', '1억' 같은 표현에서 (앞말, 금액)을 뽑는다."""
    out: List[tuple[str, int]] = []
    for m in re.finditer(r"([가-힣A-Za-z]*)\s*(\d[\d,]*)\s*(억|천만|백만|만)?\s*원?", text):
        label, num, unit = m.group(1), m.group(2).replace(",", ""), m.group(3)
        if not num.isdigit():
            continue
        val = int(num) * _UNIT.get(unit or "", 1)
        out.append((label, val))
    return out


def _build_tool_args(user: str) -> Dict[str, Any]:
    """질문에서 계산 도구와 인자를 뽑는다 (LLM 추출의 규칙 버전).

    실제 LLM 만큼 정확하진 않지만, **계산기 경로가 숫자를 받아 실제로 계산까지
    도달하는지** 확인하려면 인자를 실제로 채워야 한다. 비어 있으면 드라이런에서
    항상 '입력 부족' 으로 빠져 계산 로직이 한 번도 실행되지 않는다.
    """
    q = _q(user)
    amts = _amounts(q)
    out: Dict[str, Any] = {"tool_name": "none", "missing_args": []}

    def pick(*keys: str) -> Any:
        for label, val in amts:
            if any(k in label for k in keys):
                return val
        return None

    if re.search(r"세액공제|절세", q):
        out["tool_name"] = "tax_credit_limit"
        out["total_salary"] = pick("총급여", "연봉", "급여")
        out["pension_savings_paid"] = pick("연금저축")
        out["irp_paid"] = pick("IRP", "irp")
    elif re.search(r"퇴직소득세|퇴직금.*세금|퇴직급여.*세", q):
        out["tool_name"] = "retirement_income_tax"
        out["severance_pay"] = pick("퇴직금", "퇴직급여")
        yrs = re.search(r"(\d+)\s*년", q)
        out["service_years"] = int(yrs.group(1)) if yrs else None
    elif re.search(r"중도인출.*세금|인출.*세율|인출.*세금", q):
        out["tool_name"] = "early_withdrawal_tax"
        out["amount"] = pick("인출", "출금") or (amts[0][1] if amts else None)
        out["reason"] = "부득이한사유" if re.search(r"요양|파산|개인회생|천재지변|재난", q) else "일반"
    elif re.search(r"수령한도|연금수령한도", q):
        out["tool_name"] = "withdrawal_limit"
        out["balance"] = pick("평가액", "적립금", "잔고") or (amts[0][1] if amts else None)
        yi = re.search(r"(\d+)\s*년차", q)
        out["year_index"] = int(yi.group(1)) if yi else None

    return {k: v for k, v in out.items() if v is not None}


def _build_factcheck(_: str) -> Dict[str, Any]:
    return {"total_facts": 1, "matched_facts": 1, "missing": []}


def _build_questions(_: str) -> Dict[str, Any]:
    return {"questions": ["드라이런 역질문 1", "드라이런 역질문 2"]}


def _join_blocks(user: str) -> str:
    blocks = re.split(r"\[서브질문 \d+\][^\n]*\n", user)
    return "\n\n".join(b.strip() for b in blocks[1:] if b.strip()) or "[드라이런] 합성 답변"


_BUILDERS = {
    "SafetyCheck": _build_safety,
    "QueryAnalysis": _build_analysis,
    "RouteDecision": _build_route,
    "Decomposition": _build_decomposition,
    "Sufficiency": _build_sufficiency,
    "DocumentGrade": _build_doc_grade,
    "GeneratedAnswer": _build_answer,
    "WebAnswer": _build_web_answer,
    "HallucinationCheck": _build_halluc,
    "QualityCheck": _build_quality,
    "RewrittenQuery": _build_rewrite,
    "HypotheticalDoc": _build_hyde,
    "ToolArgs": _build_tool_args,
    "FactCheck": _build_factcheck,
    "GeneratedQuestions": _build_questions,
}
