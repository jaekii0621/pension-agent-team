"""State 정의 — 그래프를 돌아다니는 '상자'의 설계도.

선행모델(FinSight)의 state.py 와 같은 역할이다. 핵심 개념 두 개만 기억하면 된다.

1) State 는 TypedDict 다. 노드는 **바꿀 키만** dict 로 반환하면 LangGraph 가 합쳐 준다.
2) `Annotated[타입, 합치는함수]` = **리듀서(reducer)**.
   서브질문 3개를 병렬로 처리하면 3개 노드가 같은 키에 동시에 쓴다.
   리듀서가 없으면 마지막 하나만 남고 나머지가 사라진다. 그래서 리스트 필드는
   전부 `operator.add`(이어붙이기) 나 전용 병합 함수를 붙인다.

State 를 4개로 나눈 이유(선행모델과 동일한 판단):
  · OverallState  : 파이프라인 전체가 공유. 크다.
  · RAGState      : 서브질문 **하나**를 푸는 동안만 쓰는 작은 상자.
  · CalcState     : 계산기 에이전트 전용.
  · 서브그래프는 자기 일에 필요한 필드만 보게 해서 결합도를 낮춘다.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


# ==========================================================================
# 리듀서 (병렬 실행 결과를 합치는 함수들)
# ==========================================================================
def merge_dict(old: Optional[dict], new: Optional[dict]) -> dict:
    """딕셔너리 얕은 병합. 나중 값이 이긴다."""
    if not old:
        return new or {}
    if not new:
        return old
    out = dict(old)
    out.update(new)
    return out


def keep_last(old: Any, new: Any) -> Any:
    """None 이 아닌 새 값이 오면 덮어쓴다. None 이면 기존 값 유지.

    병렬 브랜치 중 하나만 값을 채우는 필드(final_answer 등)에 쓴다.
    """
    return old if new is None else new


def dedup_citations(old: List[dict], new: List[dict]) -> List[dict]:
    """인용 목록을 합치되 chunk_id 기준 중복을 제거한다.

    같은 청크가 여러 서브질문에서 인용되는 일이 흔해서, 그냥 operator.add 로
    이어붙이면 최종 답변 하단의 출처 목록에 같은 문서가 5번씩 찍힌다.
    """
    old = old or []
    new = new or []
    seen: set = set()
    out: List[dict] = []
    for c in list(old) + list(new):
        key = c.get("chunk_id") or c.get("url") or c.get("text", "")[:60]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def add_int(old: int, new: int) -> int:
    """카운터(LLM 호출 수 등)를 병렬 브랜치에서 모아 더한다."""
    return (old or 0) + (new or 0)


def or_bool(old: bool, new: bool) -> bool:
    """병렬 브랜치 중 하나라도 True 면 True.

    ※ 이 리듀서가 없으면 서브질문 2개가 동시에 web_used 에 쓰는 순간
      LangGraph 가 InvalidUpdateError("Can receive only one value per step")로 죽는다.
      **병렬로 갈라지는 그래프에서 여러 노드가 건드리는 키는 예외 없이 리듀서가 필요하다.**
      (실제로 이 프로젝트에서 그 에러를 한 번 맞고 추가한 함수다.)
    """
    return bool(old) or bool(new)


# ==========================================================================
# 서브질문 트리 노드
#   선행모델의 QuestionNode 대응. LangGraph 가 커스텀 클래스를 직렬화하지 못해서
#   to_dict()/from_dict() 로 dict 화해서 state 에 넣는다(선행모델도 같은 방식).
# ==========================================================================
class SubQuestion:
    def __init__(
        self,
        question: str,
        *,
        qid: str = "",
        round_no: int = 0,
        is_recommendation: bool = False,
        filters: Optional[dict] = None,
    ):
        self.qid = qid
        self.question = question
        self.round_no = round_no
        self.is_recommendation = is_recommendation  # 상품 추천 질문인가 → scope 결정
        self.filters = filters or {}                # 검색 필터 (아래 QueryFilters 형태)
        self.answer: Optional[str] = None
        self.citations: List[dict] = []
        self.evidence_ok: bool = False              # 근거 있는 답인가

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "question": self.question,
            "round_no": self.round_no,
            "is_recommendation": self.is_recommendation,
            "filters": self.filters,
            "answer": self.answer,
            "citations": self.citations,
            "evidence_ok": self.evidence_ok,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SubQuestion":
        o = cls(
            d["question"],
            qid=d.get("qid", ""),
            round_no=d.get("round_no", 0),
            is_recommendation=bool(d.get("is_recommendation", False)),
            filters=d.get("filters"),
        )
        o.answer = d.get("answer")
        o.citations = d.get("citations", [])
        o.evidence_ok = d.get("evidence_ok", False)
        return o


# ==========================================================================
# 검색 필터 — 선행모델의 {company_name, year} 자리에 들어가는 우리 스키마
# ==========================================================================
class QueryFilters(TypedDict, total=False):
    """질문에서 뽑아내는 검색 조건. **축이 둘뿐이다.**

    선행모델이 하드 필터로 쓴 것도 둘이었다(company_name, year). 그쪽이 LLM 으로
    뽑기까지 한 topics 를 화이트리스트에서 제외한 이유와 같다 — 필터는 질문 쪽 값과
    청크 쪽 값의 완전일치 매칭이라, **양쪽 모두 누락이 없어야** 성립한다.

    scope      : "doc"  = 규정문서(제도·세제·절차·운용규제)만
                 "all"  = 규정문서 + 투자설명서
                 서브질문마다 LLM 이 '추천인가' 하나만 판단해 정한다.
    fund_name  : 질문에 특정 펀드가 명시됐을 때만. 투자설명서 1권 = 펀드 1개라
                 그 문서의 전 청크에 대해 참인 확정값이다(= company_name 과 동격).

    나머지(주제·계좌유형·목적·자산유형·위험등급)는 전부 **점수 가중과 프롬프트**로
    내렸다. 가중은 후보를 지우지 않으므로 잘못 걸려도 순위만 밀린다.
    """
    scope: str                   # "doc" | "all"
    fund_name: List[str]         # ["미래에셋퇴직플랜증권자투자신탁1호(채권)"]


# ==========================================================================
# 최상위 State
# ==========================================================================
class OverallState(TypedDict, total=False):
    # --- 입력 ---
    session_id: str
    question: str
    user_profile: dict           # 나이·총급여·계좌종류 등 (계산기/전략이 쓴다)

    # --- 안전성 ---
    safe: bool
    safety_reason: str

    # --- 질의 분석 (선행모델의 extract_metadata 에 대응) ---
    질문유형: Literal["사실조회", "계산", "추천", "절차", "혼합"]
    filters: Annotated[Dict[str, Any], merge_dict]
    is_recommendation: bool

    # --- 라우팅 ---
    route: str                   # "rag" | "calc" | "recommend" | "out_of_scope"

    # --- 분해 & 반복(RRR) ---
    round_no: int
    sub_questions: List[dict]            # 현재 라운드의 서브질문들
    qa_pairs: Annotated[List[dict], operator.add]   # 누적 (질문,답,근거)
    asked_history: Annotated[List[str], operator.add]  # 중복 분해 방지
    sufficient: bool
    residual_question: str

    # --- 결과 ---
    draft_answer: Annotated[Optional[str], keep_last]
    final_answer: Annotated[Optional[str], keep_last]
    citations: Annotated[List[dict], dedup_citations]
    calc_result: Annotated[Dict[str, Any], merge_dict]
    strategy_note: Annotated[Optional[str], keep_last]

    # --- 검증 ---
    verification: Annotated[Dict[str, Any], merge_dict]
    verify_retries: int

    # --- 계측 (성능 측정 단계라 이게 중요하다) ---
    trace: Annotated[List[dict], operator.add]
    llm_calls: Annotated[int, add_int]
    web_used: Annotated[bool, or_bool]     # ← 병렬 rag_worker 들이 함께 쓴다


# ==========================================================================
# 서브질문 1개를 푸는 동안만 쓰는 작은 State
# ==========================================================================
class RAGState(TypedDict, total=False):
    # 입력
    qid: str
    question: str                # 서브질문 (재작성되면 바뀐다)
    original_question: str       # 사용자 원 질문 (채점 기준으로 쓴다)
    질문유형: str
    filters: Dict[str, Any]      # {"scope": "doc"|"all", "fund_name": [...]}
    is_recommendation: bool      # 이 서브질문이 상품 추천인가

    # 검색
    documents: List[dict]
    applied_filters: Dict[str, Any]
    filter_retries: int
    grading_retries: int
    irrelevance_reason: str
    hyde_doc: str                # HyDE 가 만든 가상 답변 (검색 전용, 답변 근거 아님)

    # 생성
    answer: str
    citations: List[dict]
    evidence_ok: bool

    # 검증
    verdict: Dict[str, Any]
    verify_retries: int

    # 폴백
    web_used: bool

    # 계측
    trace: Annotated[List[dict], operator.add]
    llm_calls: Annotated[int, add_int]


# ==========================================================================
# 계산기 State
# ==========================================================================
class CalcState(TypedDict, total=False):
    question: str
    user_profile: dict
    tool_name: str               # 호출할 계산 함수 이름
    tool_args: Dict[str, Any]    # LLM 이 뽑은 인자
    missing_args: List[str]      # 부족한 입력 (되물어야 할 것)
    result: Dict[str, Any]
    explanation: str
    trace: Annotated[List[dict], operator.add]
    llm_calls: Annotated[int, add_int]
