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
    """질문에서 뽑아내는 검색 조건. **코퍼스를 지우는 축이 하나도 없다.**

    필터는 질문 쪽 값과 청크 쪽 값의 완전일치 매칭이라, 양쪽 모두 누락이 없어야
    성립한다. 그 조건을 만족하지 못하는 축은 전부 배제형에서 뺐다.

    product_affinity : 0.0~1.0. **필터가 아니라 자리 배분값이다.**
                 top-k 안에서 투자설명서가 가져갈 비율을 정한다. 규정문서와
                 투자설명서 모두 최소 자리를 보장받으므로(MIN_STRATUM_QUOTA),
                 이 값이 틀려도 정답 코퍼스가 사라지지 않는다.
                 (예전 `scope: "doc"|"all"` 을 대체한다. scope 가 들어와도
                  index.store._affinity_of 가 근사값으로 변환해 준다.)
    사유         : 장르C(중도인출) 전용 **소프트 필터**. 5개 문서의 도입부가 동일해
                 검색으로는 구분되지 않아 필요하다. 결과가 모자라면 완화 사다리
                 1단계에서 자동으로 풀린다. `사유` 값을 가진 청크만 대조하므로
                 다른 코퍼스를 지우지 않는다.
    fund_name  : 유일한 배제형 조건. 투자설명서 1권 = 펀드 1개라 그 문서의 전 청크에
                 대해 참인 확정값이고(= 선행모델 company_name 과 동격), 펀드명이
                 없는 규정문서 청크는 배제하지 않는다.

    나머지(주제·계좌유형·목적·자산유형·위험등급)는 전부 **점수 가중과 프롬프트**로
    내렸다. 가중은 후보를 지우지 않으므로 잘못 걸려도 순위만 밀린다.
    """
    product_affinity: float      # 0.0(제도) ~ 1.0(상품)
    사유: List[str]              # ["주택구입"] — config.REASON_VALUES 중
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
    safety_reason: str           # 사람이 읽는 판정 이유 한 문장
    safety_category: str         # 기계가 읽는 라벨: 개인정보|위법조력|유해|범위밖|""
    safety_caution: str          # 차단은 아니지만 답변 단계가 알아야 하는 신호

    # --- 질의 분석 (선행모델의 extract_metadata 에 대응) ---
    질문유형: Literal["사실조회", "계산", "추천", "절차", "혼합"]
    filters: Annotated[Dict[str, Any], merge_dict]
    is_recommendation: bool

    # --- 슬롯 해소 / 역질문 ---
    #   답이 갈리는 값(사유)을 확정했는지, 못 했으면 무엇을 되물을지.
    #   되묻기는 **실패가 아니라 정상 종료**다 — 정보가 부족한 상태의 단정 답변보다
    #   한 번 되묻는 편이 정확하기 때문이다.
    slots: Annotated[Dict[str, Any], merge_dict]
    needs_clarification: bool
    clarification: str

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

    # 답변이 말했는데 근거에 없는 값 — 환각 후보 (2026-08-28)
    #
    # ⚠️ RAGState 쪽과 마찬가지로 **여기 선언이 없으면 LangGraph 가 버린다.**
    #    combine_answer 가 아무리 계산해서 반환해도 상태에 남지 않는다.
    #
    #   unsupported_values      최종 답변(재작성 후) 기준 — 이게 판정 대상이다
    #   unsupported_values_pre  서브답변(재작성 전) 기준 — 예전에 쓰던 값
    #
    # 둘을 다 남기는 이유: pre - final = 재작성이 걸러준 값,
    # final - pre = **재작성이 새로 만들어낸 값**. 후자가 나오면 combiner 가
    # 값을 지어내고 있다는 뜻이라 더 중요한 신호다. 최종값 하나로 덮어쓰면
    # 이 구분이 사라진다.
    unsupported_values: Annotated[Optional[List[str]], keep_last]
    unsupported_values_pre: Annotated[Optional[List[str]], keep_last]

    # --- 검증 ---
    verification: Annotated[Dict[str, Any], merge_dict]
    verify_retries: int

    # --- 계측 (성능 측정 단계라 이게 중요하다) ---
    trace: Annotated[List[dict], operator.add]
    llm_calls: Annotated[int, add_int]


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
    safety_caution: str          # 안전검사가 넘긴 주의 신호 (위법조력|유해|프롬프트공격)

    # 생성
    answer: str
    citations: List[dict]
    evidence_ok: bool

    # ⚠️ 아래 둘이 없어서 LangGraph 가 값을 버렸다 (2026-08-27 발견).
    #    TypedDict 에 선언되지 않은 키는 노드가 반환해도 상태에 남지 않는다.
    #    그래서 6차 결과에서 evidence_sufficient 는 77문항 전부 False,
    #    unsupported_values 는 전부 빈 배열이었다 — 값이 그런 게 아니라
    #    전달이 안 된 것이다. 둘 다 진단용으로 일부러 올려보내던 값이다.
    evidence_sufficient: bool          # 모델이 "근거만으로 답할 수 있다" 고 봤나
    unsupported_values: List[str]      # 답변이 말했는데 근거에 없는 값 (환각 후보)

    # 검증
    verdict: Dict[str, Any]
    verify_retries: int

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
