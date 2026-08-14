"""질의 분석 노드 — 질문에서 **질문유형과 펀드명**을 뽑는다.

■ 선행모델 대응
  FinSight 의 `nodes/metadata_extractor.py` 자리다. 그쪽은 금융 보고서라
  {company_name, filing_year, category(정량/정성)} 를 뽑았다.

■ 왜 필터를 걷어냈나 (2026-08 개편)
  예전에는 여기서 source(5종)·계좌유형·금융주제(21종)·금융목적·asset_type·
  risk_grade·연금계좌 가능여부까지 8축을 뽑아 전부 하드 필터로 걸었다.
  선행모델을 다시 읽어보니 그쪽은 정반대였다.

      METADATA_FILTER_INIT = ["company_name", "year"]
      metadata = {k: v for k, v in metadata.items() if k in metadata_filters}

  topics 를 LLM 으로 뽑아놓고 이 한 줄에서 **버린다.** 청크 쪽 topic 은 아예
  "Other" 하드코딩이고, 답변 프롬프트에서도 명시적으로 제외한다
  (field_to_ignore_from_metadata_for_generation 에 "topic" 이 두 번 적혀 있다).

  하드 필터로 쓴 건 **문서 단위로 100% 확정되는 값 둘**뿐이다. 회사명과 연도는
  문서 앞 10페이지에서 한 번 뽑아 그 문서의 모든 청크에 복사하므로 누락이
  구조적으로 불가능하다. 우리 주제 태그는 그렇지 못했고, 실측에서 정답을 지웠다.

■ 그래서 지금 뽑는 것 (둘뿐)
  1) 질문유형 — 필터가 아니라 **검색 폭·라우팅·표 가산**을 정한다
       추천이면 top_k 5→24, 계산/사실조회면 표 청크 ×1.25
  2) fund_name — 선행모델의 company_name 과 정확히 같은 성격.
       투자설명서 1권 = 펀드 1개라, 그 문서의 전 청크에 대해 참인 확정값이다.

  scope(doc / doc+투자설명서)는 여기서 정하지 않는다. **서브질문마다** 다르므로
  분해 단계에서 서브질문 단위로 판단한다 (nodes/question_decomposer.py).

■ 펀드명 목록 주입 — 선행모델의 db.get_companies() 대응
  프롬프트에 실제 펀드명 92개를 넣는다. 선행모델도 같은 방식이었다.

      companies_set = db.get_companies()
      metadata_extractor.invoke({"query": question, "company_set": companies_set})
      → "Ensure that the output is an **exact match** with one of the parent
         company names in the provided list."

  이게 없으면 LLM 이 "미래에셋 배당주 펀드" 같은 존재하지 않는 이름을 만들어내고,
  그 이름으로 필터를 걸면 후보가 0건이 되어 파이프라인 전체가 헛돈다.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

import config
from index import fund_names
from llm import llm
from prompt import prompts
from state import OverallState
from utils import Timer, step


class QueryAnalysis(BaseModel):
    질문유형: str = Field(description="사실조회 | 계산 | 추천 | 절차 | 혼합")
    fund_name: List[str] = Field(
        default_factory=list,
        description="질문에 등장하는 펀드명. 제공된 목록의 값과 정확히 일치해야 한다. "
                    "없으면 빈 배열")


# --------------------------------------------------------------------------
def _sanitize(a: QueryAnalysis, known: List[str]) -> dict:
    """LLM 이 목록 밖 값을 뱉는 경우를 막는다 (환각 필터링).

    선행모델도 topic 추출 후 `valid_topics = [t for t in topics if t in topics_set]`
    로 걸렀다. 같은 방어다 — 없는 이름으로 필터를 걸면 검색 결과가 0건이 되어
    파이프라인 전체가 헛돈다.

    프롬프트 주입(사전 방어)과 사후 검증(이 함수)을 **둘 다** 둔다.
    """
    qt = a.질문유형 if a.질문유형 in config.QUERY_TYPES else "사실조회"

    known_set = set(known)
    clean: List[str] = []
    for n in a.fund_name:
        n = (n or "").strip()
        if not n:
            continue
        if n in known_set:
            clean.append(n)
            continue
        # 부분 표현("미래에셋퇴직플랜") → 정식 명칭으로 승격. 후보가 하나일 때만.
        cands = [k for k in known if n in k]
        if len(cands) == 1:
            clean.append(cands[0])

    return {"질문유형": qt, "fund_name": clean[:3]}


def analyze_query(state: OverallState) -> dict:
    question = state["question"]
    known = fund_names()

    with Timer() as t:
        res = llm.structured(
            prompts.query_analyzer_system(known),
            f"질문: {question}",
            QueryAnalysis,
            model=config.MODEL_PLAN,
            stage="analyze",
        )

    clean = _sanitize(res, known)
    qt = clean["질문유형"]

    # 전역 필터는 기본 scope 만 담는다. 서브질문마다 분해기가 덮어쓴다.
    filters = {
        "scope": "all" if qt == "추천" else config.DEFAULT_SCOPE,
        "fund_name": clean["fund_name"],
    }

    return {
        "질문유형": qt,
        "filters": filters,
        "is_recommendation": qt == "추천",
        "llm_calls": 1,
        "trace": step(
            "analyze_query",
            summary=f"질문유형={qt}, scope={filters['scope']}"
                    + (f", 펀드={filters['fund_name']}" if filters["fund_name"] else ""),
            detail=filters,
            ms=t.ms,
        ),
    }
