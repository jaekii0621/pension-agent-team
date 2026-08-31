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

import logging
from typing import List

from pydantic import BaseModel, Field

import config
from index import (fund_core, fund_names, fund_variant, match_known_funds,
                   product_signal, sibling_funds)
from llm import llm
from prompt import prompts
from state import OverallState
from utils import Timer, step

log = logging.getLogger(__name__)


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
        # 부분 표현("미래에셋퇴직플랜") → 정식 명칭으로 승격.
        cands = [k for k in known if n in k]
        if len(cands) == 1:
            clean.append(cands[0])
        elif cands and len({fund_core(k) for k in cands}) == 1:
            # 후보가 여럿이어도 **전부 같은 상품군의 형제**면 전부 살린다
            # (2026-08-31, 9차 이슈 3).
            #
            # 예전에는 `len(cands) == 1` 이 아니면 통째로 버렸다. 그래서
            # "한국투자골드플랜연금" 처럼 상품군을 부르는 말이 들어오면 후보 3건이
            # 잡히고도 fund_name 이 빈 배열이 됐다. 형제끼리는 어느 것을 골라도
            # 틀리므로 **고르지 말고 다 넣는 것**이 맞다 — 검색은 배제형 필터라
            # 여럿을 넣어도 좁혀지기만 하지 넓어지지 않는다.
            clean.extend(sorted(cands))

    # 중복 제거(순서 유지) 후 상한. 형제군이 최대 3개라 상한을 형제 상한에 맞춘다.
    uniq = list(dict.fromkeys(clean))
    return {"질문유형": qt, "fund_name": uniq[:config.MAX_SIBLING_FUNDS]}


def _resolve_siblings(question: str, names: List[str]) -> List[str]:
    """형제펀드(같은 상품군)를 질문에 맞춰 좁히거나 넓힌다 (2026-08-31, 9차 이슈 3).

    ■ 두 방향이 모두 필요하다

      좁히기 — 질문이 변별자를 적어 왔으면 그 하나만 남긴다.
          "골드플랜연금 **주식형** 총보수?" → (주식) 하나.
          `match_known_funds` 는 접두어 동점이라 형제 셋을 다 돌려주는데,
          그대로 두면 답변 근거에 상관없는 형제 카드가 섞여 답이 흐려진다.

      넓히기 — 변별자가 없으면 형제를 **전부** 넣는다.
          "골드플랜연금은 어떤 상품인가요?" → 셋 다.
          `store._passes` 의 fund_name 은 **배제형** 필터라, 형제를 빼 두면
          그 문서가 검색 후보에서 통째로 사라진다. 어느 쪽인지 모르는 상태에서
          하나를 골라 나머지를 배제하면, 모델은 남은 하나의 수치를 다른 형제
          자리에 복사하는 수밖에 없다 — 실측 P-B1 이 정확히 그 경로였다.

    ■ 상한
      config.MAX_SIBLING_FUNDS 로 자른다. 형제군은 코퍼스 최대 3개라 여유가 있다.
    """
    if not names:
        return []

    q = (question or "").replace(" ", "")
    out: List[str] = []
    seen_core = set()
    for n in names:
        core = fund_core(n)
        if core in seen_core:
            continue
        seen_core.add(core)
        group = sibling_funds(n)
        if len(group) > 1:
            # 질문이 변별자를 명시했으면 그 형제만 남긴다.
            pinned = [g for g in group if (v := fund_variant(g)) and v in q]
            group = pinned or group
        for g in group:
            if g not in out:
                out.append(g)

    if len(out) != len(names):
        log.info("형제펀드 정리: %d건 → %d건", len(names), len(out))
    return out[:config.MAX_SIBLING_FUNDS]


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

    # ── 펀드명 2중 확보: LLM 추출 + 목록 대조 (2026-08-18 보강) ──────────
    #
    # ★ 여기가 P-F4·P-G0a·P-G0b 실패의 수정 지점이다.
    #
    # 예전에는 펀드명 추출을 **LLM 에게만** 맡겼다. 그런데 HCX-DASH-002 가
    # 세 문항에서 이걸 놓쳤다 — 셋 다 질문에 펀드명이 그대로 적혀 있었는데도다.
    #
    #     "미래에셋장기성장포커스 중간에 깨면 환매수수료 물어야 하나요?"  (P-F4)
    #     "미래에셋장기성장포커스 운용사가 … 같은 계열인가요?"            (P-G0a)
    #     "미래에셋장기성장포커스에 넣은 돈은 어디에 보관되나요?"          (P-G0b)
    #
    # 놓치면 fund_name 이 빈 배열이 되고, 관여도가 기본값 0.35 에 머문다.
    # 0.35 면 투자설명서가 top-k 8칸 중 3칸만 받는다. 답이 있는 섹션 원문에
    # 구조적으로 도달할 수 없다 — 검색 이전에 여기서 이미 진 것이다.
    #
    # 안전검사 노드는 같은 판단을 **목록 대조**로 하고 있었고 세 문항 모두 잡았다.
    # 그 함수를 index 계층으로 올려 여기서도 쓴다. 문자열 비교라 오탐이 없고
    # LLM 호출도 0회다. LLM 이 잡으면 그대로 쓰고, 놓치면 이쪽이 받친다.
    # ⚠️ 목록 대조도 **형제를 전부** 돌려받는다 (2026-08-31, 9차 이슈 3).
    #    접두어가 동점인 형제 중 하나만 잡으면 나머지가 검색에서 배제된다
    #    (store._passes 의 fund_name 은 배제형 필터다). 실측 P-B1 이 그 경로로
    #    채권 펀드 문서를 통째로 못 봤다 — loader.match_known_funds 주석 참고.
    fund_from_list = False
    if not clean["fund_name"]:
        hits = match_known_funds(question)
        if hits:
            clean["fund_name"] = hits
            fund_from_list = True
            log.info("LLM 이 놓친 펀드명을 목록 대조로 확보: %s",
                     ", ".join(h[:30] for h in hits))

    clean["fund_name"] = _resolve_siblings(question, clean["fund_name"])

    # 전역 필터는 기본값만 담는다. 서브질문마다 분해기가 덮어쓴다.
    #
    # product_affinity 는 **하드 필터가 아니라 자리 배분값**이다(config 참고).
    # 펀드명이 지목됐다면 그 자체가 "상품 정보가 필요하다"는 확정 신호이므로
    # 관여도를 올린다 — 이건 LLM 판정이 아니라 목록 대조로 확정된 값이라 안전하다.
    is_product, product_ratio = product_signal(question)
    if clean["fund_name"]:
        affinity = config.PRODUCT_INTENT_LEVELS["상품"]
    elif qt == "추천":
        affinity = config.PRODUCT_INTENT_LEVELS["상품"]
    elif is_product:
        # 펀드 이름은 없지만 **상품 어휘**로 확정되는 질문 (2026-08-20).
        #
        #   판정은 코퍼스에 직접 묻는다 — 투자설명서 BM25 가 규정문서 BM25 보다
        #   높으면 상품 질문으로 본다(index.product_signal 주석 참고).
        #   실측에서 제도 문항을 상품으로 오판한 건이 0건이라 올리는 방향으로 안전하다.
        #
        #   '상품'(0.85) 이 아니라 '혼합'(0.50) 을 쓴다. 비율 신호는 펀드명 대조만큼
        #   확정적이지 않고, 이 부류는 제도 설명이 함께 필요한 전제교정 질문이 많다.
        #   양쪽에 자리를 나눠 주는 편이 안전하다.
        affinity = config.PRODUCT_INTENT_LEVELS["혼합"]
    else:
        affinity = config.DEFAULT_PRODUCT_INTENT

    filters = {
        "product_affinity": affinity,
        "fund_name": clean["fund_name"],
    }

    return {
        "질문유형": qt,
        "filters": filters,
        "is_recommendation": qt == "추천",
        "llm_calls": 1,
        "trace": step(
            "analyze_query",
            summary=f"질문유형={qt}, 상품관여도={affinity:.2f}"
                    + (f", 펀드={filters['fund_name']}" if filters["fund_name"] else "")
                    + (", 펀드명=목록대조로 확보" if fund_from_list else "")
                    + (f", 상품신호 비율={product_ratio:.2f}"
                       if (is_product and not clean["fund_name"]) else ""),
            detail=filters,
            ms=t.ms,
        ),
    }
