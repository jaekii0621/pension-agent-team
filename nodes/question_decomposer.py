"""질문 분리 에이전트 + 충분성 판정 (RRR 반복의 두 축).

■ 선행모델 대응
  FinSight 의 `nodes/question_decomposer.py`(878줄, v1~v6 실험 버전 전부 포함) 자리다.
  우리는 최종 채택본인 **RRR(Recursive Residual Resolution)** 하나만 구현한다.

■ RRR 의 골자
    1라운드: 질문 → 서브질문 N개 → 각각 RAG → 답 모음
             → "원 질문에 충분히 답했나?" 판정
             → 부족하면 **아직 못 답한 부분(residual)만** 다시 분해
    2라운드: 위 반복 (최대 config.MAX_ROUNDS)

■ 서브질문마다 검색 범위를 정한다 (2026-08 개편)
  예전에는 서브질문마다 `source_hint`(제도/세제/일반절차/중도인출절차/상품)를 뽑아
  그 소스 문서만 보게 했고, 소스가 곧 전문가 프롬프트였다. 두 가지가 문제였다.

    · 소스 5종 분류가 LLM 판단이라 실행마다 흔들리는데, 틀리면 정답 문서군을
      통째로 못 본다. 보정 규칙(_apply_domain_rules)이 계속 늘어났다.
    · 연금 도메인에는 선행모델의 회사명·연도처럼 **문서 단위로 확정되는 축이 없다.**
      억지로 만든 축은 필터가 아니라 잡음이었다.

  그래서 판단을 하나로 줄였다 — **이 서브질문이 상품 추천인가?**

      추천이다   → scope="all"  : doc 문서 + 투자설명서 전체를 본다
      아니다     → scope="doc"  : 제도·세제·절차·운용규제 문서만 본다

  이 이분법이 유효한 이유는 코퍼스 구성 때문이다. 투자설명서가 전체의 91%를
  차지해서, 추천이 아닌 질문에 상품 문서를 열어 두면 제도·절차 문서가 그 밑에
  깔린다. 반대로 추천 질문에 doc 만 보면 펀드 정보가 아예 없다.
  실측(persona 시뮬레이션): "IRP 가입 가능한 채권형 펀드는?" 을 필터 없이 검색하면
  상위 5건이 전부 제도 Q&A 였고 펀드카드는 0건이었다.

■ 자기완결성이 필터를 대신한다
  선행모델은 분해 규칙에서 self-contained 를 강제했다.
      "Each subquestion should be self-contained with all the information
       necessary to solve it."
      "Each of the decomposed question should be dealing in only a single year"
  → 분해 후에도 회사명·연도가 문장에 남으므로, 서브질문에서 필터를 **다시 뽑아도**
    같은 값이 나온다. 우리도 같은 규칙을 넣었다. 조건이 문장에 남아 있으면
    별도의 필터 정제 노드(옛 filter_refiner)가 필요 없다 — LLM 호출 1회가 준다.
"""

from __future__ import annotations

import uuid
from typing import List

from pydantic import BaseModel, Field

import config
from llm import llm
from prompt import prompts
from state import OverallState, SubQuestion
from utils import Timer, step, truncate


# ==========================================================================
# 스키마
# ==========================================================================
class DecomposedItem(BaseModel):
    question: str = Field(description="단독으로 뜻이 통하는 서브질문")
    is_recommendation: bool = Field(
        default=False,
        description="이 서브질문이 '어떤 펀드/상품을 고를까'를 묻는가. "
                    "펀드명·수익률·총보수·위험등급·클래스를 물으면 true. "
                    "제도·세제·절차·운용규제를 물으면 false")


class Decomposition(BaseModel):
    sub_questions: List[DecomposedItem] = Field(default_factory=list)


class Sufficiency(BaseModel):
    sufficient: bool
    missing: str = Field(default="", description="빠진 부분 한 문장. 없으면 빈 문자열")


# ==========================================================================
def _scope_of(is_recommendation: bool) -> str:
    """서브질문의 검색 범위. 이것이 우리 파이프라인의 유일한 하드 필터다."""
    return "all" if is_recommendation else "doc"


def _build_subquestions(items: List[DecomposedItem], original: str,
                        filters: dict, *, round_no: int) -> List[SubQuestion]:
    """LLM 출력 → SubQuestion 객체. 서브질문별 scope 를 필터에 반영한다."""
    out: List[SubQuestion] = []
    base_funds = list((filters or {}).get("fund_name") or [])

    for item in items[: config.MAX_SUB_QUESTIONS]:
        q = (item.question or "").strip()
        if not q:
            continue
        rec = bool(item.is_recommendation)
        out.append(SubQuestion(
            q,
            qid=f"r{round_no}-{uuid.uuid4().hex[:6]}",
            round_no=round_no,
            is_recommendation=rec,
            # 펀드명은 전역에서 상속한다 — 분해하면 문장에서 빠질 수 있고,
            # 빠졌다고 해서 그 조건이 사라진 것은 아니다.
            filters={"scope": _scope_of(rec), "fund_name": base_funds},
        ))

    if not out:                                # 분해 실패 → 원문 1건으로 진행
        rec = (filters or {}).get("scope") == "all"
        out.append(SubQuestion(
            original,
            qid=f"r{round_no}-orig",
            round_no=round_no,
            is_recommendation=rec,
            filters={"scope": _scope_of(rec), "fund_name": base_funds},
        ))
    return out


# ==========================================================================
# 1라운드 분해
# ==========================================================================
def decompose_question(state: OverallState) -> dict:
    question = state["question"]
    filters = state.get("filters", {}) or {}

    if not config.WORKFLOW_SETTINGS["decompose_question"]:
        subs = _build_subquestions([], question, filters, round_no=0)
        return {
            "sub_questions": [s.to_dict() for s in subs],
            "round_no": 0,
            "trace": step("decompose", summary="비활성(설정) — 원문 1건"),
        }

    with Timer() as t:
        res = llm.structured(
            prompts.decomposer_system,
            f"질문: {question}\n"
            f"참고 - 질문유형: {state.get('질문유형')}",
            Decomposition,
            model=config.MODEL_PLAN,
            stage="decompose",
        )

    subs = _build_subquestions(res.sub_questions, question, filters, round_no=0)

    return {
        "sub_questions": [s.to_dict() for s in subs],
        "asked_history": [s.question for s in subs],
        "round_no": 0,
        "llm_calls": 1,
        "trace": step(
            "decompose",
            summary=f"{len(subs)}개로 분리 "
                    f"(추천 {sum(1 for s in subs if s.is_recommendation)}건)",
            detail={"questions": [(s.question, s.filters.get("scope")) for s in subs]},
            ms=t.ms,
        ),
    }


# ==========================================================================
# 충분성 판정 (라운드 종료 조건)
# ==========================================================================
def check_sufficiency(state: OverallState) -> dict:
    """모은 QA 쌍으로 원 질문에 답할 수 있는지 판정한다."""
    question = state["question"]
    qa_pairs = state.get("qa_pairs", [])
    round_no = state.get("round_no", 0)

    if not qa_pairs:
        return {"sufficient": True,
                "trace": step("check_sufficiency", summary="QA 없음 → 종료")}

    # 마지막 라운드면 판정 없이 종료 (LLM 호출 절약)
    if round_no >= config.MAX_ROUNDS - 1 or not config.WORKFLOW_SETTINGS["repeat_residual"]:
        return {"sufficient": True,
                "trace": step("check_sufficiency",
                              summary=f"라운드 한도({config.MAX_ROUNDS}) 도달 → 종료")}

    qa_text = "\n\n".join(
        f"[Q] {p['question']}\n[A] {truncate(p.get('answer') or '(답 없음)', 400)}"
        for p in qa_pairs
    )

    with Timer() as t:
        res = llm.structured(
            prompts.sufficiency_system,
            f"원 질문: {question}\n\n지금까지 모은 내용:\n{qa_text}",
            Sufficiency,
            model=config.MODEL_VERIFY,
            stage="sufficiency",
        )

    return {
        "sufficient": res.sufficient,
        "residual_question": res.missing,
        "llm_calls": 1,
        "trace": step("check_sufficiency",
                      summary=f"{'충분' if res.sufficient else '부족: ' + res.missing}",
                      ms=t.ms),
    }


# ==========================================================================
# 2라운드 이후 — 잔여 부분만 재분해
# ==========================================================================
def decompose_residual(state: OverallState) -> dict:
    question = state["question"]
    qa_pairs = state.get("qa_pairs", [])
    asked = state.get("asked_history", [])
    filters = state.get("filters", {}) or {}
    round_no = state.get("round_no", 0) + 1

    answered = "\n".join(f"- {p['question']}" for p in qa_pairs)

    with Timer() as t:
        res = llm.structured(
            prompts.decomposer_residual_system,
            f"원 질문: {question}\n\n"
            f"이미 답을 얻은 서브질문들:\n{answered}\n\n"
            f"부족하다고 판정된 부분: {state.get('residual_question', '(미지정)')}",
            Decomposition,
            model=config.MODEL_PLAN,
            stage="decompose_residual",
        )

    # 이미 물어본 질문은 제외 (중복 검색 = 비용 낭비)
    fresh = [i for i in res.sub_questions if i.question.strip() not in set(asked)]
    subs = _build_subquestions(fresh, question, filters, round_no=round_no) if fresh else []

    return {
        "sub_questions": [s.to_dict() for s in subs],
        "asked_history": [s.question for s in subs],
        "round_no": round_no,
        "llm_calls": 1,
        "trace": step("decompose_residual",
                      summary=f"라운드{round_no}: {len(subs)}개 추가",
                      detail={"questions": [s.question for s in subs]},
                      ms=t.ms),
    }
