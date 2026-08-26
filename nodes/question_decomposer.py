"""질문 분리 에이전트 + 충분성 판정 (RRR 반복의 두 축).

■ 선행모델 대응
  FinSight 의 `nodes/question_decomposer.py`(878줄, v1~v6 실험 버전 전부 포함) 자리다.
  우리는 최종 채택본인 **RRR(Recursive Residual Resolution)** 하나만 구현한다.

■ RRR 의 골자
    1라운드: 질문 → 서브질문 N개 → 각각 RAG → 답 모음
             → "원 질문에 충분히 답했나?" 판정
             → 부족하면 **아직 못 답한 부분(residual)만** 다시 분해
    2라운드: 위 반복 (최대 config.MAX_ROUNDS)

■ 서브질문마다 '상품 관여도'를 매긴다 (2026-08-16 재개편)

  ▸ 1세대: 서브질문마다 `source_hint`(제도/세제/절차/…) 를 뽑아 그 소스만 보게 했다.
      분류가 흔들리면 정답 문서군을 통째로 못 봤고, 보정 규칙이 계속 늘어났다.
  ▸ 2세대: 판단을 하나로 줄였다 — "이 서브질문이 상품 추천인가?"
      추천이면 scope="all", 아니면 scope="doc"(투자설명서 7,339건을 배제).
      코퍼스의 90%가 투자설명서라 규정문서가 밑에 깔리는 문제는 풀렸지만,
      **판정 1회의 실패 비용이 코퍼스 절반**이라는 병이 남았다.
        · "○○펀드 총보수 얼마?" 는 추천이 아닌데 답은 투자설명서에만 있다 → 후보 0건
        · "세액공제 한도와 담을 채권형 펀드" 는 어느 쪽으로 찍어도 절반이 사라진다
  ▸ 3세대(지금): 배제하지 않고 **자리를 나눈다.**

      product_intent  "제도" | "혼합" | "상품"   → 0.15 / 0.50 / 0.85
        검색 결과 top-k 안에서 투자설명서가 가져갈 **자리 비율**이다.
        양쪽 모두 최소 자리가 보장되므로(config.MIN_STRATUM_QUOTA)
        판정이 틀려도 정답 코퍼스가 사라지지 않는다.

  "규정문서가 상품 문서에 눌려 안 보인다"는 원래 문제도 그대로 해결된다.
  자리를 나눠 주는 것으로 충분하고, 지울 필요는 없었던 것이다.
  애매하면 "혼합" 을 고르면 반씩 들어온다 — 2세대에는 이 선택지가 없었다.

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
    product_intent: str = Field(
        default="혼합",
        description="이 서브질문에 개별 펀드 정보가 얼마나 필요한가. "
                    "'상품'=펀드명·수익률·총보수·위험등급·클래스를 묻는다 / "
                    "'제도'=제도·세제·절차·운용규제를 묻는다 / "
                    "'혼합'=둘 다 필요하거나 애매하다. 애매하면 '혼합'.")


class Decomposition(BaseModel):
    sub_questions: List[DecomposedItem] = Field(default_factory=list)


class Sufficiency(BaseModel):
    sufficient: bool
    missing: str = Field(default="", description="빠진 부분 한 문장. 없으면 빈 문자열")


# ==========================================================================
def _affinity_of(intent: str) -> float:
    """라벨 → 상품 관여도(0.0~1.0).

    **하드 필터가 아니라 자리 배분값이다.** 목록 밖 값이 와도 죽지 않고 기본값으로
    떨어진다 — 예전 scope 방식이라면 잘못된 라벨 하나가 코퍼스 절반을 지웠다.
    """
    return config.PRODUCT_INTENT_LEVELS.get(
        (intent or "").strip(), config.DEFAULT_PRODUCT_INTENT)


def _build_subquestions(items: List[DecomposedItem], original: str,
                        filters: dict, *, round_no: int) -> List[SubQuestion]:
    """LLM 출력 → SubQuestion 객체. 서브질문별 관여도를 필터에 반영한다."""
    out: List[SubQuestion] = []
    base = filters or {}
    base_funds = list(base.get("fund_name") or [])
    # 사유는 전역에서 상속한다 — 분해하면 문장에서 빠질 수 있는데,
    # 빠졌다고 해서 그 조건이 사라진 것은 아니다(펀드명과 같은 이유).
    base_reason = list(base.get("사유") or [])

    def _mk(q: str, intent: str, qid: str) -> SubQuestion:
        aff = _affinity_of(intent)
        f: dict = {"product_affinity": aff, "fund_name": base_funds}
        if base_reason:
            f["사유"] = base_reason
        return SubQuestion(
            q,
            qid=qid,
            round_no=round_no,
            is_recommendation=aff >= 0.6,     # 답변 프롬프트 선택에만 쓴다
            filters=f,
        )

    for item in items[: config.MAX_SUB_QUESTIONS]:
        q = (item.question or "").strip()
        if not q:
            continue
        out.append(_mk(q, item.product_intent, f"r{round_no}-{uuid.uuid4().hex[:6]}"))

    if not out:                                # 분해 실패 → 원문 1건으로 진행
        # 전역 관여도를 그대로 물려받는다.
        sub = _mk(original, "", f"r{round_no}-orig")
        sub.filters["product_affinity"] = base.get(
            "product_affinity", config.DEFAULT_PRODUCT_INTENT)
        sub.is_recommendation = sub.filters["product_affinity"] >= 0.6
        out.append(sub)
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
                    f"(상품위주 {sum(1 for s in subs if s.is_recommendation)}건)",
            detail={"questions": [
                (s.question, round(s.filters.get("product_affinity", 0), 2))
                for s in subs]},
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
