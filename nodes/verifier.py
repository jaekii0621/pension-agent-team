"""검증 에이전트 — 환각 검사 + 답변 품질 채점.

■ 선행모델 대응
  FinSight 는 이 역할을 두 파일로 나눠 뒀다.
    · hallucination_checker.py       (LLM 으로 검사)
    · hallucination_checker_hhem.py  (HHEM 전용 모델로 검사)
    · answer_grader.py               (답변 품질)
  요구사항에 따라 **HHEM 은 쓰지 않는다.** 전용 모델 하나를 더 띄우고 관리하는 비용
  대비, 우리 코퍼스 규모에서는 LLM 검사로 충분하다고 판단했다.
  대신 두 검사를 **한 노드 안에서 한 번의 호출로 묶어** 비용을 절반으로 줄였다.

■ 두 검사를 나눠 생각해야 하는 이유 (중요)
  · 환각 검사 = "근거에 있는 말만 했는가"   → 답이 **틀리지 않았는가**
  · 품질 채점 = "질문에 답했는가"            → 답이 **쓸모 있는가**
  둘은 독립이다. 근거만 그대로 옮겨 적고 질문엔 답 안 한 답변은
  환각 0점이지만 품질 0점이다. 반대도 성립한다.

■ 실패 시 행동 (edges/verification_check.py 가 결정)
  · 환각 있음  → 답변 재생성 (같은 근거, 더 보수적으로)
  · 품질 미달  → 질문 재작성 후 재검색
  · 재시도 한도 초과 → 웹 검색 폴백
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

import config
from index import format_chunk_for_llm
from llm import llm
from prompt import prompts
from state import RAGState
from utils import Timer, log_message, step, truncate


class HallucinationCheck(BaseModel):
    grounded: bool = Field(description="모든 문장이 근거로 뒷받침되면 true")
    unsupported: List[str] = Field(default_factory=list, description="근거 없는 문장들")
    reason: str = Field(default="")


class QualityCheck(BaseModel):
    addresses_question: bool
    complete: bool
    actionable: bool = True
    score: float = Field(description="0.0 ~ 1.0")
    missing: str = Field(default="")


# --------------------------------------------------------------------------
def verify_answer(state: RAGState) -> dict:
    """환각 + 품질을 검사해 verdict 를 만든다."""
    answer = state.get("answer", "") or ""
    docs = state.get("documents", []) or []
    question = state.get("original_question") or state["question"]
    qid = state.get("qid", "")

    ws = config.WORKFLOW_SETTINGS
    if not (ws["check_hallucination"] or ws["grade_answer"]) or not answer:
        return {"verdict": {"pass": True, "skipped": True},
                "trace": step("verify", qid=qid, summary="검증 비활성/답변 없음")}

    verdict = {"pass": True, "grounded": True, "score": 1.0,
               "unsupported": [], "missing": ""}
    used = 0

    with Timer() as t:
        # ── 1. 환각 검사 ──────────────────────────────────────────────
        if ws["check_hallucination"] and docs:
            context = "\n\n".join(
                format_chunk_for_llm(c, idx=i + 1) for i, c in enumerate(docs))
            try:
                h = llm.structured(
                    prompts.hallucination_system,
                    f"[근거]\n{context}\n\n[생성된 답변]\n{answer}",
                    HallucinationCheck, model=config.MODEL_VERIFY,
                    max_tokens=800, stage="verify:halluc")
                used += 1
                verdict["grounded"] = h.grounded
                verdict["unsupported"] = h.unsupported[:5]
                if not h.grounded:
                    verdict["pass"] = False
            except Exception as e:
                # 검증 실패로 답변을 버리면 안 된다 — 통과시키고 기록만 남긴다
                log_message(f"환각 검사 실패(통과 처리): {e}", "warning")
                verdict["halluc_error"] = str(e)[:120]

        # ── 2. 품질 채점 ──────────────────────────────────────────────
        if ws["grade_answer"]:
            try:
                q = llm.structured(
                    prompts.answer_quality_system,
                    f"[질문]\n{question}\n\n[답변]\n{answer}",
                    QualityCheck, model=config.MODEL_VERIFY,
                    max_tokens=600, stage="verify:quality")
                used += 1
                verdict["score"] = float(q.score)
                verdict["addresses"] = q.addresses_question
                verdict["complete"] = q.complete
                verdict["missing"] = q.missing
                if q.score < 0.7 or not q.addresses_question:
                    verdict["pass"] = False
            except Exception as e:
                log_message(f"품질 채점 실패(통과 처리): {e}", "warning")
                verdict["quality_error"] = str(e)[:120]

    tag = "통과" if verdict["pass"] else "실패"
    detail = (f"근거일치={verdict['grounded']}, 점수={verdict['score']:.2f}"
              + (f", 부족={truncate(verdict['missing'], 40)}" if verdict.get("missing") else ""))

    return {
        "verdict": verdict,
        "verify_retries": state.get("verify_retries", 0) + 1,
        "llm_calls": used,
        "trace": step("verify", qid=qid, summary=f"{tag} — {detail}",
                      detail=verdict, ms=t.ms),
    }
