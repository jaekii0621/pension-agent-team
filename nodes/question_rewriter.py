"""질문 재작성 노드 — 검색 실패를 복구한다. **재작성 + HyDE** 두 갈래를 함께 쓴다.

■ 선행모델 대응
  FinSight 의 `nodes/question_rewriter.py` (`rewrite_question` / `rewrite_with_hyde`).
  선행모델은 두 결과를 `xxxxxxxxxx` 라는 구분자로 한 문자열에 이어붙여 retriever 로
  넘기고, retriever 가 그걸 split 해서 각각 검색한 뒤 결과를 합쳤다.
  State 에 필드를 추가하기 싫어서 쓴 꼼수였다.
  → 우리는 `hyde_doc` 이라는 **명시적 State 필드**로 넘긴다. 같은 효과인데 읽기 쉽다.

■ HyDE (Hypothetical Document Embeddings) 란
  질문으로 검색하는 대신, LLM 에게 **"정답이라면 이렇게 쓰여 있을 것"** 이라는
  가상의 문서를 먼저 쓰게 하고, 그 가상 문서로 검색한다.

      질문:      "돈 미리 빼 쓸 수 있나요?"
      가상 문서:  "IRP는 원칙적으로 만 55세 이전에는 인출할 수 없으나, 근로자퇴직급여
                  보장법 시행령이 정한 사유(무주택자의 주택 구입, 6개월 이상 요양,
                  파산 선고 등)에 해당하면 중도인출이 가능하다. …"

  왜 되나 — 벡터 검색은 **질문 벡터와 문서 벡터의 거리**를 잰다. 그런데 질문과 문서는
  애초에 문체가 다르다(질문은 짧고 구어체, 문서는 길고 서술형). 가상 문서는 실제
  문서와 형태가 같으므로 벡터 공간에서 훨씬 가깝다. BM25 에도 도움이 된다 —
  가상 문서가 "근로자퇴직급여보장법", "시행령" 같은 **문서에 실제로 등장하는 용어**를
  끌고 오기 때문이다.

  주의 — 가상 문서는 **검색 질의로만 쓴다. 절대 답변의 근거가 아니다.**
  LLM 이 지어낸 내용이므로, 그걸 근거로 답하면 그게 바로 환각이다.
  이 파일은 `hyde_doc` 을 State 에 넣고, retriever 만 그것을 읽는다.
  answer_generator 는 `documents`(실제 검색된 청크)만 본다.

■ 3갈래 검색으로 재현율을 올린다
  재작성 후 retriever 는 다음 셋으로 각각 검색해 결과를 합친다.
      ① 정규화된 질문 (문서 용어로 바꾼 것)
      ② 동의어 확장 질의
      ③ HyDE 가상 문서
  하나가 실패해도 나머지가 건진다.

■ 재작성이 호출되는 조건 (edges 가 판단)
  1) 검색 결과가 관련성 채점을 통과하지 못했다
  2) 답변 품질 점수가 기준 미달이다
"""

from __future__ import annotations

from pydantic import BaseModel, Field

import config
from llm import llm
from prompt import prompts
from state import RAGState
from utils import Timer, log_message, step, truncate


class RewrittenQuery(BaseModel):
    rewritten: str = Field(description="문서 용어로 정규화한 질문")
    expanded: str = Field(default="", description="동의어를 덧붙인 검색용 확장 질의")
    changed: str = Field(default="", description="무엇을 바꿨는지 한 문장")


class HypotheticalDoc(BaseModel):
    passage: str = Field(description="정답이 실려 있을 법한 가상의 문서 단락")


def rewrite_question(state: RAGState) -> dict:
    """질문을 문서 용어로 정규화하고, HyDE 가상 문서를 함께 만든다."""
    question = state["question"]
    original = state.get("original_question") or question
    # HyDE 가상 문서의 문체를 맞추기 위한 힌트. 추천이면 투자설명서 문체,
    # 아니면 규정문서 문체로 쓰게 한다.
    doc_kind = ("펀드 투자설명서(상품 요약·클래스별 보수·위험등급)"
                if state.get("is_recommendation")
                else "퇴직연금 규정·세제·업무절차 문서")
    reason = (state.get("irrelevance_reason", "")
              or (state.get("verdict") or {}).get("missing", ""))
    qid = state.get("qid", "")

    # ── ① 재작성 (용어 정규화 + 동의어 확장) ──────────────────────────
    with Timer() as t1:
        res = llm.structured(
            prompts.rewriter_system,
            f"원 질문: {original}\n"
            f"직전 검색에 쓴 질의: {question}\n"
            f"검색 실패 사유: {reason or '검색 결과가 질문과 맞지 않음'}",
            RewrittenQuery, model=config.MODEL_PLAN,
            max_tokens=500, stage="rewrite")

    new_q = (res.expanded or res.rewritten or question).strip()

    # ── ② HyDE (가상 문서 생성) ───────────────────────────────────────
    hyde_doc = ""
    hyde_ms = 0
    if config.WORKFLOW_SETTINGS["rewrite_with_hyde"]:
        try:
            with Timer() as t2:
                h = llm.structured(
                    prompts.hyde_system,
                    f"질문: {res.rewritten or original}\n"
                    f"이 질문의 답이 실려 있을 문서 종류: {doc_kind}",
                    HypotheticalDoc, model=config.MODEL_PLAN,
                    max_tokens=config.HYDE_MAX_TOKENS, stage="rewrite:hyde")
            hyde_doc = (h.passage or "").strip()
            hyde_ms = t2.ms
        except Exception as e:
            # HyDE 실패로 재작성 전체를 버리면 안 된다 — 재작성 질의만으로 계속 간다
            log_message(f"HyDE 생성 실패(재작성 질의만 사용): {e}", "warning")

    used = 2 if hyde_doc else 1
    summary = f"'{truncate(question, 26)}' → '{truncate(new_q, 34)}' ({res.changed})"
    if hyde_doc:
        summary += f" | HyDE {len(hyde_doc)}자"

    return {
        "question": new_q,
        "hyde_doc": hyde_doc,
        "documents": [],                       # 이전 검색 결과는 버린다
        "llm_calls": used,
        "trace": step("rewrite_question", qid=qid, summary=summary,
                      detail={"rewritten": res.rewritten, "expanded": res.expanded,
                              "hyde": truncate(hyde_doc, 200)},
                      ms=t1.ms + hyde_ms),
    }
