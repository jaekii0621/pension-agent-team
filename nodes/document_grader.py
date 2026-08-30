"""문서 관련성 채점 노드.

■ 왜 필요한가 (선행모델과 동일한 이유)
  벡터/BM25 검색은 "가장 가까운 k개"를 **무조건** 돌려준다.
  5개 전부 엉뚱해도 5개를 준다. 그 상태로 답변을 만들면 환각이 시작된다.
  그래서 LLM 에게 한 번 더 "이게 답에 쓸 내용이냐"를 묻는다.

■ 선행모델에서 가져온 구현 포인트
  `ThreadPoolExecutor` 로 **문서 k개를 동시에 채점**한다.
  순차로 하면 5개 × 1.5초 = 7.5초, 병렬이면 ~1.5초. 체감 차이가 크다.

■ 비용 관리
  채점은 문서당 LLM 1회다. 추천 질문처럼 24건을 가져오면 24회가 된다.
  그래서 채점 대상을 상위 GRADE_LIMIT 개로 제한하고, 나머지는 검색 점수 순서를
  믿고 통과시킨다. (성능 측정 단계에서 비용이 실험 횟수를 제한하므로 중요)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List

from pydantic import BaseModel, Field

import config
from index import format_chunk_for_llm
from llm import llm
from prompt import prompts
from state import RAGState
from utils import Timer, log_message, step, truncate

# 채점에 LLM 을 쓸 최대 문서 수 (그 아래 순위는 검색 점수를 신뢰)
# 서버에서 채점 요청량 자체를 줄일 수 있게 env 로 뺀다 (2026-08-29, 재희님 요청).
# 기본값 8 그대로라 동작은 바뀌지 않는다.
GRADE_LIMIT = config.env_int("PENSION_GRADE_LIMIT", 8)


class DocumentGrade(BaseModel):
    relevant: str = Field(description="yes | no")
    reason: str = Field(default="", description="한 문장 근거")


def _grade_one(question: str, chunk: dict) -> dict:
    try:
        res = llm.structured(
            prompts.doc_grader_system,
            f"질문: {question}\n\n문서 조각:\n{format_chunk_for_llm(chunk)}",
            DocumentGrade,
            # 문서 채점은 yes/no 이진 판정에 한 줄 사유가 전부다. 출력이 짧고
            # 정답 공간이 좁아 경량 모델로 충분하다. 반면 호출 수는 문서 수만큼
            # 늘어난다(질문당 최대 8회) → 여기를 내리는 효과가 가장 크다.
            model=config.MODEL_GRADE,
            max_tokens=300,
            stage="grade_doc",
        )
        return {"ok": res.relevant.strip().lower().startswith("y"),
                "reason": res.reason, "chunk": chunk, "llm": 1}
    except Exception as e:
        # 채점 실패는 문서를 버리는 근거가 못 된다 → 통과시킨다 (보수적 폴백)
        log_message(f"문서 채점 실패(통과 처리): {e}", "warning")
        return {"ok": True, "reason": "채점 실패", "chunk": chunk, "llm": 0}


def grade_documents(state: RAGState) -> dict:
    question = state.get("original_question") or state["question"]
    docs: List[dict] = state.get("documents", []) or []
    qid = state.get("qid", "")

    if not config.WORKFLOW_SETTINGS["grade_documents"] or not docs:
        return {"grading_retries": state.get("grading_retries", 0) + 1,
                "trace": step("grade_documents", qid=qid,
                              summary=f"건너뜀 (문서 {len(docs)}건)")}

    head, tail = docs[:GRADE_LIMIT], docs[GRADE_LIMIT:]

    with Timer() as t:
        # 동시 요청 수는 설정으로 뺀다 (2026-08-27). 8 고정일 때 6차 실행에서
        # P-B1 이 CLOVA 레이트리밋(429) 재시도 4회를 모두 소진하고 죽었다.
        workers = max(1, min(config.GRADE_WORKERS, len(head)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda d: _grade_one(question, d), head))

    kept = [r["chunk"] for r in results if r["ok"]] + tail
    dropped = [r for r in results if not r["ok"]]
    reasons = " | ".join(truncate(r["reason"], 60) for r in dropped[:3])
    used = sum(r["llm"] for r in results)

    return {
        "documents": kept,
        "irrelevance_reason": reasons,
        "grading_retries": state.get("grading_retries", 0) + 1,
        "llm_calls": used,
        "trace": step("grade_documents", qid=qid,
                      summary=f"{len(head)}건 채점 → {len(kept)}건 통과 "
                              f"(탈락 {len(dropped)}건)",
                      detail={"drop_reasons": reasons},
                      ms=t.ms),
    }
