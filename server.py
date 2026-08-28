"""HTTP 서버 — 대회 규격 `/answer` 엔드포인트 (스켈레톤 테스트용)

요구사항 (8/17 팀 공지, 대회 슬라이드 "평가용 API 서버 스키마 예시" 확인 완료):
  · GET /answer
  · 파라미터: question_id, question
  · 300초 타임아웃
  · 응답 JSON: question_id, question, retrieved_context, think_trace, answer (전부 string)

⚠️ 이 파일은 "스켈레톤 테스트"용이다 — 목적은 로직 정확도가 아니라
   "서버로서 300초 내에 스키마를 맞춰 응답하는가"만 확인하는 것.
   따라서 retrieved_context/think_trace 의 내용 품질은 이 단계에서 중요하지 않다.

⚠️ 미확정 → 실행 전 반드시 확인 필요:
  1. requirements.txt 에 fastapi/uvicorn 있는지 확인 안 됨 — 없으면 설치 필요:
         pip install fastapi uvicorn
  2. run_agent() 는 main_graph.py 의 run() 을 그대로 쓴다 — 동기 함수라
     ThreadPoolExecutor 로 감싸서 비동기 서버 안에서 블로킹 없이 돌게 했다.
     LLM 키(CLOVA_API_KEY) 없으면 llm/__init__.py 경고대로 실제 호출 시점에
     LLMError 로 실패한다 — 이것도 서버 자체 동작 확인에는 문제 없다
     (500 에러로 스키마는 지켜서 응답하니까).
  3. retrieved_context/think_trace를 OverallState의 어느 필드로 채울지도 근사치임
     (final_answer/qa_pairs/trace/citations 중 qa_pairs, trace 사용) — 팀 합의 필요.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from workflows import run as run_agent  # main_graph.run

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pension_agent_server")

app = FastAPI(title="Pension Agent API (skeleton test)")

TIMEOUT_SECONDS = 300
_executor = ThreadPoolExecutor(max_workers=4)


def _build_response(question_id: str, question: str, result: dict) -> dict:
    """OverallState → 대회 스키마 5개 string 필드로 변환.

    ⚠️ retrieved_context / think_trace 매핑은 확정 스펙이 아니라 근사치다.
       팀 합의 후 다시 다듬어야 한다.
    """
    answer = result.get("final_answer") or ""

    qa_pairs = result.get("qa_pairs") or []
    retrieved_context = "\n\n".join(
        f"[{p.get('qid', '')}] {p.get('answer', '')}" for p in qa_pairs
    ) or ""

    trace = result.get("trace") or []
    think_trace = "\n".join(
        (t.get("summary", "") if isinstance(t, dict) else str(t)) for t in trace
    ) or ""

    return {
        "question_id": str(question_id),
        "question": str(question),
        "retrieved_context": str(retrieved_context),
        "think_trace": str(think_trace),
        "answer": str(answer),
    }


@app.get("/answer")
async def answer(
    question_id: str = Query(..., description="평가 질문 ID (예: Q-001)"),
    question: str = Query(..., description="평가 질문 원문"),
):
    start = time.time()
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, lambda: run_agent(question)),
            timeout=TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        log.error(f"/answer 타임아웃 ({elapsed:.1f}초, 한도 {TIMEOUT_SECONDS}초) — question_id={question_id!r} question={question!r}")
        return JSONResponse(
            status_code=504,
            content={
                "question_id": question_id,
                "question": question,
                "retrieved_context": "",
                "think_trace": "",
                "answer": "",
            },
        )
    except Exception as e:
        log.exception("agent 실행 중 오류")
        return JSONResponse(
            status_code=500,
            content={
                "question_id": question_id,
                "question": question,
                "retrieved_context": "",
                "think_trace": "",
                "answer": f"오류: {e}",
            },
        )

    elapsed = time.time() - start
    log.info(f"/answer 완료 {elapsed:.1f}초 — question_id={question_id!r} question={question!r}")
    return _build_response(question_id, question, result)


@app.get("/health")
async def health():
    """배포 확인용 — 대회 스펙 아님, 트러블슈팅 편의를 위해 추가."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # 핸드오프 문서 기준 ACG 인바운드가 이미 8000 포트로 열려 있음
    uvicorn.run(app, host="0.0.0.0", port=8000)