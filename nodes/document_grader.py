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

    # ── 이어붙인 표 조각은 채점 대상에서 뺀다 (2026-09-02, 10차 검증 실행 회귀) ──
    #
    # ■ 무엇이 깨졌었나
    #   `_stitch_table_continuations` 가 잘린 표의 짝을 **문서 목록 뒤에**
    #   덧붙인다. 그런데 아래 `head/tail` 분할은 상위 GRADE_LIMIT(8)개만 채점하고
    #   **나머지는 무조건 통과**시킨다. top_k=8 일 때 예전에는 tail 이 비어서
    #   "전부 탈락 → kept 가 빈 목록 → 전멸 폴백" 이 돌았는데, 이어붙인 조각이
    #   tail 로 들어가면서 kept 가 절대 비지 않게 됐다.
    #
    #   결과: 폴백이 죽고, 살아남은 근거가 **열 머리 없는 표 뒷조각뿐**이 된다.
    #   10차 검증 실행 실측 — 신규 게이트 거절 6건(D-A4·D-D2·D-F5·D-H3·P-E1·P-E4)이
    #   전부 이어붙이기가 발동한 문항이었다. D-A4 는 evidence_ok 가 True→False,
    #   인용 1→0 으로 뒤집혔다.
    #
    # ■ 짝은 짝의 운명을 따른다
    #   뒷조각은 독립된 근거가 아니라 **앞조각의 나머지**다. 따로 채점하면
    #   맥락이 없어 무관 판정이 나고, 안 하고 통과시키면 위처럼 폴백을 막는다.
    #   그래서 채점에서 빼 두었다가 **앞조각이 살아남았을 때만** 다시 붙인다.
    #   채점 호출 수도 예전과 같아진다 (이어붙인 만큼 늘지 않는다).
    stitched = [d for d in docs if d.get("stitched_from")]
    docs = [d for d in docs if not d.get("stitched_from")]

    # ── 질문이 지목한 펀드의 카드는 채점에서 뺀다 (2026-09-02, 4차수 회귀) ──
    #
    # ■ 무엇이 반복됐나
    #   8·9·10차와 검증 실행 내내 상품 트랙 문항이 "제공된 자료에서 근거를 찾지 못했습니다"
    #   로 나갔다. **걸리는 문항이 매 차수 달라서** 특정 질문의 문제로 안 보였다.
    #
    #   10차 검증 실행 실측 — 실패한 상품 문항 7건 전부 `retrieval` 분포에
    #   `prospectus/fund_card: 1` 이 찍혀 있다. 즉 **검색은 그 펀드 카드를 이미
    #   가져왔다.** 그리고 그 카드에 답이 글자 그대로 있다.
    #
    #       P-E4 삼성ABF   카드: "매매회전율: 해당없음(주식 매매 지표 없음)"
    #                      기대: 해당없음과 미확인을 구분하라
    #       P-E1 키움연금   카드: "총보수: 0.31% (종류형 아님 — 클래스 구분 없음)"
    #                      기대: 종류형이 아니라고 답하라. 결손이 아니다
    #
    #   버린 것은 검색이 아니라 **이 채점 단계**다. 채점은 LLM 이라 판정이 흔들리고,
    #   그래서 매 차수 다른 문항이 걸린다(9차 검토서 §5 가 관측한 그 변동성이다).
    #
    # ■ 왜 카드만 고정하나
    #   펀드카드·클래스카드는 우리가 그 펀드 하나를 정리해 만든 **정본**이다.
    #   "이 펀드에 대한 질문" 에서 그 펀드의 카드가 무관일 수 없다 — 무관 판정이
    #   나오면 그건 채점기가 틀린 것이다. 반대로 섹션 원문은 88개나 되므로
    #   채점이 걸러 줄 값어치가 있다. 그래서 **지목된 펀드의 카드에만** 건다.
    #
    #   질문이 펀드를 지목하지 않았으면(`fund_name` 필터 없음) 아무것도 고정하지
    #   않는다 — 예전 동작 그대로다.
    #   ⚠️ `applied_filters` 가 아니라 **원래 필터**를 본다. 완화 2단계는 후보를
    #      넓히려고 fund_name 을 비우는 것이지, 질문이 그 펀드를 안 말한 게 아니다
    #      (`_ensure_named_fund_cards` 가 `base_filters` 를 보는 것과 같은 이유).
    pinned_names = {n for n in ((state.get("filters") or {}).get("fund_name")
                                or []) if n}
    pinned = [d for d in docs
              if d.get("chunk_type") in ("fund_card", "class_card")
              and (d.get("fund") or {}).get("name") in pinned_names]
    if pinned:
        pin_ids = {d.get("chunk_id") for d in pinned}
        docs = [d for d in docs if d.get("chunk_id") not in pin_ids]

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

    # ── 전멸 폴백 (2026-08-31 신설, 9차 이슈 5) ───────────────────────
    #
    # head 가 **전부** 기각되고 tail 도 비면 kept 가 빈 리스트가 된다. 그러면
    # 답변 생성이 "근거 0건 → 미확인 응답" 으로 빠지고 근거 게이트에서 거절된다.
    # 9차에 7건(78문항의 9%)이 이 경로였고 8차와 겹치는 건 1건뿐이다 —
    # **매 실행마다 다른 문항이 걸린다.** 특정 질문의 문제가 아니라 채점기가
    # 가끔 통째로 엄격해지는 문제다.
    #
    # 위 `_grade_one` 은 개별 채점이 실패하면 이미 통과시킨다 — "채점 실패는
    # 문서를 버리는 근거가 못 된다". 전멸도 같은 성격이라 같은 철학을 적용한다.
    #
    # ⚠️ 삭제한 '상위 3건 폴백'과 다르다. 그건 답변과 무관한 청크를 **인용**으로
    #    갖다 붙였고(P-A1 오출처), 이건 검색 결과를 답변 생성에 넘기는 것뿐이다.
    #    인용은 여전히 값 대조·모델 판단을 거치고, 근거가 정말 무관하면 답변은
    #    "확인되지 않습니다" 로 간다 — 거절과 결과가 같고 잃는 게 없다.
    rescued = 0
    if not kept and head and config.GRADE_WIPEOUT_KEEP > 0:
        kept = head[:config.GRADE_WIPEOUT_KEEP]
        rescued = len(kept)
        log_message(f"문서 채점 전멸({len(head)}건 전부 탈락) → 검색 상위 "
                    f"{rescued}건을 살립니다", "warning")

    # 지목 펀드 카드는 채점 결과와 무관하게 **맨 앞에** 되돌린다 (위 주석 참고).
    kept = pinned + kept

    # 살아남은 앞조각의 짝만 되붙인다 (위 주석 참고).
    keep_ids = {c.get("chunk_id") for c in kept}
    restitched = [c for c in stitched if c.get("stitched_from") in keep_ids]
    kept = kept + restitched

    return {
        "documents": kept,
        "irrelevance_reason": reasons,
        "grading_retries": state.get("grading_retries", 0) + 1,
        "llm_calls": used,
        "trace": step("grade_documents", qid=qid,
                      summary=f"{len(head)}건 채점 → {len(kept)}건 통과 "
                              f"(탈락 {len(dropped)}건)"
                              + (f" ⚠전멸 폴백 {rescued}건" if rescued else "")
                              + (f", 지목펀드 카드 {len(pinned)}건 고정"
                                 if pinned else "")
                              + (f", 표 짝 {len(restitched)}건 유지"
                                 if restitched else "")
                              + (f", 표 짝 {len(stitched) - len(restitched)}건 탈락"
                                 if len(stitched) > len(restitched) else ""),
                      detail={"drop_reasons": reasons},
                      ms=t.ms),
    }
