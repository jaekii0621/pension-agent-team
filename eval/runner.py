"""평가 실행기.

    python -m eval.runner                     # 전체 골드셋
    python -m eval.runner --limit 3           # 앞 3건만
    python -m eval.runner --no-llm-judge      # LLM 심판 없이 (검색 지표만, 비용 0 추가)
    python -m eval.runner --retrieval-only    # 파이프라인 안 돌리고 검색만 측정 (가장 쌈)
    python -m eval.runner --tag baseline      # 결과 파일에 이름표

■ 실험 설계 (config.WORKFLOW_SETTINGS 를 바꿔가며 비교한다)
    1) baseline      : grade_documents=False, check_hallucination=False, grade_answer=False
                       → 순수 검색+생성. 나머지 구성요소의 기여도를 재는 기준선.
    2) +grading      : grade_documents=True
    3) +verify       : check_hallucination=True, grade_answer=True
    4) +RRR          : repeat_residual=True, MAX_ROUNDS=2
    5) full          : 전부 True
  각각 --tag 를 다르게 주고 돌린 뒤 results/ 의 summary 를 비교하면
  선행모델 논문의 ablation 표와 같은 형태가 나온다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import config
from index import get_index
from llm import embedder
from nodes.document_retriever import _top_k_for
from utils import log_message, render_trace

from . import metrics


# ==========================================================================
def load_goldset(path=None, limit: int = 0) -> List[dict]:
    p = path or config.GOLDSET_PATH
    rows = []
    with open(p, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


# ==========================================================================
def eval_retrieval_only(cases: List[dict]) -> Dict[str, Any]:
    """파이프라인을 돌리지 않고 **검색 성능만** 측정한다 (LLM 호출 0회).

    검색이 정답 청크를 못 가져오면 그 뒤 무엇을 해도 정답이 안 나온다.
    그래서 파이프라인을 손보기 전에 항상 이걸 먼저 본다.
    """
    index = get_index()
    emb = embedder()
    rows: List[dict] = []

    for c in cases:
        gold = c.get("gold_chunk_ids") or []
        if not gold:
            continue
        qvec = None
        try:
            qvec = emb.embed_query(c["question"])
        except Exception:
            pass

        # 파이프라인과 **같은** top_k 규칙을 쓴다 (추천 질문은 넓게 본다).
        # 여기서 규칙이 어긋나면 검색 지표가 실제 파이프라인 성능과 달라진다.
        qt = c.get("질문유형", "사실조회")
        scope = "all" if qt == "추천" else "doc"
        top_k = _top_k_for(qt, {"scope": scope})

        # 필터 없이 / scope 필터를 걸고 — 두 조건을 비교한다
        for label, flt in (("nofilter", {}),
                           ("scopefilter", {"scope": scope})):
            hits = index.search(c["question"], top_k=top_k,
                                filters=flt, query_vector=qvec, 질문유형=qt)
            ids = [h["chunk"]["chunk_id"] for h in hits]
            m = metrics.retrieval_metrics(ids, gold)
            rows.append({"qid": c["qid"], "mode": label, **m})
            log_message(f"  {c['qid']:5s} [{label:11s}] recall={m['recall']:.2f} "
                        f"mrr={m['mrr']:.2f} hit={m['hit']:.0f}")

    by_mode = {}
    for mode in ("nofilter", "scopefilter"):
        sub = [r for r in rows if r["mode"] == mode]
        by_mode[mode] = metrics.aggregate_scores(
            [{k: v for k, v in r.items() if k != "mode"} for r in sub])
    return {"mode": "retrieval_only", "by_mode": by_mode, "rows": rows}


# ==========================================================================
def eval_case(case: dict, *, llm_judge: bool = True) -> Dict[str, Any]:
    """골드셋 1건을 파이프라인에 태우고 지표를 계산한다."""
    from workflows import run

    q = case["question"]
    log_message(f"\n{'=' * 70}\n[{case['qid']}] {q}")

    t0 = time.time()
    try:
        # qkey 로 이 질문의 LLM 호출을 표시해 두면 usage 가 이 질문 것만 담는다.
        # ⚠️ reset_usage=False 여야 한다 — True 면 질문마다 기록이 지워져서
        #    마지막 질문의 내역만 남고 usage_by_question() 이 비어 버린다.
        #    (전체 초기화는 main() 이 루프 시작 전에 한 번만 한다)
        result = run(q, session_id=f"eval-{case['qid']}", qkey=case["qid"],
                     reset_usage=False)
        error = None
    except Exception as e:
        log_message(f"실행 실패: {e}", "error")
        return {"qid": case["qid"], "error": str(e)[:200]}
    elapsed = time.time() - t0

    answer = result.get("final_answer") or ""
    cites = result.get("citations") or []
    retrieved_ids = [c.get("chunk_id") for c in cites if c.get("chunk_id")]

    row: Dict[str, Any] = {"qid": case["qid"]}

    # --- 1층: 결정론적 ---
    row.update(metrics.retrieval_metrics(retrieved_ids, case.get("gold_chunk_ids") or []))
    row.update(metrics.routing_accuracy(result.get("질문유형", ""), case.get("질문유형")))
    row.update(metrics.cost_metrics(result, elapsed))
    row.update(metrics.pipeline_health(result))

    # --- 질문 1건당 토큰 (성능 측정의 비용 축) ---
    usage = result.get("usage") or {}
    row["tokens_in"] = usage.get("in_tokens", 0)
    row["tokens_out"] = usage.get("out_tokens", 0)
    row["tokens_total"] = usage.get("total_tokens", 0)
    row["scope"] = (result.get("filters") or {}).get("scope", "")

    # --- 2·3층: LLM 심판 ---
    if llm_judge:
        row.update(metrics.llm_correctness(answer, case.get("reference_answer", "")))
        contexts = [c.get("snippet", "") for c in cites]
        row.update(metrics.llm_faithfulness(answer, contexts))
        row.update(metrics.response_relevancy(q, answer))

    log_message(render_trace(result.get("trace", [])))
    log_message(f"[지표] " + ", ".join(
        f"{k}={v}" for k, v in row.items()
        if k != "qid" and isinstance(v, (int, float))))

    return {**row, "_answer": answer, "_n_cites": len(cites)}


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="연금 에이전트 성능 평가")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-llm-judge", action="store_true",
                    help="LLM 심판 지표를 빼고 결정론적 지표만 (추가 비용 0)")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="파이프라인 없이 검색 성능만 (LLM 호출 0회)")
    ap.add_argument("--tag", default="run", help="결과 파일 이름표 (실험 구분용)")
    ap.add_argument("--goldset", default=None)
    a = ap.parse_args()

    cases = load_goldset(a.goldset, a.limit)
    log_message(f"골드셋 {len(cases)}건 로드")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if a.retrieval_only:
        report = eval_retrieval_only(cases)
        report["settings"] = {"top_k": config.NUM_DOCS_TO_RETRIEVE,
                              "bm25_w": config.BM25_WEIGHT,
                              "vec_w": config.VEC_WEIGHT,
                              "embed": config.EMBED_MODEL}
    else:
        from llm import llm

        llm.reset_usage()
        rows = [eval_case(c, llm_judge=not a.no_llm_judge) for c in cases]
        clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

        # 질문별 토큰 — 어느 질문이 비싼지, 어느 단계가 먹는지 한눈에 본다
        per_q = llm.usage_by_question()
        totals = [v.get("total_tokens", 0) for v in per_q.values()]

        report = {
            "mode": "end_to_end",
            "summary": metrics.aggregate_scores(clean),
            "rows": rows,
            "tokens": {
                "per_question": per_q,
                "avg_total_per_question": round(sum(totals) / len(totals), 1) if totals else 0,
                "max_total_per_question": max(totals) if totals else 0,
                "sum_total": sum(totals),
                "overall": llm.usage_summary(),
            },
            "settings": dict(config.WORKFLOW_SETTINGS),
            "params": {
                "top_k": config.NUM_DOCS_TO_RETRIEVE,
                "top_k_recommend": config.NUM_DOCS_RECOMMEND,
                "max_rounds": config.MAX_ROUNDS,
                "max_subq": config.MAX_SUB_QUESTIONS,
                "bm25_w": config.BM25_WEIGHT,
                "vec_w": config.VEC_WEIGHT,
                "tier_trust": config.TIER_TRUST,
                "models": {"plan": config.MODEL_PLAN, "grade": config.MODEL_GRADE,
                           "answer": config.MODEL_ANSWER, "verify": config.MODEL_VERIFY,
                           "embed": config.EMBED_MODEL},
            },
        }

    out = config.RESULTS_DIR / f"{a.tag}_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(json.dumps(report.get("summary") or report.get("by_mode"),
                     ensure_ascii=False, indent=2))
    print(f"\n결과 저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
