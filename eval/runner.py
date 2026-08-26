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
def load_goldset(path=None, limit: int = 0, split: str = "") -> List[dict]:
    """골드셋 로드.

    split: "dev"(튜닝용) / "holdout"(최종 확인용) / ""(전체).
      같은 문항으로 파라미터를 만지고 같은 문항으로 성능을 재면 과적합이 안 보인다.
      튜닝 중에는 dev 만 보고, 다 만진 뒤 holdout 으로 **한 번만** 확인한다.
      split 필드가 없는 문항은 dev 로 본다(기존 골드셋 파일 호환).
    """
    p = path or config.GOLDSET_PATH
    rows = []
    with open(p, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))

    # 확정 형식의 골드셋은 식별자를 `id` 로 쓴다. 이 아래 경로들(eval_case /
    # eval_retrieval_only)은 전부 `qid` 를 읽으므로 여기서 한 번만 맞춰 준다.
    # 맞추지 않으면 KeyError 로 검색 지표 측정 자체가 안 돈다.
    for r in rows:
        if "qid" not in r and r.get("id"):
            r["qid"] = r["id"]
    if split:
        rows = [r for r in rows if (r.get("split") or "dev") == split]
    return rows[:limit] if limit else rows


def load_compact(path: str) -> List[dict]:
    """run_queryset 이 낸 `*.compact.jsonl` 을 읽는다 (첫 줄은 _meta).

    ■ 왜 이 경로가 필요한가 (2026-08-18 신설)

      runner 는 지금까지 **파이프라인을 직접 돌려야만** 채점할 수 있었다.
      그런데 78문항 한 바퀴가 토큰 260만·지연 최대 114초다. 채점 규칙을 고칠
      때마다 다시 돌리는 것은 비용이 감당이 안 된다.

      run_queryset 은 이미 채점에 필요한 것을 전부 남긴다(답변·인용·검색분포·
      안전범주·계산도구). 그 파일을 그대로 먹여서 **LLM 호출 0회로** 결정론적
      지표를 다시 계산할 수 있게 한다. 채점 규칙만 바뀌었을 때 쓰는 경로다.
    """
    rows = []
    with open(path, encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0 and "_meta" in obj:      # 머리글 줄
                continue
            rows.append(obj)
    return rows


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
        # gold_chunk_ids 뿐 아니라 evidence[].chunk_id 도 받는다.
        # (문항을 인용 단위로 적으면 gold_chunk_ids 가 비어서 지표가 통째로
        #  사라졌다 — metrics.gold_ids_for 주석 참고)
        gold = metrics.gold_ids_for(c)
        gold_docs = metrics.gold_doc_ids_for(c)
        if not gold:
            continue
        qvec = None
        try:
            qvec = emb.embed_query(c["question"])
        except Exception:
            pass

        # 파이프라인과 **같은** top_k 규칙을 쓴다 (상품 질문은 넓게 본다).
        # 여기서 규칙이 어긋나면 검색 지표가 실제 파이프라인 성능과 달라진다.
        qt = c.get("질문유형", "사실조회")
        aff = (config.PRODUCT_INTENT_LEVELS["상품"] if qt == "추천"
               else config.PRODUCT_INTENT_LEVELS["제도"])
        top_k = _top_k_for(qt, {"product_affinity": aff})

        # 배분 없이(중립) / 관여도를 반영해서 — 두 조건을 비교한다.
        # 예전에는 "필터 없음 vs scope 필터"를 비교했는데, 이제 코퍼스를 지우는
        # 필터가 없으므로 **자리 배분의 효과**를 재는 비교로 바뀌었다.
        for label, flt in (("neutral", {"product_affinity": 0.5}),
                           ("affinity", {"product_affinity": aff})):
            hits = index.search(c["question"], top_k=top_k,
                                filters=flt, query_vector=qvec, 질문유형=qt)
            ids = [h["chunk"]["chunk_id"] for h in hits]
            m = metrics.retrieval_metrics(ids, gold)
            m.update(metrics.retrieval_metrics_doc(ids, gold_docs))
            rows.append({"qid": c["qid"], "mode": label, **m})
            log_message(f"  {c['qid']:5s} [{label:11s}] recall={m['recall']:.2f} "
                        f"mrr={m['mrr']:.2f} hit={m['hit']:.0f}")

    by_mode = {}
    for mode in ("neutral", "affinity"):
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
    row.update(metrics.retrieval_metrics(retrieved_ids, metrics.gold_ids_for(case)))
    row.update(metrics.retrieval_metrics_doc(retrieved_ids,
                                             metrics.gold_doc_ids_for(case)))
    row.update(metrics.citation_integrity(answer, cites))
    row.update(metrics.routing_accuracy(result.get("질문유형", ""), case.get("질문유형")))
    row.update(metrics.cost_metrics(result, elapsed))
    row.update(metrics.pipeline_health(result))

    # --- 안전 판정은 점수가 아니라 사실로 남긴다 ---
    #   정당한 거절(개인정보)과 오차단(위법조력 오판)은 같은 '차단' 이지만
    #   채점이 정반대여야 한다. 범주를 그대로 남기고 판정은 채점자에게 맡긴다.
    row["blocked"] = not result.get("safe", True)
    row["safety_category"] = result.get("safety_category", "")
    row["safety_caution"] = result.get("safety_caution", "")

    # --- 질문 1건당 토큰 (성능 측정의 비용 축) ---
    usage = result.get("usage") or {}
    row["tokens_in"] = usage.get("in_tokens", 0)
    row["tokens_out"] = usage.get("out_tokens", 0)
    row["tokens_total"] = usage.get("total_tokens", 0)
    row["product_affinity"] = (result.get("filters") or {}).get("product_affinity")
    row["clarified"] = bool(result.get("needs_clarification"))

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

    # 축별 점수 — 자동 2축은 채우고, 사람 2축은 None 으로 비워 둔다.
    # (LLM 심판 점수를 사람 점수 자리에 넣으면 "사람이 확인했다" 와
    #  "모델이 추측했다" 가 구분되지 않는다. metrics.axis_scores 주석 참고)
    axes = metrics.axis_scores(row, result)

    return {**row, "axes": axes, "_answer": answer, "_n_cites": len(cites)}


# ==========================================================================
def score_compact(compact_path: str, cases: List[dict]) -> Dict[str, Any]:
    """run_queryset 산출물을 **LLM 호출 0회로** 채점한다.

    파이프라인을 다시 돌리지 않으므로, 채점 규칙만 고쳤을 때 쓴다.
    골드셋과 qid 로 짝지어 검색 지표를 계산하고, 골드셋에 없는 문항은
    인용 무결성·안전 판정만 기록한다(관찰 문항).
    """
    by_qid = {c["qid"]: c for c in cases}
    rows: List[Dict[str, Any]] = []

    for r in load_compact(compact_path):
        qid = r.get("qid")
        case = by_qid.get(qid, {})
        cites = r.get("citations") or []
        ids = [c.get("chunk_id") for c in cites if c.get("chunk_id")]
        answer = r.get("answer") or ""

        row: Dict[str, Any] = {"qid": qid}
        row.update(metrics.retrieval_metrics(ids, metrics.gold_ids_for(case)))
        row.update(metrics.retrieval_metrics_doc(ids, metrics.gold_doc_ids_for(case)))
        row.update(metrics.citation_integrity(answer, cites))
        row["n_citations"] = r.get("n_citations", len(cites))
        row["llm_calls"] = r.get("llm_calls", 0)
        row["tokens_total"] = r.get("tokens_total", 0)
        row["latency_sec"] = r.get("latency_sec", 0)
        row["blocked"] = bool(r.get("blocked"))
        row["safety_category"] = r.get("safety_category", "")
        row["safety_caution"] = r.get("safety_caution", "")

        # compact 행은 이미 state 가 아니라 평면 dict 라, axis_scores 가 기대하는
        # 모양으로 최소한만 맞춰 준다.
        pseudo_result = {
            "safe": not r.get("blocked"),
            "safety_category": r.get("safety_category", ""),
            "safety_caution": r.get("safety_caution", ""),
            "needs_clarification": bool(r.get("clarified")),
        }
        rows.append({**row, "axes": metrics.axis_scores(row, pseudo_result),
                     "_answer": answer, "_n_cites": len(cites)})

    clean = [{k: v for k, v in r.items() if not k.startswith("_") and k != "axes"}
             for r in rows]
    return {
        "mode": "score_compact",
        "source": compact_path,
        "summary": metrics.aggregate_scores(clean),
        "rows": rows,
    }


# ==========================================================================
# 파이썬 API — 노트북에서 그대로 부른다
#
# 코랩에서 `!python -m eval.runner ...` 로 부르면 **호출마다 새 프로세스**가 뜬다.
# 그때마다 코퍼스·벡터 아티팩트·임베딩 모델을 통째로 다시 올리므로, 조건을 바꿔
# 여러 번 돌리는 ablation 에서는 준비 작업이 실행 시간의 대부분을 차지한다.
# 아래 함수들은 **이미 메모리에 올라간 인덱스를 재사용**한다.
#
# CLI(main)도 같은 함수를 부른다 — 두 경로의 결과가 갈리지 않게 하기 위해서다.
# ==========================================================================
def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_report(report: Dict[str, Any], tag: str = "run",
                stamp: str = "") -> Any:
    """리포트를 `eval/results/<tag>_<stamp>.json` 에 쓰고 요약을 찍는다."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / f"{tag}_{stamp or _now_stamp()}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("\n" + "=" * 70)
    print(json.dumps(report.get("summary") or report.get("by_mode"),
                     ensure_ascii=False, indent=2))
    if report.get("not_run"):
        print(f"\n⚠️ 실행 기록이 없어 채점하지 못한 문항 {len(report['not_run'])}건: "
              f"{report['not_run']}")
    print(f"\n결과 저장: {out}")
    return out


def run_gold(split: str = "dev", track: str = "", tag: str = "baseline",
             stamp: str = "") -> Dict[str, Any]:
    """골드셋 문항을 파이프라인에 태워 **실행 기록만** 남긴다. 채점은 하지 않는다.

        from eval import runner
        res = runner.run_gold(split="dev", track="policy", tag="baseline")
        res["compact"]        # ← 평가 담당에게 넘길 파일

    채점 로직은 이 저장소에 두지 않는다. 판정을 두 곳에 두면 반드시 갈라지므로,
    정답 조건과 채점기는 평가 담당이 소유한다. 여기가 만드는 것은 **답안지**뿐이다.
    """
    from . import gold as goldmod
    from llm import llm as _llm
    from workflows import run as _run

    stamp = stamp or _now_stamp()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    items = goldmod.load_goldset(split=split, track=track)
    log_message(f"골드셋 {len(items)}건 — {goldmod.stats(items)}")

    _llm.reset_usage()
    rows: List[dict] = []
    for i, it in enumerate(items, 1):
        log_message(f"--- [{i}/{len(items)}] {it['id']} ---")
        t0 = time.time()
        res = _run(it["question"], session_id=f"gold-{it['id']}",
                   qkey=it["id"], reset_usage=False)
        rows.append({
            # ⚠️ qid 는 골드셋 id 와 **글자 그대로** 같아야 한다.
            #    다르면 채점기가 그 문항을 '결과에 없는 문항'으로 떨군다.
            "qid": it["id"],
            "kind": it.get("kind", ""),
            "track": it.get("track", ""),
            "split": it.get("split", "dev"),
            "question": it["question"],

            # --- 채점기가 실제로 읽는 여섯 필드 ---
            "answer": res.get("final_answer") or "",
            "citations": res.get("citations") or [],
            "blocked": not res.get("safe", True),
            "safety_category": res.get("safety_category", ""),
            "safety_caution": res.get("safety_caution", ""),
            "clarified": bool(res.get("needs_clarification")),

            "llm_calls": res.get("llm_calls", 0),
            "latency_sec": round(time.time() - t0, 1),
        })

    # ── compact.jsonl — 평가 담당에게 넘기는 파일 ─────────────────────
    #
    # 첫 줄 _meta 는 채점기가 건너뛴다. 실험 조건을 여기 박아 두지 않으면 나중에
    # 어느 파일이 어느 조건이었는지 알 수 없고, ablation 자체가 무의미해진다.
    compact = config.RESULTS_DIR / f"{tag}_{stamp}.compact.jsonl"
    meta = {"_meta": {
        "stamp": stamp, "tag": tag,
        "split": split or "(전체)", "track": track or "(전체)",
        "n": len(rows),
        "params": {
            "top_k": config.NUM_DOCS_TO_RETRIEVE,
            "max_rounds": config.MAX_ROUNDS,
            "max_subq": config.MAX_SUB_QUESTIONS,
            "bm25_w": config.BM25_WEIGHT,
            "vec_w": config.VEC_WEIGHT,
            "card_boost": config.CARD_BOOST,
        },
        "workflow": dict(config.WORKFLOW_SETTINGS),
    }}
    with compact.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "mode": "goldset_run",
        "summary": {
            "문항": len(rows),
            "차단": sum(1 for r in rows if r["blocked"]),
            "인용 0건": sum(1 for r in rows if not r["citations"]),
            "답변 빈칸": sum(1 for r in rows if not r["answer"]),
            "llm_calls": sum(r["llm_calls"] for r in rows),
            "compact": str(compact),
        },
        "rows": rows,
        "settings": {"split": split or "(전체)", "track": track or "(전체)"},
    }
    out = save_report(report, tag, stamp)
    print(f"\n★ 채점용 실행 기록 → {compact}")
    return {"compact": compact, "json": out, "report": report, "rows": rows}


def run_retrieval(split: str = "", limit: int = 0, goldset: str = None,
                  tag: str = "retrieval", stamp: str = "") -> Dict[str, Any]:
    """검색 성능만 측정한다 (LLM 호출 0회).

        report = runner.run_retrieval(tag="colab_retrieval")["report"]
    """
    cases = load_goldset(goldset, limit, split)
    log_message(f"골드셋 {len(cases)}건 로드"
                + (f" (split={split})" if split else ""))
    report = eval_retrieval_only(cases)
    report["settings"] = {"top_k": config.NUM_DOCS_TO_RETRIEVE,
                          "bm25_w": config.BM25_WEIGHT,
                          "vec_w": config.VEC_WEIGHT,
                          "embed": config.EMBED_MODEL}
    return {"json": save_report(report, tag, stamp), "report": report}


def run_pipeline(split: str = "", limit: int = 0, goldset: str = None,
                 llm_judge: bool = False, tag: str = "pipeline",
                 stamp: str = "") -> Dict[str, Any]:
    """파이프라인을 끝까지 돌려 경로·비용·결정론적 지표를 잰다.

    `llm_judge=True` 면 정확도·충실도까지 재지만 비용이 크게 는다.
    """
    from llm import llm

    cases = load_goldset(goldset, limit, split)
    log_message(f"골드셋 {len(cases)}건 로드"
                + (f" (split={split})" if split else ""))

    llm.reset_usage()
    rows = [eval_case(c, llm_judge=llm_judge) for c in cases]
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
            "card_boost": config.CARD_BOOST,
            "min_section_quota": config.MIN_SECTION_QUOTA,
            "scope_min_bm25": config.SCOPE_MIN_BM25,
            "harm_min_bm25": config.HARM_MIN_BM25,
            "llm_budget": config.LLM_CALL_BUDGET,
            "llm_hard_cap": config.LLM_CALL_HARD_CAP,
            "split": split or "(전체)",
            "tier_trust": config.TIER_TRUST,
            "models": {"plan": config.MODEL_PLAN, "grade": config.MODEL_GRADE,
                       "answer": config.MODEL_ANSWER, "verify": config.MODEL_VERIFY,
                       "embed": config.EMBED_MODEL},
        },
    }
    return {"json": save_report(report, tag, stamp), "report": report}


def rescore_compact(compact_path: str, split: str = "", limit: int = 0,
                    goldset: str = None, tag: str = "rescore",
                    stamp: str = "") -> Dict[str, Any]:
    """이미 있는 `*.compact.jsonl` 로 결정론적 지표만 다시 잰다 (LLM 0회)."""
    cases = load_goldset(goldset, limit, split)
    report = score_compact(compact_path, cases)
    report["settings"] = {"split": split or "(전체)",
                          "note": "기존 산출물 재채점 — 파이프라인 미실행"}
    return {"json": save_report(report, tag, stamp), "report": report}


# ==========================================================================
# CLI — 위 함수들을 부르는 얇은 껍데기
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
    ap.add_argument("--split", default="", choices=["", "dev", "holdout"],
                    help="dev(튜닝용) / holdout(최종 확인용) 만. "
                         "split 필드가 없는 문항은 dev 로 본다")
    ap.add_argument("--from-compact", dest="from_compact", default=None,
                    help="run_queryset 이 낸 *.compact.jsonl 로 결정론적 지표만 "
                         "다시 계산 (파이프라인 미실행, LLM 호출 0회)")
    ap.add_argument("--gold", action="store_true",
                    help="골드셋(eval/gold/*.jsonl) 문항을 파이프라인에 태워 "
                         "*.compact.jsonl 실행 기록을 남긴다. 채점은 하지 않는다")
    ap.add_argument("--track", default="", choices=["", "product", "policy"],
                    help="골드셋 트랙 한쪽만")
    a = ap.parse_args()

    stamp = _now_stamp()

    if a.gold:
        if a.from_compact:
            print("--gold 는 실행 전용입니다. 이미 있는 기록을 채점하려면 "
                  "평가 담당의 채점기를 쓰세요.")
            return 1
        run_gold(split=a.split or "dev", track=a.track, tag=a.tag, stamp=stamp)
    elif a.from_compact:
        rescore_compact(a.from_compact, split=a.split, limit=a.limit,
                        goldset=a.goldset, tag=a.tag, stamp=stamp)
    elif a.retrieval_only:
        run_retrieval(split=a.split, limit=a.limit, goldset=a.goldset,
                      tag=a.tag, stamp=stamp)
    else:
        run_pipeline(split=a.split, limit=a.limit, goldset=a.goldset,
                     llm_judge=not a.no_llm_judge, tag=a.tag, stamp=stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
