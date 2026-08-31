"""질의 세트 실행기 — 답변 + **분석용 기록**을 남긴다.

    python -m eval.run_queryset                    # 전체 78문항
    python -m eval.run_queryset --limit 5          # 앞 5건만 (배선 확인용)
    python -m eval.run_queryset --track 상품        # 한 트랙만
    python -m eval.run_queryset --section C        # 섹션 접두사로 고르기
    python -m eval.run_queryset --qid P-C5,D-A1    # 특정 문항만
    python -m eval.run_queryset --tag baseline     # 결과 파일 이름표
    python -m eval.run_queryset --resume           # 이전 실행 이어서 (중단 복구)

■ eval/runner.py 와 무엇이 다른가
  runner.py 는 **골드셋 기반 자동 채점기**다(recall/MRR/정답문자열 비교).
  이쪽은 골드셋이 아직 없는 단계에서 쓰는 **관찰 기록기**다. 점수를 매기지 않고,
  사람(또는 LLM)이 성공/실패를 판정하는 데 필요한 것을 빠짐없이 남긴다.

  자동 채점을 하지 않는 이유가 있다. 이 질의 세트의 절반은 "정답 문자열"이 아니라
  **행동**을 본다 — 되물었는가, 전제를 바로잡았는가, 단정을 피했는가,
  '해당없음'과 '미확인'을 구분했는가. 문자열 일치로는 잴 수 없다.

■ 남기는 것 (실패 분류에 필요한 최소 집합)
    답변 본문        무엇을 답했는가
    역질문 여부      정보 부족 시 되물었는가 (정보한계 대응)
    인용 근거        어떤 청크를 근거로 썼는가 (근거 완전성·근거 오염)
    검색된 청크      인용 안 된 것까지 (검색은 됐는데 안 쓴 건지, 검색부터 실패인지)
    계층 분포        규정문서 vs 투자설명서 몇 건이 들어왔는가 (카드 밀림 판별)
    실행 경로        어느 노드를 거쳤는가 (루프 이상·폴백 발동)
    상품관여도/사유   검색 조건이 어떻게 잡혔는가
    비용             토큰·지연·LLM 호출 수

■ 출력 3종
    <tag>_<stamp>.json          전체 기록 (사람이 안 읽어도 되는 원본)
    <tag>_<stamp>.compact.jsonl 분석용 요약 — **이 파일을 분석에 쓴다**
    <tag>_<stamp>.report.md     사람이 훑어보는 용도
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import utils  # noqa: F401  — stdout 을 UTF-8 로 고정 (Windows cp949 대비)
from utils import log_message, truncate

QUERYSET_PATH = Path(__file__).resolve().parent / "queryset.jsonl"


# ==========================================================================
def load_queryset(path: Optional[str] = None) -> List[dict]:
    rows: List[dict] = []
    with open(path or QUERYSET_PATH, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def _select(cases: List[dict], a: argparse.Namespace) -> List[dict]:
    out = cases
    # split: 문항을 dev(튜닝용) / holdout(최종 확인용) 으로 가른다.
    #
    # 같은 문항으로 튜닝하고 같은 문항으로 성능을 재면 과적합이 보이지 않는다.
    # 파라미터를 만질 때는 dev 만 보고, 다 만진 뒤 holdout 으로 한 번만 확인한다.
    # 질의 세트에 split 필드가 없는 문항은 dev 로 본다(기존 파일 호환).
    if getattr(a, "split", ""):
        want = a.split.strip()
        out = [c for c in out if (c.get("split") or "dev") == want]
    if a.track:
        out = [c for c in out if c.get("track") == a.track]
    if a.section:
        out = [c for c in out if (c.get("section") or "").startswith(a.section)]
    if a.qid:
        want = {q.strip() for q in a.qid.split(",") if q.strip()}
        out = [c for c in out if c["qid"] in want]
    if a.kind:
        out = [c for c in out if c.get("expect_kind") == a.kind]
    return out[: a.limit] if a.limit else out


# ==========================================================================
def _trace_path(trace: List[dict]) -> List[str]:
    """실행 경로를 노드 이름 나열로 압축한다 (연속 중복은 접는다)."""
    path: List[str] = []
    for ev in trace:
        n = ev.get("node", "")
        if not path or path[-1].split("×")[0] != n:
            path.append(n)
        else:
            head = path[-1].split("×")[0]
            cnt = int(path[-1].split("×")[1]) if "×" in path[-1] else 1
            path[-1] = f"{head}×{cnt + 1}"
    return path


def _retrieval_detail(trace: List[dict]) -> List[dict]:
    """retrieve 노드가 남긴 기록을 뽑는다 — 검색이 무엇을 어떻게 가져왔는지.

    summary(사람용 한 줄) 뿐 아니라 **chunk_type 분포를 dict 그대로** 싣는다.
    실패 문항을 볼 때 이게 없으면 "검색 실패" 와 "인용 실패" 를 구분할 수 없다.
    """
    out: List[dict] = []
    for ev in trace:
        if ev.get("node") != "retrieve":
            continue
        d = ev.get("detail") or {}
        row = {"qid": ev.get("qid", ""), "summary": ev.get("summary", "")}
        if d.get("chunk_type_dist"):
            row["chunk_type_dist"] = d["chunk_type_dist"]
        if d.get("top_k"):
            row["top_k"] = d["top_k"]
        out.append(row)
    return out


def _retrieval_dist_total(trace: List[dict]) -> Dict[str, int]:
    """이 문항이 검색해 온 청크의 chunk_type 분포 합계 (서브질문 전부 합산).

    인용 분포(cite_dist)와 **짝을 이룬다.**
        검색 분포에 원하는 타입이 없다  → 검색 실패. 쿼터·가중·필터를 봐야 한다.
        검색 분포엔 있는데 인용에는 없다 → 인용 실패. 답변 프롬프트를 봐야 한다.
    """
    total: Dict[str, int] = {}
    for ev in trace:
        if ev.get("node") != "retrieve":
            continue
        for k, v in ((ev.get("detail") or {}).get("chunk_type_dist") or {}).items():
            total[k] = total.get(k, 0) + v
    return total


def _calc_detail(result: Dict[str, Any], trace: List[dict]) -> Dict[str, Any]:
    """계산이 **파이썬 도구를 탔는지, LLM 이 암산했는지**를 남긴다.

    산술 오류를 볼 때 이 구분이 없으면 원인을 좁힐 수 없다.
        tool 이 있다  → 도구는 탔다. 인자 추출이나 도구 자체를 봐야 한다.
        tool 이 없다  → LLM 이 직접 계산했다. 그러면 틀리는 게 당연하고,
                        고칠 곳은 도구 선택(calculator_select_system) 이다.
    """
    calc = result.get("calc_result") or {}
    if not calc:
        return {}
    out: Dict[str, Any] = {
        "tool_name": calc.get("tool") or calc.get("tool_name") or "",
        "used_tool": bool(calc.get("tool") or calc.get("tool_name")),
    }
    if calc.get("skipped"):
        out["skipped_reason"] = calc.get("reason", "")
        out["used_tool"] = False
    if calc.get("needs_input"):
        out["needs_input"] = calc["needs_input"]
    if calc.get("error"):
        out["error"] = calc["error"]
    if calc.get("args"):
        out["args"] = calc["args"]
    # 숫자 결과만 추린다 (설명문·steps 는 answer 에 이미 들어 있다)
    nums = {k: v for k, v in calc.items()
            if isinstance(v, (int, float)) and k not in ("llm_calls",)}
    if nums:
        out["numbers"] = nums
    return out


def _cite_rows(cites: List[dict]) -> List[dict]:
    out = []
    for c in cites:
        row = {
            "n": c.get("n"),
            "chunk_id": c.get("chunk_id"),
            "doc_id": c.get("doc_id"),
            "tier": c.get("tier"),
            "type": c.get("chunk_type"),
            "page": c.get("page"),
            "section": truncate(c.get("section") or "", 40),
        }
        if c.get("page_unverified"):
            row["page추정"] = True
        if c.get("fund_name"):
            row["fund"] = c["fund_name"]
        if c.get("사유"):
            row["사유"] = c["사유"]

        # ── 인용이 **어느 경로로** 붙었는지 (2026-08-27 신설) ─────────────
        #   경로가 셋인데 결과 파일에는 구분이 안 남아서, 6차 실행 결과로도
        #   "P-A4 오출처가 ①이었나 ②였나" 를 특정하지 못했다.
        #     (없음)   모델이 구조화 출력의 citations 배열에 직접 준 것
        #     복원     본문에 쓴 [n] 번호에서 되찾은 것 (파싱 실패 시 경로)
        #     값대조   코드가 수치 일치로 붙인 것
        #   quote 유무도 같이 남긴다 — 복원 경로는 quote 가 반드시 빈다.
        # 세 경로를 **전부 명시**한다 (2026-08-28). 예전에는 모델직접이
        # 키 없음으로 저장돼서, 집계할 때 "키 없음 = 모델직접" 이라는 암묵
        # 규칙을 알아야 했다.
        if c.get("restored"):
            row["경로"] = "복원"
        elif c.get("auto"):
            row["경로"] = "값대조"
        else:
            row["경로"] = "모델직접"
        row["quote있음"] = bool(c.get("quote"))

        # ── quote **원문**을 남긴다 (2026-08-31 신설, 9차 이슈 4) ──────────
        #
        # `attach_citations_by_value` 는 인용에 어떤 값이 걸려서 붙었는지를
        # `quote="값 일치: C-e"` 형태로 적어 둔다. 그런데 결과 파일에는
        # `quote있음: true` 라는 불리언만 남고 **텍스트가 통째로 사라졌다.**
        # compact 뿐 아니라 full json 에서도 없었다.
        #
        # 이게 남아 있었으면 P-E5 에서 `"값 일치: C-e"` 가 그대로 보여 값 대조
        # 오탐을 즉시 알 수 있었다. 실제로는 인용된 청크 3건을 손으로 대조해
        # 역산해야 했고, "클래스코드 매칭이 다른 문항에서는 정당했는지" 는
        # 집계 자체가 불가능했다.
        #
        # 값 대조 인용의 진단 정보는 이 한 줄이 전부다. 길이만 자른다.
        if c.get("quote"):
            row["quote"] = truncate(c["quote"], 100)
        out.append(row)
    return out


# 4000 → 8000 (2026-08-28). P-A4 처럼 표가 긴 답변이 기록 단계에서 잘려
# 분석 때 뒷부분을 못 봤다. 이건 모델 출력 한도가 아니라 **결과 파일 기록**
# 한도라, 올려도 파이프라인 동작은 그대로다.
def run_case(case: dict, *, max_answer: int = 8000) -> Dict[str, Any]:
    """질의 1건을 파이프라인에 태우고 분석에 필요한 것을 전부 남긴다."""
    from llm import llm            # 구제 경로 기록을 읽는다 (아래 salvage)
    from workflows import run

    q = case["question"]
    log_message(f"\n{'=' * 72}\n[{case['qid']}] {q}")

    t0 = time.time()
    try:
        result = run(q, session_id=f"qs-{case['qid']}", qkey=case["qid"],
                     reset_usage=False)
        err = None
    except Exception as e:                       # 한 건이 죽어도 나머지는 돈다
        log_message(f"실행 실패: {e}", "error")
        return {
            **{k: case[k] for k in ("qid", "track", "section", "난이도",
                                    "유형", "expect_kind", "question", "expect")
               if k in case},
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-1500:],
            "latency_sec": round(time.time() - t0, 1),
        }
    elapsed = time.time() - t0

    trace = result.get("trace", []) or []
    cites = result.get("citations", []) or []
    qa_pairs = result.get("qa_pairs", []) or []
    usage = result.get("usage") or {}
    filters = result.get("filters") or {}

    # 인용된 근거의 계층 분포 — "카드 밀림"·"필터 탈락" 판별에 쓴다
    dist: Dict[str, int] = {}
    for c in cites:
        k = f"{c.get('tier')}/{c.get('chunk_type')}"
        dist[k] = dist.get(k, 0) + 1

    answer = result.get("final_answer") or ""

    return {
        # --- 문항 정보 (기대값 포함 — 채점자가 이 파일만 보고 판정할 수 있게) ---
        "qid": case["qid"],
        "track": case.get("track"),
        "section": case.get("section"),
        "난이도": case.get("난이도"),
        "유형": case.get("유형"),
        "expect_kind": case.get("expect_kind"),
        "question": q,
        "expect": case.get("expect"),

        # --- 결과 ---
        "answer": truncate(answer, max_answer) if max_answer else answer,
        "answer_len": len(answer),
        "answer_truncated": bool(max_answer and len(answer) > max_answer),

        # --- 행동 관찰 (이 질의 세트의 절반은 여기서 성패가 갈린다) ---
        "clarified": bool(result.get("needs_clarification")),
        "slots": result.get("slots") or {},
        "blocked": not result.get("safe", True),
        # safety_reason  : 사람이 읽는 판정 이유 한 문장
        # safety_category: 기계가 읽는 라벨 (개인정보|위법조력|유해|범위밖)
        #                  → 정당한 거절과 오차단을 **다르게 채점**하려면 이게 있어야 한다
        # safety_caution : 차단은 안 했지만 주의로 표시한 범주
        "safety_reason": result.get("safety_reason", ""),
        "safety_category": result.get("safety_category", ""),
        "safety_caution": result.get("safety_caution", ""),

        # --- 검색·근거 ---
        "질문유형": result.get("질문유형"),
        "product_affinity": filters.get("product_affinity"),
        "fund_name": filters.get("fund_name") or [],
        "n_citations": len(cites),
        "cite_dist": dist,                       # 실제로 **인용된** 청크의 분포
        "retrieval_dist": _retrieval_dist_total(trace),   # **검색된** 청크의 분포
        "dangling_cites": result.get("dangling_cites") or [],
        "citations": _cite_rows(cites),
        "retrieval": _retrieval_detail(trace),
        "calc": _calc_detail(result, trace),

        # --- 분해 ---
        "sub_questions": [
            {"q": p.get("question"),
             "evidence_sufficient": p.get("evidence_sufficient"),
             "evidence_ok": p.get("evidence_ok"),
             "verified": p.get("verified"),
             "score": p.get("score")}
            for p in qa_pairs
        ],

        # 구조화 파싱이 구제 경로로 빠진 기록 (2026-08-25)
        #   citations 가 비는 가장 큰 원인이라 결과에 남긴다.
        "salvage": [f"{s['stage']}:{s['kind']}"
                    for s in llm.salvage_events(case["qid"])],

        # 서브답변별 '근거만으로 답할 수 있었나' 원값 (2026-08-25)
        #   sufficient=False → 답변 프롬프트 문제
        #   sufficient=True + 인용 0 → 인용 표기 문제
        "evidence_sufficient": [bool(p.get("evidence_sufficient"))
                                for p in qa_pairs],

        # 서브답변들의 '근거 없는 값' 을 문항 단위로 모은다 (2026-08-23)
        # ── 근거 없는 값 — 재작성 전/후를 **둘 다** 남긴다 (2026-08-28) ──
        #
        #   unsupported_values      최종 답변 기준. 판정할 가치가 있는 목록
        #   unsupported_values_pre  서브답변 기준. 예전에 쓰던 값
        #   unsupported_values_new  final - pre = 재작성이 새로 만든 값
        #
        # 예전에는 pre 만 기록하면서 이름을 `unsupported_values` 로 달아, 최종
        # 답변에 없는 문자열을 환각으로 판정할 뻔했다(7차 P-B3·P-D5).
        #
        # ⚠️ `_new` 를 곧바로 환각으로 읽지 마라. 재작성이 표기를 바꾼 것
        #    ("12.5%" → "12.5퍼센트")도 여기 들어온다. 건수부터 본다.
        "unsupported_values": result.get("unsupported_values") or [],
        "unsupported_values_pre": result.get("unsupported_values_pre") or [],
        "unsupported_values_new": sorted(
            set(result.get("unsupported_values") or [])
            - set(result.get("unsupported_values_pre") or [])),

        # ── 값 대조가 못 잡는 두 유형 (2026-08-31, 9차 이슈 1·6-b) ────────
        #   unknown_funds       코퍼스에 없는 자리표시자 상품명
        #   asset_claim_facts   답변 서술과 어긋나는 카드 사실
        #   all_subs_unverified 서브답변이 전부 검증에 떨어졌나
        # 셋 다 문자열 대조로 계산한다 — LLM 호출이 늘지 않는다.
        "unknown_funds": result.get("unknown_funds") or [],
        "asset_claim_facts": result.get("asset_claim_facts") or [],
        "all_subs_unverified": bool(result.get("all_subs_unverified")),

        # ── 429 재시도 기록 (2026-08-31 신설) ────────────────────────────
        # 지금까지 재시도는 로그에만 찍히고 결과 파일 어디에도 안 남았다.
        # 그래서 9차에 "REQUEST_GAP 이 429 를 막은 것인지 재시도 상향이 흡수한
        # 것인지" 를 결과 파일만으로는 판정할 수 없었다 — 서버·코랩 비교의
        # 근거가 통째로 없었다는 뜻이다.
        "retries": llm.retries_for(case["qid"]),
        "retries_429": llm.retries_for(case["qid"], code=429),

        # --- 실행 경로·비용 ---
        "path": _trace_path(trace),
        "rounds": (result.get("round_no", 0) or 0) + 1,
        "llm_calls": result.get("llm_calls", 0),
        "tokens_in": usage.get("in_tokens", 0),
        "tokens_out": usage.get("out_tokens", 0),
        "tokens_total": usage.get("total_tokens", 0),
        "latency_sec": round(elapsed, 1),
    }


# ==========================================================================
# 출력
# ==========================================================================
# compact.jsonl 에 실을 키. **여기 없으면 분석 파일에 안 실린다.**
#
# ⚠️ 여기에 safety_reason 이 빠져 있어서 오차단 7건의 원인을 확정하지 못했다.
#    수집은 하고 있었는데(위 _row 참고) 산출물에만 안 실린 것이다.
#    한 줄 누락이 분석 한 사이클을 통째로 막았다 — 필드를 추가할 때는
#    **수집·기록 두 군데를 같이** 봐야 한다.
COMPACT_KEYS = [
    "qid", "track", "section", "난이도", "expect_kind", "question", "expect",
    "answer", "clarified", "slots",

    # --- 안전 판정 (정당한 거절 vs 오차단 구분에 필수) ---
    "blocked", "safety_reason", "safety_category", "safety_caution",

    "질문유형", "product_affinity", "fund_name",

    # --- 근거 (검색 실패인지 인용 실패인지 갈리려면 둘 다 필요) ---
    "n_citations", "cite_dist", "retrieval_dist", "dangling_cites",
    "unsupported_values", "unsupported_values_pre", "unsupported_values_new",
    # 값 대조가 원리상 못 잡는 서술형 오류 두 종 (2026-08-31)
    "unknown_funds", "asset_claim_facts", "all_subs_unverified",
    "evidence_sufficient", "salvage",
    "citations", "retrieval",

    # --- 계산 (산술 오류의 원인이 도구인지 LLM 암산인지) ---
    "calc",

    # --- 답변 기록이 잘렸는지 (기록 상한은 실제 답변 길이와 다르다) ---
    #
    # ⚠️ 이 두 필드가 빠져 있어서 진단이 한 번 어긋났다.
    #    4차 실행 분석에서 "답변 9건이 정확히 1,201자에서 끊긴다 → 출처 섹션이
    #    통째로 날아간다" 는 결론이 나왔는데, 실제로는 **답변이 아니라 기록이**
    #    잘린 것이었다(--max-answer 1200 + 말줄임표 1자).
    #    실제 답변에는 출처 목록이 정상적으로 붙어 있었다.
    #    기록 상한에 걸렸다는 사실 자체를 실어야 이런 오독을 막는다.
    "answer_len", "answer_truncated",

    "sub_questions", "path", "llm_calls", "tokens_total", "latency_sec",
    # 429 재시도 (서버·코랩 비교의 근거. 지금까지 결과 파일에 없었다)
    "retries", "retries_429",
    "error",
]


def write_outputs(rows: List[dict], tag: str, stamp: str,
                  meta: Dict[str, Any]) -> Dict[str, Path]:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = config.RESULTS_DIR / f"{tag}_{stamp}"

    full = base.with_suffix(".json")
    full.write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # 분석용 요약 — 한 줄 = 한 문항. 필요한 키만 남긴다.
    compact = Path(str(base) + ".compact.jsonl")
    with compact.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for r in rows:
            slim = {k: r[k] for k in COMPACT_KEYS if k in r and r[k] not in ([], {}, "")}
            fp.write(json.dumps(slim, ensure_ascii=False) + "\n")

    report = Path(str(base) + ".report.md")
    report.write_text(_render_report(rows, meta), encoding="utf-8")
    return {"full": full, "compact": compact, "report": report}


def _quantile(vals: List[float], q: float) -> float:
    """작은 표본용 분위수 (numpy 없이). 78문항 규모에서는 이걸로 충분하다."""
    if not vals:
        return 0.0
    xs = sorted(vals)
    i = min(int(round(q * (len(xs) - 1))), len(xs) - 1)
    return xs[i]


def _cost_section(ok: List[dict], meta: Dict[str, Any]) -> List[str]:
    """비용·지연 요약.

    ■ 왜 분위수를 보나 — 평균은 타임아웃을 숨긴다

      실측: 지연 중앙 45초인데 **최대 114초**. 평가 서버가 응답 시간 제한을 두면
      중앙값이 아니라 꼬리에서 터진다. 그래서 p50 뿐 아니라 p90·max 를 같이 본다.

    ■ 왜 예산 초과 문항을 따로 세나

      LLM_CALL_BUDGET=40 인데 실측 최대 95회가 나왔다. 예산 검사가 라운드
      경계에만 있었기 때문이다(지금은 호출 레벨로 옮겼다).
      이 줄이 0 이 아니면 게이트가 여전히 새고 있다는 뜻이다.
    """
    if not ok:
        return []
    lat = [r.get("latency_sec", 0) or 0 for r in ok]
    tok = [r.get("tokens_total", 0) or 0 for r in ok]
    calls = [r.get("llm_calls", 0) or 0 for r in ok]

    budget = (meta.get("params") or {}).get("llm_budget")
    hard = (meta.get("params") or {}).get("llm_hard_cap")
    over = [r for r in ok if budget and (r.get("llm_calls", 0) or 0) > budget]

    L = ["### 비용·지연", "",
         "| 지표 | 중앙(p50) | p90 | 최대 |",
         "|---|---|---|---|",
         f"| 지연(초) | {_quantile(lat, .5):.1f} | {_quantile(lat, .9):.1f} "
         f"| {max(lat):.1f} |",
         f"| 토큰 | {_quantile(tok, .5):,.0f} | {_quantile(tok, .9):,.0f} "
         f"| {max(tok):,} |",
         f"| LLM 호출 | {_quantile(calls, .5):.0f} | {_quantile(calls, .9):.0f} "
         f"| {max(calls)} |",
         ""]
    if budget:
        L.append(f"- LLM 예산 {budget}회 / 하드 상한 {hard}회 — "
                 f"**예산 초과 {len(over)}건**"
                 + (f" ({', '.join(r['qid'] for r in over[:10])})" if over else ""))
        L.append("")
    return L


def _safety_section(ok: List[dict]) -> List[str]:
    """차단 판정을 **범주별로** 센다.

    차단 총계만 보면 정당한 거절과 오차단이 섞여서 판단할 수 없다.
    개인정보 차단은 맞은 것이고, 위법조력·유해 차단은 오판일 수 있다.
    """
    blocked = [r for r in ok if r.get("blocked")]
    if not blocked:
        return []
    by_cat: Dict[str, List[str]] = {}
    for r in blocked:
        by_cat.setdefault(r.get("safety_category") or "(미기재)", []).append(r["qid"])

    L = ["### 차단 판정", "", f"- 차단 {len(blocked)}건", ""]
    for cat, qids in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        L.append(f"  - **{cat}** {len(qids)}건 — {', '.join(qids)}")
    cautioned = [r["qid"] for r in ok if r.get("safety_caution")]
    if cautioned:
        L += ["", f"- 주의 플래그로 통과 {len(cautioned)}건 — {', '.join(cautioned)}"]
    L.append("")
    return L


def _render_report(rows: List[dict], meta: Dict[str, Any]) -> str:
    ok = [r for r in rows if not r.get("error")]
    L: List[str] = [
        "# 질의 세트 실행 기록", "",
        f"- 실행 시각: {meta['stamp']}",
        f"- 문항: {len(rows)}건 (실행 실패 {len(rows) - len(ok)}건)",
        f"- 역질문 발생: {sum(1 for r in ok if r.get('clarified'))}건",
        f"- 근거 0건: {sum(1 for r in ok if not r.get('n_citations'))}건",
        f"- 총 토큰: {sum(r.get('tokens_total', 0) for r in ok):,}",
        "",
    ]
    L += _cost_section(ok, meta)
    L += _safety_section(ok)
    L += [
        "", "---", "",
        "## 한눈에 보기", "",
        "| qid | 섹션 | 기대 | 역질문 | 근거 | 분포 | 초 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            L.append(f"| {r['qid']} | {r.get('section','')} | "
                     f"{r.get('expect_kind','')} | — | ❌실행실패 | — | "
                     f"{r.get('latency_sec','')} |")
            continue
        dist = " ".join(f"{k.split('/')[0][:4]}{v}" for k, v in
                        (r.get("cite_dist") or {}).items()) or "-"
        L.append(f"| {r['qid']} | {r.get('section','')} | {r.get('expect_kind','')} "
                 f"| {'✔' if r.get('clarified') else ''} | {r.get('n_citations',0)} "
                 f"| {dist} | {r.get('latency_sec','')} |")

    L += ["", "---", "", "## 문항별 상세", ""]
    for r in rows:
        L += [f"### {r['qid']} — {r.get('section','')} ({r.get('expect_kind','')})", "",
              f"**질문**: {r['question']}", "",
              f"**기대**: {r.get('expect','')}", ""]
        if r.get("error"):
            L += [f"**실행 실패**: `{r['error']}`", "", "---", ""]
            continue
        L += [f"**답변**:", "", "```", r.get("answer", ""), "```", ""]
        if r.get("clarified"):
            L.append("→ **역질문으로 종료** (검색 미실행)")
            L.append("")
        if r.get("citations"):
            L.append("**근거**: " + ", ".join(
                f"`{c['chunk_id']}`({c.get('type')})" for c in r["citations"][:8]))
            L.append("")
        if r.get("safety_category") or r.get("safety_caution"):
            bits = []
            if r.get("safety_category"):
                bits.append(f"차단범주={r['safety_category']}")
            if r.get("safety_caution"):
                bits.append(f"주의={r['safety_caution']}")
            if r.get("safety_reason"):
                bits.append(f"사유: {r['safety_reason']}")
            L += ["**안전판정**: " + " / ".join(bits), ""]
        if r.get("retrieval_dist"):
            L += ["**검색 분포**: " + ", ".join(
                f"{k} {v}" for k, v in sorted(
                    r["retrieval_dist"].items(), key=lambda kv: -kv[1])), ""]
        if r.get("calc"):
            c = r["calc"]
            L += [f"**계산**: 도구={c.get('tool_name') or '없음(LLM 직접)'}"
                  + (f", 인자={c.get('args')}" if c.get("args") else ""), ""]
        if r.get("dangling_cites"):
            L += [f"**⚠️ 출처 목록에 없어 제거된 인용 번호**: {r['dangling_cites']}", ""]
        L += [f"**경로**: {' → '.join(r.get('path', []))}", "",
              f"**검색**: " + " / ".join(
                  d["summary"] for d in r.get("retrieval", [])) or "(없음)", "",
              "---", ""]
    return "\n".join(L)


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="질의 세트 실행 + 기록 생성")
    ap.add_argument("--limit", type=int, default=0, help="앞 N건만")
    ap.add_argument("--track", default="", help="상품 | 제도")
    ap.add_argument("--section", default="", help="섹션 접두사 (예: C)")
    ap.add_argument("--kind", default="", help="expect_kind 로 고르기 (예: 역질문)")
    ap.add_argument("--split", default="", choices=["", "dev", "holdout"],
                    help="dev(튜닝용) / holdout(최종 확인용) 만 실행. "
                         "split 필드가 없는 문항은 dev 로 본다")
    ap.add_argument("--qid", default="", help="쉼표로 구분한 qid 목록")
    ap.add_argument("--tag", default="queryset", help="결과 파일 이름표")
    ap.add_argument("--queryset", default=None, help="질의 세트 경로")
    ap.add_argument("--max-answer", type=int, default=8000,
                    help="기록에 남길 답변 최대 길이 (0=자르지 않음)")
    ap.add_argument("--resume", action="store_true",
                    help="같은 tag 의 최신 기록에 있는 문항은 건너뛴다")
    a = ap.parse_args()

    cases = _select(load_queryset(a.queryset), a)
    if not cases:
        print("실행할 문항이 없습니다. 필터 조건을 확인하세요.")
        return 1

    done: Dict[str, dict] = {}
    if a.resume:
        prev = sorted(config.RESULTS_DIR.glob(f"{a.tag}_*.json"))
        if prev:
            old = json.loads(prev[-1].read_text(encoding="utf-8"))
            done = {r["qid"]: r for r in old.get("rows", []) if not r.get("error")}
            log_message(f"이어서 실행: 기존 {len(done)}건 재사용 ({prev[-1].name})")

    from llm import llm

    if not config.CLOVA_API_KEY:
        print("CLOVA_API_KEY 가 없습니다. 질의 세트 실행에는 LLM 이 필요합니다.\n"
              "  · .env 에 CLOVA_API_KEY 를 넣고 다시 실행하세요.\n"
              "  · 키 없이 확인할 것: `python run_cli.py --check` (검색·코퍼스 점검)")
        return 1

    llm.reset_usage()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: List[dict] = []

    for i, c in enumerate(cases, 1):
        if c["qid"] in done:
            rows.append(done[c["qid"]])
            continue
        log_message(f"--- [{i}/{len(cases)}] ---")
        rows.append(run_case(c, max_answer=a.max_answer))
        # 문항 사이 간격 — 레이트리밋이 잦으면 config.REQUEST_GAP 을 올린다.
        if config.REQUEST_GAP and i < len(cases):
            time.sleep(config.REQUEST_GAP)

    meta = {
        "stamp": stamp,
        "n_cases": len(rows),
        "corpus": _corpus_meta(),
        "settings": dict(config.WORKFLOW_SETTINGS),
        "params": {
            "top_k": config.NUM_DOCS_TO_RETRIEVE,
            "top_k_recommend": config.NUM_DOCS_RECOMMEND,
            "min_stratum_quota": config.MIN_STRATUM_QUOTA,
            "max_per_doc": config.MAX_CHUNKS_PER_DOC,
            "bm25_w": config.BM25_WEIGHT,
            "vec_w": config.VEC_WEIGHT,
            "card_boost": config.CARD_BOOST,
            "min_section_quota": config.MIN_SECTION_QUOTA,
            "scope_min_bm25": config.SCOPE_MIN_BM25,
            "harm_min_bm25": config.HARM_MIN_BM25,
            "product_signal_ratio": config.PRODUCT_SIGNAL_RATIO,
            "llm_budget": config.LLM_CALL_BUDGET,
            "llm_hard_cap": config.LLM_CALL_HARD_CAP,
            "use_confidence": config.USE_CONFIDENCE_WEIGHT,

            # ── 재시도·동시성 3종 + 단계별 상한 (2026-08-31 신설) ─────────
            #
            # 9차에 `CLOVA_MAX_RETRY` 4→6, `BACKOFF_BASE` 3.0→5.0,
            # `REQUEST_GAP` 0→0.5 를 **동시에** 바꿨는데, 어느 것이 429 를
            # 막았는지 결과 파일만으로는 알 수 없었다 — 셋 다 기록이 없었다.
            # 그중 `REQUEST_GAP` 은 배치 전용이라 서버에는 그 보호가 없다.
            # 무엇으로 돌았는지가 안 남으면 서버·코랩 비교가 성립하지 않는다.
            "llm_max_retry": config.LLM_MAX_RETRY,
            "backoff_base": config.LLM_BACKOFF_BASE,
            "backoff_max": config.LLM_BACKOFF_MAX,
            "request_gap": config.REQUEST_GAP,          # ⚠️ 배치 전용
            "llm_concurrency": config.LLM_MAX_CONCURRENCY,
            "grade_workers": config.GRADE_WORKERS,
            "grade_limit": config.env_int("PENSION_GRADE_LIMIT", 8),
            "grade_wipeout_keep": config.GRADE_WIPEOUT_KEEP,
            "combine_max_tokens": config.COMBINE_MAX_TOKENS,
            "max_sibling_funds": config.MAX_SIBLING_FUNDS,

            "models": {"plan": config.MODEL_PLAN, "grade": config.MODEL_GRADE,
                       "answer": config.MODEL_ANSWER, "verify": config.MODEL_VERIFY,
                       "decompose": config.MODEL_DECOMPOSE,
                       "safety": config.MODEL_SAFETY,
                       "repair": config.MODEL_REPAIR,
                       "embed": config.EMBED_MODEL},
        },
    }

    paths = write_outputs(rows, a.tag, stamp, meta)

    n_err = sum(1 for r in rows if r.get("error"))
    print("\n" + "=" * 72)
    print(f"문항 {len(rows)}건 실행 완료 (실행 실패 {n_err}건)")
    print(f"  역질문 {sum(1 for r in rows if r.get('clarified'))}건 / "
          f"근거 0건 {sum(1 for r in rows if not r.get('n_citations') and not r.get('error'))}건 / "
          f"총 {sum(r.get('tokens_total', 0) for r in rows):,} 토큰")
    print("\n생성된 기록")
    print(f"  전체     : {paths['full']}")
    print(f"  ★ 분석용 : {paths['compact']}")
    print(f"  사람용   : {paths['report']}")
    print("\n→ 분석을 맡기려면 **compact.jsonl** 파일을 전달하세요.")
    return 0


def _corpus_meta() -> dict:
    try:
        from index import corpus_stats
        s = corpus_stats()
        return {"total": s["total"], "by_tier": s["by_tier"],
                "by_chunk_type": s["by_chunk_type"]}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    sys.exit(main())
