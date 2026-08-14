"""터미널 실행기 — 질문 하나를 넣고 전 과정을 눈으로 확인한다.

    python run_cli.py "IRP 중도인출 사유가 뭔가요?"
    python run_cli.py                          # 대화형
    python run_cli.py --check                  # 설정·데이터 점검 (API 키 불필요)
    python run_cli.py "..." --mermaid          # 실행 경로를 mermaid 로 출력
    python run_cli.py "..." --profile '{"total_salary":60000000}'

선행모델의 `chatbot.py` 에 대응한다(웹·DB 없이 그래프만 직접 돌리는 학습·디버깅용).
"""

from __future__ import annotations

import argparse
import json
import sys

import config


# ==========================================================================
def cmd_check() -> int:
    """실행 전 점검. API 키 없이도 대부분 확인할 수 있다."""
    print("=" * 68)
    print("설정 점검")
    print("=" * 68)

    ok = True

    # 1. 데이터
    print(f"\n[1] 데이터 디렉토리: {config.DATA_DIR}")
    found = 0
    for fname, corpus, tier, ctype in config.CORPUS_FILES:
        p = config.DATA_DIR / fname
        if p.exists():
            found += 1
            print(f"    ✓ {fname:26s} → {tier}/{ctype} ({corpus or '레코드 source'})")
        else:
            print(f"    · {fname:26s} (없음, 건너뜀)")
    if not found:
        ok = False

    # 2. 코퍼스 로딩
    if ok:
        try:
            from index import corpus_stats
            s = corpus_stats()
            print(f"\n[2] 코퍼스: 총 {s['total']} 청크")
            print(f"    출처(tier)  : {s['by_tier']}")
            print(f"    장르(corpus): {s['by_corpus']}")
            print(f"    청크형태    : {s['by_chunk_type']}")
            print(f"    펀드 수     : {s['funds']}개 (질의분석 프롬프트에 주입)")
            print(f"    distractor  : {s['distractor']}건 (검색 시 감점)")
            print(f"    표 청크     : {s['table_chunks']}건 (계산·사실조회 가산)")
            print(f"    저신뢰 파싱 : {s['low_confidence']}건 (confidence<0.8, 감점)")
        except Exception as e:
            print(f"    ✗ 코퍼스 로딩 실패: {e}")
            ok = False

    # 3. LLM
    print(f"\n[3] CLOVA Studio")
    print(f"    API 키      : {'설정됨' if config.CLOVA_API_KEY else '✗ 없음'}")
    print(f"    판정/분류   : {config.MODEL_PLAN}  (분해·라우팅·재작성·HyDE)")
    print(f"    문서채점    : {config.MODEL_GRADE}")
    print(f"    답변생성    : {config.MODEL_ANSWER}")
    print(f"    검증        : {config.MODEL_VERIFY}")
    print(f"    토큰 계측   : usage={'ON' if config.COUNT_TOKENS else 'OFF'}, "
          f"사전계산(Tokenizer API)={'ON' if config.PRECOUNT_TOKENS else 'OFF'}")

    # 4. 임베딩
    print(f"\n[4] 임베딩 ({config.EMBED_MODEL})")
    try:
        import sentence_transformers  # noqa: F401
        print(f"    라이브러리  : 설치됨")
    except ImportError:
        print(f"    라이브러리  : ✗ 없음 → BM25 단독 "
              f"(pip install sentence-transformers)")
    print(f"    접두사      : 질의='{config.EMBED_QUERY_PREFIX}' "
          f"문서='{config.EMBED_PASSAGE_PREFIX}'")
    print(f"    하이브리드  : BM25 {config.BM25_WEIGHT} : 벡터 {config.VEC_WEIGHT}")
    vec = config.CACHE_DIR / f"vectors_{config.EMBED_SLUG}.npy"
    print(f"    벡터 캐시   : {'있음' if vec.exists() else '없음 → BM25 단독 (python -m index.build_index)'}")

    # 5. 웹 검색
    print(f"\n[5] 네이버 웹 검색 폴백")
    print(f"    키          : {'설정됨' if config.NAVER_CLIENT_ID else '✗ 없음 (폴백 비활성)'}")

    # 6. 워크플로 스위치
    print(f"\n[6] 활성 구성요소")
    on = [k for k, v in config.WORKFLOW_SETTINGS.items() if v]
    off = [k for k, v in config.WORKFLOW_SETTINGS.items() if not v]
    print(f"    ON : {', '.join(on)}")
    print(f"    OFF: {', '.join(off) or '(없음)'}")

    # 7. 검색 단독 테스트 (LLM 불필요)
    if ok:
        print(f"\n[7] 검색 동작 확인 (LLM 호출 없음)")
        try:
            from index import get_index
            idx = get_index()
            for q, f in [("IRP 중도인출 사유", {"scope": "doc"}),
                         ("주택구입 중도인출 서류", {"scope": "doc"}),
                         ("퇴직연금 적립금 위험자산 투자한도", {"scope": "doc"}),
                         ("퇴직연금 IRP 가입 가능한 채권형 펀드", {"scope": "all"})]:
                hits = idx.search(q, top_k=3, filters=f)
                print(f"    '{q}' (scope={f['scope']}) → {len(hits)}건")
                for h in hits[:2]:
                    c = h["chunk"]
                    print(f"        {h['score']:.3f} {c['chunk_id']:24s} "
                          f"{(c.get('section') or c['text'])[:38]}")
        except Exception as e:
            print(f"    ✗ 검색 실패: {e}")
            ok = False

    print("\n" + "=" * 68)
    print("점검 완료" if ok else "⚠️ 문제가 있습니다 (위 ✗ 항목 확인)")
    return 0 if ok else 1


# ==========================================================================
def ask(question: str, profile: dict, *, show_mermaid: bool = False,
        show_json: bool = False) -> None:
    from llm import dry_run_banner
    from utils import render_mermaid, render_trace
    from workflows import run

    print("\n" + "=" * 68)
    print(f"질문: {question}")
    banner = dry_run_banner()
    if banner:
        print(banner)
    print("=" * 68)

    result = run(question, user_profile=profile)

    print("\n" + render_trace(result.get("trace", [])))

    print("\n" + "─" * 68)
    print("답변")
    print("─" * 68)
    print(result.get("final_answer") or "(답변 없음)")

    # ── 이 질문 1건이 쓴 토큰 ────────────────────────────────────────
    u = result.get("usage") or {}
    total = u.get("total_tokens") or (u.get("in_tokens", 0) + u.get("out_tokens", 0))
    print("\n" + "─" * 68)
    # usage_for() 는 "ms", usage_summary() 는 "total_ms" 로 준다.
    ms = u.get("ms", u.get("total_ms", 0))

    if u.get("mode") == "dry-run":
        # 토큰 0 을 그냥 찍으면 "왜 0이지?" 로 읽힌다. 이유를 함께 쓴다.
        print(f"토큰 측정 불가 — 드라이런이라 CLOVA 를 호출하지 않았습니다 "
              f"(노드 실행 {u.get('calls', 0)}회).")
        print("  실제 토큰을 재려면 CLOVA_API_KEY 를 설정하고 다시 실행하세요.")
    else:
        print(f"질문 1건 토큰: 총 {total:,}  "
              f"(입력 {u.get('in_tokens', 0):,} + 출력 {u.get('out_tokens', 0):,})  |  "
              f"LLM 호출 {u.get('calls', 0)}회  |  {ms / 1000:.1f}초")
        if u.get("precounted_in_tokens"):
            print(f"    ※ Tokenizer API 사전계산 입력 토큰: "
                  f"{u['precounted_in_tokens']:,}")

        if u.get("by_model"):
            print("  모델별")
            for m, d in sorted(u["by_model"].items(), key=lambda x: -x[1]["total"]):
                share = 100 * d["total"] / total if total else 0
                print(f"    {m:16s} {d['n']:>2}회  {d['total']:>7,} 토큰 ({share:4.1f}%)  "
                      f"{d['ms']:>6}ms")

        if u.get("by_stage"):
            print("  단계별")
            for stage, d in sorted(u["by_stage"].items(), key=lambda x: -x[1]["total"]):
                print(f"    {stage:22s} {d['n']:>2}회  {d['ms']:>6}ms  "
                      f"in={d['in']:>6} out={d['out']:>5} total={d['total']:>6}")

    if show_mermaid:
        print("\n[mermaid]\n" + render_mermaid(result.get("trace", [])))
    if show_json:
        printable = {k: v for k, v in result.items() if k != "trace"}
        print("\n[state]\n" + json.dumps(printable, ensure_ascii=False,
                                         indent=2, default=str)[:4000])


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="연금 에이전트 CLI")
    ap.add_argument("question", nargs="?", help="질문 (생략 시 대화형)")
    ap.add_argument("--check", action="store_true", help="설정·데이터 점검")
    ap.add_argument("--mermaid", action="store_true", help="실행 경로 mermaid 출력")
    ap.add_argument("--json", action="store_true", help="최종 state 출력")
    ap.add_argument("--profile", default="", help='사용자 정보 JSON (예: \'{"age":57}\')')
    a = ap.parse_args()

    if a.check:
        return cmd_check()

    profile = {}
    if a.profile:
        try:
            profile = json.loads(a.profile)
        except json.JSONDecodeError:
            print("--profile 은 JSON 형식이어야 합니다.")
            return 1

    if a.question:
        ask(a.question, profile, show_mermaid=a.mermaid, show_json=a.json)
        return 0

    # 대화형
    print("연금 에이전트 (종료: exit / 빈 줄)")
    while True:
        try:
            q = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit", "q"):
            break
        try:
            ask(q, profile, show_mermaid=a.mermaid, show_json=a.json)
        except Exception as e:
            print(f"오류: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
