"""
투자설명서 파싱·청킹 전체 파이프라인

⚠️ 파일명 숫자 ≠ 실행 순서.
   pass8~10(청커)은 pass11(검증)·pass12(수작업 보정) **뒤에** 실행해야 한다.
   검증이 격리한 값과 수작업으로 되살린 값이 청크에 반영돼야 하기 때문이다.

⚠️ pass3~7은 six_axis.json을 순차적으로 덧씌운다. 순서를 바꾸면 앞 결과가 지워진다.

사용법
  python run_all.py            # 전체
  python run_all.py --from 3   # 3패스부터 (덤프 캐시 활용)
  python run_all.py --chunk    # 청커만 재생성 (파싱 결과 그대로)
  python run_all.py --skip 1   # 특정 단계 건너뛰기

환경변수: RAW_DIR (기본 data/raw/prospectus)
"""
import os, sys, subprocess, time, argparse

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "src", "parse")

# (단계번호, 표시명, 파일명, 그룹)
STEPS = [
    (0,  "중복 제거",        "pass0_dedup.py",         "parse"),
    (1,  "텍스트 덤프",      "pass1_dump.py",          "parse"),
    (2,  "섹션 지도",        "pass2_sections.py",      "parse"),
    (3,  "6축 필드",         "pass3_extract.py",       "parse"),
    (4,  "수익률·시장잔고",  "pass4_perf.py",          "parse"),
    (5,  "가입자격",         "pass5_eligibility.py",   "parse"),
    (6,  "깨짐 복구",        "pass6_broken.py",        "parse"),
    (7,  "보수 복구",        "pass7_fee_recover.py",   "parse"),
    (8,  "범위 검증",        "pass11_validate.py",     "parse"),
    (9,  "수작업 보정",      "pass12_override.py",     "parse"),
    (10, "펀드 카드",        "pass8_chunk_fund.py",    "chunk"),
    (11, "클래스 카드",      "pass9_chunk_class.py",   "chunk"),
    (12, "섹션 원문",        "pass10_chunk_section.py", "chunk"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0,
                    help="이 단계부터 실행 (0~12)")
    ap.add_argument("--chunk", action="store_true", help="청커(10~12)만 실행")
    ap.add_argument("--skip", type=int, nargs="*", default=[], help="건너뛸 단계 번호")
    args = ap.parse_args()

    os.environ.setdefault("RAW_DIR", os.path.join(BASE, "data", "raw"))
    steps = [s for s in STEPS if s[0] >= args.start and s[0] not in args.skip]
    if args.chunk:
        steps = [s for s in STEPS if s[3] == "chunk"]

    print(f"RAW_DIR = {os.environ['RAW_DIR']}")
    print(f"실행 단계 {len(steps)}개\n")

    t0 = time.time()
    for num, name, fname, _ in steps:
        path = os.path.join(SRC, fname)
        if not os.path.exists(path):
            print(f"[!] {name}: {fname} 없음 — 중단")
            sys.exit(1)
        print(f"\n{'='*62}\n[{num:>2}] {name}   ({fname})\n{'='*62}")
        t = time.time()
        r = subprocess.run([sys.executable, path], cwd=BASE)
        if r.returncode != 0:
            print(f"\n[!] {name} 실패 (exit {r.returncode}) — 중단")
            sys.exit(1)
        print(f"    ⏱ {time.time()-t:.1f}s")

    print(f"\n{'='*62}")
    print(f"완료 — 총 {time.time()-t0:.0f}초")
    for f in ["fund_cards.jsonl", "class_cards.jsonl", "section_chunks.jsonl"]:
        p = os.path.join(BASE, "data", "chunks", f)
        if os.path.exists(p):
            n = sum(1 for _ in open(p, encoding="utf-8"))
            print(f"  {f:<24} {n:>5}개")


if __name__ == "__main__":
    main()
