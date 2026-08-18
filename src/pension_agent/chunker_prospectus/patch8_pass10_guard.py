# -*- coding: utf-8 -*-
"""
patch8 — pass10이 입력 없이 기존 청크 파일을 날리지 않게.

사고 사례: dump가 없는 상태로 --from 5 를 돌렸더니 섹션 청크 0개가 만들어지고
section_chunks.jsonl 이 빈 파일로 덮어써졌다(3,514개 소실, 백업에서 복구).

  G-1  입력(덤프)이 하나도 없으면 파일을 쓰지 않고 exit 1
  G-2  기존 파일 대비 50% 미만으로 급감하면 경고 + 중단 (FORCE=1 로 우회)
  G-3  min()/max() 빈 리스트 크래시 제거

멱등, .bak8 백업.
"""
import os, glob, shutil

PARSE_DIR = ""
SEARCH_ROOTS = ["/content", ".", "/content/drive/MyDrive"]


def find_parse_dir():
    if PARSE_DIR:
        return PARSE_DIR
    for root in SEARCH_ROOTS:
        hits = [h for h in glob.glob(os.path.join(root, "**", "pass10_chunk_section.py"),
                                     recursive=True) if "__pycache__" not in h]
        if hits:
            return os.path.dirname(sorted(hits, key=len)[0])
    raise SystemExit("[!] pass10_chunk_section.py 를 못 찾았습니다.")


DIR = find_parse_dir()
path = os.path.join(DIR, "pass10_chunk_section.py")
src = open(path, encoding="utf-8").read()
print(f"PARSE_DIR = {DIR}\n")

OLD = '''    path = os.path.join(CHUNK_DIR, "section_chunks.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\\n")'''

NEW = '''    path = os.path.join(CHUNK_DIR, "section_chunks.jsonl")

    # ★ 입력이 없는데 파일을 쓰면 기존 청크가 통째로 사라진다.
    #   실제로 dump 없이 실행해 3,514개가 빈 파일로 덮어써진 적이 있다.
    prev = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prev = sum(1 for _ in f)
    force = os.environ.get("FORCE") == "1"

    if not out:
        print(f"[!] 생성된 청크가 0개입니다 — 파일을 쓰지 않고 중단합니다 "
              f"(기존 {prev}개 보존).")
        print("    덤프(data/dump)와 six_axis.json 이 있는지 확인하세요.")
        sys.exit(1)
    if prev and len(out) < prev * 0.5 and not force:
        print(f"[!] 청크가 {prev} → {len(out)} 로 급감했습니다 — 중단합니다.")
        print("    의도한 변경이면 FORCE=1 을 붙여 다시 실행하세요.")
        sys.exit(1)

    with open(path, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\\n")'''

if "생성된 청크가 0개입니다" in src:
    print("• 이미 적용됨")
elif src.count(OLD) != 1:
    print(f"❌ 앵커 {src.count(OLD)}회 매칭 (1회여야 함)")
    raise SystemExit(1)
else:
    if not os.path.exists(path + ".bak8"):
        shutil.copy2(path, path + ".bak8")
    open(path, "w", encoding="utf-8").write(src.replace(OLD, NEW, 1))
    print("✅ 가드 추가")

import py_compile
py_compile.compile(path, doraise=True)
print("✅ 문법 정상")
print("\n확인: FORCE=1 없이 빈 입력으로 실행하면 exit 1 로 멈추고 기존 파일이 남습니다.")
