# -*- coding: utf-8 -*-
"""
backup_to_drive.py — 지금까지의 작업을 구글 드라이브에 스냅샷으로 남긴다.

담는 것:  src/ (패치된 pass 파일 + .bak*), data/tables/, data/chunks/,
          루트의 패치·유틸 스크립트, 실행 로그
빼는 것:  data/raw (원본 PDF — 용량이 크고 바뀌지 않는다)
          data/dump (재생성 가능. INCLUDE_DUMP = True 로 켤 수 있다)

같은 날 여러 번 돌려도 폴더가 겹치지 않는다(시각까지 붙는다).
"""
import os, sys, glob, json, shutil, subprocess, datetime

PROJECT = "/content/pension"
DRIVE_DIR = "/content/drive/MyDrive/pension_backup"
INCLUDE_DUMP = True
MAKE_TARBALL = True          # 폴더 복사 외에 tar.gz도 하나 남긴다

# ── 드라이브 마운트 ──────────────────────────────────
if not os.path.ismount("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
dest = os.path.join(DRIVE_DIR, stamp)
os.makedirs(dest, exist_ok=True)

TARGETS = [
    ("src", os.path.join(PROJECT, "src")),
    ("data/tables", os.path.join(PROJECT, "data", "tables")),
    ("data/chunks", os.path.join(PROJECT, "data", "chunks")),
]
if INCLUDE_DUMP:
    TARGETS.append(("data/dump", os.path.join(PROJECT, "data", "dump")))


def dsize(p):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(p) for f in fs)


print(f"백업 위치: {dest}\n")
total = 0
for name, src in TARGETS:
    if not os.path.isdir(src):
        print(f"  · {name:<14} 없음 — 건너뜀")
        continue
    out = os.path.join(dest, name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copytree(src, out, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    sz = dsize(out)
    total += sz
    n = sum(len(fs) for _, _, fs in os.walk(out))
    print(f"  ✅ {name:<14} {n:>4}개  {sz/1e6:>7.1f} MB")

# 루트의 패치·유틸 스크립트
os.makedirs(os.path.join(dest, "scripts"), exist_ok=True)
scripts = []
for pat in ["patch*.py", "fit_embed_text.py", "check_after.py",
            "refine_evidence.py", "run_all.py", "backup_to_drive.py"]:
    for f in glob.glob(os.path.join(PROJECT, pat)) + glob.glob(os.path.join("/content", pat)):
        shutil.copy2(f, os.path.join(dest, "scripts", os.path.basename(f)))
        scripts.append(os.path.basename(f))
print(f"  ✅ {'scripts':<14} {len(set(scripts)):>4}개")

# ── 현재 상태 요약을 같이 남긴다 ─────────────────────
manifest = {"시각": stamp, "project": PROJECT, "scripts": sorted(set(scripts))}
try:
    T = os.path.join(PROJECT, "data", "tables")
    six = json.load(open(os.path.join(T, "six_axis.json"), encoding="utf-8"))
    manifest["문서수"] = len(six)
    manifest["필드"] = {
        k: sum(1 for r in six if r.get(k) not in (None, [], {}, ""))
        for k in ["fund_name", "위험등급", "상품분류", "시장잔고_백만원",
                  "펀드수익률", "보수", "클래스", "가입자격", "수작업근거"]}
    for f in ["fund_cards", "class_cards", "section_chunks"]:
        p = os.path.join(PROJECT, "data", "chunks", f + ".jsonl")
        if os.path.exists(p):
            manifest[f] = sum(1 for _ in open(p, encoding="utf-8"))
except Exception as e:
    manifest["요약실패"] = str(e)[:120]

try:
    manifest["git"] = subprocess.run(["git", "-C", PROJECT, "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True).stdout.strip() or None
except Exception:
    pass

json.dump(manifest, open(os.path.join(dest, "MANIFEST.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ── tar.gz ──────────────────────────────────────────
if MAKE_TARBALL:
    tar = os.path.join(DRIVE_DIR, f"pension_{stamp}.tar.gz")
    subprocess.run(["tar", "-czf", tar, "-C", dest, "."], check=True)
    print(f"  ✅ {'tarball':<14}       {os.path.getsize(tar)/1e6:>7.1f} MB")

print(f"\n총 {total/1e6:.1f} MB → {dest}")
print("\n요약:", json.dumps(manifest.get("필드", {}), ensure_ascii=False))
print("청크:", {k: manifest.get(k) for k in
                ["fund_cards", "class_cards", "section_chunks"]})

# ── 오래된 스냅샷 정리 안내 ──────────────────────────
snaps = sorted(d for d in glob.glob(os.path.join(DRIVE_DIR, "20*")) if os.path.isdir(d))
if len(snaps) > 5:
    print(f"\n스냅샷 {len(snaps)}개 누적. 오래된 것 정리:")
    print(f"  !rm -rf {snaps[0]}")
