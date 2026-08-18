# -*- coding: utf-8 -*-
"""
patch13 — 섹션 청크를 계약(INTERFACES.md §1-3)에 맞춘다

계약이 "✅ 포함/정본"으로 지정한 섹션 중 빠져 있던 것을 넣는다.

  추가(기본 ON)
    13항 보수·수수료   ✅ 정본 — 환매수수료·판매수수료 원문. 카드엔 총보수만 있어
                              "환매수수료 있나요?"에 답할 근거가 지금 없다
    제3부 재무·운용실적  ✅ 포함 — 수익률 원문 + 진짜 수익률기준일
    제1부 1항          ✅ 정본 — 전 클래스 목록 + 협회 펀드코드

  추가(기본 OFF — ADD_OPTIONAL=True 로 켬)
    요약정보           ✅ 포함이나 내용이 펀드 카드와 거의 겹친다(+908 청크)
    제4부              ✅ 포함이나 질의 세트에 해당 문항이 없다(+1,152 청크)

⚠️ 청크 수가 3,514 → 약 6,700개가 된다(옵션까지 켜면 8,762).
   임베딩을 다시 태워야 하므로 서윤님과 시점을 맞출 것.

계약이 ❌로 지정한 것(연혁·12항·14항과세·제5부·엄브렐러 목록)은 그대로 제외한다.
투자위험 가/다 세분화(계약은 '나. 특수위험'만)는 보류 — 소제목 판정이 필요하고,
patch4로 잘게 쪼개져 있어 손해가 작다.

멱등, .bak22 백업.  실행 후: python run_all.py --chunk && python fit_embed_text.py
"""
import os, glob, shutil

PARSE_DIR = ""
ADD_OPTIONAL = False        # True 로 바꾸면 요약정보·제4부까지 추가
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
PATH = os.path.join(DIR, "pass10_chunk_section.py")
src = open(PATH, encoding="utf-8").read()
print(f"PARSE_DIR = {DIR}   ADD_OPTIONAL = {ADD_OPTIONAL}\n")

OLD = '''# (섹션키, span_max, 표시명)
KEEP = [
    ("투자대상",     5, "투자대상"),
    ("투자전략",     5, "투자전략"),
    ("투자위험",     5, "투자위험"),
    ("11항매입환매", 6, "매입·환매"),
    ("별첨1",        4, "별첨1"),
]'''

NEW = '''# (섹션키, span_max, 표시명)
# ★ 계약 INTERFACES.md §1-3의 "✅ 포함/정본" 목록을 따른다.
#   ⚠️ 카드에 값이 있다고 원문을 빼면 안 된다. 카드는 총보수만 담고 있어서
#      "환매수수료 있나요?" 같은 질의에 답할 근거가 사라졌다(13항 누락).
KEEP = [
    ("요약정보",     4, "요약정보"),          # 계약 ✅ 포함 — OPTIONAL
    ("제1부1항",     3, "제1부 클래스"),       # 계약 ✅ 정본 — 클래스 목록·협회코드
    ("투자대상",     5, "투자대상"),
    ("투자전략",     5, "투자전략"),
    ("투자위험",     5, "투자위험"),
    ("11항매입환매", 6, "매입·환매"),
    ("13항보수",     8, "보수·수수료"),        # 계약 ✅ 정본 — 환매·판매수수료 원문
    ("별첨1",        4, "별첨1"),
    ("제3부",        8, "재무·운용실적"),      # 계약 ✅ 포함 — 수익률 원문·기준일
    ("제4부",        6, "관련회사"),           # 계약 ✅ 포함(1·3항) — OPTIONAL
]

# 계약에 있으나 질의 세트에 대응 문항이 없고 청크만 2,060개 늘어나는 둘은
# 기본적으로 뺀다. 필요해지면 이 집합을 비우면 된다.
_OPTIONAL = {"요약정보", "제4부"}
if os.environ.get("ADD_OPTIONAL") != "1":
    KEEP = [k for k in KEEP if k[0] not in _OPTIONAL]

# 계약이 ❌로 지정해 넣지 않는 것:
#   연혁(과거 등급·보수 오염원) / 12항 기준가격산정 / 14항 과세(distractor) /
#   제5부(법정 문구) / 엄브렐러 전환 대상 목록(다른 펀드 정보)'''

if "계약 INTERFACES.md §1-3" in src:
    print("• 이미 적용됨")
elif src.count(OLD) != 1:
    print(f"❌ 앵커 {src.count(OLD)}회 (1회여야 함)")
    print("   pass10_chunk_section.py 의 KEEP 목록이 이미 수정된 상태일 수 있습니다.")
    raise SystemExit(1)
else:
    if not os.path.exists(PATH + ".bak22"):
        shutil.copy2(PATH, PATH + ".bak22")
    open(PATH, "w", encoding="utf-8").write(src.replace(OLD, NEW, 1))
    print("✅ KEEP 목록 교체")

import py_compile
py_compile.compile(PATH, doraise=True)
print("✅ 문법 정상")

# ── 적용 후 예상 규모 ─────────────────────────────────
import sys, json, importlib
sys.path.insert(0, os.path.dirname(DIR))
if ADD_OPTIONAL:
    os.environ["ADD_OPTIONAL"] = "1"
m = importlib.import_module("parse.pass10_chunk_section")
importlib.reload(m)
from parse.common import load_json, TABLE_DIR
from parse.pass3_extract import span

smap = load_json(os.path.join(TABLE_DIR, "section_map.json"))
tot = 0
per = {}
for did, v in smap.items():
    for key, smax, label in m.KEEP:
        r = span(v.get("sec", {}), key, v.get("pages", 0), smax)
        if r:
            n = r[1] - r[0] + 1
            tot += n
            per[label] = per.get(label, 0) + n
print(f"\n대상 섹션: {[k[2] for k in m.KEEP]}")
print(f"페이지 합 {tot}p  → 예상 청크 약 {int(tot * 2.49)}개 (현재 3,514)")
for k, v in sorted(per.items(), key=lambda x: -x[1]):
    print(f"   {k:<14}{v:>5}p")
print("\n다음: python run_all.py --chunk  &&  python fit_embed_text.py")
