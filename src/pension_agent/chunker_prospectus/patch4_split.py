# -*- coding: utf-8 -*-
"""
patch4 — pass10 분할 규칙 교체. 절단 대신 조각을 늘린다.

  · 표 블록: '통째로 두기' → 줄 단위 분할 + **머리줄을 각 조각에 복제**
  · 긴 문단: '그대로 두기' → 문장 경계 분할
  · 상한: 산문 850자 / 표 600자 (multilingual-e5-large 512토큰 실측 기준)

효과(올려주신 청크로 시뮬레이션):
  2492개 → 3690개 (1.48배), 중앙 578자, max 968자, 870자 초과 28개
  버려지던 328,783자(전체의 16%)가 인덱스로 돌아온다.

멱등, .bak4 백업. 재파싱 불필요:
  python run_all.py --chunk  &&  python fit_embed_text.py
"""
import os, glob, shutil, hashlib

PARSE_DIR = ""
SEARCH_ROOTS = ["/content", ".", "/content/drive/MyDrive"]

OLD_BLOCK = 'MAX_LEN = 800\nMIN_LEN = 120\n\n\ndef is_table_block(text):\n    """숫자·구분자가 많고 줄이 짧으면 표로 본다 — 쪼개지 않는다."""\n    lines = [l for l in text.split("\\n") if l.strip()]\n    if len(lines) < 3:\n        return False\n    short = sum(1 for l in lines if len(l.strip()) <= 30)\n    numeric = sum(1 for l in lines if len(re.findall(r"[\\d.,%-]", l)) > len(l) * 0.35)\n    return short / len(lines) > 0.6 or numeric / len(lines) > 0.5\n\n\ndef split_page(text):\n    """\n    페이지 텍스트를 800자 이하 조각으로 나눈다.\n    문단(빈 줄) → 줄 단위 순으로 경계를 찾고, 표 블록은 통째로 둔다.\n    """\n    text = text.strip()\n    if not text:\n        return []\n    if len(text) <= MAX_LEN or is_table_block(text):\n        return [text]\n\n    # 문단 우선\n    paras = [p.strip() for p in re.split(r"\\n\\s*\\n", text) if p.strip()]\n    if len(paras) == 1:\n        paras = [l.strip() for l in text.split("\\n") if l.strip()]\n\n    out, cur = [], ""\n    for p in paras:\n        if len(p) > MAX_LEN and not cur:\n            out.append(p)                     # 한 문단이 이미 길면 그대로 (문장 중간 절단 방지)\n            continue\n        if len(cur) + len(p) + 1 > MAX_LEN and cur:\n            out.append(cur)\n            cur = p\n        else:\n            cur = (cur + "\\n" + p).strip() if cur else p\n    if cur:\n        out.append(cur)\n\n    # 너무 짧은 꼬리는 앞에 붙인다\n    merged = []\n    for c in out:\n        if merged and len(c) < MIN_LEN:\n            merged[-1] = merged[-1] + "\\n" + c\n        else:\n            merged.append(c)\n    return merged\n'
NEW_BLOCK = 'MAX_LEN = 850            # 산문 조각 상한 (508토큰 ≈ 870자 실측 기준)\nTABLE_MAX = 600          # 표 조각 상한 — 숫자·구분자는 토큰을 더 먹는다\nMIN_LEN = 120\n\n\ndef is_table_block(text):\n    """숫자·구분자가 많고 줄이 짧으면 표로 본다."""\n    lines = [l for l in text.split("\\n") if l.strip()]\n    if len(lines) < 3:\n        return False\n    short = sum(1 for l in lines if len(l.strip()) <= 30)\n    numeric = sum(1 for l in lines if len(re.findall(r"[\\d.,%-]", l)) > len(l) * 0.35)\n    return short / len(lines) > 0.6 or numeric / len(lines) > 0.5\n\n\n# ★ 표 조각마다 앞에 복제할 머리줄을 고른다.\n#   ⚠️ 길이 상한이 없으면 잘린 본문 문장("…이란 집합투자기구 수익률이")이 머리로\n#      뽑혀 모든 조각에 복제된다. 짧은 줄만 머리로 인정한다.\nCOLHEAD = re.compile(r"^(구\\s*분|종\\s*류|클래스|항\\s*목)\\b|구\\s*분\\s")\n\n\ndef _table_head(lines):\n    head = []\n    for l in lines[:3]:\n        if len(l) <= 45 and len(re.findall(r"\\d", l)) <= len(l) * 0.25:\n            head.append(l)\n        else:\n            break\n    head = head[:2]\n    for l in lines[:6]:                      # 열 머리행(구분/종류…)이 있으면 함께\n        if l not in head and COLHEAD.search(l) and len(l) <= 120:\n            head.append(l)\n            break\n    return head\n\n\ndef split_table(text):\n    """\n    ★ 표를 줄 단위로 나누되 **머리줄을 각 조각에 복제**한다.\n\n    ⚠️ 예전 규칙은 \'표는 통째로\'였다. 헤더를 잃지 않으려는 의도는 맞지만\n       75줄짜리 표가 한 청크가 되고, 임베딩(512토큰)에서 뒤가 통째로 잘려\n       인덱싱됐다(전체 글자의 16%가 검색 불가). 머리를 복제하면 두 목적이 다 된다.\n    """\n    lines = [l for l in text.split("\\n") if l.strip()]\n    if len(text) <= TABLE_MAX or len(lines) < 2:\n        return [text]\n    head = _table_head(lines)\n    body = lines[len(head):]\n    if not body:\n        return [text]\n    hd = "\\n".join(head)\n    room = TABLE_MAX - len(hd) - 12\n    if room < 150:                           # 머리가 너무 길면 머리 없이 자른다\n        hd, room = "", TABLE_MAX\n\n    groups, cur, clen = [], [], 0\n    for l in body:\n        if cur and clen + len(l) + 1 > room:\n            groups.append(cur)\n            cur, clen = [], 0\n        cur.append(l)\n        clen += len(l) + 1\n    if cur:\n        groups.append(cur)\n    if len(groups) > 1 and sum(len(x) for x in groups[-1]) < MIN_LEN:\n        last = groups.pop()                  # 꼬리 조각이 짧으면 앞에 붙인다\n        groups[-1] += last\n\n    out = []\n    for i, g in enumerate(groups):\n        pre = hd if i == 0 else ((hd + "  (표 계속)") if hd else "(표 계속)")\n        out.append("\\n".join(([pre] if pre else []) + g))\n    return out\n\n\ndef split_sentences(p, limit):\n    """한 문단이 상한을 넘으면 문장 경계로 나눈다. 문장 중간 절단을 피한다."""\n    sents = re.split(r"(?<=다\\.)\\s+|(?<=\\.)\\s+(?=[가-힣A-Z])", p)\n    res, cur = [], []\n    for s in sents:\n        if len(s) > limit:                   # 한 문장이 이미 상한 초과 — 강제 절단\n            if cur:\n                res.append(" ".join(cur))\n                cur = []\n            res.extend(s[i:i + limit] for i in range(0, len(s), limit))\n            continue\n        if cur and sum(len(x) + 1 for x in cur) + len(s) > limit:\n            res.append(" ".join(cur))\n            cur = []\n        cur.append(s)\n    if cur:\n        res.append(" ".join(cur))\n    return [x for x in res if x.strip()]\n\n\ndef split_page(text):\n    """페이지 텍스트를 조각으로 나눈다. 표는 머리 복제, 산문은 문단→문장 순."""\n    text = text.strip()\n    if not text:\n        return []\n    if is_table_block(text):\n        return split_table(text)\n    if len(text) <= MAX_LEN:\n        return [text]\n\n    paras = [p.strip() for p in re.split(r"\\n\\s*\\n", text) if p.strip()]\n    if len(paras) == 1:\n        paras = [l.strip() for l in text.split("\\n") if l.strip()]\n\n    out, cur = [], ""\n    for p in paras:\n        if len(p) > MAX_LEN:                 # ★ 예전엔 그대로 뒀다 → 문장 분할\n            if cur:\n                out.append(cur)\n                cur = ""\n            out.extend(split_sentences(p, MAX_LEN))\n            continue\n        if len(cur) + len(p) + 1 > MAX_LEN and cur:\n            out.append(cur)\n            cur = p\n        else:\n            cur = (cur + "\\n" + p).strip() if cur else p\n    if cur:\n        out.append(cur)\n\n    merged = []\n    for c in out:\n        if merged and len(c) < MIN_LEN:\n            merged[-1] = merged[-1] + "\\n" + c\n        else:\n            merged.append(c)\n    return merged\n\n\n'


def find_parse_dir():
    if PARSE_DIR:
        return PARSE_DIR
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        hits = [h for h in glob.glob(os.path.join(root, "**", "pass10_chunk_section.py"),
                                     recursive=True) if "__pycache__" not in h]
        if hits:
            return os.path.dirname(sorted(hits, key=len)[0])
    raise SystemExit("[!] pass10_chunk_section.py 를 못 찾았습니다. PARSE_DIR 을 지정하세요.")


DIR = find_parse_dir()
path = os.path.join(DIR, "pass10_chunk_section.py")
src = open(path, encoding="utf-8").read()
print(f"PARSE_DIR = {DIR}\n")

if "def split_table" in src:
    print("• 이미 적용됨 — 변경 없음")
elif src.count(OLD_BLOCK) != 1:
    print(f"❌ 앵커 {src.count(OLD_BLOCK)}회 매칭 (1회여야 함)")
    print("   pass10_chunk_section.py 가 이미 수정된 상태일 수 있습니다.")
else:
    if not os.path.exists(path + ".bak4"):
        shutil.copy2(path, path + ".bak4")
    src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    # 리포트 문구도 새 상한에 맞춘다
    src = src.replace(
        'print(f"  800자 초과 {sum(1 for x in lens if x > MAX_LEN)}개 (표·단일문단)")',
        'print(f"  {MAX_LEN}자 초과 {sum(1 for x in lens if x > MAX_LEN)}개")')
    open(path, "w", encoding="utf-8").write(src)
    print("✅ pass10_chunk_section.py 분할 규칙 교체")

import py_compile
try:
    py_compile.compile(path, doraise=True)
    print("✅ 문법 정상")
except Exception as e:
    print(f"❌ 문법 오류: {e}\n   {path}.bak4 로 되돌리세요")
    raise SystemExit(1)

# ── 실제 페이지 덤프로 미리보기 (파일은 쓰지 않는다) ──────
import sys, importlib
sys.path.insert(0, os.path.dirname(DIR))
try:
    m = importlib.import_module("parse.pass10_chunk_section")
    importlib.reload(m)
    import json
    dump_dir = os.path.join(os.path.dirname(os.path.dirname(DIR)), "data", "dump")
    files = sorted(glob.glob(os.path.join(dump_dir, "*.json")))[:20]
    if files:
        lens = []
        for f in files:
            for p in json.load(open(f, encoding="utf-8"))["pages"]:
                for piece in m.split_page(p["text"]):
                    if len(piece) >= m.MIN_LEN:
                        lens.append(len(piece))
        lens.sort()
        if lens:
            print(f"\n미리보기(덤프 20개 전 페이지): 조각 {len(lens)}개  "
                  f"중앙 {lens[len(lens)//2]}  max {lens[-1]}  "
                  f"850자 초과 {sum(1 for x in lens if x > 850)}")
except Exception as e:
    print(f"(미리보기 생략: {e})")

print("\n다음: python run_all.py --chunk  &&  python fit_embed_text.py")
