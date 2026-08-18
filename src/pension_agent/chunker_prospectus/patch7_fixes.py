# -*- coding: utf-8 -*-
"""
patch7 — patch6 이후 남은 세 가지.

  X-1  pass5 : 근거페이지 역탐색을 다중 키로 (앞/뒤/최장 한글구간) + 섹션 우선
               → 79/335행에서 크게 올라가야 한다
  X-2  pass9 : 수수료 칸이 가입자격으로 들어온 36행 제거
  X-3  pass9 : 설명(판매유형)과 충돌하는 연금 판정 무시 (C-Pe 뒤집힘 2건)

멱등, .bak7 백업.  실행 후: python run_all.py --from 5
"""
import os, glob, shutil

PARSE_DIR = ""
SEARCH_ROOTS = ["/content", ".", "/content/drive/MyDrive"]
_LOG = []


def find_parse_dir():
    if PARSE_DIR:
        return PARSE_DIR
    for root in SEARCH_ROOTS:
        hits = [h for h in glob.glob(os.path.join(root, "**", "pass9_chunk_class.py"),
                                     recursive=True) if "__pycache__" not in h]
        if hits:
            return os.path.dirname(sorted(hits, key=len)[0])
    raise SystemExit("[!] pass9_chunk_class.py 를 못 찾았습니다.")


def patch(fname, tag, old, new, done_marker=None):
    path = os.path.join(DIR, fname)
    if not os.path.exists(path):
        _LOG.append(("FAIL", tag, f"{fname} 없음")); return
    src = open(path, encoding="utf-8").read()
    if done_marker and done_marker in src:
        _LOG.append(("SKIP", tag, "이미 적용됨")); return
    n = src.count(old)
    if n == 0:
        _LOG.append(("SKIP", tag, "앵커 없음 — patch6을 먼저 적용했는지 확인")); return
    if n > 1:
        _LOG.append(("FAIL", tag, f"앵커 {n}회 매칭")); return
    if not os.path.exists(path + ".bak7"):
        shutil.copy2(path, path + ".bak7")
    open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
    _LOG.append(("OK", tag, fname))


DIR = find_parse_dir()
print(f"PARSE_DIR = {DIR}\n")

patch("pass5_eligibility.py", "X-1a 시그니처",
      "def locate_pages(pages, elig):", "def locate_pages(pages, elig, info=None):",
      done_marker="def locate_pages(pages, elig, info=None)")
patch("pass5_eligibility.py", "X-1b 다중 키 탐색",
      '    by = {p["page"]: re.sub(r"\\s+", "", p["text"]) for p in pages}\n    out = {}\n    for c, v in elig.items():\n        key = re.sub(r"\\s+", "", v)[:24]\n        if len(key) < 8:\n            continue\n        for pg in sorted(by):\n            if key in by[pg]:\n                out[c] = pg\n                break\n    return out', '    by = {p["page"]: re.sub(r"\\s+", "", p["text"]) for p in pages}\n    # ★ 섹션 범위를 먼저 본다. 같은 문구가 목차·다른 항에도 나올 수 있다.\n    pref = []\n    if info:\n        for key, smax in [("11항매입환매", 6), ("제2부6항구조", 6), ("13항보수", 8)]:\n            rng = span(info.get("sec", {}), key, info.get("pages", 0), smax)\n            if rng:\n                pref += list(range(rng[0], rng[1] + 1))\n    order = [p for p in pref if p in by] + [p for p in sorted(by) if p not in pref]\n\n    out = {}\n    for c, v in elig.items():\n        s = re.sub(r"\\s+", "", v)\n        # ⚠️ 앞 24자 하나로만 찾으면 실패한다. 5패스가 여러 칸을 이어붙인 값이라\n        #   그 순서 그대로 원문에 연속해 나오지 않는 경우가 많다(254/335행 실패).\n        #   앞·뒤·가장 긴 한글 구간을 모두 후보로 쓴다.\n        keys = []\n        if len(s) >= 12:\n            keys.append(s[:24])\n        if len(s) >= 30:\n            keys.append(s[-24:])\n        keys += sorted(re.findall(r"[가-힣]{12,}", s), key=len, reverse=True)[:2]\n        keys = [k for k in dict.fromkeys(keys) if len(k) >= 12]\n        if not keys:\n            continue\n        for pg in order:\n            if any(k in by[pg] for k in keys):\n                out[c] = pg\n                break\n    return out', done_marker="keys += sorted(re.findall")
patch("pass5_eligibility.py", "X-1c 호출부",
      'rec["가입자격_페이지"] = locate_pages(pages, elig)',
      'rec["가입자격_페이지"] = locate_pages(pages, elig, info)',
      done_marker="locate_pages(pages, elig, info)")

patch("pass9_chunk_class.py", "X-2a 수수료 패턴",
      '_NONE = re.compile(r"제한\\s*없음|가입\\s*제한이?\\s*없|제한없|해당\\s*없음|누구나")', '_NONE = re.compile(r"제한\\s*없음|가입\\s*제한이?\\s*없|제한없|해당\\s*없음|누구나")\n# ⚠️ 5패스가 수수료 칸을 가입자격으로 긁어오는 경우가 있다(335행 중 36행).\n#   "수수료선취-오프라인 납입금액의0.02% 이내" → 가입자격이 아니다. 버린다.\n_FEE = re.compile(r"납입금액|판매수수료율|%\\s*이내|이내의?\\s*범위|환매금액")', done_marker="_FEE = re.compile(")
patch("pass9_chunk_class.py", "X-2b 필터 적용",
      '        if not _ARTICLE.search(s[:30]):          # 규약 조문이면 버린다', '        if not _ARTICLE.search(s[:30]) and not _FEE.search(s):   # 조문·수수료칸 제외', done_marker="not _FEE.search(s)")
patch("pass9_chunk_class.py", "X-3 연금 충돌 해소",
      '    if tags:\n        return ", ".join(tags), "문서"\n\n    # ── ③ 코드·설명 규칙 ──\n    #   ⚠️ \'제한 없음\'을 여기서 추론하지 않는다. 원문에 없는 단정은 넣지 않는다.\n    pt = (crec or {}).get("연금유형")', '    # ★ 설명(판매유형)과 충돌하면 설명을 따른다.\n    #   가입자격 문장이 옆 행 것과 섞여 C-Pe(개인연금)가 퇴직연금으로 뒤집히는 사례가 있다.\n    pt = (crec or {}).get("연금유형")\n    if tags and pt == "개인연금":\n        tags = [t for t in tags if "퇴직연금" not in t]\n    elif tags and pt == "퇴직연금":\n        tags = [t for t in tags if "연금저축" not in t]\n    if tags:\n        return ", ".join(tags), "문서"\n\n    # ── ③ 코드·설명 규칙 ──\n    #   ⚠️ \'제한 없음\'을 여기서 추론하지 않는다. 원문에 없는 단정은 넣지 않는다.',
      done_marker="설명(판매유형)과 충돌하면")

w = max(len(t) for _, t, _ in _LOG)
for st, tag, msg in _LOG:
    print(f"{ {'OK':'✅','SKIP':'•','FAIL':'❌'}[st] } {tag:<{w}}  {msg}")
print(f"\n적용 {sum(1 for s,_,_ in _LOG if s=='OK')} / 건너뜀 "
      f"{sum(1 for s,_,_ in _LOG if s=='SKIP')} / 실패 {sum(1 for s,_,_ in _LOG if s=='FAIL')}")

import py_compile
bad = []
for f in ["pass5_eligibility.py", "pass9_chunk_class.py"]:
    try:
        py_compile.compile(os.path.join(DIR, f), doraise=True)
    except Exception as e:
        bad.append((f, str(e)[:150]))
print("❌ 문법 오류: " + str(bad) if bad else "✅ 문법 정상")
print("\n다음: python run_all.py --from 5  &&  python fit_embed_text.py")
