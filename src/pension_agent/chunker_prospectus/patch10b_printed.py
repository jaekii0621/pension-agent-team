# -*- coding: utf-8 -*-
"""
patch10b — 인쇄 라벨 맵이 ["유형", 페이지] 형태인 것에 맞춘다. (patch10 뒤에 적용)

  printed_pension_map.json = {doc_id: {class: ["개인연금", 3], ...}}

  P-1  _norm_pension 이 리스트/튜플을 풀도록
       (지금은 str(리스트)에 '개인연금'이 들어 있어 우연히 동작한다 — 우연에 기대지 않는다)
  P-2  인쇄 라벨의 **페이지**를 요약줄 근거로 노출
       "연금저축계좌 가입 가능 클래스: C-P, C-Pe (근거 p.3, p.28)"
  P-3  청크에 연금유형_근거페이지 필드 추가

멱등, .bak10b 백업.
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


def patch(tag, old, new, marker=None):
    global SRC
    if marker and marker in SRC:
        _LOG.append(("SKIP", tag, "이미 적용됨")); return
    n = SRC.count(old)
    if n != 1:
        _LOG.append(("FAIL", tag, f"앵커 {n}회 (1회여야 함)")); return
    SRC = SRC.replace(old, new, 1)
    _LOG.append(("OK", tag, ""))


DIR = find_parse_dir()
PATH = os.path.join(DIR, "pass9_chunk_class.py")
SRC = open(PATH, encoding="utf-8").read()
print(f"PARSE_DIR = {DIR}\n")

if "_pension_of" not in SRC:
    raise SystemExit("[!] patch10 을 먼저 적용하세요.")

# ── P-1 리스트 언랩 ──────────────────────────────────
patch("P-1 리스트 형태 처리",
'''def _norm_pension(v):''',
'''def _unwrap(v):
    """["개인연금", 3] 처럼 [유형, 페이지] 로 저장된 값을 (유형, 페이지)로 푼다."""
    if isinstance(v, (list, tuple)):
        t = v[0] if v else None
        pg = v[1] if len(v) > 1 and isinstance(v[1], int) else None
        return t, pg
    if isinstance(v, dict):
        return (v.get("유형") or v.get("연금유형") or v.get("type"),
                v.get("page") or v.get("페이지") or v.get("근거페이지"))
    return v, None


def _printed_entry(rec, cls):
    """인쇄 라벨 (유형, 페이지). 없으면 (None, None)."""
    m = (_printed_map().get(rec["doc_id"]) or {})
    if isinstance(m, dict) and cls in m:
        t, pg = _unwrap(m[cls])
        return _norm_pension(t), pg
    return None, None


def _norm_pension(v):''',
      marker="def _printed_entry(")

patch("P-1b _pension_of 에서 사용",
'''    m = (_printed_map().get(rec["doc_id"]) or {})
    v = _norm_pension(m.get(cls) if isinstance(m, dict) else None)
    if v:
        return v, "인쇄"''',
'''    v, _pg = _printed_entry(rec, cls)
    if v:
        return v, "인쇄"''',
      marker="v, _pg = _printed_entry(rec, cls)")

# ── P-2 요약줄에 근거 페이지 ─────────────────────────
patch("P-2 요약줄 근거",
'''    mark = "(코드규칙 추정 포함)" if guessed else ""
    if p_save:
        lines.append(f"\\n연금저축계좌 가입 가능 클래스: {', '.join(p_save)} {mark}".rstrip())
    if p_ret:
        lines.append(f"퇴직연금(IRP) 가입 가능 클래스: {', '.join(p_ret)} {mark}".rstrip())''',
'''    mark = "(코드규칙 추정 포함)" if guessed else ""

    def _pgs(codes):
        # ★ 인쇄 라벨에 근거 페이지가 함께 저장돼 있다. 요약줄은 가장 자주 인용되는
        #   줄이므로 여기에 근거를 붙인다.
        s = sorted({_printed_entry(rec, c)[1] for c in codes
                    if _printed_entry(rec, c)[1]})
        return " (근거 " + ", ".join(f"p.{p}" for p in s) + ")" if s else ""

    if p_save:
        lines.append(f"\\n연금저축계좌 가입 가능 클래스: {', '.join(p_save)}"
                     f"{_pgs(p_save)} {mark}".rstrip())
    if p_ret:
        lines.append(f"퇴직연금(IRP) 가입 가능 클래스: {', '.join(p_ret)}"
                     f"{_pgs(p_ret)} {mark}".rstrip())''',
      marker="def _pgs(codes):")

# ── P-3 청크 필드 ────────────────────────────────────
patch("P-3 청크 필드",
'''        "총보수_근거페이지": {c: p for c, p in _fee_pages(rec).items() if p} or None,''',
'''        "총보수_근거페이지": {c: p for c, p in _fee_pages(rec).items() if p} or None,
        "연금유형_근거페이지": {c["class"]: _printed_entry(rec, c["class"])[1]
                        for c in (rec.get("클래스") or [])
                        if _printed_entry(rec, c["class"])[1]} or None,''',
      marker='"연금유형_근거페이지"')

if any(s == "OK" for s, _, _ in _LOG):
    if not os.path.exists(PATH + ".bak10b"):
        shutil.copy2(PATH, PATH + ".bak10b")
    open(PATH, "w", encoding="utf-8").write(SRC)

w = max(len(t) for _, t, _ in _LOG)
for st, tag, msg in _LOG:
    print(f"{ {'OK': '✅', 'SKIP': '•', 'FAIL': '❌'}[st] } {tag:<{w}}  {msg}")
print(f"\n적용 {sum(1 for s,_,_ in _LOG if s=='OK')} / 건너뜀 "
      f"{sum(1 for s,_,_ in _LOG if s=='SKIP')} / 실패 {sum(1 for s,_,_ in _LOG if s=='FAIL')}")

import py_compile
py_compile.compile(PATH, doraise=True)
print("✅ 문법 정상")

import sys, importlib
sys.path.insert(0, os.path.dirname(DIR))
m = importlib.import_module("parse.pass9_chunk_class")
importlib.reload(m)
pm = m._printed_map()
print(f"\n인쇄 라벨 맵 {len(pm)}개 문서")
if pm:
    k = list(pm)[0]
    cls = list(pm[k])[0]
    print(f"  {k} / {cls} → {m._printed_entry({'doc_id': k}, cls)}   (유형, 페이지)")
