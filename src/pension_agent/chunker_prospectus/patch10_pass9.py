# -*- coding: utf-8 -*-
"""
patch10 — pass9 세 가지 수정

  Z-1  연금유형 판정에 **인쇄 라벨 우선순위** 적용
       인쇄 라벨 > 원문 설명 > 코드규칙 > 미확인
       ⚠️ C-P=개인연금 코드규칙이 코퍼스 10% 이상에서 틀렸다(C-P가 퇴직연금인 문서,
          C-P2가 개인연금인 문서가 둘 다 존재). 대소문자도 의미를 가진다(C-P vs Cp).
       → 요약줄이 이 판정을 쓰도록 바꾸고, 코드규칙으로 채운 건 표시한다.

  Z-2  가입자격 칸 오염 차단
       5패스 문장이 옆 행 내용을 끌고 오는 경우가 있다(Ce·S 행). 문장 안에
       **다른 클래스 코드**가 보이면 그 앞에서 자르거나 버린다.

  Z-3  총보수 근거 페이지를 카드에 노출
       보수 행별 페이지가 기록돼 있으면 "총보수 근거: p.29~30"을 붙이고
       청크에 총보수_근거페이지 필드를 넣는다.

인쇄 라벨 저장 위치를 몰라 **여러 후보를 자동 탐색**한다(아래 _printed_map 참조).
실제 위치가 다르면 PRINTED_KEYS / PRINTED_FILE 만 고치면 된다.

멱등, .bak10 백업.
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

# ══════════════════════════════════════════════════════
# Z-1  인쇄 라벨 우선 판정
# ══════════════════════════════════════════════════════
patch("Z-1a 판정 함수 추가",
'''def _elig(rec, cls, crec=None):''',
'''# ★ 인쇄 라벨 저장 위치 자동 탐색.
#   실제 구조가 다르면 아래 두 상수만 고치면 된다.
PRINTED_FILE = "printed_pension_map.json"          # TABLE_DIR 아래
PRINTED_KEYS = ("연금유형_인쇄", "인쇄연금유형", "printed_연금유형", "인쇄라벨")
_PMAP = None


def _printed_map():
    """{doc_id: {class: '개인연금'|'퇴직연금'}} 형태를 기대한다. 없으면 빈 dict."""
    global _PMAP
    if _PMAP is None:
        p = os.path.join(TABLE_DIR, PRINTED_FILE)
        try:
            _PMAP = load_json(p) if os.path.exists(p) else {}
        except Exception:
            _PMAP = {}
    return _PMAP


def _norm_pension(v):
    """'개인연금/연금저축' → 개인연금, '퇴직연금/IRP' → 퇴직연금."""
    s = str(v or "").replace(" ", "")
    if not s:
        return None
    if "퇴직" in s or "IRP" in s.upper():
        return "퇴직연금"
    if "개인연금" in s or "연금저축" in s:
        return "개인연금"
    return None


def _pension_of(rec, cls, crec=None):
    """
    (연금유형, 출처) 반환. 출처: '인쇄' / '문서' / '코드규칙' / None

    ⚠️ 우선순위가 핵심이다. C-P=개인연금 코드규칙이 코퍼스 10% 이상에서 틀렸다.
       C-P가 퇴직연금인 문서, C-P2가 개인연금인 문서가 둘 다 있고,
       한 문서 안에서 C-P(개인연금)와 Cp(퇴직연금)를 대소문자로만 구분하기도 한다.
       → 클래스 코드를 정규화하지 않는다. 인쇄 라벨이 있으면 그것이 정본이다.
    """
    # ① 인쇄 라벨 — 별도 맵 또는 rec/클래스 레코드 안의 키
    m = (_printed_map().get(rec["doc_id"]) or {})
    v = _norm_pension(m.get(cls) if isinstance(m, dict) else None)
    if v:
        return v, "인쇄"
    for k in PRINTED_KEYS:
        src = rec.get(k)
        if isinstance(src, dict):
            v = _norm_pension(src.get(cls))
            if v:
                return v, "인쇄"
        if crec is not None:
            v = _norm_pension(crec.get(k))
            if v:
                return v, "인쇄"

    # ② 원문 설명 문구 — 그 행만의 값이라 문장보다 정확하다
    d = ((crec or {}).get("설명") or "").replace(" ", "")
    if "개인연금" in d or "연금저축" in d:
        return "개인연금", "문서"
    if "퇴직연금" in d:
        return "퇴직연금", "문서"

    # ③ 3패스가 코드에서 추론한 값 — 신뢰도가 낮다
    v = _norm_pension((crec or {}).get("연금유형"))
    if v:
        return v, "코드규칙"
    return None, None


def _elig(rec, cls, crec=None):''',
      marker="def _pension_of(")

patch("Z-1b _elig 판정 교체",
'''    _d = ((crec or {}).get("설명") or "").replace(" ", "")
    pt = ("개인연금" if ("개인연금" in _d or "연금저축" in _d)
          else "퇴직연금" if "퇴직연금" in _d
          else (crec or {}).get("연금유형"))''',
'''    pt, _psrc = _pension_of(rec, cls, crec)''',
      marker="pt, _psrc = _pension_of(")

patch("Z-1c 코드규칙 출처 구분",
'''    if pt == "개인연금":
        return "연금저축계좌 전용", "코드규칙"
    if pt == "퇴직연금":
        return "퇴직연금·IRP 전용", "코드규칙"''',
'''    if pt == "개인연금":
        return "연금저축계좌 전용", ("문서" if _psrc in ("인쇄", "문서") else "코드규칙")
    if pt == "퇴직연금":
        return "퇴직연금·IRP 전용", ("문서" if _psrc in ("인쇄", "문서") else "코드규칙")''',
      marker='("문서" if _psrc in ("인쇄", "문서")')

patch("Z-1d 요약줄 교체",
'''    p_save = [c["class"] for c in classes if c.get("연금유형") == "개인연금"]
    p_ret = [c["class"] for c in classes if c.get("연금유형") == "퇴직연금"]
    if p_save:
        lines.append(f"\\n연금저축계좌 가입 가능 클래스: {', '.join(p_save)}")
    if p_ret:
        lines.append(f"퇴직연금(IRP) 가입 가능 클래스: {', '.join(p_ret)}")
    if not (p_save or p_ret):
        lines.append(f"\\n연금계좌 전용 클래스: {NA}")''',
'''    # ★ 요약줄은 카드에서 가장 자주 인용된다. 코드규칙 추론이 아니라
    #   인쇄 라벨·원문 설명을 우선으로 판정한다(_pension_of).
    p_save, p_ret, guessed = [], [], False
    for c in classes:
        pt_, src_ = _pension_of(rec, c["class"], c)
        if pt_ == "개인연금":
            p_save.append(c["class"])
        elif pt_ == "퇴직연금":
            p_ret.append(c["class"])
        if pt_ and src_ == "코드규칙":
            guessed = True
    mark = "(코드규칙 추정 포함)" if guessed else ""
    if p_save:
        lines.append(f"\\n연금저축계좌 가입 가능 클래스: {', '.join(p_save)} {mark}".rstrip())
    if p_ret:
        lines.append(f"퇴직연금(IRP) 가입 가능 클래스: {', '.join(p_ret)} {mark}".rstrip())
    if not (p_save or p_ret):
        # ⚠️ 인쇄에 라벨이 없다고 '가입 불가'로 뒤집지 않는다. 미확인이다.
        lines.append(f"\\n연금계좌 전용 클래스: {NA}")''',
      marker="p_save, p_ret, guessed = [], [], False")

patch("Z-1e 판매유형 칸 정합",
'''def _route(desc):
    d = (desc or "").replace(" ", "")''',
'''def _route(desc, pt=None):
    """
    판매경로 표기. pt를 주면 연금 구분은 그 값을 따른다.
    ⚠️ 설명 문구와 인쇄 라벨이 어긋나는 문서가 있다(C-P가 퇴직연금인 문서 등).
       판매유형 칸과 가입자격 칸이 서로 다른 근거를 쓰면 카드 안에서 모순이 된다.
    """
    d = (desc or "").replace(" ", "")''',
      marker="def _route(desc, pt=None)")

patch("Z-1f 연금 표기 치환",
'''    if "개인연금" in d or "연금저축" in d:
        parts.append("개인연금")
    elif "퇴직연금" in d:
        parts.append("퇴직연금")
    return "·".join(parts) if parts else NA''',
'''    if pt:
        parts.append(pt)
    elif "개인연금" in d or "연금저축" in d:
        parts.append("개인연금")
    elif "퇴직연금" in d:
        parts.append("퇴직연금")
    return "·".join(parts) if parts else NA''',
      marker="    if pt:\n        parts.append(pt)")

patch("Z-1g 호출부",
'''            _route(c.get("설명")),''',
'''            _route(c.get("설명"), _pension_of(rec, cls, c)[0]),''',
      marker='_route(c.get("설명"), _pension_of(')

# ══════════════════════════════════════════════════════
# Z-2  가입자격 칸 오염 차단
# ══════════════════════════════════════════════════════
patch("Z-2 다른 클래스 문구 차단",
'''            # 표 칸이므로 첫 절만 남긴다. 전문은 섹션 원문 청크에 있다.
            s = re.split(r"(?:하며|으로서|이며|에 한하여|를 통하여)", s)[0].strip(" ,")''',
'''            # ★ 5패스 문장이 옆 행 내용을 끌고 오는 경우가 있다(Ce·S 행에서 관찰).
            #   문장 안에 **다른 클래스 코드**가 보이면 그 앞에서 자른다.
            others = [x["class"] for x in (rec.get("클래스") or [])
                      if x["class"] != cls and x["class"]]
            for o in sorted(others, key=len, reverse=True):
                m_ = re.search(r"종류\\s*" + re.escape(o) + r"\\b", s)
                if m_ and m_.start() > 0:
                    s = s[:m_.start()].strip(" ,")
                elif m_:
                    s = ""          # 아예 남의 행 문장이다
                    break
            if not s:
                return NA, None
            # 표 칸이므로 첫 절만 남긴다. 전문은 섹션 원문 청크에 있다.
            s = re.split(r"(?:하며|으로서|이며|에 한하여|를 통하여)", s)[0].strip(" ,")''',
      marker="아예 남의 행 문장이다")

# ══════════════════════════════════════════════════════
# Z-3  총보수 근거 페이지
# ══════════════════════════════════════════════════════
patch("Z-3a 근거줄 추가",
'''    # ★ 원문 문장에서 온 가입자격의 근거 페이지. (코드규칙) 표기분은 제외한다.''',
'''    # ★ 총보수 근거 페이지 — 보수표가 두 쪽에 걸치는 문서가 많고
    #   하필 연금 클래스가 뒤쪽에 몰려 있어 문서 단위 페이지로는 인용이 어긋난다.
    fpg = sorted({p for p in (_fee_pages(rec).values()) if p})
    if fpg:
        lines.append("총보수 근거: " + ", ".join(f"p.{p}" for p in fpg))
    # ★ 원문 문장에서 온 가입자격의 근거 페이지. (코드규칙) 표기분은 제외한다.''',
      marker="총보수 근거: ")

patch("Z-3b 페이지 수집 함수",
'''def build_text(rec):''',
'''def _fee_pages(rec):
    """
    클래스 → 총보수 근거 페이지. 행별 키를 자동 탐색하고,
    없으면 문서 단위 보수_근거페이지로 대체한다.
    """
    doc_pg = rec.get("보수_근거페이지")
    out = {}
    for r in (rec.get("보수") or []):
        pg = None
        for k in ("근거페이지", "page", "페이지", "보수_근거페이지"):
            if isinstance(r.get(k), int):
                pg = r[k]
                break
        out[r.get("class")] = pg or doc_pg
    return out


def build_text(rec):''',
      marker="def _fee_pages(")

patch("Z-3c 청크 필드",
'''        "chunk_type": "class_card",''',
'''        "chunk_type": "class_card",
        "총보수_근거페이지": {c: p for c, p in _fee_pages(rec).items() if p} or None,''',
      marker='"총보수_근거페이지"')

# ══════════════════════════════════════════════════════
if any(s == "OK" for s, _, _ in _LOG):
    if not os.path.exists(PATH + ".bak10"):
        shutil.copy2(PATH, PATH + ".bak10")
    open(PATH, "w", encoding="utf-8").write(SRC)

w = max(len(t) for _, t, _ in _LOG)
for st, tag, msg in _LOG:
    print(f"{ {'OK': '✅', 'SKIP': '•', 'FAIL': '❌'}[st] } {tag:<{w}}  {msg}")
print(f"\n적용 {sum(1 for s,_,_ in _LOG if s=='OK')} / 건너뜀 "
      f"{sum(1 for s,_,_ in _LOG if s=='SKIP')} / 실패 {sum(1 for s,_,_ in _LOG if s=='FAIL')}")

import py_compile
py_compile.compile(PATH, doraise=True)
print("✅ 문법 정상")

# 인쇄 맵이 실제로 읽히는지 확인
import sys, importlib
sys.path.insert(0, os.path.dirname(DIR))
m = importlib.import_module("parse.pass9_chunk_class")
importlib.reload(m)
pm = m._printed_map()
print(f"\n인쇄 라벨 맵: {len(pm)}개 문서" if pm else
      "\n⚠️ 인쇄 라벨 맵을 못 찾았습니다 — PRINTED_FILE / PRINTED_KEYS 확인 필요")
if pm:
    k = list(pm)[0]
    print(f"  예: {k} → {dict(list((pm[k] or {}).items())[:4])}")
