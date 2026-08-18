# -*- coding: utf-8 -*-
"""
patch2 — patch_passes.py 적용 후 발견된 클래스 회귀를 되돌린다.

  C-1  normalize_class : 끝 구분자 제거 + 코드 뒤에 한글 설명이 붙는 칸 인정
  C-2  pass11 필드표    : 빈 리스트를 '유효'로 세지 않게
  C-3  pass3 클래스 출처 : 좌표 폴백과 보수표 폴백 중 더 많이 찾은 쪽을 쓴다
                          (APPLY_C3 = False 로 끄면 C-1/C-2만 적용)

patch_passes.py 와 같은 규칙: 멱등, .bak2 백업, 앵커 불일치 시 FAIL 표시.
"""
import os, glob, shutil

PARSE_DIR = ""
APPLY_C3 = True
SEARCH_ROOTS = ["/content", ".", "/content/drive/MyDrive"]


def find_parse_dir():
    if PARSE_DIR:
        return PARSE_DIR
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        hits = [h for h in glob.glob(os.path.join(root, "**", "pass3_extract.py"),
                                     recursive=True) if "__pycache__" not in h]
        if hits:
            return os.path.dirname(sorted(hits, key=len)[0])
    raise SystemExit("[!] pass3_extract.py 를 못 찾았습니다. PARSE_DIR 을 직접 지정하세요.")


_LOG = []


def patch(fname, tag, old, new, done_marker=None):
    path = os.path.join(DIR, fname)
    if not os.path.exists(path):
        _LOG.append(("FAIL", tag, f"{fname} 없음"))
        return
    src = open(path, encoding="utf-8").read()
    if done_marker and done_marker in src:
        _LOG.append(("SKIP", tag, "이미 적용됨"))
        return
    n = src.count(old)
    if n == 0:
        _LOG.append(("SKIP", tag, "앵커 없음 — 이미 적용된 것으로 보임"))
        return
    if n > 1:
        _LOG.append(("FAIL", tag, f"앵커 {n}회 매칭 (1회여야 함)"))
        return
    bak = path + ".bak2"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
    _LOG.append(("OK", tag, fname))


DIR = find_parse_dir()
print(f"PARSE_DIR = {DIR}   APPLY_C3 = {APPLY_C3}\n")

# ══════════════════════════════════════════════════════
# C-1  normalize_class
# ══════════════════════════════════════════════════════
patch(
    "pass3_extract.py", "C-1 normalize_class 복구",
    '''    s = (s or "").strip()
    if not s:
        return None
    m = re.search(r"\\(\\s*(?:종류)?\\s*([A-Za-z][A-Za-z0-9\\-]{0,6})\\s*\\)\\s*$", s)
    if m:
        return m.group(1)
    m = re.fullmatch(r"(?:종류|Class)?\\s*([A-Za-z][A-Za-z0-9\\-]{0,6})", s)
    if m:
        return m.group(1)
    return None''',
    '''    s = (s or "").strip()
    if not s:
        return None

    NOT_CODE = {"class", "total", "type"}

    def _ok(code):
        # ★ 끝의 구분자를 떼어낸다. 'C-F-' 'I-' 가 그대로 나가면 pass11 CLASS_OK
        #   (마지막 글자는 영숫자)에 걸려 그 문서 클래스가 통째로 None이 된다.
        code = code.strip().rstrip("-·ㆍ_")
        if code.lower() in NOT_CODE:
            return None
        return code if re.fullmatch(
            r"[A-Za-z][A-Za-z0-9\\-]{0,5}[A-Za-z0-9]|[A-Za-z]", code) else None

    m = re.search(r"\\(\\s*(?:종류)?\\s*([A-Za-z][A-Za-z0-9\\-]{0,6})\\s*\\)\\s*$", s)
    if m:
        return _ok(m.group(1))
    m = re.fullmatch(r"(?:종류|Class)?\\s*([A-Za-z][A-Za-z0-9\\-]{0,6})", s)
    if m:
        return _ok(m.group(1))
    # ★ 코드 바로 뒤에 한글 설명이 붙는 칸 — norm()이 공백을 지운 결과다.
    #     "종류A수수료미징구-오프라인-개인연금" → A
    #   전체일치만 보면 이런 칸이 전부 버려진다(클래스 64→54 회귀의 원인).
    #   설명이 **먼저** 오는 칸("수수료미징구-…")은 여전히 버린다 — 시작이 한글이라
    #   아래 패턴에 걸리지 않는다.
    m = re.match(r"^(?:종류|Class)?\\s*([A-Za-z][A-Za-z0-9\\-]{0,6})(?=[가-힣])", s)
    if m:
        return _ok(m.group(1))
    return None''',
    done_marker="NOT_CODE = ")

# ══════════════════════════════════════════════════════
# C-2  pass11 필드표 — 빈 리스트를 유효로 세지 않는다
# ══════════════════════════════════════════════════════
patch(
    "pass11_validate.py", "C-2 필드표 빈리스트 처리",
    '''        ok = sum(1 for r in six if r.get(k) is not None)''',
    '''        # ★ 클래스·보수는 []로 남는다. is not None 이면 빈 표가 '유효'로 잡혀
        #   보수 92/92, 클래스 91/92 처럼 실제(74·54)와 딴판인 숫자가 찍힌다.
        ok = sum(1 for r in six if r.get(k) not in (None, [], {}, ""))''',
    done_marker="not in (None, [], {}, \"\")")

# ══════════════════════════════════════════════════════
# C-3  pass3 — 좌표 폴백과 보수표 폴백 중 많이 찾은 쪽
# ══════════════════════════════════════════════════════
if APPLY_C3:
    patch(
        "pass3_extract.py", "C-3 클래스 출처 우선순위",
        '''    # ★ 3차 폴백 — 13항 텍스트가 세로 분해된 문서는 좌표 재조립으로 클래스를 얻는다
    if not uniq and pdf_path:
        rng13 = span(sec, "13항보수", npages, 8)
        if rng13:
            uniq = classes_from_coord(pdf_path, rng13)''',
        '''    # ★ 3차 폴백 — 13항 텍스트가 세로 분해된 문서는 좌표 재조립으로 클래스를 얻는다
    #   ⚠️ 'not uniq' 조건만 두면 보수표 폴백이 좌표 폴백을 선점한다.
    #      같은 문서에서 좌표 16개 vs 보수표 7개가 나오는 사례가 있고(25건 확인),
    #      좌표 쪽은 "수수료미징구-온라인-개인연금(C-Pe)" 설명까지 들고 온다
    #      (연금유형·판매유형 판정이 이 설명에서 나온다).
    #      → 보수표에서 온 목록이면 좌표도 돌려보고 많이 찾은 쪽을 쓴다.
    from_fee = bool(uniq) and all(c.get("출처") == "13항" for c in uniq)
    if pdf_path and (not uniq or from_fee):
        rng13 = span(sec, "13항보수", npages, 8)
        if rng13:
            coord = classes_from_coord(pdf_path, rng13)
            if len(coord) > len(uniq):
                uniq = coord''',
        done_marker="from_fee = bool(uniq)")
else:
    _LOG.append(("SKIP", "C-3 클래스 출처 우선순위", "APPLY_C3=False"))


w = max(len(t) for _, t, _ in _LOG)
for st, tag, msg in _LOG:
    print(f"{ {'OK': '✅', 'SKIP': '•', 'FAIL': '❌'}[st] } {tag:<{w}}  {msg}")
print(f"\n적용 {sum(1 for s,_,_ in _LOG if s=='OK')} / 건너뜀 "
      f"{sum(1 for s,_,_ in _LOG if s=='SKIP')} / 실패 "
      f"{sum(1 for s,_,_ in _LOG if s=='FAIL')}")

import py_compile
bad = []
for f in ["pass3_extract.py", "pass11_validate.py"]:
    try:
        py_compile.compile(os.path.join(DIR, f), doraise=True)
    except Exception as e:
        bad.append((f, str(e)[:120]))
if bad:
    print("\n❌ 문법 오류 — .bak2 로 되돌리세요")
    for f, e in bad:
        print(f"   {f}: {e}")
else:
    print("✅ 문법 정상")

# ── normalize_class 단위 확인 (PDF 없이 즉시 검증) ──────
import sys, importlib
sys.path.insert(0, os.path.dirname(DIR))
try:
    m = importlib.import_module("parse.pass3_extract")
    importlib.reload(m)
    CASES = [
        ("종류A", "A"),
        ("종류A수수료미징구-오프라인-개인연금", "A"),
        ("C-P수수료미징구-온라인-개인연금", "C-P"),
        ("수수료미징구-오프라인-개인연금(C-Pe)", "C-Pe"),
        ("수수료미징구-오", None),
        ("수수료미징구-", None),
        ("C-F-", "C-F"),
        ("I-", "I"),
        ("A2", "A2"),
        ("#1", None),
    ]
    print("\n── normalize_class 확인 ──")
    for src, want in CASES:
        got = m.normalize_class(src)
        print(f"  {'✅' if got == want else '❌'} {src!r:<40} → {got!r}  (기대 {want!r})")
except Exception as e:
    print(f"\n(단위 확인 생략: {e})")
