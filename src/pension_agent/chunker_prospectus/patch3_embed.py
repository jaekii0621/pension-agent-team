# -*- coding: utf-8 -*-
"""
patch3 — 임베딩 절단 대비. 인용용 원문(text)은 그대로 두고 embed_text를 따로 만든다.

  E-1  common.py : EMBED_MAX / make_embed_text() 추가
  E-2  pass10    : 섹션 청크에 embed_text (머리 = 펀드명·섹션)
  E-3  pass9     : 클래스 카드에 embed_text (표 아래 연금 요약줄을 꼬리로 보존)
  E-4  pass8     : 펀드 카드에 embed_text (300자대 — 원문 그대로)

멱등, .bak3 백업, 앵커 불일치 시 FAIL.
청커만 다시 돌리면 되므로 재파싱 불필요: python run_all.py --chunk
"""
import os, glob, shutil

PARSE_DIR = ""
EMBED_MAX = 786          # 한글 70% 기준 512토큰 ≈ 600~750자. 보수적으로 600.
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
    bak = path + ".bak3"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
    _LOG.append(("OK", tag, fname))


DIR = find_parse_dir()
print(f"PARSE_DIR = {DIR}   EMBED_MAX = {EMBED_MAX}\n")

# ══════════════════════════════════════════════════════
# E-1  common.py — 헬퍼 추가
# ══════════════════════════════════════════════════════
HELPER = '''

# ── 임베딩용 텍스트 ★ 512토큰 절단 대비 ────────────────
#   text(인용용 원문)는 자르지 않는다. 근거 표시가 깨지기 때문이다.
#   대신 임베딩 입력만 따로 만든다. 로컬 임베딩 모델 max_seq_len이 512면
#   한글 70% 기준 대략 600~750자에서 뒤가 통째로 잘려 인덱싱된다.
#   ⚠️ 앞에서 자르면 표 아래 요약줄(연금 가능 클래스 등)이 사라진다.
#      머리(펀드명·섹션)와 꼬리(요약줄)는 남기고 본문 가운데를 줄인다.
EMBED_MAX = __MAX__


def make_embed_text(head, body, tail="", budget=EMBED_MAX):
    head = re.sub(r"\\s+", " ", head or "").strip(" ·")
    tail = re.sub(r"[ \\t]+", " ", tail or "").strip()
    body = re.sub(r"\\s+", " ", body or "").strip()
    room = budget - len(head) - len(tail) - 2
    if room < 60:                       # 머리+꼬리만으로 예산을 다 쓴 경우
        return ("\\n".join(x for x in (head, tail) if x))[:budget]
    if len(body) > room:
        cut = body[:room]
        p = max(cut.rfind(". "), cut.rfind("다 "), cut.rfind(" "))
        if p > room * 0.6:              # 단어·문장 경계에서 자른다
            cut = cut[:p]
        body = cut.rstrip() + "…"
    return "\\n".join(x for x in (head, body, tail) if x)
'''.replace("__MAX__", str(EMBED_MAX))

_p = os.path.join(DIR, "common.py")
_src = open(_p, encoding="utf-8").read()
if "def make_embed_text" in _src:
    _LOG.append(("SKIP", "E-1 common 헬퍼", "이미 적용됨"))
else:
    if not os.path.exists(_p + ".bak3"):
        shutil.copy2(_p, _p + ".bak3")
    open(_p, "w", encoding="utf-8").write(_src.rstrip() + "\n" + HELPER)
    _LOG.append(("OK", "E-1 common 헬퍼", "common.py"))

# ══════════════════════════════════════════════════════
# E-2  pass10 섹션 청크
# ══════════════════════════════════════════════════════
patch(
    "pass10_chunk_section.py", "E-2 섹션 청크 embed_text",
    "from parse.common import DUMP_DIR, TABLE_DIR, load_json",
    "from parse.common import DUMP_DIR, TABLE_DIR, load_json, make_embed_text",
    done_marker="load_json, make_embed_text")

patch(
    "pass10_chunk_section.py", "E-2b 필드 추가",
    '''                    "parse_method": "text",
                    "confidence": 1.0,''',
    '''                    "parse_method": "text",
                    "confidence": 1.0,
                    # ★ 인덱싱 입력. text는 인용용이라 자르지 않는다.
                    "embed_text": make_embed_text(
                        f"{rec.get('fund_name') or ''} · {label}", piece),''',
    done_marker='"embed_text": make_embed_text(')

# ══════════════════════════════════════════════════════
# E-3  pass9 클래스 카드
# ══════════════════════════════════════════════════════
patch(
    "pass9_chunk_class.py", "E-3 import",
    "from parse.common import TABLE_DIR, load_json",
    "from parse.common import TABLE_DIR, load_json, make_embed_text",
    done_marker="load_json, make_embed_text")

patch(
    "pass9_chunk_class.py", "E-3b 필드 추가",
    '''def build(rec):
    page = card_page(rec)
    return {''',
    '''# ★ 표 아래 요약줄이 검색 키워드를 담고 있다("연금저축계좌 가입 가능 클래스: C-P, C-Pe").
#   앞에서만 자르면 이게 통째로 사라져 연금 질의가 이 카드를 못 찾는다.
_TAIL_KEYS = ("연금저축계좌 가입", "퇴직연금(IRP) 가입", "연금계좌 전용", "총보수 범위")


def build(rec):
    page = card_page(rec)
    txt = build_text(rec)
    tail = "\\n".join(l for l in txt.split("\\n") if l.startswith(_TAIL_KEYS))
    return {''',
    done_marker="_TAIL_KEYS = (")

patch(
    "pass9_chunk_class.py", "E-3c text 재사용",
    '''        "text": build_text(rec),
        "parse_method": parse_method_of(rec),''',
    '''        "text": txt,
        "embed_text": make_embed_text(
            f"{rec.get('fund_name') or rec['doc_id']} · 클래스별 보수·가입자격",
            txt, tail=tail),
        "parse_method": parse_method_of(rec),''')

# ══════════════════════════════════════════════════════
# E-4  pass8 펀드 카드 (스키마 통일용 — 300자대라 절단 없음)
# ══════════════════════════════════════════════════════
patch(
    "pass8_chunk_fund.py", "E-4 필드 추가",
    '''        "text": build_text(rec),
        "parse_method": "manual" if rec.get("수작업보정") else "text",''',
    '''        "text": build_text(rec),
        "embed_text": build_text(rec),   # 300자대 — 절단 위험 없음. 스키마 통일용
        "parse_method": "manual" if rec.get("수작업보정") else "text",''')


w = max(len(t) for _, t, _ in _LOG)
for st, tag, msg in _LOG:
    print(f"{ {'OK': '✅', 'SKIP': '•', 'FAIL': '❌'}[st] } {tag:<{w}}  {msg}")
print(f"\n적용 {sum(1 for s,_,_ in _LOG if s=='OK')} / 건너뜀 "
      f"{sum(1 for s,_,_ in _LOG if s=='SKIP')} / 실패 "
      f"{sum(1 for s,_,_ in _LOG if s=='FAIL')}")

import py_compile
bad = []
for f in ["common.py", "pass8_chunk_fund.py", "pass9_chunk_class.py",
          "pass10_chunk_section.py"]:
    try:
        py_compile.compile(os.path.join(DIR, f), doraise=True)
    except Exception as e:
        bad.append((f, str(e)[:120]))
print("❌ 문법 오류: " + str(bad) if bad else "✅ 문법 정상")
