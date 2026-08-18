# -*- coding: utf-8 -*-
"""
patch6 — 가입자격을 원문에서 채우고, 출처와 근거 페이지를 남긴다.

  W-1  pass5 : 가입자격 문장이 나온 페이지를 rec["가입자격_페이지"] 로 기록
  W-2  pass9 : _elig() 교체 — 판매유형 접두어 제거, '제한없음'류 인식,
               (문구, 출처) 반환. 출처는 문서 / 코드규칙 / 미확인.
  W-3  pass9 : 카드에 "가입자격 근거: p.21, p.22" 줄 + 청크에 근거 필드

배경
  5패스가 이미 가입자격 335행(55문서)을 원문에서 뽑아뒀고, 그중 44행은
  '제한없음'류다. PDF를 다시 열 필요가 없다. 9패스가 문장 앞의 판매유형
  설명("수수료미징구-오프라인-개인연금 …")을 못 떼서 미확인으로 떨어뜨렸을 뿐이다.

멱등, .bak6 백업.
실행: python patch6_eligibility.py  →  python run_all.py --from 5
      (5패스를 고치므로 파싱부터 다시 탄다. 6·7패스 포함 약 2분)
"""
import os, glob, shutil

PARSE_DIR = ""
SEARCH_ROOTS = ["/content", ".", "/content/drive/MyDrive"]

ELIG_OLD = 'def _elig(rec, cls):\n    """\n    가입자격 — 코드 규칙이 확실한 경우가 1순위, 그다음 문장, 마지막이 판정 플래그.\n\n    ★ C1~C4는 \'선행 클래스를 1년 이상 보유한 투자자\'만 전환으로 취득하는 클래스라\n      신규 가입이 불가능하다(5패스 독스트링의 최상세 정본 규칙). 문서 문장이\n      "수익증권으로 전환"처럼 잘려 나와도 코드 규칙이 더 정확하므로 우선한다.\n    """\n    if re.fullmatch(r"C[1-4]", cls):\n        return "신규 가입 불가(선행 클래스 보유 필요)"\n    if cls in ("I", "C-F"):\n        return "고액·기관 전용"\n\n    e = (rec.get("가입자격") or {}).get(cls)\n    if e:\n        s = re.sub(r"\\s+", " ", e).strip()\n        if not _ARTICLE.search(s[:30]):          # 규약 조문이면 버린다\n            # 표 칸이므로 첫 절만 남긴다. 전문은 섹션 원문 청크에 있다.\n            s = re.split(r"(?:하며|으로서|이며|에 한하여|를 통하여)", s)[0].strip(" ,")\n            if len(s) >= 4:\n                return s[:40] + ("…" if len(s) > 40 else "")\n    flags = (rec.get("가입자격_판정") or {}).get(cls) or {}\n    tags = []\n    if flags.get("신규가입불가"):\n        tags.append("신규 가입 불가")\n    if flags.get("고액전용"):\n        tags.append("고액·기관 전용")\n    if flags.get("연금저축"):\n        tags.append("연금저축계좌 전용")\n    if flags.get("퇴직연금"):\n        tags.append("퇴직연금·IRP 전용")\n    if tags:\n        return ", ".join(tags)\n    return NA\n\n\n'
ELIG_NEW = '# ★ \'제한 없음\'류 — 원문에 실제로 적혀 있는 표기 변형들.\n#   "제한 없음" / "제한없음" / "가입제한이 없으며, 선취판매수수료가 징구되는 수익증권"\n#   5패스가 이미 335행을 뽑아뒀는데, 아래 접두어를 못 떼서 카드에서 미확인으로 떨어졌다.\n_NONE = re.compile(r"제한\\s*없음|가입\\s*제한이?\\s*없|제한없|해당\\s*없음|누구나")\n# 가입자격 문장 앞에 판매유형 설명이 붙어 오는 문서가 있다\n#   "수수료미징구-오프라인-개인연금 가입제한이 없으며, …"\n_DESC_HEAD = re.compile(\n    r"^(?:종류\\s*)?(?:수수료(?:미징구|선취|후취)[^\\s]*|온라인[^\\s]*|오프라인[^\\s]*)\\s*")\n\n\ndef _elig(rec, cls, crec=None):\n    """\n    가입자격과 그 출처를 함께 돌려준다 → (문구, 출처)\n\n    출처: "문서"(원문 문장) / "코드규칙"(코드·설명에서 추론) / None(미확인)\n    ★ 근거 표시가 평가 항목이므로 어디서 온 값인지 구분해서 남긴다.\n      코드규칙으로 채운 값은 카드에 (코드규칙)으로 표시해 원문 인용과 섞이지 않게 한다.\n\n    ★ C1~C4는 선행 클래스를 1년 이상 보유해야 취득하는 클래스라 신규 가입이 불가능하다\n      (5패스 독스트링의 최상세 정본 규칙). 문서 문장이 잘려 나와도 코드 규칙이 우선한다.\n    """\n    if re.fullmatch(r"C[1-4]", cls):\n        return "신규 가입 불가(선행 클래스 보유 필요)", "코드규칙"\n    if cls in ("I", "C-F"):\n        return "고액·기관 전용", "코드규칙"\n\n    # ── ① 원문 문장 ──\n    e = (rec.get("가입자격") or {}).get(cls)\n    if e:\n        s = re.sub(r"\\s+", " ", e).strip()\n        if not _ARTICLE.search(s[:30]):          # 규약 조문이면 버린다\n            s = _DESC_HEAD.sub("", s).strip(" ,-")\n            if _NONE.search(s):\n                return "제한 없음", "문서"\n            # 표 칸이므로 첫 절만 남긴다. 전문은 섹션 원문 청크에 있다.\n            s = re.split(r"(?:하며|으로서|이며|에 한하여|를 통하여)", s)[0].strip(" ,")\n            if len(s) >= 4:\n                return s[:40] + ("…" if len(s) > 40 else ""), "문서"\n\n    # ── ② 문장에서 뽑은 판정 플래그 ──\n    flags = (rec.get("가입자격_판정") or {}).get(cls) or {}\n    tags = []\n    if flags.get("신규가입불가"):\n        tags.append("신규 가입 불가")\n    if flags.get("고액전용"):\n        tags.append("고액·기관 전용")\n    if flags.get("연금저축"):\n        tags.append("연금저축계좌 전용")\n    if flags.get("퇴직연금"):\n        tags.append("퇴직연금·IRP 전용")\n    if tags:\n        return ", ".join(tags), "문서"\n\n    # ── ③ 코드·설명 규칙 ──\n    #   ⚠️ \'제한 없음\'을 여기서 추론하지 않는다. 원문에 없는 단정은 넣지 않는다.\n    pt = (crec or {}).get("연금유형")\n    if pt == "개인연금":\n        return "연금저축계좌 전용", "코드규칙"\n    if pt == "퇴직연금":\n        return "퇴직연금·IRP 전용", "코드규칙"\n    d = (crec or {}).get("설명") or ""\n    d = d.replace(" ", "")\n    if re.search(r"기관|고액", d):\n        return "고액·기관 전용", "코드규칙"\n    if "온라인슈퍼" in d:\n        return "온라인 판매채널 전용", "코드규칙"\n    return NA, None\n\n\n'
ROWS_OLD = '    rows = []\n    for c in classes:\n        cls = c["class"]\n        fee = fees.get(cls)\n        rows.append((\n            cls,\n            f"{fee}%" if fee is not None else NA,\n            _route(c.get("설명")),\n            _elig(rec, cls),\n        ))\n\n'
ROWS_NEW = '    rows, srcs = [], {}\n    for c in classes:\n        cls = c["class"]\n        fee = fees.get(cls)\n        elig, src = _elig(rec, cls, c)\n        if src:\n            srcs[cls] = src\n        rows.append((\n            cls,\n            f"{fee}%" if fee is not None else NA,\n            _route(c.get("설명")),\n            elig + ("(코드규칙)" if src == "코드규칙" else ""),\n        ))\n\n'


def find_parse_dir():
    if PARSE_DIR:
        return PARSE_DIR
    for root in SEARCH_ROOTS:
        hits = [h for h in glob.glob(os.path.join(root, "**", "pass9_chunk_class.py"),
                                     recursive=True) if "__pycache__" not in h]
        if hits:
            return os.path.dirname(sorted(hits, key=len)[0])
    raise SystemExit("[!] pass9_chunk_class.py 를 못 찾았습니다. PARSE_DIR 을 지정하세요.")


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
        _LOG.append(("FAIL", tag, f"앵커 {n}회 매칭"))
        return
    if not os.path.exists(path + ".bak6"):
        shutil.copy2(path, path + ".bak6")
    open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
    _LOG.append(("OK", tag, fname))


DIR = find_parse_dir()
print(f"PARSE_DIR = {DIR}\n")

# ══════════════════════════════════════════════════════
# W-1  pass5 — 가입자격 문장의 페이지 기록
# ══════════════════════════════════════════════════════
patch(
    "pass5_eligibility.py", "W-1a 페이지 탐색 함수",
    '''def main():
    six = load_json(os.path.join(TABLE_DIR, "six_axis.json"))''',
    '''def locate_pages(pages, elig):
    """
    ★ 가입자격 문장이 어느 페이지에서 왔는지 되찾는다.

    text_of()가 여러 페이지를 이어붙이면서 페이지 정보가 사라진다.
    문장이 충분히 길고 특징적이므로, 공백 제거 후 앞 24자로 역탐색하면
    안전하게 찾을 수 있다(짧은 값은 오탐 위험이 있어 건너뛴다).
    """
    by = {p["page"]: re.sub(r"\\s+", "", p["text"]) for p in pages}
    out = {}
    for c, v in elig.items():
        key = re.sub(r"\\s+", "", v)[:24]
        if len(key) < 8:
            continue
        for pg in sorted(by):
            if key in by[pg]:
                out[c] = pg
                break
    return out


def main():
    six = load_json(os.path.join(TABLE_DIR, "six_axis.json"))''',
    done_marker="def locate_pages(")

patch(
    "pass5_eligibility.py", "W-1b 기록",
    '''        if elig:
            rec["가입자격"] = elig''',
    '''        if elig:
            rec["가입자격"] = elig
            rec["가입자격_페이지"] = locate_pages(pages, elig)''',
    done_marker='rec["가입자격_페이지"]')

patch(
    "pass5_eligibility.py", "W-1c 리포트",
    '''    print(f"가입자격 {ok_e}/{n}   기준가격적용 {ok_p}/{n}")''',
    '''    pg_rows = sum(len(r.get("가입자격_페이지") or {}) for r in six)
    el_rows = sum(len(r.get("가입자격") or {}) for r in six)
    print(f"가입자격 {ok_e}/{n}   기준가격적용 {ok_p}/{n}   "
          f"근거페이지 {pg_rows}/{el_rows}행")''',
    done_marker="근거페이지 {pg_rows}")

# ══════════════════════════════════════════════════════
# W-2  pass9 — _elig() 교체
# ══════════════════════════════════════════════════════
patch("pass9_chunk_class.py", "W-2 _elig 교체", ELIG_OLD, ELIG_NEW,
      done_marker="_DESC_HEAD = re.compile(")

patch("pass9_chunk_class.py", "W-2b 표 생성", ROWS_OLD, ROWS_NEW,
      done_marker="rows, srcs = [], {}")

# ══════════════════════════════════════════════════════
# W-3  pass9 — 근거 페이지 노출
# ══════════════════════════════════════════════════════
patch(
    "pass9_chunk_class.py", "W-3a 카드에 근거 줄",
    '''    if fees:
        lo = min(fees.values())
        hi = max(fees.values())
        lines.append(f"총보수 범위: {lo}% ~ {hi}%")
    return "\\n".join(lines)''',
    '''    if fees:
        lo = min(fees.values())
        hi = max(fees.values())
        lines.append(f"총보수 범위: {lo}% ~ {hi}%")
    # ★ 원문 문장에서 온 가입자격의 근거 페이지. (코드규칙) 표기분은 제외한다.
    pmap = rec.get("가입자격_페이지") or {}
    pgs = sorted({pmap[c] for c, s in srcs.items() if s == "문서" and c in pmap})
    if pgs:
        lines.append("가입자격 근거: " + ", ".join(f"p.{p}" for p in pgs))
    return "\\n".join(lines)''',
    done_marker="가입자격 근거: ")

patch(
    "pass9_chunk_class.py", "W-3b 청크 필드",
    '''        "chunk_type": "class_card",''',
    '''        "chunk_type": "class_card",
        "가입자격_근거페이지": {c: p for c, p in (rec.get("가입자격_페이지") or {}).items()
                        if c in {x["class"] for x in (rec.get("클래스") or [])}} or None,''',
    done_marker='"가입자격_근거페이지"')


# ══════════════════════════════════════════════════════
# W-4  pass5 — 문장이 클래스 설명을 덮어쓰지 않게
# ══════════════════════════════════════════════════════
patch("pass5_eligibility.py", "W-4 연금유형 덮어쓰기 방지",
      '                    if f["퇴직연금"]:\n                        c["연금유형"] = "퇴직연금"\n                    elif f["연금저축"]:\n                        c["연금유형"] = "개인연금"',
      '                    # ★ 3패스가 설명("…-개인연금")에서 이미 판정한 값이 더 정확하다.\n                    #   가입자격 문장은 한 문단에 개인연금·퇴직연금이 같이 나오는 경우가\n                    #   있어, 무조건 덮어쓰면 C-Pe(개인연금)가 퇴직연금으로 뒤집힌다.\n                    if not c.get("연금유형"):\n                        if f["퇴직연금"]:\n                            c["연금유형"] = "퇴직연금"\n                        elif f["연금저축"]:\n                            c["연금유형"] = "개인연금"',
      done_marker='if not c.get("연금유형"):')

# ══════════════════════════════════════════════════════
w = max(len(t) for _, t, _ in _LOG)
for st, tag, msg in _LOG:
    print(f"{ {'OK': '✅', 'SKIP': '•', 'FAIL': '❌'}[st] } {tag:<{w}}  {msg}")
print(f"\n적용 {sum(1 for s,_,_ in _LOG if s=='OK')} / 건너뜀 "
      f"{sum(1 for s,_,_ in _LOG if s=='SKIP')} / 실패 "
      f"{sum(1 for s,_,_ in _LOG if s=='FAIL')}")

import py_compile
bad = []
for f in ["pass5_eligibility.py", "pass9_chunk_class.py"]:
    try:
        py_compile.compile(os.path.join(DIR, f), doraise=True)
    except Exception as e:
        bad.append((f, str(e)[:150]))
print("❌ 문법 오류: " + str(bad) if bad else "✅ 문법 정상")
print("\n다음: python run_all.py --from 5  &&  python fit_embed_text.py")
