# -*- coding: utf-8 -*-
"""
patch5 — 수작업 근거를 파이프라인에 연결한다.

  V-1  pass12 : override_evidence.json 을 읽어 rec["수작업근거"] 로 붙인다
  V-2  pass12 : 근거 커버리지를 리포트에 출력
  V-3  pass8  : 펀드카드 연금 줄에 "(근거 p.35)" 표기 + 청크에 근거페이지 필드

멱등, .bak5 백업.
실행: python patch5_evidence.py  →  python run_all.py --from 9  (12패스부터)
"""
import os, glob, shutil

PARSE_DIR = ""
SEARCH_ROOTS = ["/content", ".", "/content/drive/MyDrive"]


def find_parse_dir():
    if PARSE_DIR:
        return PARSE_DIR
    for root in SEARCH_ROOTS:
        hits = [h for h in glob.glob(os.path.join(root, "**", "pass12_override.py"),
                                     recursive=True) if "__pycache__" not in h]
        if hits:
            return os.path.dirname(sorted(hits, key=len)[0])
    raise SystemExit("[!] pass12_override.py 를 못 찾았습니다. PARSE_DIR 을 지정하세요.")


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
    if not os.path.exists(path + ".bak5"):
        shutil.copy2(path, path + ".bak5")
    open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
    _LOG.append(("OK", tag, fname))


DIR = find_parse_dir()
print(f"PARSE_DIR = {DIR}\n")

# ══════════════════════════════════════════════════════
# V-1  pass12 — 근거 파일 로드 + rec에 부착
# ══════════════════════════════════════════════════════
patch(
    "pass12_override.py", "V-1a 근거 파일 경로",
    'OV_PATH = os.path.join(TABLE_DIR, "manual_override.json")',
    '''OV_PATH = os.path.join(TABLE_DIR, "manual_override.json")
# ★ 수작업 값의 근거(페이지·원문 한 줄). refine_evidence.py 가 만든다.
#   확인필요=True 인 항목은 붙이지 않는다 — 틀린 근거를 다는 것보다 없는 게 낫다.
EV_PATH = os.path.join(TABLE_DIR, "override_evidence.json")''',
    done_marker="EV_PATH = ")

patch(
    "pass12_override.py", "V-1b 로드",
    '''    ov = load_json(OV_PATH)

    by_id = {r["doc_id"]: r for r in six}''',
    '''    ov = load_json(OV_PATH)
    ev = load_json(EV_PATH) if os.path.exists(EV_PATH) else {}
    if not ev:
        print("  (근거 파일 없음 — refine_evidence.py 를 먼저 돌리면 근거가 붙습니다)")

    by_id = {r["doc_id"]: r for r in six}''',
    done_marker="ev = load_json(EV_PATH)")

patch(
    "pass12_override.py", "V-1c rec에 부착",
    '''        rec["수작업보정"] = sorted(set((rec.get("수작업보정") or []) + list(patch)))''',
    '''        rec["수작업보정"] = sorted(set((rec.get("수작업보정") or []) + list(patch)))
        # ★ 근거 부착 — 페이지가 있고 확인필요가 아닌 것만.
        #   답변에서 "연금저축 가입 가능"을 인용할 때 쓸 span이 여기서 나온다.
        src_ev = {k: v for k, v in (ev.get(did) or {}).items()
                  if isinstance(v, dict) and v.get("page") and not v.get("확인필요")}
        if src_ev:
            rec["수작업근거"] = {k: {"page": v["page"], "quote": v.get("quote")}
                             for k, v in src_ev.items()}''',
    done_marker='rec["수작업근거"] = {')

patch(
    "pass12_override.py", "V-2 커버리지 리포트",
    '''    print("\\n※ 수작업 값은 파싱 값을 덮어쓴다. 어느 필드인지는 '수작업보정'에 기록된다.")''',
    '''    ev_docs = sum(1 for r in six if r.get("수작업근거"))
    ev_flds = sum(len(r.get("수작업근거") or {}) for r in six)
    tgt = sum(len([k for k in p if not k.startswith("_")]) for p in ov.values())
    print(f"\\n근거 부착 {ev_flds}/{tgt}건 (문서 {ev_docs}개)")
    miss = [r["doc_id"] for r in six
            if r.get("수작업보정") and not r.get("수작업근거")]
    if miss:
        print(f"  근거 없는 문서 {len(miss)}건: {miss[:5]}")
    print("\\n※ 수작업 값은 파싱 값을 덮어쓴다. 어느 필드인지는 '수작업보정'에 기록된다.")''',
    done_marker="근거 부착 {ev_flds}")

# ══════════════════════════════════════════════════════
# V-3  pass8 — 카드에 근거 페이지 노출
# ══════════════════════════════════════════════════════
patch(
    "pass8_chunk_fund.py", "V-3a 연금 줄에 근거 표기",
    '''def _pension_line(rec):''',
    '''def _ev_pages(rec, fields):
    """수작업 근거 페이지 목록. 없으면 빈 문자열."""
    ev = rec.get("수작업근거") or {}
    pgs = sorted({ev[f]["page"] for f in fields
                  if isinstance(ev.get(f), dict) and ev[f].get("page")})
    return " (근거 " + ", ".join(f"p.{p}" for p in pgs) + ")" if pgs else ""


def _pension_line(rec):''',
    done_marker="def _ev_pages(")

patch(
    "pass8_chunk_fund.py", "V-3b 반환문",
    '''    return "연금계좌: " + ", ".join(tags)''',
    '''    # ★ 수작업으로 확인한 값이면 근거 페이지를 문장에 직접 박는다.
    #   별도 필드로 두면 에이전트가 인용할 때 놓친다.
    return ("연금계좌: " + ", ".join(tags)
            + _ev_pages(rec, ("연금저축_가능", "퇴직연금_가능")))''',
    done_marker='+ _ev_pages(rec, ("연금저축_가능"')

patch(
    "pass8_chunk_fund.py", "V-3c 청크 필드",
    '''        "chunk_type": "fund_card",''',
    '''        "chunk_type": "fund_card",
        "근거페이지": {k: v["page"] for k, v in (rec.get("수작업근거") or {}).items()
                   if v.get("page")} or None,''',
    done_marker='"근거페이지": {')


w = max(len(t) for _, t, _ in _LOG)
for st, tag, msg in _LOG:
    print(f"{ {'OK': '✅', 'SKIP': '•', 'FAIL': '❌'}[st] } {tag:<{w}}  {msg}")
print(f"\n적용 {sum(1 for s,_,_ in _LOG if s=='OK')} / 건너뜀 "
      f"{sum(1 for s,_,_ in _LOG if s=='SKIP')} / 실패 "
      f"{sum(1 for s,_,_ in _LOG if s=='FAIL')}")

import py_compile
bad = []
for f in ["pass8_chunk_fund.py", "pass12_override.py"]:
    try:
        py_compile.compile(os.path.join(DIR, f), doraise=True)
    except Exception as e:
        bad.append((f, str(e)[:120]))
print("❌ 문법 오류: " + str(bad) if bad else "✅ 문법 정상")
print("\n다음: python run_all.py --from 9")
