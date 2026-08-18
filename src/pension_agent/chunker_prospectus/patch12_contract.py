# -*- coding: utf-8 -*-
"""
patch12 — INTERFACES.md 계약 준수 (짧은 것 3종)

  K-1  confidence 를 계약대로 되돌린다
       계약 §2-1: "confidence: OCR만. text는 1.0"
       지금 쓰던 충족도 계산값은 fill_ratio 필드로 옮겨 정보를 잃지 않는다.
       ⚠️ 안내문서 코퍼스(재희)는 계약을 지키고 있어 상품 쪽만 어긋난 상태였다.
          같은 필드명이 두 코퍼스에서 다른 뜻이면 랭킹·필터가 어긋난다.

  K-2  table_ref 추가 — 계약 §2-1 "표 셀 유래면 기입, 아니면 null"
       클래스 카드는 13항 보수표에서 왔으므로 표 참조를 남긴다.

  K-3  수익률기준일 노출 — 계약 §2-3 "문서마다 다름. 비교 시 필수"
       §4-1d expects_caveat 케이스(기준일 10개월 편차)와 직결.
       six_axis에 별도 필드가 없으면 작성기준일을 쓰되 그렇다고 명시한다.

멱등, .bak21 백업.  실행 후: python run_all.py --chunk && python fit_embed_text.py
"""
import os, glob, shutil

PARSE_DIR = ""
SEARCH_ROOTS = ["/content", ".", "/content/drive/MyDrive"]
_LOG = []


def find_parse_dir():
    if PARSE_DIR:
        return PARSE_DIR
    for root in SEARCH_ROOTS:
        hits = [h for h in glob.glob(os.path.join(root, "**", "pass8_chunk_fund.py"),
                                     recursive=True) if "__pycache__" not in h]
        if hits:
            return os.path.dirname(sorted(hits, key=len)[0])
    raise SystemExit("[!] pass8_chunk_fund.py 를 못 찾았습니다.")


def patch(fname, tag, old, new, marker=None):
    path = os.path.join(DIR, fname)
    if not os.path.exists(path):
        _LOG.append(("FAIL", tag, f"{fname} 없음")); return
    src = open(path, encoding="utf-8").read()
    if marker and marker in src:
        _LOG.append(("SKIP", tag, "이미 적용됨")); return
    n = src.count(old)
    if n != 1:
        _LOG.append(("FAIL", tag, f"앵커 {n}회 (1회여야 함)")); return
    if not os.path.exists(path + ".bak21"):
        shutil.copy2(path, path + ".bak21")
    open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
    _LOG.append(("OK", tag, fname))


DIR = find_parse_dir()
print(f"PARSE_DIR = {DIR}\n")

# ══════════════════════════════════════════════════════
# K-1a  pass8 — confidence 계약 준수 + fill_ratio 분리
# ══════════════════════════════════════════════════════
patch("pass8_chunk_fund.py", "K-1a 펀드카드 confidence",
'''def confidence_of(rec):
    """카드 신뢰도 — 핵심 6축 중 몇 개가 채워졌는지."""
    keys = ["fund_name", "위험등급", "상품분류", "시장잔고_백만원"]
    filled = sum(1 for k in keys if rec.get(k) is not None)
    base = 0.7 + 0.3 * (filled / len(keys))
    if rec.get("보수_신뢰불가") or rec.get("수익률_수동확인필요"):
        base -= 0.1
    return round(min(base, 1.0), 2)''',
'''def confidence_of(rec):
    """
    ★ 계약(INTERFACES.md §2-1): "confidence: OCR만. text는 1.0"

    ⚠️ 예전엔 여기서 필드 충족도를 계산했다. 그런데 안내문서 코퍼스는 계약대로
       상수(text=1.0 / ocr=0.95)를 쓰고 있어서, 같은 필드명이 두 코퍼스에서
       전혀 다른 뜻이 된다. 검색 쪽에서 이 값으로 랭킹을 주면 어긋난다.
       → 계약을 따르고, 충족도는 fill_ratio로 따로 낸다.
    """
    return 1.0


def fill_ratio_of(rec):
    """카드 충실도 — 핵심 축 중 몇 개가 채워졌는지. confidence와 별개 지표."""
    keys = ["fund_name", "위험등급", "상품분류", "시장잔고_백만원"]
    filled = sum(1 for k in keys if rec.get(k) is not None)
    base = 0.7 + 0.3 * (filled / len(keys))
    if rec.get("보수_신뢰불가") or rec.get("수익률_수동확인필요"):
        base -= 0.1
    return round(min(base, 1.0), 2)''',
      marker="def fill_ratio_of(")

patch("pass8_chunk_fund.py", "K-1b 펀드카드 필드",
'''        "confidence": confidence_of(rec),''',
'''        "confidence": confidence_of(rec),      # 계약: text=1.0
        "fill_ratio": fill_ratio_of(rec),      # 충실도(자체 지표)
        "table_ref": None,                     # 계약 §2-1. 펀드카드는 표 유래 아님''',
      marker='"fill_ratio": fill_ratio_of(rec)')

# ══════════════════════════════════════════════════════
# K-1c  pass9 — 같은 처리 + table_ref 기입
# ══════════════════════════════════════════════════════
patch("pass9_chunk_class.py", "K-1c 클래스카드 confidence",
'''def confidence_of(rec):
    classes = rec.get("클래스") or []
    fees = _fee_map(rec)
    if not classes:
        return 0.0
    filled = sum(1 for c in classes if c["class"] in fees)
    base = 0.6 + 0.4 * (filled / len(classes))
    if rec.get("보수_신뢰불가"):
        base -= 0.1
    if parse_method_of(rec) == "coord":
        base -= 0.05
    return round(max(0.5, min(base, 1.0)), 2)''',
'''def confidence_of(rec):
    """★ 계약(INTERFACES.md §2-1): text는 1.0. 좌표 재조립분만 낮춘다."""
    return 0.95 if parse_method_of(rec) == "coord" else 1.0


def fill_ratio_of(rec):
    """클래스 중 총보수가 매핑된 비율. confidence와 별개 지표."""
    classes = rec.get("클래스") or []
    fees = _fee_map(rec)
    if not classes:
        return 0.0
    filled = sum(1 for c in classes if c["class"] in fees)
    base = 0.6 + 0.4 * (filled / len(classes))
    if rec.get("보수_신뢰불가"):
        base -= 0.1
    return round(max(0.5, min(base, 1.0)), 2)''',
      marker="def fill_ratio_of(")

patch("pass9_chunk_class.py", "K-2 클래스카드 table_ref",
'''        "confidence": confidence_of(rec),''',
'''        "confidence": confidence_of(rec),      # 계약: text=1.0 / coord=0.95
        "fill_ratio": fill_ratio_of(rec),      # 총보수 매핑률(자체 지표)
        # ★ 계약 §2-1 — 이 카드는 13항 보수표에서 재구성했다
        "table_ref": f"{rec['doc_id']}#p{page}#t13항보수",''',
      marker='"table_ref": f"{rec[')

# ══════════════════════════════════════════════════════
# K-1d  pass10 — 섹션 청크에도 필드 통일
# ══════════════════════════════════════════════════════
patch("pass10_chunk_section.py", "K-1d 섹션청크 필드",
'''                    "parse_method": "text",
                    "confidence": 1.0,''',
'''                    "parse_method": "text",
                    "confidence": 1.0,          # 계약 §2-1
                    "table_ref": None,''',
      marker='"table_ref": None,')

# ══════════════════════════════════════════════════════
# K-3  수익률기준일 — 펀드카드에 명시
# ══════════════════════════════════════════════════════
patch("pass8_chunk_fund.py", "K-3 수익률기준일",
'''    if rec.get("작성기준일"):
        lines.append(f"작성기준일: {rec['작성기준일']}")''',
'''    if rec.get("작성기준일"):
        lines.append(f"작성기준일: {rec['작성기준일']}")
    # ★ 계약 §2-3: 수익률기준일은 "문서마다 다름. 비교 시 필수".
    #   §4-1d에 '기준일 10개월 편차'가 expects_caveat 케이스로 등재돼 있다.
    #   별도 필드가 없으면 작성기준일을 쓰되 그 사실을 밝힌다(추정으로 단정 금지).
    base = rec.get("수익률기준일")
    if base:
        lines.append(f"수익률 기준일: {base}")
    elif rec.get("작성기준일") and not rec.get("수익률_해당없음"):
        lines.append(f"수익률 기준일: 작성기준일({rec['작성기준일']}) 기준으로 추정 "
                     f"— 펀드마다 다를 수 있어 기간 비교 시 확인 필요")''',
      marker="수익률 기준일:")

# ══════════════════════════════════════════════════════
w = max(len(t) for _, t, _ in _LOG)
for st, tag, msg in _LOG:
    print(f"{ {'OK': '✅', 'SKIP': '•', 'FAIL': '❌'}[st] } {tag:<{w}}  {msg}")
print(f"\n적용 {sum(1 for s,_,_ in _LOG if s=='OK')} / 건너뜀 "
      f"{sum(1 for s,_,_ in _LOG if s=='SKIP')} / 실패 {sum(1 for s,_,_ in _LOG if s=='FAIL')}")

import py_compile
bad = []
for f in ["pass8_chunk_fund.py", "pass9_chunk_class.py", "pass10_chunk_section.py"]:
    try:
        py_compile.compile(os.path.join(DIR, f), doraise=True)
    except Exception as e:
        bad.append((f, str(e)[:150]))
print("❌ 문법 오류: " + str(bad) if bad else "✅ 문법 정상")
print("\n다음: python run_all.py --chunk  &&  python fit_embed_text.py")
