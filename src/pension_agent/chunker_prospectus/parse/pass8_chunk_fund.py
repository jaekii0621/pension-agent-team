"""
8패스 — 펀드 카드 청크 생성 (문서당 1개, 92개)

six_axis.json의 6축을 '사람이 읽는 문장'으로 재구성한다.
이름·운용사·"총보수 얼마" 같은 질의가 이 카드에 걸린다.

⚠️ 설계 원칙 두 가지
  ① 빈 값은 줄을 생략하지 않고 "자료 미확인"으로 **명시**한다.
     생략하면 LLM이 다른 펀드 숫자를 끌어와 채운다(정보한계대응 감점).
  ② 신뢰할 수 없는 값은 카드에 넣지 않는다.
     보수_신뢰불가(클래스 매핑 미상) / 수익률_수동확인필요 → 미확인으로 표기.

스키마: 팀 INTERFACES.md 준수
  필수 7 : chunk_id, doc_id, source, page, text, parse_method, confidence
  조건부  : fund_code, 위험등급, 자산유형, 연금저축_가능, 퇴직연금_가능
  chunk_id 형식 = {doc_id}#p{페이지}#c{순번3자리}

산출: data/chunks/fund_cards.jsonl
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import TABLE_DIR, load_json

CHUNK_DIR = os.path.join(os.path.dirname(TABLE_DIR), "chunks")
NA = "자료 미확인"


def _fmt_pct(v):
    return f"{v}%" if isinstance(v, (int, float)) else NA


def _fmt_aum(v):
    """백만원 단위 → 사람이 읽는 표기. 1억원 = 100백만원."""
    if not isinstance(v, (int, float)):
        return NA
    if v >= 1_000_000:                       # 1조원 이상
        return f"{v/1_000_000:,.2f}조원"
    if v >= 100:                             # 1억원 이상
        return f"{v/100:,.0f}억원"
    return f"{v:,.0f}백만원"


def _returns(rec):
    """최근 1년 수익률 + 비교지수. 수동확인필요 건은 미확인 처리."""
    if rec.get("수익률_해당없음"):
        why = rec.get("수익률_사유") or "문서 미기재"
        return f"해당없음({why})", "해당없음"
    if rec.get("수익률_수동확인필요"):
        return NA, NA
    f = rec.get("펀드수익률")
    b = rec.get("비교지수수익률")
    return (_fmt_pct(f[0]) if f else NA), (_fmt_pct(b[0]) if b else NA)


# ★ 과제 정의상 수익률 축은 "1년~3년"이다. 카드에 1년치만 쓰면 비교 질의에 못 답한다.
#   연평균수익률 표의 열 순서(확인 결과 기준). 값이 모자라면 있는 만큼만 쓴다.
_PERIODS = ["1년", "2년", "3년", "5년", "설정이후"]


def _returns_multi(rec):
    """최근 1~3년 수익률을 한 줄로. 값이 없으면 None."""
    if rec.get("수익률_해당없음") or rec.get("수익률_수동확인필요"):
        return None
    f = rec.get("펀드수익률") or []
    b = rec.get("비교지수수익률") or []
    if len(f) < 2:
        return None
    parts = []
    for i, lab in enumerate(_PERIODS[:3]):
        if i >= len(f):
            break
        bi = f" (비교지수 {_fmt_pct(b[i])})" if i < len(b) else ""
        parts.append(f"{lab} {_fmt_pct(f[i])}{bi}")
    return "연평균 수익률: " + " / ".join(parts)


def _fee_line(rec):
    """
    클래스별 총보수. ⚠️ 보수_신뢰불가(클래스 매핑 미상)면 숫자를 쓰지 않는다.
    잘못 매핑된 보수를 카드에 박으면 그대로 인용된다.
    """
    # ⚠️ 문서 단위로 차단하면 정상 행까지 사라진다. 아래 필터가 이미 '클래스명미상'
    #   행을 걸러내므로 행 단위로 판단하면 충분하다.
    #   (18행 중 17행이 정상인데 #N 하나 때문에 총보수 줄이 통째로 빠지던 사례)
    rows = [r for r in (rec.get("보수") or [])
            if isinstance(r.get("총보수"), (int, float)) and not r.get("클래스명미상")]
    if not rows:
        return f"총보수: {NA} (클래스 매핑 확인 필요)"
    # ★ 종류형이 아닌 펀드는 클래스 구분이 없다. 코드 자리에 '-'가 나가면 어색하다.
    if len(rows) == 1 and rows[0]["class"] in ("-", "#1", ""):
        return f"총보수: {rows[0]['총보수']}% (종류형 아님 — 클래스 구분 없음)"
    parts = [f"{r['class']} {r['총보수']}%" for r in rows[:8]]
    return "클래스별 총보수: " + ", ".join(parts)


def _ev_pages(rec, fields):
    """수작업 근거 페이지 목록. 없으면 빈 문자열."""
    ev = rec.get("수작업근거") or {}
    pgs = sorted({ev[f]["page"] for f in fields
                  if isinstance(ev.get(f), dict) and ev[f].get("page")})
    return " (근거 " + ", ".join(f"p.{p}" for p in pgs) + ")" if pgs else ""


def _pension_line(rec):
    tags = []
    if rec.get("연금저축_가능"):
        tags.append("연금저축 가입 가능")
    if rec.get("퇴직연금_가능"):
        tags.append("퇴직연금(IRP) 가입 가능")
    if not tags:
        # 수작업으로 확인한 문서는 False가 '해당 클래스 없음'으로 확정된 값이다
        # 클래스 목록을 충분히 읽었는데 연금 클래스가 없으면 그것도 확인된 결과다
        if ("연금저축_가능" in (rec.get("수작업보정") or [])
                or len(rec.get("클래스") or []) >= 5):
            return "연금계좌: 연금저축·퇴직연금 전용 클래스 없음"
        return f"연금계좌 적격 여부: {NA}"
    # ★ 수작업으로 확인한 값이면 근거 페이지를 문장에 직접 박는다.
    #   별도 필드로 두면 에이전트가 인용할 때 놓친다.
    return ("연금계좌: " + ", ".join(tags)
            + _ev_pages(rec, ("연금저축_가능", "퇴직연금_가능")))


def build_text(rec):
    name = rec.get("fund_name") or rec["doc_id"]
    grade = rec.get("위험등급")
    label = rec.get("위험라벨")
    grade_s = f"{grade}등급" + (f"({label})" if label else "") if grade else NA
    r1, b1 = _returns(rec)

    lines = [
        name,
        f"운용사: {rec.get('운용사') or NA}",
        f"상품분류: {rec.get('상품분류') or NA}",
        f"자산유형: {rec.get('자산유형') or NA}",
        f"투자위험등급: {grade_s}",
        _pension_line(rec),
        _fee_line(rec),
        f"최근 1년 수익률: {r1} (비교지수 {b1})",
        _returns_multi(rec),
        f"비교지수: {rec.get('비교지수') or NA}",
        f"순자산(시장잔고): {_fmt_aum(rec.get('시장잔고_백만원'))}",
    ]
    to = rec.get("매매회전율")
    # ★ 0은 파싱 오류가 아니라 원문에 명시된 값이다(dump 전수 대조, 43건).
    #   매매회전율은 정의상 '주식' 매매 지표라 채권형의 0은 정상이고,
    #   "회전율 얼마예요?" 질의에 답하려면 카드에 있어야 한다.
    if rec.get("매매회전율_해당없음"):
        lines.append("매매회전율: 해당없음(주식 매매 지표 없음)")
    elif isinstance(to, (int, float)):
        if to == 0:
            lines.append("매매회전율: 0% (해당 기간 주식 매매 없음)")
        else:
            lines.append(f"매매회전율: {to:g}%")
    else:
        lines.append(f"매매회전율: {NA}")      # 빈 값은 생략하지 않고 명시
    if rec.get("작성기준일"):
        lines.append(f"작성기준일: {rec['작성기준일']}")
    # ★ 계약 §2-3: 수익률기준일은 "문서마다 다름. 비교 시 필수".
    #   §4-1d에 '기준일 10개월 편차'가 expects_caveat 케이스로 등재돼 있다.
    #   별도 필드가 없으면 작성기준일을 쓰되 그 사실을 밝힌다(추정으로 단정 금지).
    base = rec.get("수익률기준일")
    if base:
        lines.append(f"수익률 기준일: {base}")
    elif rec.get("작성기준일") and not rec.get("수익률_해당없음"):
        lines.append(f"수익률 기준일: 작성기준일({rec['작성기준일']}) 기준으로 추정 "
                     f"— 펀드마다 다를 수 있어 기간 비교 시 확인 필요")
    return "\n".join(l for l in lines if l)   # ★ None 줄 제거


def confidence_of(rec):
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
    return round(min(base, 1.0), 2)


def card_page(rec):
    """카드의 근거 페이지 — 요약정보가 정본, 없으면 제1부1항, 없으면 1."""
    src = rec.get("src") or {}
    for k in ("요약정보", "제1부1항"):
        if isinstance(src.get(k), int):
            return src[k]
    return 1


def build(rec):
    page = card_page(rec)
    return {
        # ── 필수 7 ──
        "chunk_id": f"{rec['doc_id']}#p{page}#c001",
        "doc_id": rec["doc_id"],
        "source": "상품",
        "page": page,
        "section": "펀드요약",
        "text": build_text(rec),
        "embed_text": build_text(rec),   # 300자대 — 절단 위험 없음. 스키마 통일용
        "parse_method": "manual" if rec.get("수작업보정") else "text",
        "confidence": confidence_of(rec),      # 계약: text=1.0
        "fill_ratio": fill_ratio_of(rec),      # 충실도(자체 지표)
        "table_ref": None,                     # 계약 §2-1. 펀드카드는 표 유래 아님
        # ── 조건부 ──
        "chunk_type": "fund_card",
        "근거페이지": {k: v["page"] for k, v in (rec.get("수작업근거") or {}).items()
                   if v.get("page")} or None,
        "fund_code": rec.get("fund_codes") or None,
        "fund_name": rec.get("fund_name"),
        "위험등급": rec.get("위험등급"),
        "자산유형": rec.get("자산유형"),
        "연금저축_가능": rec.get("연금저축_가능"),
        "퇴직연금_가능": rec.get("퇴직연금_가능"),
    }


def main():
    six = load_json(os.path.join(TABLE_DIR, "six_axis.json"))
    os.makedirs(CHUNK_DIR, exist_ok=True)
    out = [build(r) for r in six]

    path = os.path.join(CHUNK_DIR, "fund_cards.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ── 검증 ──
    import re
    ID = re.compile(r"^.+#p\d+#c\d{3}$")
    bad_id = [c["chunk_id"] for c in out if not ID.match(c["chunk_id"])]
    dup = len(out) - len({c["chunk_id"] for c in out})
    na_cnt = {k: sum(1 for c in out if NA in c["text"].split("\n")[i])
              for i, k in enumerate(["이름"])}

    print(f"펀드 카드 {len(out)}개 → {path}")
    print(f"  chunk_id 형식 위반 {len(bad_id)} / 중복 {dup}")
    print(f"  평균 길이 {sum(len(c['text']) for c in out)//len(out)}자")
    print(f"  confidence 평균 {sum(c['confidence'] for c in out)/len(out):.2f}")
    for k in ["위험등급", "자산유형", "fund_code"]:
        print(f"  {k} 채워짐 {sum(1 for c in out if c.get(k))}/{len(out)}")
    print("\n--- 샘플 ---")
    print(json.dumps(out[0], ensure_ascii=False, indent=1)[:900])


if __name__ == "__main__":
    main()
