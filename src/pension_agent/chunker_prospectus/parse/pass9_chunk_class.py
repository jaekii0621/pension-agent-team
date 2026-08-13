"""
9패스 — 클래스 카드 청크 생성 (문서당 1개)

클래스별 총보수·판매경로·가입자격을 **표 하나 = 청크 하나**로 보존한다.
"C-P2가 뭔가요", "IRP로 살 수 있는 클래스는", "C1은 왜 못 사나요" 같은
클래스 단위 질의가 여기 걸린다.

⚠️ 설계 판단
  ① 클래스마다 청크를 만들지 않는다(447개가 되고 서로 비슷해 검색이 뭉개진다).
     형제 펀드 문제가 청크 레벨에서 재현되는 것을 막는다.
  ② 표는 통째로 — 길이 300~800자 규칙의 **예외**.
     쪼개면 헤더를 잃어 "이 숫자가 총보수인지 판매보수인지" 모르게 되고,
     클래스 간 비교가 한 청크 안에서 불가능해진다.
  ③ 값이 없는 칸은 빈칸이 아니라 '미확인'으로 적는다.
     빈칸이면 LLM이 다른 행 값을 끌어와 채운다.

입력: six_axis.json 의 세 소스를 클래스 코드로 조인
  · 클래스   (61/92) — class, 설명, 협회펀드코드, 연금유형
  · 보수     (69/92) — class, 총보수
  · 가입자격 (55/92) — class → 조건 문장

산출: data/chunks/class_cards.jsonl
"""
import os, re, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import TABLE_DIR, load_json, make_embed_text

CHUNK_DIR = os.path.join(os.path.dirname(TABLE_DIR), "chunks")
NA = "미확인"

# 설명 문구에서 판매경로를 뽑는다
ROUTE = [("온라인슈퍼", "온라인슈퍼"), ("온라인", "온라인"), ("오프라인", "오프라인")]
FEE_TYPE = [("수수료미징구", "미징구"), ("수수료선취", "선취"), ("수수료후취", "후취")]


def _route(desc):
    d = (desc or "").replace(" ", "")
    parts = []
    for key, label in FEE_TYPE:
        if key in d:
            parts.append(label)
            break
    for key, label in ROUTE:
        if key in d:
            parts.append(label)
            break
    if "개인연금" in d or "연금저축" in d:
        parts.append("개인연금")
    elif "퇴직연금" in d:
        parts.append("퇴직연금")
    return "·".join(parts) if parts else NA


# ⚠️ 5패스가 집합투자규약 조문을 가입자격으로 긁어오는 경우가 있다.
#   "③ 제1항의 규정에 따라 전환하는 경우에 적용되는 기준가격은…" → 가입자격이 아니다
_ARTICLE = re.compile(r"^[①-⑳]|제\s*\d+\s*[항조]|규정에\s*따라|기준가격은")


# ★ '제한 없음'류 — 원문에 실제로 적혀 있는 표기 변형들.
#   "제한 없음" / "제한없음" / "가입제한이 없으며, 선취판매수수료가 징구되는 수익증권"
#   5패스가 이미 335행을 뽑아뒀는데, 아래 접두어를 못 떼서 카드에서 미확인으로 떨어졌다.
_NONE = re.compile(r"제한\s*없음|가입\s*제한이?\s*없|제한없|해당\s*없음|누구나")
# ⚠️ 5패스가 수수료 칸을 가입자격으로 긁어오는 경우가 있다(335행 중 36행).
#   "수수료선취-오프라인 납입금액의0.02% 이내" → 가입자격이 아니다. 버린다.
_FEE = re.compile(r"납입금액|판매수수료율|%\s*이내|이내의?\s*범위|환매금액")
# 가입자격 문장 앞에 판매유형 설명이 붙어 오는 문서가 있다
#   "수수료미징구-오프라인-개인연금 가입제한이 없으며, …"
_DESC_HEAD = re.compile(
    r"^(?:종류\s*)?(?:수수료(?:미징구|선취|후취)[^\s]*|온라인[^\s]*|오프라인[^\s]*)\s*")


def _elig(rec, cls, crec=None):
    """
    가입자격과 그 출처를 함께 돌려준다 → (문구, 출처)

    출처: "문서"(원문 문장) / "코드규칙"(코드·설명에서 추론) / None(미확인)
    ★ 근거 표시가 평가 항목이므로 어디서 온 값인지 구분해서 남긴다.
      코드규칙으로 채운 값은 카드에 (코드규칙)으로 표시해 원문 인용과 섞이지 않게 한다.

    ★ C1~C4는 선행 클래스를 1년 이상 보유해야 취득하는 클래스라 신규 가입이 불가능하다
      (5패스 독스트링의 최상세 정본 규칙). 문서 문장이 잘려 나와도 코드 규칙이 우선한다.
    """
    if re.fullmatch(r"C[1-4]", cls):
        return "신규 가입 불가(선행 클래스 보유 필요)", "코드규칙"
    if cls in ("I", "C-F"):
        return "고액·기관 전용", "코드규칙"

    # ── ① 원문 문장 ──
    e = (rec.get("가입자격") or {}).get(cls)
    if e:
        s = re.sub(r"\s+", " ", e).strip()
        if not _ARTICLE.search(s[:30]) and not _FEE.search(s):   # 조문·수수료칸 제외
            s = _DESC_HEAD.sub("", s).strip(" ,-")
            if _NONE.search(s):
                return "제한 없음", "문서"
            # 표 칸이므로 첫 절만 남긴다. 전문은 섹션 원문 청크에 있다.
            s = re.split(r"(?:하며|으로서|이며|에 한하여|를 통하여)", s)[0].strip(" ,")
            if len(s) >= 4:
                return s[:40] + ("…" if len(s) > 40 else ""), "문서"

    # ── ② 문장에서 뽑은 판정 플래그 ──
    flags = (rec.get("가입자격_판정") or {}).get(cls) or {}
    tags = []
    if flags.get("신규가입불가"):
        tags.append("신규 가입 불가")
    if flags.get("고액전용"):
        tags.append("고액·기관 전용")
    if flags.get("연금저축"):
        tags.append("연금저축계좌 전용")
    if flags.get("퇴직연금"):
        tags.append("퇴직연금·IRP 전용")
    # ★ 설명(판매유형)과 충돌하면 설명을 따른다.
    #   가입자격 문장이 옆 행 것과 섞여 C-Pe(개인연금)가 퇴직연금으로 뒤집히는 사례가 있다.
    # ★ 설명(판매유형)이 명시한 쪽을 따른다.
    #   가입자격 문장은 뒤에 옆 행 내용이 붙어 오는 경우가 있어 연금저축·퇴직연금이
    #   동시에 True가 된다. 반면 설명("…-온라인-개인연금")은 그 행만의 값이라 정확하다.
    #   ⚠️ 연금유형 필드는 이미 오염돼 있을 수 있으므로 설명 문자열을 직접 본다.
    _d = ((crec or {}).get("설명") or "").replace(" ", "")
    pt = ("개인연금" if ("개인연금" in _d or "연금저축" in _d)
          else "퇴직연금" if "퇴직연금" in _d
          else (crec or {}).get("연금유형"))
    if tags and pt == "개인연금":
        tags = [t for t in tags if "퇴직연금" not in t]
    elif tags and pt == "퇴직연금":
        tags = [t for t in tags if "연금저축" not in t]
    if tags:
        return ", ".join(tags), "문서"

    # ── ③ 코드·설명 규칙 ──
    #   ⚠️ '제한 없음'을 여기서 추론하지 않는다. 원문에 없는 단정은 넣지 않는다.
    if pt == "개인연금":
        return "연금저축계좌 전용", "코드규칙"
    if pt == "퇴직연금":
        return "퇴직연금·IRP 전용", "코드규칙"
    d = (crec or {}).get("설명") or ""
    d = d.replace(" ", "")
    if re.search(r"기관|고액", d):
        return "고액·기관 전용", "코드규칙"
    if "온라인슈퍼" in d:
        return "온라인 판매채널 전용", "코드규칙"
    return NA, None


def _fee_map(rec):
    """클래스 → 총보수. 신뢰불가 문서는 전부 미확인 처리."""
    if rec.get("보수_신뢰불가"):
        return {}
    out = {}
    for r in (rec.get("보수") or []):
        if r.get("클래스명미상"):
            continue
        v = r.get("총보수")
        if isinstance(v, (int, float)) and 0 < v <= 5:
            out.setdefault(r["class"], v)
    return out


def build_text(rec):
    classes = rec.get("클래스") or []
    fees = _fee_map(rec)
    name = rec.get("fund_name") or rec["doc_id"]

    rows, srcs = [], {}
    for c in classes:
        cls = c["class"]
        fee = fees.get(cls)
        elig, src = _elig(rec, cls, c)
        if src:
            srcs[cls] = src
        rows.append((
            cls,
            f"{fee}%" if fee is not None else NA,
            _route(c.get("설명")),
            elig + ("(코드규칙)" if src == "코드규칙" else ""),
        ))

    HEAD = ("클래스", "총보수", "판매유형", "가입자격")
    w = [max(len(r[i]) for r in rows + [HEAD]) for i in range(4)]
    head = "  ".join(HEAD[i].ljust(w[i]) for i in range(3)) + "  " + HEAD[3]
    lines = [f"{name} — 클래스별 보수 및 가입자격", "", head, "-" * (sum(w) + 6)]
    for r in rows:
        lines.append(f"{r[0].ljust(w[0])}  {r[1].ljust(w[1])}  {r[2].ljust(w[2])}  {r[3]}")

    # 연금 적격 요약 — 검색에 걸리는 키워드를 본문에 남긴다
    p_save = [c["class"] for c in classes if c.get("연금유형") == "개인연금"]
    p_ret = [c["class"] for c in classes if c.get("연금유형") == "퇴직연금"]
    if p_save:
        lines.append(f"\n연금저축계좌 가입 가능 클래스: {', '.join(p_save)}")
    if p_ret:
        lines.append(f"퇴직연금(IRP) 가입 가능 클래스: {', '.join(p_ret)}")
    if not (p_save or p_ret):
        lines.append(f"\n연금계좌 전용 클래스: {NA}")
    if fees:
        lo = min(fees.values())
        hi = max(fees.values())
        lines.append(f"총보수 범위: {lo}% ~ {hi}%")
    # ★ 원문 문장에서 온 가입자격의 근거 페이지. (코드규칙) 표기분은 제외한다.
    pmap = rec.get("가입자격_페이지") or {}
    pgs = sorted({pmap[c] for c, s in srcs.items() if s == "문서" and c in pmap})
    if pgs:
        lines.append("가입자격 근거: " + ", ".join(f"p.{p}" for p in pgs))
    return "\n".join(lines)


def card_page(rec):
    """13항 보수가 정본. 없으면 제1부1항."""
    src = rec.get("src") or {}
    for k in ("13항보수", "제1부1항", "요약정보"):
        if isinstance(src.get(k), int):
            return src[k]
    return 1


def parse_method_of(rec):
    """좌표 재조립으로 얻은 클래스는 'ocr'에 준하는 별도 표기."""
    srcs = {c.get("출처") for c in (rec.get("클래스") or [])}
    return "coord" if "13항좌표" in srcs else "text"


def confidence_of(rec):
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
    return round(max(0.5, min(base, 1.0)), 2)


# ★ 표 아래 요약줄이 검색 키워드를 담고 있다("연금저축계좌 가입 가능 클래스: C-P, C-Pe").
#   앞에서만 자르면 이게 통째로 사라져 연금 질의가 이 카드를 못 찾는다.
_TAIL_KEYS = ("연금저축계좌 가입", "퇴직연금(IRP) 가입", "연금계좌 전용", "총보수 범위")


def build(rec):
    page = card_page(rec)
    txt = build_text(rec)
    tail = "\n".join(l for l in txt.split("\n") if l.startswith(_TAIL_KEYS))
    return {
        # ── 필수 7 ──
        "chunk_id": f"{rec['doc_id']}#p{page}#c002",
        "doc_id": rec["doc_id"],
        "source": "상품",
        "page": page,
        "section": "13항보수",
        "text": txt,
        "embed_text": make_embed_text(
            f"{rec.get('fund_name') or rec['doc_id']} · 클래스별 보수·가입자격",
            txt, tail=tail),
        "parse_method": parse_method_of(rec),
        "confidence": confidence_of(rec),
        # ── 조건부 ──
        "chunk_type": "class_card",
        "가입자격_근거페이지": {c: p for c, p in (rec.get("가입자격_페이지") or {}).items()
                        if c in {x["class"] for x in (rec.get("클래스") or [])}} or None,
        "fund_code": rec.get("fund_codes") or None,
        "fund_name": rec.get("fund_name"),
        "위험등급": rec.get("위험등급"),
        "자산유형": rec.get("자산유형"),
        "연금저축_가능": rec.get("연금저축_가능"),
        "퇴직연금_가능": rec.get("퇴직연금_가능"),
        "클래스목록": [c["class"] for c in (rec.get("클래스") or [])],
    }


def main():
    six = load_json(os.path.join(TABLE_DIR, "six_axis.json"))
    os.makedirs(CHUNK_DIR, exist_ok=True)

    # ★ 클래스가 없는 문서는 카드를 만들지 않는다 (빈 표는 검색 노이즈)
    targets = [r for r in six if r.get("클래스")]
    out = [build(r) for r in targets]

    path = os.path.join(CHUNK_DIR, "class_cards.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    ID = re.compile(r"^.+#p\d+#c\d{3}$")
    bad = [c["chunk_id"] for c in out if not ID.match(c["chunk_id"])]
    lens = [len(c["text"]) for c in out]
    print(f"클래스 카드 {len(out)}개 (전체 {len(six)}개 문서 중) → {path}")
    print(f"  chunk_id 위반 {len(bad)} / 중복 {len(out) - len({c['chunk_id'] for c in out})}")
    print(f"  길이 min {min(lens)} / 중앙 {sorted(lens)[len(lens)//2]} / max {max(lens)}")
    print(f"  800자 초과 {sum(1 for x in lens if x > 800)}개 (표는 통째 보존 — 정상)")
    print(f"  confidence 평균 {sum(c['confidence'] for c in out)/len(out):.2f}")
    print(f"  parse_method: coord {sum(1 for c in out if c['parse_method']=='coord')} / "
          f"text {sum(1 for c in out if c['parse_method']=='text')}")
    print("\n--- 샘플 ---")
    print(out[0]["text"][:1000])


if __name__ == "__main__":
    main()