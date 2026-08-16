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


def _route(desc, pt=None):
    """
    판매경로 표기. pt를 주면 연금 구분은 그 값을 따른다.
    ⚠️ 설명 문구와 인쇄 라벨이 어긋나는 문서가 있다(C-P가 퇴직연금인 문서 등).
       판매유형 칸과 가입자격 칸이 서로 다른 근거를 쓰면 카드 안에서 모순이 된다.
    """
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
    if pt:
        parts.append(pt)
    elif "개인연금" in d or "연금저축" in d:
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


# ★ 인쇄 라벨 저장 위치 자동 탐색.
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


def _unwrap(v):
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
    v, _pg = _printed_entry(rec, cls)
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
    # ⚠️ C-F를 코드만 보고 '고액·기관'으로 확정하면 안 된다.
    #   퇴직연금·직판인 문서가 있다(미래에셋퇴직플랜 계열 2건 확인).
    #   설명에 '기관/고액'이 있으면 아래 ③ 코드·설명 규칙에서 잡힌다.
    #   I는 코퍼스에서 예외가 확인되지 않아 남긴다.
    if cls == "I":
        return "고액·기관 전용", "코드규칙"

    # ── ① 원문 문장 ──
    e = (rec.get("가입자격") or {}).get(cls)
    if e:
        s = re.sub(r"\s+", " ", e).strip()
        if not _ARTICLE.search(s[:30]) and not _FEE.search(s):   # 조문·수수료칸 제외
            s = _DESC_HEAD.sub("", s).strip(" ,-")
            if _NONE.search(s):
                return "제한 없음", "문서"
            # ★ 5패스 문장이 옆 행 내용을 끌고 오는 경우가 있다(Ce·S 행에서 관찰).
            #   문장 안에 **다른 클래스 코드**가 보이면 그 앞에서 자른다.
            others = [x["class"] for x in (rec.get("클래스") or [])
                      if x["class"] != cls and x["class"]]
            for o in sorted(others, key=len, reverse=True):
                m_ = re.search(r"종류\s*" + re.escape(o) + r"\b", s)
                if m_ and m_.start() > 0:
                    s = s[:m_.start()].strip(" ,")
                elif m_:
                    s = ""          # 아예 남의 행 문장이다
                    break
            if not s:
                return NA, None
            # ★ 문장 앞부분이 잘려 괄호 닫힘이나 조사로 시작하는 조각은 버린다.
            #   "성질을가진것을포함) 기관투자자또는…" 처럼 카드에 그대로 나가면
            #   읽는 사람이 문맥을 알 수 없다. 아래 규칙/플래그로 넘긴다.
            if re.match(r"^[)\]}）]|^(?:을|를|이|가|은|는|에|의|로|와|과)\s", s):
                s = ""
            if s and re.search(r"기관투자자|전문투자자|국가재정법", s[:40]) \
                    and not re.search(r"^[가-힣A-Za-z]{2,6}(은|는|이|가)\s", s):
                return "고액·기관 전용", "문서"
            if not s:
                return NA, None
            # ★ 긴 정형 문구는 요약한다. 표 칸에 두 줄짜리 설명이 들어가면
            #   카드가 넓어지고 '집합투자증 권'처럼 줄바꿈 흔적이 그대로 노출된다.
            #   전문은 섹션 원문 청크에 있으므로 여기서는 짧은 라벨이면 충분하다.
            _sq = s.replace(" ", "")
            for _pat, _lab in (
                (r"투자자문업자로부터투자자문을받고", "투자자문 경유(무권유저비용)"),
                (r"공동판매채널로서의역할", "온라인 판매채널 전용"),
                (r"일임형|WRAPAccount|랩어카운트|특정금전신탁", "랩·금전신탁 전용"),
                (r"전문투자자|기관투자자|국가재정법에따른기금", "고액·기관 전용"),
                (r"최초매수일로부터\d+년", "신규 가입 불가(선행 클래스 보유 필요)"),
            ):
                if re.search(_pat, _sq):
                    return _lab, "문서"
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
    pt, _psrc = _pension_of(rec, cls, crec)
    if tags and pt == "개인연금":
        tags = [t for t in tags if "퇴직연금" not in t]
    elif tags and pt == "퇴직연금":
        tags = [t for t in tags if "연금저축" not in t]
    if tags:
        return ", ".join(tags), "문서"

    # ── ③ 코드·설명 규칙 ──
    #   ⚠️ '제한 없음'을 여기서 추론하지 않는다. 원문에 없는 단정은 넣지 않는다.
    if pt == "개인연금":
        return "연금저축계좌 전용", ("문서" if _psrc in ("인쇄", "문서") else "코드규칙")
    if pt == "퇴직연금":
        return "퇴직연금·IRP 전용", ("문서" if _psrc in ("인쇄", "문서") else "코드규칙")
    d = (crec or {}).get("설명") or ""
    d = d.replace(" ", "")
    if re.search(r"기관|고액", d):
        return "고액·기관 전용", "코드규칙"
    if "온라인슈퍼" in d:
        return "온라인 판매채널 전용", "코드규칙"
    return NA, None


def _fee_map(rec):
    """클래스 → 총보수. 신뢰불가 문서는 전부 미확인 처리."""
    # ⚠️ 문서 단위로 전부 버리면 정상 행까지 사라진다. 행 단위 '클래스명미상'
    #   플래그가 이미 있으므로 그 행만 제외하면 된다(아래 루프에서 처리).
    #   예: 18행 중 17행이 정상 코드인데 #N 하나 때문에 전부 미확인이 되던 사례.
    out = {}
    for r in (rec.get("보수") or []):
        if r.get("클래스명미상"):
            continue
        v = r.get("총보수")
        if isinstance(v, (int, float)) and 0 < v <= 5:
            out.setdefault(r["class"], v)
    return out


def _fee_pages(rec):
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
            _route(c.get("설명"), _pension_of(rec, cls, c)[0]),
            elig + ("(코드규칙)" if src == "코드규칙" else ""),
        ))

    HEAD = ("클래스", "총보수", "판매유형", "가입자격")
    w = [max(len(r[i]) for r in rows + [HEAD]) for i in range(4)]
    head = "  ".join(HEAD[i].ljust(w[i]) for i in range(3)) + "  " + HEAD[3]
    lines = [f"{name} — 클래스별 보수 및 가입자격", "", head, "-" * (sum(w) + 6)]
    for r in rows:
        lines.append(f"{r[0].ljust(w[0])}  {r[1].ljust(w[1])}  {r[2].ljust(w[2])}  {r[3]}")

    # 연금 적격 요약 — 검색에 걸리는 키워드를 본문에 남긴다
    # ★ 요약줄은 카드에서 가장 자주 인용된다. 코드규칙 추론이 아니라
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

    def _pgs(codes):
        # ★ 인쇄 라벨에 근거 페이지가 함께 저장돼 있다. 요약줄은 가장 자주 인용되는
        #   줄이므로 여기에 근거를 붙인다.
        s = sorted({_printed_entry(rec, c)[1] for c in codes
                    if _printed_entry(rec, c)[1]})
        return " (근거 " + ", ".join(f"p.{p}" for p in s) + ")" if s else ""

    if p_save:
        lines.append(f"\n연금저축계좌 가입 가능 클래스: {', '.join(p_save)}"
                     f"{_pgs(p_save)} {mark}".rstrip())
    if p_ret:
        # ★ 연금저축 줄이 없으면 여기가 첫 요약줄이므로 빈 줄로 표와 갈라준다
        lines.append(f"{'' if p_save else chr(10)}퇴직연금(IRP) 가입 가능 클래스: {', '.join(p_ret)}"
                     f"{_pgs(p_ret)} {mark}".rstrip())
    if not (p_save or p_ret):
        # ⚠️ 인쇄에 라벨이 없다고 '가입 불가'로 뒤집지 않는다. 미확인이다.
        lines.append(f"\n연금계좌 전용 클래스: {NA}")
    if fees:
        lo = min(fees.values())
        hi = max(fees.values())
        if lo == hi:
            only = [c for c, v in fees.items() if v == lo]
            lines.append(f"총보수: {lo}% ({', '.join(only[:3])} 확인분)")
        else:
            lines.append(f"총보수 범위: {lo}% ~ {hi}%")
    # ★ 총보수 근거 페이지 — 보수표가 두 쪽에 걸치는 문서가 많고
    #   하필 연금 클래스가 뒤쪽에 몰려 있어 문서 단위 페이지로는 인용이 어긋난다.
    fpg = sorted({p for p in (_fee_pages(rec).values()) if p})
    if fpg:
        lines.append("총보수 근거: " + ", ".join(f"p.{p}" for p in fpg))
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
        "confidence": confidence_of(rec),      # 계약: text=1.0 / coord=0.95
        "fill_ratio": fill_ratio_of(rec),      # 총보수 매핑률(자체 지표)
        # ★ 계약 §2-1 — 이 카드는 13항 보수표에서 재구성했다
        "table_ref": f"{rec['doc_id']}#p{page}#t13항보수",
        # ── 조건부 ──
        "chunk_type": "class_card",
        "총보수_근거페이지": {c: p for c, p in _fee_pages(rec).items() if p} or None,
        "연금유형_근거페이지": {c["class"]: _printed_entry(rec, c["class"])[1]
                        for c in (rec.get("클래스") or [])
                        if _printed_entry(rec, c["class"])[1]} or None,
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