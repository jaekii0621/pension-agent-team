"""
3패스 — 섹션 지도를 따라 6축 필드 추출

지도가 나왔으니 필요한 페이지만 연다. 표는 pdfplumber(MIT)로 뜯는다.

산출: data/tables/six_axis.json  (92행)
"""
import os, re, sys, glob, json, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import (DUMP_DIR, TABLE_DIR, flat, norm, norm_company,
                          parse_grade_basis,
                          parse_grade, parse_var, grade_from_var,
                          load_json, save_json)
warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def span(sec, start_key, doc_pages, span_max=6):
    """
    섹션 시작 페이지 ~ 다음 섹션 시작 직전까지. 고정 오프셋 금지.
    ★ 같은 페이지에 두 섹션이 있으면(연혁이 제1부1항과 같은 11p) 끝 < 시작이 된다.
      이때는 최소 1페이지를 보장한다. 섹션 내부 경계는 3패스에서 헤딩 문자열로 자른다.
    """
    if start_key not in sec:
        return None
    s = sec[start_key]
    later = [v for v in sec.values() if v > s]
    e = min(later) - 1 if later else min(s + span_max, doc_pages)
    e = max(e, s)                      # ← 최소 1페이지 보장
    # ★ 다음 섹션이 바로 다음 페이지에 있으면 표가 잘린다.
    #   섹션 헤딩은 페이지 중간에도 오므로, 최소 2페이지는 보되 span_max를 넘지 않는다.
    e = max(e, min(s + 1, doc_pages))
    return (s, min(e, s + span_max))


def text_of(pages, rng):
    if not rng:
        return ""
    a, b = rng
    return "\n".join(p["text"] for p in pages if a <= p["page"] <= b)


# ── ⑤⑥ 펀드명 폴백 ★ 요약정보/제1부1항 범위 밖에 이름이 있는 문서용 ──
#   ⑤ 앞 3페이지 라벨형. 기존 ①과 달리 콜론이 없어도(공백 정렬) 잡고,
#      자간이 깨진 문서(공백 전부 소실)는 norm() 사본에서 다시 본다.
#        "1. 집합투자기구 명칭            신영밸류고배당증권자투자신탁(주식)"
#   ⑥ 머리글(running header) 빈도. 여러 페이지 맨 위에 같은 줄이 반복되면 펀드명이다.
#      1페이지가 절단되거나("증권투자신탁(") 라벨이 아예 없는 문서에서 유일하게 남는 단서.
_TAIL = r"(?:투자신탁|투자회사)"
# 모투자신탁·다른 자투자신탁 나열·일반 서술을 배제한다
_BAD = re.compile(r"모투자신탁|수익증권|집합투자업자|판매회사|이\s*투자신탁|"
                  r"명칭\s*변경|변경\(|→|증권의?\s*종류")


def _clean_name(s):
    s = re.sub(r"\s+", " ", (s or "")).strip()
    s = re.sub(r"^[\d.\s]*", "", s)
    s = re.sub(r"\s*[\[\(]\s*펀드\s*코드.*$", "", s)
    return s.strip(" :：·-")


def _plausible(s):
    if not s or _BAD.search(s):
        return False
    s = _clean_name(s)
    return 8 <= len(s) <= 60 and re.search(_TAIL, s) is not None


def name_front(pages, npages=3):
    """⑤ 앞 3페이지에서 '집합투자기구 명칭' 라벨을 찾는다 (콜론·공백정렬·자간깨짐)."""
    LAB = [r"집합투자기구\s*(?:의\s*)?명칭\s*[:：]\s*(.{6,60})",
           r"집합투자기구\s*(?:의\s*)?명칭\s{2,}(.{6,60})"]      # 공백 정렬형
    for p in pages[:npages]:
        for line in p["text"].split("\n"):
            L = line.strip()
            if "종류형 명칭" in L or "펀드코드" in L and not re.search(_TAIL, L):
                continue
            for pat in LAB:
                m = re.search(pat, L)
                if m and _plausible(m.group(1)):
                    return _clean_name(m.group(1))
        # 자간 깨짐 — 공백을 지운 사본에서 재시도
        ns = norm(p["text"])
        m = re.search(r"집합투자기구의?명칭[:：]?([가-힣A-Za-z0-9]{6,50}" + _TAIL
                      + r"[가-힣A-Za-z0-9]{0,12}(?:\([^)]{1,12}\))?)", ns)
        if m and _plausible(m.group(1)):
            return _clean_name(m.group(1))
    return None


def name_header(pages, scan=15, min_hits=3):
    """⑥ 여러 페이지에 반복되는 짧은 줄 = 머리글 = 펀드명."""
    from collections import Counter
    seen = Counter()
    for p in pages[:scan]:
        top = [l.strip() for l in p["text"].split("\n")[:4] if l.strip()]
        for L in set(top):                       # 같은 페이지 중복은 1회로
            c = _clean_name(L)
            if _plausible(c) and re.search(_TAIL + r"\s*(?:\(|\[|\d|$)", c):
                seen[c] += 1
    if not seen:
        return None
    best, hits = seen.most_common(1)[0]
    return best if hits >= min_hits else None


# ★ 상품분류 값 자체의 형태 — 라벨('분류:') 없이 값만 적는 문서가 많다
#   "투자신탁, 증권(채권형), 개방형(중도환매가능), 추가형(추가납입가능), 종류형, 모자형"
CLASSIFY_VAL = re.compile(r"^(?:투자신탁|투자회사)\s*,\s*증권형?\s*\(?[^\n]{0,90}$")


def normalize_class(s):
    """
    ★ 클래스 코드 정규화 — 저장 직전의 마지막 관문.

    ⚠️ 13항 보수표 폴백 경로에서 설명 문구가 클래스로 들어왔다(335건 중 150건).
         "수수료미징구-온"  "수수료선취-오프"  ← 8자에서 잘린 설명
       올바른 형태는 괄호 안 코드다:
         "수수료미징구-온라인-개인연금(C-Pe)" → "C-Pe"
       코드로 볼 수 없으면 None을 돌려 **버린다**. 잘린 줄
       ("수수료미징구-오프라인-")은 클래스가 아니다.
    """
    s = (s or "").strip()
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
            r"[A-Za-z][A-Za-z0-9\-]{0,5}[A-Za-z0-9]|[A-Za-z]", code) else None

    m = re.search(r"\(\s*(?:종류)?\s*([A-Za-z][A-Za-z0-9\-]{0,6})\s*\)\s*$", s)
    if m:
        return _ok(m.group(1))
    m = re.fullmatch(r"(?:종류|Class)?\s*([A-Za-z][A-Za-z0-9\-]{0,6})", s)
    if m:
        return _ok(m.group(1))
    # ★ 코드 바로 뒤에 한글 설명이 붙는 칸 — norm()이 공백을 지운 결과다.
    #     "종류A수수료미징구-오프라인-개인연금" → A
    #   전체일치만 보면 이런 칸이 전부 버려진다(클래스 64→54 회귀의 원인).
    #   설명이 **먼저** 오는 칸("수수료미징구-…")은 여전히 버린다 — 시작이 한글이라
    #   아래 패턴에 걸리지 않는다.
    m = re.match(r"^(?:종류|Class)?\s*([A-Za-z][A-Za-z0-9\-]{0,6})(?=[가-힣])", s)
    if m:
        return _ok(m.group(1))
    return None


def classify_front(pages, npages=12):
    """
    ★ 상품분류 폴백 — 요약정보 섹션이 없어 라벨을 못 찾는 문서용(22건 중 19건).
    라벨을 찾는 대신 값의 형태로 직접 인식한다. '증권(○형)'이 포함돼야 하므로
    일반 서술문이 걸릴 위험은 낮다.
    """
    for p in pages[:npages]:
        for line in p["text"].split("\n"):
            L = line.strip()
            if CLASSIFY_VAL.match(L) and re.search(r"증권형?\s*\(.{1,10}형", L):
                return L[:90]
    return None


def date_front(pages, npages=10):
    """
    ★ 작성기준일 폴백 — 요약정보 밖(표지·앞장)에만 있는 문서용(23건 중 21건).
    ⚠️ 표기가 갈린다:
        "작성기준일 : 2025.10.31" / "작성기준일: 2025. 09. 04"
        "기 준 일 : 2025년 9월 25"        ← 라벨에 공백이 끼어든다
        "작성기준일   2025년 5월 25"      ← 콜론 없이 공백 정렬
    """
    PAT = re.compile(r"(?:작성\s*)?기\s*준\s*일\s*[:：]?\s*"
                     r"([0-9]{4}\s*[년.\-/]\s*[0-9]{1,2}\s*[월.\-/]\s*[0-9]{1,2})")
    for p in pages[:npages]:
        for line in p["text"].split("\n"):
            m = PAT.search(line)
            if m:
                return re.sub(r"\s+", "", m.group(1))
    return None


def grade_front(pages, npages=6):
    """
    ★ 위험등급 폴백 — 등급 문구가 표지(1p)에만 있는 문서용 (22건 중 17건).
      "…위험도를 감안하여 5등급으로 분류하였습니다"
    ⚠️ 같은 페이지에 '투기등급자산', '신용등급의 하락' 등 무관한 '등급'이 섞여 있어
       느슨한 패턴은 쓸 수 없다. parse_grade의 확정 패턴만 사용한다.
    """
    for p in pages[:npages]:
        g, label = parse_grade(p["text"])
        if g and 1 <= g <= 6:
            return g, label, parse_grade_basis(p["text"])
    return None, None, None


def company_front(pages, npages=3):
    """
    ★ 운용사 폴백 — 집합투자업자 명칭이 표지(1p)에만 있는 문서용.
      "2. 집합투자업자의 명칭 : 우리자산운용㈜"
    2패스에서 표지를 섹션 검출 대상에서 뺀 뒤로 part1 범위에 안 들어온다.
    """
    LAB = [r"집합투자업자\s*(?:의\s*)?명칭\s*[:：]\s*(.{2,30})",
           r"집합투자업자\s*(?:의\s*)?명칭\s{2,}(.{2,30})"]     # 공백 정렬형
    TAIL = r"(?:자산운용|투자신탁운용|투신운용|자산운용사)"
    for p in pages[:npages]:
        for line in p["text"].split("\n"):
            for pat in LAB:
                m = re.search(pat, line)
                if m and re.search(TAIL, m.group(1)):
                    return m.group(1).strip()
        ns = norm(p["text"])                      # 자간 깨짐 대응
        m = re.search(r"집합투자업자의?명칭[:：]?([가-힣A-Za-z]{2,14}" + TAIL
                      + r"[^\s,(]{0,8})", ns)
        if m:
            return m.group(1)
    return None


def classes_from_coord(pdf_path, rng):
    """
    ★ 3차 폴백 — 13항 수수료 표에서 좌표 재조립으로 클래스 목록을 뽑는다.

    ⚠️ 일부 문서는 13항 텍스트가 2~3글자씩 세로로 분해된다:
         '구분' '투자' '기타' '총보수･' … '수수료미징구' (숫자 없음)
       extract_tables()도 텍스트 줄 파싱도 실패하지만, pass6의 reassemble()로
       좌표 복원하면 클래스 행이 살아난다:
         "수수료미징구-오프라인-개인연금(C)없음없음없음" → C

    보수 **숫자**는 헤더가 깨져 매핑 불가지만, 클래스 목록과 설명(개인연금/
    퇴직연금)은 온전하다. 연금 적격 판정이 이 설명에서 나오므로 가치가 크다.
    """
    import pdfplumber
    from parse.pass6_broken import reassemble
    out, seen = [], set()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            lo = max(1, rng[0] - 1)                 # 헤더가 앞 페이지에 걸치는 경우
            hi = min(rng[1], len(pdf.pages))
            for i in range(lo, hi + 1):
                for line in reassemble(pdf.pages[i - 1]):
                    L = line.replace(" ", "")
                    # ① 괄호형: "수수료미징구-오프라인-개인연금(C)없음없음"
                    m = re.match(r"^(수수료[^()]{2,30}?)\(([A-Za-z][A-Za-z0-9\-]{0,6})\)", L)
                    if m:
                        c, desc = m.group(2), m.group(1)
                    else:
                        # ② 코드가 줄 맨 앞: "C-P오프라인-판매수수료및보수ᆞ비용"
                        #    설명이 별도 줄이라 연금 판정은 코드 규칙(-P/-P2)에 의존
                        m = re.match(r"^([A-Za-z](?:[A-Za-z0-9\-]{0,5}[A-Za-z0-9])?)"
                                     r"(?:오프라인|온라인슈퍼|온라인)?[-]?판매수수료및보수", L)
                        if not m:
                            continue
                        c, desc = m.group(1), ""
                    if c in seen:
                        continue
                    seen.add(c)
                    out.append({"class": c, "설명": desc[:40],
                                "협회펀드코드": None, "출처": "13항좌표"})
    except Exception:
        return []
    return out




# ══════════════════════════════════════════════════════════════
# ★ 4차 폴백 — 13항 표가 **칸 단위로 세로 분해**된 문서
#
# 좌표 재조립(3차)도 실패하는 문서들이 있다. 원인은 PDF 표의 한 칸이
# 여러 줄로 흘러나오는 것이고, 운용사마다 형태가 다르다:
#
#   신영    'A(수수료' + '선취-오프' + '라인)'        ← 칸이 3줄로 쪼개짐
#   KB      'C' / '수수료미징구-오프라인'              ← 코드와 설명이 다른 줄
#   코레이트 '수수료미징구-' + '오프라인(C)'            ← 설명이 2줄, 코드가 끝
#
# 줄을 다시 붙이면(merge_cells) 셋 다 읽을 수 있는 형태가 된다. PDF 불필요.
#
# ⚠️ **클래스가 하나도 없는 문서에만** 적용한다.
#    기존 목록이 있는 문서에 돌려보니 다른 집합을 찾는다(11개 → 1개인 사례).
#    어느 쪽이 맞는지 모르는 상태에서 덮으면 회귀가 된다.
# ══════════════════════════════════════════════════════════════
_CS_CODE = r"[A-Za-z][A-Za-z0-9\-]{0,7}"
_CS_DESC = re.compile(r"^수수료(미징구|선취|후취)")
_CS_FRONT = re.compile(rf"^({_CS_CODE})\s*\(\s*(수수료.*?)\s*\)\s*$")
_CS_BACK = re.compile(rf"^(수수료.*?)\s*\(\s*(?:종류)?\s*({_CS_CODE})\s*\)\s*$")
_CS_BARE = re.compile(rf"^(?:종류)?\s*({_CS_CODE})$")
_CS_STOP = re.compile(r"^(선취판매수수료|후취판매수수료|환매수수료|전환수수료|기타비용|"
                      r"총보수|판매보수|운용보수|수탁보수|사무관리)")


def _cs_open(cand):
    """이 칸이 아직 안 끝났는가."""
    if cand.count("(") > cand.count(")"):
        return True
    # ⚠️ '-' 한 글자는 빈 칸 표시지 이어지는 칸이 아니다.
    #    이어붙이면 다음 클래스 코드까지 삼켜 'C'가 '-C'가 된다.
    return cand.endswith("-") and len(cand) >= 3 and re.search(r"[가-힣]", cand)


def merge_cells(lines):
    """줄바꿈으로 쪼개진 표의 한 칸을 다시 붙인다."""
    out, buf = [], ""
    for raw in lines:
        L = raw.strip()
        if not L:
            continue
        # ★ 하이픈으로 **시작**하는 줄도 앞 칸의 꼬리다(흥국 형식):
        #     '수수료선취' + '-오프라인(A)'
        #     '수수료미징구' + '-오프라인' + '-퇴직연금(C)'
        #   앞 칸이 하이픈으로 끝나는 경우(_cs_open)와 반대 방향이다.
        if L.startswith("-") and len(L) > 1 and not buf and out and out[-1].startswith("수수료"):
            out[-1] = out[-1] + L
            if _cs_open(out[-1]):
                buf = out.pop()
            continue
        # 괄호로 시작하는 짧은 줄은 앞 칸의 꼬리다: '무권유저비용' + '(C-G)'
        if L.startswith("(") and not buf and out and len(L) <= 12:
            out[-1] = out[-1] + L
            continue
        cand = (buf + L) if buf else L
        if _cs_open(cand):
            buf = cand
            if len(buf) > 60:
                out.append(buf)
                buf = ""
            continue
        if buf:
            out.append(cand)
            buf = ""
        else:
            out.append(L)
    if buf:
        out.append(buf)
    return out


def classes_from_cellstream(lines):
    """복원한 칸에서 (클래스, 설명) 목록을 뽑는다."""
    found, pending = [], None
    for c in merge_cells(lines):
        s = c.replace(" ", "")
        if _CS_STOP.match(s):
            pending = None
            continue
        m = _CS_FRONT.match(s)
        if m:
            found.append((m.group(1), m.group(2)))
            pending = None
            continue
        m = _CS_BACK.match(s)
        if m:
            found.append((m.group(2), m.group(1)))
            pending = None
            continue
        m = _CS_BARE.match(s)
        if m:
            pending = m.group(1)
            continue
        if pending and _CS_DESC.match(s):
            found.append((pending, s[:40]))
            pending = None
            continue
        if len(s) > 3:
            pending = None
    uniq = []
    for cls, desc in found:
        c = normalize_class(cls) or cls
        if c not in [x[0] for x in uniq]:
            uniq.append((c, desc))
    return uniq


def extract(dump, sec, npages, fee_rows=None, pdf_path=None):
    pages = dump["pages"]
    rec = {"doc_id": dump["doc_id"], "fund_codes": dump["fund_codes"], "pages": npages}

    summary = text_of(pages, span(sec, "요약정보", npages, 4))
    part1   = text_of(pages, span(sec, "제1부1항", npages, 3))
    risk10  = text_of(pages, span(sec, "투자위험", npages, 2))

    # ── 이름 ──
    # ── 펀드명 ★ 운용사별 형식 4가지 ──
    #   ① "집합투자기구 명칭 : ○○신탁"            (라벨형)
    #   ② "○○증권투자신탁[채권] 61842"            (표 형식, 코드가 뒤)
    #   ③ "○○투자신탁 1호(국공채) (펀드코드 : 31405)"  (한국투자)
    #   ④ "이 투자설명서는 ○○신탁에 대한 …"        (폴백)
    # ⚠️ "집합투자기구 명칭(종류형 명칭) 금융투자협회 펀드코드"는 표 헤더 — 배제
    NAME_TAIL = r"(?:투자신탁|투자회사|모투자신탁)"
    fname = None
    for src in (part1, summary):
        if not src:
            continue
        for line in src.split("\n"):
            L = line.strip()
            if not L or "펀드코드" in L and NAME_TAIL not in L:
                continue
            if "종류형 명칭" in L:
                continue
            # ③ 펀드코드 괄호형
            m = re.match(r"^([가-힣A-Za-z].{3,60}?" + NAME_TAIL + r"[^\(]{0,25})\s*\(\s*펀드코드", L)
            if m:
                fname = m.group(1); break
            # ① 라벨형
            m = re.match(r"^집합투자기구\s*(?:의\s*)?명칭\s*[:：]\s*(.+)$", L)
            if m and "펀드코드" not in m.group(1):
                fname = m.group(1); break
            # ② 표 형식 — 이름 + 코드
            m = re.match(r"^([가-힣A-Za-z].{3,60}?" + NAME_TAIL + r"[^\s]{0,25})\s+[A-Z0-9]{5,6}\s*$", L)
            if m:
                fname = m.group(1); break
        if fname:
            break
    if not fname:
        # ④ 폴백
        for pat in (r"이\s*투자설명서는\s*(.+?)에\s*대한",
                    r"이\s*요약정보는\s*[''\"]?(.+?)[''\"]?\s*의\s*투자설명서"):
            m = re.search(pat, flat(summary))
            if m:
                fname = m.group(1); break
    if not fname:
        fname = name_front(pages)      # ⑤ 앞장 라벨형 (콜론 없는 공백정렬·자간깨짐 포함)
    if not fname:
        fname = name_header(pages)     # ⑥ 머리글 반복 빈도
    rec["fund_name"] = re.sub(r"\s+", " ", fname).strip()[:70] if fname else None

    # ── 운용사 (표기 정규화) ──
    m = re.search(r"집합투자업자\s*(?:의\s*)?명칭\s*[:：]?\s*(.+)", flat(part1 or summary))
    if not m:
        m = re.search(r"([가-힣A-Za-z\-]{2,14}(?:자산운용|투자신탁운용|투신운용)[^\s,\(]{0,12})",
                      flat(summary))
    cname = m.group(1) if m else company_front(pages)   # ★ 표지 폴백
    rec["운용사"] = norm_company(cname) if cname else None

    # ── 위험등급 ★ 연혁 배제 + VaR 교차검증 ──
    # 반드시 요약정보 범위 안에서만 뽑는다. 연혁(12~14p)에 과거 등급이 있다.
    g, label = parse_grade(summary)
    basis = parse_grade_basis(summary)
    if g is None:
        # ★ 폴백 — 등급 문구가 표지에만 있는 문서 (요약정보 섹션이 없는 21건 등)
        g, label, b2 = grade_front(pages)
        basis = basis or b2
        if g:
            rec["위험등급_출처"] = "표지"
    rec["위험등급"], rec["위험라벨"] = g, label
    rec["등급산정기준"] = basis
    # ★ VaR 문구는 요약정보·투자위험 밖에도 있다. 두 섹션은 73/92·76/92만 검출되고,
    #   "…(97.5% VaR)은 52.95%로서 위험등급 6등급 중 1등급에 해당합니다" 같은
    #   서술문이 위험등급 설명 문단(앞쪽 페이지)에 있는 문서가 많다.
    #   ⚠️ 연혁에는 과거 등급·변동성이 섞여 있으므로 연혁 페이지는 제외한다.
    # ★ VaR 서술문의 실제 위치는 11항 매입·환매 근처다(미검출 35건 중 30건).
    #   "…(97.5% VaR)은 52.95%로서 위험등급 6등급 중 1등급에 해당합니다"
    #   위험등급 산정 근거 문단이 11항 뒤에 붙는 형식이 많다. 중앙값 22p라
    #   '앞 10페이지' 폴백으로는 1건밖에 못 잡았다.
    #   ⚠️ 연혁은 과거 등급·변동성이 섞여 있어 제외한다.
    var = parse_var(summary) or parse_var(risk10)
    if var is None:
        var = parse_var(text_of(pages, span(sec, "11항매입환매", npages, 6)))
    if var is None:
        hist = sec.get("연혁")
        for _p in pages:
            if hist and _p["page"] == hist:
                continue
            var = parse_var(_p["text"])
            if var is not None:
                break
    rec["VaR_97.5"] = var
    gv = grade_from_var(var)
    rec["등급_VaR역산"] = gv
    # ⚠️ 자산 기준(설정 3년 미만) 문서는 변동성과 무관하게 등급을 매기므로
    #    VaR 역산 교차검증을 적용하면 안 된다 — 불일치로 세지 않는다.
    rec["등급_불일치"] = (g is not None and gv is not None and g != gv
                       and basis != "자산기준")

    # ── 분류 / 법적구조 ──
    m = re.search(r"\n\s*분류\s*[:：]?\s*(.+)", flat(summary))
    rec["상품분류"] = m.group(1).strip()[:90] if m else None
    if not rec["상품분류"]:
        rec["상품분류"] = classify_front(pages)      # ★ 라벨 없는 문서 폴백
        if rec["상품분류"]:
            rec["상품분류_출처"] = "본문형태"
    # ── 자산유형 (분류 문자열에서 파생) ──
    cls = rec["상품분류"] or ""
    m2 = re.search(r"증권형?\s*\(?\s*(주식혼합|채권혼합|주식|채권|재간접|파생|혼합)", cls)
    rec["자산유형"] = m2.group(1) if m2 else None
    rec["법적구조"] = "투자회사" if "투자회사" in (rec["fund_name"] or "") else "투자신탁"
    rec["is_모투자신탁"] = "모투자신탁" in (rec["fund_name"] or "")

    # ── 작성기준일 ──
    m = re.search(r"작성\s*기준일\s*[:：]?\s*([0-9]{4}[년.\-/ ]+[0-9]{1,2}[월.\-/ ]+[0-9]{1,2})",
                  flat(summary))
    rec["작성기준일"] = m.group(1).strip() if m else None
    if not rec["작성기준일"]:
        rec["작성기준일"] = date_front(pages)      # ★ 표지·앞장 폴백

    # ── 클래스 + 협회 펀드코드 (제1부 1항이 정본) ★ ──
    # 운용사별로 줄 형식이 3가지다:
    #   A형 미래에셋 : "종류C-P  수수료미징구-오프라인-개인연금  CO698"
    #   B형 하나·키움: "C-P  수수료미징구-오프라인-개인연금  BL297"   ('종류' 없음)
    #   C형 한국투자 : "수수료미징구-오프라인-개인연금 (C)  AQ443"     (설명이 앞, 클래스가 괄호)
    CLASS_PATS = [
        # A형·B형 — 클래스명이 앞
        (r"^(?:종류)?\s*([A-Za-z][A-Za-z0-9\-]{0,7})\s+(수수료[^\s]*[^\s]*.*?)\s+([A-Z0-9]{5,6})\s*$", "front"),
        # C형 — 설명이 앞, 클래스가 괄호 안:  "수수료미징구-오프라인-개인연금 (C)  AQ443"
        (r"^(수수료.*?)\s*\(\s*([A-Za-z][A-Za-z0-9\-]{0,7})\s*\)\s+([A-Z0-9]{5,6})\s*$", "paren"),
        # D형 신한 — 괄호 안에 '종류'가 붙음:  "수수료선취-오프라인(종류A1)  AO422"
        (r"^(수수료.*?)\s*\(\s*종류\s*([A-Za-z][A-Za-z0-9\-]{0,7})\s*\)\s+([A-Z0-9]{5,6})\s*$", "paren"),
    ]
    cls = []
    for line in part1.split("\n"):
        L = line.strip()
        if re.match(r"^2\s*[\.．]?\s*집합투자기구의?\s*연혁", L):
            break                              # 연혁이 같은 페이지에 이어짐
        for pat, kind in CLASS_PATS:
            m = re.match(pat, L)
            if not m:
                continue
            if kind == "front":
                cname, desc, code = m.group(1), m.group(2), m.group(3)
            else:
                desc, cname, code = m.group(1), m.group(2), m.group(3)
            cname = normalize_class(cname) or cname
            cls.append({"class": cname, "설명": desc.strip()[:40], "협회펀드코드": code})
            break
    seen, uniq = set(), []
    for c in cls:
        if c["class"] in seen: continue
        seen.add(c["class"]); uniq.append(c)
    # ★ 폴백 — 제1부1항 헤딩이 없는 문서는 13항 보수표에서 클래스를 가져온다
    if not uniq and fee_rows:
        seenf = set()
        for r in fee_rows:
            c = normalize_class(r.get("class"))      # ★ 설명 문구 오염 차단
            if c and c not in seenf:
                seenf.add(c)
                uniq.append({"class": c, "설명": r.get("설명", ""),
                             "협회펀드코드": None, "출처": "13항"})
    # ★ 3차 폴백 — 13항 텍스트가 세로 분해된 문서는 좌표 재조립으로 클래스를 얻는다
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
                uniq = coord
    # ★ 4차 폴백 — 칸 단위 세로 분해. 클래스가 **아예 없을 때만** 적용한다.
    if not uniq:
        rng13 = span(sec, "13항보수", npages, 8)
        if rng13:
            ls = text_of(pages, rng13).split("\n")
            for i, l in enumerate(ls):        # ★ 13항 헤딩 뒤부터 (앞은 평가방법 표)
                if re.match(r"^\s*13\s*[\.．]\s*보수", l):
                    ls = ls[i + 1:]
                    break
            cs = classes_from_cellstream(ls)
            # ★ 원문 대조 — 문서 전체에서 2회 미만 등장하는 코드는 버린다.
            #   칸을 잘못 이어붙이면 'Ae' 'e' 'Cy' 같은 없는 코드가 만들어진다.
            #   실제로 한 운용사 4개 문서가 똑같은 목록을 뱉었고 그중 절반이 0회였다.
            whole = re.sub(r"\s+", "", "\n".join(p_["text"] for p_ in pages))
            cs = [(c, d) for c, d in cs
                  if whole.count(f"종류{c}") + whole.count(f"({c})") >= 2]
            uniq = [{"class": c, "설명": d, "협회펀드코드": None, "출처": "13항셀"}
                    for c, d in cs]
    rec["클래스"] = uniq

    # 연금 클래스 판정 — 설명의 '개인연금/퇴직연금'이 1순위, 클래스명은 보조
    def pension_type(c):
        d = c.get("설명", "")
        if "퇴직연금" in d: return "퇴직연금"
        if "개인연금" in d: return "개인연금"
        if re.search(r"P2", c["class"], re.I): return "퇴직연금"
        if re.search(r"-P(E|e)?$", c["class"]): return "개인연금"
        return None
    for c in uniq:
        c["연금유형"] = pension_type(c)
    rec["연금저축_가능"] = any(c["연금유형"] == "개인연금" for c in uniq)
    rec["퇴직연금_가능"] = any(c["연금유형"] == "퇴직연금" for c in uniq)

    # ── 비교지수 (투자전략이 정본, 요약본은 보조) ──
    strat = text_of(pages, span(sec, "투자전략", npages, 3))
    m = re.search(r"(?:비교|참조)지수\s*[:：]?\s*([^\n(<]{3,60})", flat(strat) or "")
    if not m:
        m = re.search(r"(?:비교|참조)지수\s*[:：]?\s*([^\n(<]{3,60})", flat(summary))
    rec["비교지수"] = m.group(1).strip() if m else None

    rec["src"] = {k: sec.get(k) for k in
                  ["요약정보", "제1부1항", "투자전략", "13항보수", "제3부"]}
    return rec


def extract_fees(path, sec, npages, dump_pages=None, known_classes=None):
    """
    13항 보수 표. 헤더가 텍스트로 살아있어 bbox 불필요.

    ⚠️ 함정 3가지
      · 헤더가 2~3행에 걸쳐 있다 → 상위 3행을 열별로 합침
      · '총 보수' / '총보수·비용' / '합성총보수' 등 표기가 다양 → 공백 제거 후 매칭
      · '동종유형총보수'는 평균값이지 이 펀드 값이 아니다 → 배제
      · extract_tables()가 표를 못 잡는 문서 존재 → 텍스트 줄 폴백
    """
    import pdfplumber
    rng = span(sec, "13항보수", npages, 6)
    if not rng:
        return []
    a, b = rng
    rows = []

    def pick_total_col(head):
        """'총보수' 컬럼 인덱스. 동종유형·합성은 후순위."""
        pure = [j for j, h in enumerate(head)
                if "총보수" in h and "동종" not in h and "합성" not in h]
        if pure:
            return pure[0]
        syn = [j for j, h in enumerate(head) if "총보수" in h and "동종" not in h]
        return syn[0] if syn else None

    # ── 1차: 표 파싱 ──
    with pdfplumber.open(path) as pdf:
        for i in range(a, min(b, len(pdf.pages)) + 1):
            for tb in (pdf.pages[i - 1].extract_tables() or []):
                if not tb or len(tb) < 2:
                    continue
                ncol = max(len(r) for r in tb[:3])
                head = []
                for j in range(ncol):
                    head.append(norm("".join((tb[k][j] or "")
                                for k in range(min(3, len(tb))) if j < len(tb[k]))))
                tot = pick_total_col(head)
                if tot is None:
                    continue
                for r in tb[2:]:
                    if not r or not r[0]:
                        continue
                    cname = norm(r[0])
                    val = r[tot] if tot < len(r) else None
                    if val and re.match(r"^[\d.]+$", norm(val)):
                        # ★ 첫 칸이 설명 문구인 문서가 많다("수수료미징구-오프라인-…").
                        #   코드로 볼 수 없으면 class를 비우고 아래에서 순서 인덱스를 준다.
                        #   값 자체는 정확하므로 버리지 않는다.
                        c = normalize_class(cname)
                        rows.append({"class": c,
                                     "설명": cname[:40],
                                     "총보수": float(norm(val)),
                                     "출처": "table",
                                     "클래스명미상": c is None})
    for i, r in enumerate(rows):
        if r["class"] is None:
            r["class"] = f"#{i+1}"
    if rows:
        return rows

    # ── 2차: 전치(transpose) 표 ──
    # 행이 클래스가 아니라 **보수 항목**이고, 열이 클래스인 형식이 있다.
    #   집합투자업자보수  0.0750 0.0750 0.0750 …
    #   판매회사보수      0.7000 0.5000 0.3000 …
    #   총보수            0.8200 0.6200 0.4200 …
    # → '총보수' 행의 n번째 값이 n번째 클래스의 총보수다.
    if dump_pages:
        head_classes, total_row = [], None
        for p in dump_pages:
            if not (a <= p["page"] <= b):
                continue
            for line in p["text"].split("\n"):
                L = line.strip()
                # 클래스 헤더 줄. 앞에 '구분/종류' 라벨이 붙거나 줄바꿈으로 잘리기도 한다.
                #   "종류A 종류C 종류C-P …"  /  "구분 A C C-P …"  /  "A C C-P …"
                if not head_classes:
                    toks = re.sub(r"^(구분|종류|클래스)\s+", "", L).split()
                    cls_toks = [t for t in toks
                                if re.fullmatch(r"(?:종류)?[A-Za-z][A-Za-z0-9\-]{0,7}", t)]
                    # 토큰 대부분이 클래스명 형태이고 2개 이상이면 헤더로 본다
                    if len(cls_toks) >= 2 and len(cls_toks) >= len(toks) * 0.8:
                        head_classes = [re.sub(r"^종류", "", t) for t in cls_toks]
                        continue
                m = re.match(r"^(?:합성\s*)?총\s*보수[^\d]{0,12}((?:[\d.]+\s+){1,14}[\d.]+)\s*$", L)
                if m and total_row is None:
                    total_row = [float(x) for x in m.group(1).split()]
        # 헤더를 못 찾았으면 이미 파싱된 클래스 목록으로 매핑한다 (순서가 같다)
        if not head_classes and known_classes and total_row:
            if len(known_classes) == len(total_row):
                head_classes = known_classes
        if head_classes and total_row:
            n = min(len(head_classes), len(total_row))
            for c, v in zip(head_classes[:n], total_row[:n]):
                rows.append({"class": c, "총보수": v, "출처": "transpose"})
            return rows

    # ── 3차 폴백: 텍스트 줄 파싱 ──
    # "종류A  0.030  0.700  0.030  0.020  0.030  0.813  0.900"
    # 앞이 클래스명, 뒤가 숫자 나열. 총보수는 보통 뒤에서 2번째(동종유형 앞).
    if dump_pages:
        for p in dump_pages:
            if not (a <= p["page"] <= b):
                continue
            for line in p["text"].split("\n"):
                L = line.strip()
                # 실제 줄: "C-P 0.2 0.3 0.02 0.015 0.535 0.0070 0.5420 - 0.5518 0.0016"
                #   · 정수(0.2)와 소수(0.0070)가 섞임
                #   · '-'(해당없음)가 중간에 낀다
                m = re.match(r"^(?:종류)?\s*([A-Za-z][A-Za-z0-9\-]{0,7})"
                             r"\s+((?:(?:\d+(?:\.\d+)?|-)\s+){3,10}"
                             r"(?:\d+(?:\.\d+)?|-))\s*$", L)
                if not m:
                    continue
                toks = m.group(2).split()
                nums = [None if t == "-" else float(t) for t in toks]
                real = [x for x in nums if x is not None]
                if len(real) < 4:
                    continue
                # 총보수는 보통 뒤에서 2번째 실수값(맨 뒤는 증권거래비용 또는 동종유형)
                rows.append({"class": m.group(1),
                             "총보수": real[-2],
                             "전체값": nums,
                             "출처": "text"})
    return rows


def main():
    docs = {d["doc_id"]: d for d in load_json(os.path.join(TABLE_DIR, "docs.json"))}
    smap = load_json(os.path.join(TABLE_DIR, "section_map.json"))
    out = []
    for f in sorted(glob.glob(os.path.join(DUMP_DIR, "*.json"))):
        dump = load_json(f)
        did = dump["doc_id"]
        info = smap.get(did, {})
        sec, npages = info.get("sec", {}), info.get("pages", len(dump["pages"]))
        # 1) 보수를 먼저 뽑고 → 2) 클래스 파싱 → 3) 클래스를 알았으면 보수 재시도
        try:
            fees = extract_fees(docs[did]["path"], sec, npages,
                                dump_pages=dump["pages"])
        except Exception as e:
            fees, err = [], str(e)[:60]
        else:
            err = None
        rec = extract(dump, sec, npages, fee_rows=fees,
                      pdf_path=docs[did]["path"] if did in docs else None)
        if not fees and rec.get("클래스"):
            try:
                fees = extract_fees(docs[did]["path"], sec, npages,
                                    dump_pages=dump["pages"],
                                    known_classes=[c["class"] for c in rec["클래스"]])
            except Exception:
                pass
        rec["보수"] = fees
        # ★ 클래스 매핑이 없는 보수 행이 하나라도 있으면 문서 단위로 표시한다.
        #   pass8 _fee_line / pass9 _fee_map 의 방어가 이 플래그로 작동한다.
        if any(r.get("클래스명미상") for r in fees):
            rec["보수_신뢰불가"] = True
        if err:
            rec["보수_err"] = err
        out.append(rec)
    save_json(out, os.path.join(TABLE_DIR, "six_axis.json"))
    report(out)


def report(out):
    n = len(out)
    print(f"\n6축 테이블 {n}행")
    for k in ["fund_name", "운용사", "위험등급", "상품분류", "작성기준일", "비교지수"]:
        c = sum(1 for r in out if r.get(k))
        print(f"  {k:<12} {c:>3}/{n}")
    print(f"  {'클래스≥1':<12} {sum(1 for r in out if r.get('클래스')):>3}/{n}")
    print(f"  {'보수행≥1':<12} {sum(1 for r in out if r.get('보수')):>3}/{n}")
    bad = [r["doc_id"] for r in out if r.get("등급_불일치")]
    if bad:
        print(f"\n⚠️ 등급 vs VaR 불일치 {len(bad)}건 — 연혁 오염 의심: {bad[:5]}")
    from collections import Counter
    print("\n등급 분포:", sorted(Counter(r["위험등급"] for r in out).items(),
                              key=lambda t: (t[0] is None, t[0])))


if __name__ == "__main__":
    main()