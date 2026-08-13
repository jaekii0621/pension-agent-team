"""공통 상수·유틸. 모든 패스가 여기를 참조한다."""
import re, json, hashlib, os, glob

# ── 경로 ──────────────────────────────────────────────
RAW_DIR   = os.environ.get("RAW_DIR", "data/raw/prospectus")  # KR*/R2_*.pdf
DUMP_DIR  = "data/dump"
TABLE_DIR = "data/tables"

# ── 텍스트 정규화 ─────────────────────────────────────
def norm(s: str) -> str:
    """공백 제거. 앵커 매칭 전 필수 — 자간 삽입(투 자 설 명 서)이 흔하다."""
    return re.sub(r"\s+", "", s or "")

def flat(s: str) -> str:
    """줄바꿈은 살리고 가로 공백만 정리."""
    return re.sub(r"[ \t]+", " ", s or "")

# ── 목차 판정 ★ 두 조건을 OR로 ─────────────────────────
def is_toc(text: str) -> bool:
    """
    목차 페이지 판정 — 오검출·미검출 양쪽을 다 막아야 한다.

    ⚠️ "번호 헤딩이 여러 개면 목차" 규칙은 위험하다.
       실제 본문 섹션 페이지에도 하위 번호(1) 2) …)가 많이 나온다.
    목차의 진짜 특징은 **번호 헤딩 뒤에 페이지 번호가 붙는다**는 것이다.
        "13. 보수 및 수수료에 관한 사항 ......... 26"
        "13. 보수 및 수수료에 관한 사항        26"
    """
    if len(re.findall(r"제\s*\d\s*부", text)) >= 2:
        return True
    if text.count(". 집합투자기구의") >= 4:
        return True
    # ① 번호 헤딩 + 줄 끝 페이지번호 조합이 3개 이상 (점선/공백 목차)
    toc_lines = re.findall(
        r"^\s*\d{1,2}\s*[\.．][^\n]{2,50}?[\.\s]{2,}\d{1,3}\s*$", text, re.M)
    if len(toc_lines) >= 3:
        return True

    # ② 페이지번호가 아예 없는 목차 (삼성 형식)
    #    번호 헤딩(N.)이 4개 이상이고, 가/나/다 하위항목이 5개 이상이며,
    #    본문 문장(40자 초과 줄)이 거의 없다 → 목차
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        num_heads = sum(1 for l in lines if re.match(r"^\d{1,2}\s*[\.．]\s*\S", l))
        sub_heads = sum(1 for l in lines if re.match(r"^[가-하]\s*[\.．]\s*\S", l))
        long_lines = sum(1 for l in lines if len(l) > 40)
        if num_heads >= 4 and sub_heads >= 5 and long_lines <= len(lines) * 0.15:
            return True
        # ③ 시도했다가 **철회**한 조건 — 다시 넣지 말 것.
        #    "제N부 + 번호헤딩 4개 이상 + 긴 줄 없음"으로 부(部)별 목차를 잡으려 했으나,
        #    보수표(13항)·재무정보(제3부) 페이지가 같은 형태여서 각각 11건·9건이
        #    통째로 목차로 오판됐다. 검출률: 13항보수 92→81, 제3부 91→83.
        #    부(部) 목차 오검출은 pass2의 섹션 순서 검증(제N부 역전 제거)으로 처리한다.
    return False


# ── 운용사명 정규화 (31종 표기 → 실제 26개사) ──────────
def norm_company(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"^[:：\s]+", "", s)
    s = re.sub(r"\(?[☎(]?\s*0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}\s*\)?", "", s)  # 전화번호
    s = re.sub(r"(주식회사|㈜|\(주\)|주식회사)\s*$", "", s)
    s = re.sub(r"(는|은|이|가)$", "", s)                                       # 조사
    return s.strip()

# ── 위험등급 표기 20종 대응 ★ ─────────────────────────
_GRADE_PATTERNS = [
    r"(\d)\s*등급\s*[\[\(]\s*([^\]\)]{2,12}?)\s*[\]\)]",   # 5등급[낮은 위험] / 5등급 (낮은 위험)
    r"감안하여\s*(\d)\s*등급으로\s*분류",                     # 신한 — 라벨 분리형
    r"투자위험등급\s*[:：]?\s*(\d)\s*등급",                   # 키움 — 콜론
    # ★ 줄바꿈으로 '분류'가 잘리는 문서 — flat()은 \n을 남기므로
    #   '…감안하여 5등급으' 에서 끝나면 위 2번 패턴이 실패한다
    r"위험도를?\s*감안하여\s*(\d)\s*등급",
    r"변동성을?\s*감안하여\s*(\d)\s*등급",
]

# ── 등급 산정 기준 ★ VaR 역산 검증의 적용 가능 여부를 가른다 ──
def parse_grade_basis(text: str):
    """
    설정 3년 이상 → 수익률 변동성 기준 / 3년 미만 → 투자대상자산 기준.
    ⚠️ 자산 기준 문서에는 VaR 역산 교차검증을 적용하면 안 된다(원리적으로 무관).
    """
    t = flat(text)
    if re.search(r"투자대상자산의?\s*종류\s*및?\s*위험도", t):
        return "자산기준"
    if re.search(r"(?:수익률\s*)?변동성을?\s*감안", t):
        return "변동성기준"
    return None

def parse_grade(text: str):
    """(등급, 라벨) 반환. 실패 시 (None, None)."""
    t = flat(text)
    for i, pat in enumerate(_GRADE_PATTERNS):
        m = re.search(pat, t)
        if m:
            g = int(m.group(1))
            label = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else None
            return g, label
    return None, None

# ── VaR 기준표 — 등급 교차검증 ★ ──────────────────────
def grade_from_var(var: float):
    """97.5% VaR로 등급 역산. 연혁 오염 검출용."""
    if var is None: return None
    if var > 50: return 1
    if var > 30: return 2
    if var > 20: return 3
    if var > 10: return 4
    if var > 1:  return 5
    return 6

def parse_var(text: str):
    r"""
    변동성(97.5% VaR) 추출.

    ⚠️ [^\d]{0,20}이 줄바꿈을 넘어가면 '투자신탁 재산의 50% 이상' 같은
       무관한 비중 숫자를 문다. 실제로 6개 문서에서 VaR=50.0이 나왔고,
       그 4건이 등급 교차검증 '불일치'로 잡혔다(전부 허위).
       → 같은 줄 안(\n 제외)에서 8자 이내만 본다.
    """
    for pat in (r"변동성\s*\(?97\.5%?\s*VaR\)?[^\n\d]{0,8}([\d.]+)\s*%",
                r"97\.5%\s*VaR[^\n\d]{0,8}([\d.]+)\s*%"):
        m = re.search(pat, flat(text))
        if not m:
            continue
        try:
            v = float(m.group(1).rstrip("."))
        except ValueError:
            continue
        # ★ 비중 표현 오인 방지 — 실제 변동성은 소수점 표기(37.49, 0.54 …)다.
        #   딱 떨어지는 정수 비중값(50/60/80/100)은 버린다.
        if v > 100 or v in (50.0, 60.0, 80.0, 100.0):
            return None
        return v
    return None

def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


# ── 임베딩용 텍스트 ★ 512토큰 절단 대비 ────────────────
#   text(인용용 원문)는 자르지 않는다. 근거 표시가 깨지기 때문이다.
#   대신 임베딩 입력만 따로 만든다. 로컬 임베딩 모델 max_seq_len이 512면
#   한글 70% 기준 대략 600~750자에서 뒤가 통째로 잘려 인덱싱된다.
#   ⚠️ 앞에서 자르면 표 아래 요약줄(연금 가능 클래스 등)이 사라진다.
#      머리(펀드명·섹션)와 꼬리(요약줄)는 남기고 본문 가운데를 줄인다.
EMBED_MAX = 786


def make_embed_text(head, body, tail="", budget=EMBED_MAX):
    head = re.sub(r"\s+", " ", head or "").strip(" ·")
    tail = re.sub(r"[ \t]+", " ", tail or "").strip()
    body = re.sub(r"\s+", " ", body or "").strip()
    room = budget - len(head) - len(tail) - 2
    if room < 60:                       # 머리+꼬리만으로 예산을 다 쓴 경우
        return ("\n".join(x for x in (head, tail) if x))[:budget]
    if len(body) > room:
        cut = body[:room]
        p = max(cut.rfind(". "), cut.rfind("다 "), cut.rfind(" "))
        if p > room * 0.6:              # 단어·문장 경계에서 자른다
            cut = cut[:p]
        body = cut.rstrip() + "…"
    return "\n".join(x for x in (head, body, tail) if x)
