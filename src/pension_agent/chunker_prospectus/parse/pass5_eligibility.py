"""
5패스 — 11항에서 가입자격 + 기준가격 적용일 추출

⚠️ '가입자격'은 별도 섹션 헤딩이 없다. 11항(매입·환매)의 하위 항목이라
   2패스 섹션 지도로는 0/91이 나온다. 11항 범위 안에서 하위 헤딩을 찾아야 한다.

여기가 클래스별 가입자격의 **최상세 정본**이다(INTERFACES.md 1-3l).
  · C-P  = 연금저축계좌 전용
  · C-P2 = 퇴직연금·IRP 전용
  · C1~C4 = 신규 가입 불가 (1년 이상 선행 클래스 보유 필요)
  · I / C-F = 고액(50억↑) 또는 기관 전용 → 개인 추천 금지

산출: six_axis.json 에 병합
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import TABLE_DIR, DUMP_DIR, flat, load_json, save_json
from parse.pass3_extract import span, text_of

# 하위 헤딩 — 번호 형식이 (2)/2)/없음 으로 갈린다
# ⚠️ 표기 변형: "(2) 수익증권 종류별 매입자격" / "나. 종류별가입자격"
ELIG_HEAD = re.compile(r"^\(?\d?\)?\s*(?:종류별\s*)?가입\s*자격")
STOP_HEAD = re.compile(r"^\(?\d\)?\s*(?:매입청구|환매|전환|기준가격)")


def find_eligibility(text):
    """
    11항 본문에서 가입자격 블록을 잘라 클래스별 설명을 뽑는다.
    두 형식:
      A형: "종류C-P  소득세법 제20조의3 … 연금저축계좌를 통하여 가입할 수 있으며"
      B형: "수수료미징구-오프라인-개인연금 (C)  없음  없음"   ← 설명이 앞
    """
    lines = [l.strip() for l in text.split("\n")]
    start = None
    for i, L in enumerate(lines):
        if ELIG_HEAD.match(L):
            start = i + 1
            break
    if start is None:
        return {}

    block, cur = [], None
    for L in lines[start:start + 60]:
        if not L:
            continue
        if STOP_HEAD.match(L):
            break
        # A형 — 클래스가 앞
        m = re.match(r"^종류\s*([A-Za-z][A-Za-z0-9\-]{0,7})\s+(.+)$", L)
        if not m:
            m = re.match(r"^([A-Za-z][A-Za-z0-9\-]{0,7})\s{2,}(.+)$", L)
        if m:
            cur = {"class": m.group(1), "가입자격": m.group(2).strip()}
            block.append(cur)
            continue
        # B형 — 설명이 앞, 클래스가 괄호
        m = re.match(r"^(.*?)\s*\(\s*(?:종류)?\s*([A-Za-z][A-Za-z0-9\-]{0,7})\s*\)", L)
        if m and ("수수료" in m.group(1) or "연금" in m.group(1)):
            cur = {"class": m.group(2), "가입자격": m.group(1).strip()}
            block.append(cur)
            continue
        # 이어지는 설명 줄 (줄바꿈으로 잘린 문장)
        if cur and len(L) > 4 and not re.match(r"^[-─※(]", L):
            cur["가입자격"] = (cur["가입자격"] + " " + L).strip()[:200]

    out = {}
    for b in block:
        out.setdefault(b["class"], b["가입자격"])
    return out


def scan_eligibility(text):
    """
    ★ 헤딩 없는 폴백 — '가입자격' 하위 헤딩이 아예 없는 문서용.

    문서 전체(11항·6항 범위)에서 "클래스 + 가입자격 설명" 줄을 직접 긁는다.
    설명에 가입/전용/투자자/수익증권 같은 단어가 있어야 하며,
    숫자만 나열된 줄(보수 표)은 배제한다.
    """
    KEY = re.compile(r"가입|전용|투자자|수익증권|매입자격|해당하는")
    out = {}
    cur = None
    for line in text.split("\n"):
        L = line.strip()
        if not L or len(L) > 160:
            cur = None
            continue
        m = re.match(r"^(?:종류)?\s*([A-Za-z][A-Za-z0-9\-]{0,7})\s+(.{8,})$", L)
        if m:
            desc = m.group(2).strip()
            # 보수 표 줄 배제 — 숫자와 '-'만으로 이루어진 경우
            if re.fullmatch(r"[\d.,\s\-]+", desc):
                cur = None
                continue
            if KEY.search(desc):
                cur = m.group(1)
                out.setdefault(cur, desc[:200])
                continue
        # 이어지는 설명 줄
        if cur and not re.match(r"^[-─※(주\d]", L) and KEY.search(L):
            out[cur] = (out[cur] + " " + L)[:200]
    return out




# ==============================================================
# ★ 3차 폴백 — 가입자격 표가 **칸 단위로 세로 분해**된 문서
#
#   find_eligibility / scan_eligibility 가 둘 다 실패하는 문서가 많다(511행 미확인).
#   원인은 보수 표와 같다: PDF 표의 한 칸이 여러 줄로 흘러나온다.
#
#     A형   '종류 C-P' / 조건문장 …
#     B형   '종류C-P' / '수수료미징구-…' / 조건문장 / '있음'|'없음'
#
#   '종류X' 줄을 기준으로 다음 '종류Y'까지를 모으고,
#   판매유형·부과여부 칸만 걸러낸다.
# ==============================================================
CLS = r"[A-Za-z][A-Za-z0-9\-]{0,7}"
HEAD = re.compile(rf"^종\s*류\s*({CLS})\s*$")          # '종류 C-P' 단독 줄
# ⚠️ '가. 집합투자기구'처럼 조건 본문이 가/나/다로 시작하는 문서가 있다(C-F 사례).
#   섹션 헤딩만 STOP으로 본다.
STOP = re.compile(r"^(부과기준|주\s*\d+\s*\)|\(\s*\d+\s*\)\s*[가-힣]|\d+\s*[.．]\s*[가-힣]{2,})")
DROP = re.compile(r"^(있음|없음|해당\s*없음|-|—|구\s*분|종\s*류|판매수수료|부과여부|"
                  r"수수료(선취|미징구|후취)[^\s]*)$")
PAGE = re.compile(r"^-?\s*\d{1,3}\s*-?$")
# 실제 가입자격 문장에 나타나는 말들 — 이게 있으면 판매유형 꼬리가 아니다
_COND = re.compile(r"제한|없음|투자자|가입|전용|자격|기관|법|계좌|이상|한하여|대상|"
                   r"온라인|누구나|개인|법인|기금")


def eligibility_cells(lines):
    """{class: 조건문장}. '종류X' 줄 다음부터 다음 '종류Y' 전까지를 모은다."""
    out, cur, buf = {}, None, []
    fee_tail = False        # 직전 줄이 판매유형 칸이면 그 꼬리를 흘려보낸다

    def flush():
        if cur and buf:
            s = " ".join(buf).strip()
            s = re.sub(r"\s+", " ", s)
            if len(s) >= 3:
                out.setdefault(cur, s[:200])

    for raw in lines:
        L = raw.strip()
        if not L or PAGE.match(L):
            continue
        m = HEAD.match(L.replace(" ", "").replace("종류", "종류 ", 1)) or HEAD.match(L)
        if m:
            flush()
            cur, buf = m.group(1), []
            continue
        if STOP.match(L):
            flush()
            cur, buf = None, []
            continue
        if cur is None:
            continue
        # ★ 판매유형 칸이 두 줄로 쪼개진 경우의 꼬리("…무권유저" + "비용")
        #   ⚠️ 짧다고 무조건 버리면 '제한없음', '온라인 투자자' 같은 실제 조건까지 사라진다.
        #   조건 키워드가 하나도 없는 짧은 조각만 꼬리로 본다.
        if fee_tail and not buf and len(L) <= 12 and not _COND.search(L):
            fee_tail = False
            continue
        fee_tail = False
        if DROP.match(L.replace(" ", "")):
            if re.match(r"^수수료(선취|미징구|후취)", L.replace(" ", "")):
                fee_tail = True
            # 판매유형/부과여부 칸은 조건이 아니다. 다만 조건을 이미 모았으면 거기서 끊는다
            if buf:
                flush()
                cur, buf = None, []
            continue
        buf.append(L)
    flush()
    return out

def classify(text):
    """가입자격 문장에서 판정 플래그를 뽑는다."""
    # ★ PDF 줄바꿈이 단어 안에 끼어든다("연금저축계\n좌", "퇴직연\n금").
    #   flat()은 가로 공백만 정리하므로 여기서 모든 공백을 없앤 사본으로 판정한다.
    #   실제로 C-Pe(개인연금)가 퇴직연금으로 뒤집힌 사례가 있었다.
    t = re.sub(r"\s+", "", flat(text))
    return {
        "연금저축": bool(re.search(r"연금저축계좌|개인연금", t)),
        "퇴직연금": bool(re.search(r"퇴직연금|개인형퇴직연금|IRP", t)),
        # C2 = "종류 C1 수익증권의 최초 매수일로부터 1년 이상된 투자자" → 신규 가입 불가
        #   ⚠️ flat()이 공백을 남기므로 '종류C1'과 '종류 C1' 둘 다 걸려야 한다
        "신규가입불가": bool(
            re.search(r"종류\s*[A-Za-z][A-Za-z0-9\-]{0,4}\s*수익증권의?\s*최초\s*매수일", t)
            or re.search(r"\d\s*년\s*이상된?\s*투자자", t)
            or re.search(r"\d\s*년\s*이상\s*종류\s*[A-Za-z]", t)),
        "고액전용": bool(re.search(r"\d+\s*억원\s*이상|기관투자자|고액|기금", t)),
        "온라인전용": bool(re.search(r"온라인", t)),
    }


def find_price_rule(text):
    """
    기준가격 적용일 — 매입·환매 각각.

    ⚠️ 표기 변형이 크다
      · 기준 시각: "17시" / "오후 5시" / "15시 30분"      ← 펀드마다 다르다
      · 영업일 표기: "2영업일" / "제2영업일" / "제2영업일(D+1)"
      · 매입과 환매가 각각 있다 (환매가 보통 1영업일 더 걸린다)
    """
    CUT  = r"(\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?|오후\s*\d{1,2}\s*시)"
    DAY  = r"제?\s*(\d)\s*영업일"
    BEF  = r"(?:이전|이내)"
    AFT  = r"(?:경과|이후)"

    def grab(kind_pat):
        b = re.search(CUT + r"[^\n]{0,12}" + BEF + r"[^\n]{0,60}?" + kind_pat
                      + r"[^\n]{0,40}?" + DAY, text)
        a = re.search(CUT + r"[^\n]{0,12}" + AFT + r"[^\n]{0,60}?" + kind_pat
                      + r"[^\n]{0,40}?" + DAY, text)
        if not (b or a):
            return None
        m0 = b or a
        return {"기준시각": re.sub(r"\s+", "", m0.group(1)),
                "이전": f"{b.group(2)}영업일" if b else None,
                "이후": f"{a.group(2)}영업일" if a else None}

    buy = grab(r"(?:자금을?\s*납입|매입)")
    sell = grab(r"환매")
    if not (buy or sell):
        # 시각 없이 서술만 있는 경우
        m = re.search(r"자금을\s*납입한\s*영업일로부터\s*" + DAY, text)
        if m:
            buy = {"기준시각": None, "이전": f"{m.group(1)}영업일", "이후": None}
    if not (buy or sell):
        return None
    out = {}
    if buy:  out["매입"] = buy
    if sell: out["환매"] = sell
    return out


def locate_pages(pages, elig, info=None):
    """
    ★ 가입자격 문장이 어느 페이지에서 왔는지 되찾는다.

    text_of()가 여러 페이지를 이어붙이면서 페이지 정보가 사라진다.
    문장이 충분히 길고 특징적이므로, 공백 제거 후 앞 24자로 역탐색하면
    안전하게 찾을 수 있다(짧은 값은 오탐 위험이 있어 건너뛴다).
    """
    by = {p["page"]: re.sub(r"\s+", "", p["text"]) for p in pages}
    # ★ 섹션 범위를 먼저 본다. 같은 문구가 목차·다른 항에도 나올 수 있다.
    pref = []
    if info:
        for key, smax in [("11항매입환매", 6), ("제2부6항구조", 6), ("13항보수", 8)]:
            rng = span(info.get("sec", {}), key, info.get("pages", 0), smax)
            if rng:
                pref += list(range(rng[0], rng[1] + 1))
    order = [p for p in pref if p in by] + [p for p in sorted(by) if p not in pref]

    out = {}
    for c, v in elig.items():
        s = re.sub(r"\s+", "", v)
        # ⚠️ 앞 24자 하나로만 찾으면 실패한다. 5패스가 여러 칸을 이어붙인 값이라
        #   그 순서 그대로 원문에 연속해 나오지 않는 경우가 많다(254/335행 실패).
        #   앞·뒤·가장 긴 한글 구간을 모두 후보로 쓴다.
        keys = []
        if len(s) >= 12:
            keys.append(s[:24])
        if len(s) >= 30:
            keys.append(s[-24:])
        keys += sorted(re.findall(r"[가-힣]{12,}", s), key=len, reverse=True)[:2]
        keys = [k for k in dict.fromkeys(keys) if len(k) >= 12]
        if not keys:
            continue
        for pg in order:
            if any(k in by[pg] for k in keys):
                out[c] = pg
                break
    return out


def main():
    six = load_json(os.path.join(TABLE_DIR, "six_axis.json"))
    smap = load_json(os.path.join(TABLE_DIR, "section_map.json"))
    ok_e = ok_p = 0
    for rec in six:
        did = rec["doc_id"]
        dp = os.path.join(DUMP_DIR, did + ".json")
        info = smap.get(did, {})
        sec0 = info.get("sec", {})
        if not os.path.exists(dp) or not ({"11항매입환매", "제2부6항구조"} & set(sec0)):
            continue
        pages = load_json(dp)["pages"]
        sec = info["sec"]
        t = (text_of(pages, span(sec, "11항매입환매", info["pages"], 6))
             if "11항매입환매" in sec else "")

        elig = find_eligibility(t)
        # ★ 폴백 — 일부 문서는 11항에서 "제2부 6. 집합투자기구의 구조 참고"로 넘긴다
        if not elig and "제2부6항구조" in sec:
            t6 = text_of(pages, span(sec, "제2부6항구조", info["pages"], 4))
            elig = find_eligibility(t6)
            t = t + "\n" + t6          # 기준가격 탐색에도 함께 쓴다
        # ★ 2차 폴백 — 하위 헤딩이 아예 없는 문서는 줄 단위로 직접 스캔
        if not elig:
            elig = scan_eligibility(t)
        # ★ 3차 폴백 — 표가 칸 단위로 세로 분해된 문서
        #   ⚠️ 기존 값은 건드리지 않는다. 빠진 클래스만 채운다.
        _cells = eligibility_cells(t.split("\n"))
        for _k, _v in _cells.items():
            elig.setdefault(_k, _v)
        if elig:
            rec["가입자격"] = elig
            rec["가입자격_페이지"] = locate_pages(pages, elig, info)
            flags = {c: classify(v) for c, v in elig.items()}
            rec["가입자격_판정"] = flags
            # 클래스 레코드에 병합
            for c in rec.get("클래스", []):
                if c["class"] in elig:
                    c["가입자격"] = elig[c["class"]][:120]
                    f = flags[c["class"]]
                    c["신규가입가능"] = not f["신규가입불가"]
                    # ★ 3패스가 설명("…-개인연금")에서 이미 판정한 값이 더 정확하다.
                    #   가입자격 문장은 한 문단에 개인연금·퇴직연금이 같이 나오는 경우가
                    #   있어, 무조건 덮어쓰면 C-Pe(개인연금)가 퇴직연금으로 뒤집힌다.
                    if not c.get("연금유형"):
                        if f["퇴직연금"]:
                            c["연금유형"] = "퇴직연금"
                        elif f["연금저축"]:
                            c["연금유형"] = "개인연금"
            # 펀드 수준 재판정 (가입자격 문장이 더 정확하다)
            rec["연금저축_가능"] = any(f["연금저축"] for f in flags.values()) or rec.get("연금저축_가능")
            rec["퇴직연금_가능"] = any(f["퇴직연금"] for f in flags.values()) or rec.get("퇴직연금_가능")
            rec["신규가입불가_클래스"] = [c for c, f in flags.items() if f["신규가입불가"]]
            rec["고액전용_클래스"] = [c for c, f in flags.items() if f["고액전용"]]
            ok_e += 1

        pr = find_price_rule(t)
        if pr:
            rec["기준가격적용"] = pr
            ok_p += 1

    save_json(six, os.path.join(TABLE_DIR, "six_axis.json"))
    n = len(six)
    pg_rows = sum(len(r.get("가입자격_페이지") or {}) for r in six)
    el_rows = sum(len(r.get("가입자격") or {}) for r in six)
    print(f"가입자격 {ok_e}/{n}   기준가격적용 {ok_p}/{n}   "
          f"근거페이지 {pg_rows}/{el_rows}행")
    for r in six[:4]:
        e = r.get("가입자격") or {}
        print(f"  {(r['fund_name'] or '?')[:30]:<32} 클래스 {len(e):>2}  "
              f"연금저축 {r.get('연금저축_가능')}  퇴직연금 {r.get('퇴직연금_가능')}  "
              f"신규불가 {r.get('신규가입불가_클래스')}")


if __name__ == "__main__":
    main()
