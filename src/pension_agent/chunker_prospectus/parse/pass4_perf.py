"""
4패스 — 제3부에서 수익률·시장잔고 추출

⚠️ 6축 중 이 둘만 앞 8페이지가 아니라 제3부(중앙값 41p)에 있다.
   그리고 요약본(3~4p)에도 축약된 수익률이 있어 그걸 뽑으면 대표 클래스만 나온다.
   반드시 제3부 범위 안에서 뽑는다.

  · 시장잔고 = 재무상태표 '자산총계' (단위 백만원, 3개년)
  · 수익률   = '3. 집합투자기구의 운용실적' 이후 연평균수익률 표 (클래스별)

산출: data/tables/six_axis.json 에 병합
"""
import os, re, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import TABLE_DIR, DUMP_DIR, flat, load_json, save_json

NUM = r"-?[\d,]+(?:\.\d+)?"
_DOCS = {}   # main()에서 docs.json으로 채운다

# ── 클래스 라벨 판별 ★ 지수명·지표명 오검출 방지 ──
#   실제 클래스: A, C, C1, C-P, C-P2, Ae, S …  (짧고 형태가 정해져 있다)
CLASS_LABELED = re.compile(r"^(?:종류|Class)\s*([A-Za-z][A-Za-z0-9\-]{0,7})\s")
CLASS_BARE    = re.compile(r"^([A-Za-z](?:\d{1,2}|-[A-Za-z]\d?|[A-Za-z])?)\s")
NOT_CLASS = {"VAR", "BBB", "MMF", "ETF", "YTM", "BM", "NO", "KOSPI", "KOSDAQ",
             "MSCI", "REPO", "TOTAL", "IT", "AI", "ESG", "USD", "KRW"}

def _f(x):
    try:
        return float(x.replace(",", ""))
    except Exception:
        return None


def _read_below(lines, i, k=8):
    """
    ★ 세로형 표 대응 — 라벨 줄 아래에 숫자가 한 줄에 하나씩 오는 형식.

    fitz가 표를 뜯을 때 라벨 열과 숫자 열을 따로 흘려보내는 문서가 있다:
        '자산총계'
        '5,711'
        '9,235'
        '7,381'
        '   운용부채'      ← 숫자가 아니므로 여기서 종료
    6패스의 '글자 단위 분해'와는 다른 종류(셀 단위 분해)라 좌표 재조립이 필요 없다.
    """
    vals = []
    for L in lines[i + 1:i + 1 + k]:
        s = L.strip()
        if not s or s in "-–—":
            if vals:
                break              # 숫자 시작 후 빈칸/대시 → 열 끝
            continue
        if not re.fullmatch(r"-?[\d,]+(?:\.\d+)?", s):
            break                  # 다음 라벨 또는 문장 → 종료
        vals.append(float(s.replace(",", "")))
    return vals

def find_aum(pages, p3):
    """
    재무상태표 '자산총계' — 최신 연도가 맨 앞.

    ⚠️ 단위가 문서마다 다르다. '(단위 : 원)' / '(단위 : 백만원)' / '(단위 : 천원)'
       그냥 숫자만 뽑으면 5,711(백만원)과 208,907,969,455(원)이 섞인다.
       → 같은 페이지의 단위 표기를 읽어 백만원으로 환산한다.
    """
    UNIT = {"원": 1e-6, "천원": 1e-3, "백만원": 1.0, "억원": 100.0}
    for p in pages:
        if p["page"] < p3 or p["page"] > p3 + 14:
            continue
        lines = p["text"].split("\n")
        mult = 1.0
        for i, line in enumerate(lines):
            L = line.strip()
            # 단위 표기는 자산총계 줄보다 위에 나온다 → 훑으면서 갱신
            # ⚠️ 괄호가 소괄호가 아닐 수 있다: '[단위:원]' '[단위:원,%]' '단위: 천원'
            #    ★ '백만원'을 '원'보다 앞에 둘 것. 순서가 반대면 '백만원'의 끝글자에
            #      '원'이 먼저 매칭돼 전 문서가 원 단위로 오인된다.
            mu = re.search(r"단위\s*[:：]?\s*[^)\]\n]{0,6}?(백만원|천원|억원|원)", L)
            if mu:
                mult = UNIT.get(mu.group(1), 1.0)
            s = L.replace(" ", "")
            # ⚠️ "매매회전율이란 …" 같은 설명 문장이 걸리지 않게 길이 상한을 둔다
            if not s.startswith("자산총계") or len(s) > 60:
                continue
            # ★ 한 줄에 두 항목이 오는 형식이 있다:
            #   "자산 총계 1,819.1 1,393.5 영업이익 126.6 46.3"
            #   → '$' 앵커를 쓰면 실패. 뒤에 다른 항목이 와도 앞부분만 잡는다.
            m = re.match(r"^자산\s*총\s*계\s+((?:%s\s+)*%s)(?:\s|$)" % (NUM, NUM), L)
            if m:                                      # 가로형
                vals = [_f(v) for v in m.group(1).split()]
            else:                                      # ★ 세로형 폴백
                vals = _read_below(lines, i)
            vals = [round(v * mult, 1) for v in vals if v is not None]   # ★ 0.0도 유효값
            # ★ 안전망 — 단위 표기를 못 읽은 경우 대비.
            #   국내 공모펀드 AUM은 10조원(=1e7 백만원)을 넘지 않는다.
            while vals and vals[0] > 1e7:
                vals = [round(v * 1e-6, 1) for v in vals]
            if vals:
                return vals[0], vals                   # (최신, 추이) — 단위 백만원
    return None, None

def find_aum_coord(pdf_path, p3):
    """
    ★ 시장잔고 좌표 폴백 — 라벨 열과 숫자 열이 완전히 분리된 문서용(5건).

    텍스트 추출로는 '자산총계' 다음에 날짜·항목명이 오고 숫자는 한참 뒤에 있어
    _read_below로 못 잡는다. 좌표 재조립하면 한 행으로 붙는다:
        '자산총계424,168,172,451408,510,430,501323,560,418,824'
    ⚠️ 숫자가 붙어 있지만 3자리 콤마 그룹이라 분리 가능하다.
    """
    import pdfplumber
    from parse.pass6_broken import reassemble
    UNIT = {"원": 1e-6, "천원": 1e-3, "백만원": 1.0, "억원": 100.0}
    GROUP = re.compile(r"\d{1,3}(?:,\d{3})+")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i in range(p3, min(p3 + 14, len(pdf.pages)) + 1):
                lines = reassemble(pdf.pages[i - 1])
                mult = 1.0
                for L in lines:
                    mu = re.search(r"단위\s*[:：]?\s*[^)\]\n]{0,6}?(백만원|천원|억원|원)",
                                   L.replace(" ", ""))
                    if mu:
                        mult = UNIT.get(mu.group(1), 1.0)
                for L in lines:
                    t = L.replace(" ", "")
                    if not t.startswith("자산총계"):
                        continue
                    vals = [float(x.replace(",", "")) for x in GROUP.findall(t)]
                    vals = [round(v * mult, 1) for v in vals if v > 0]
                    while vals and vals[0] > 1e7:      # 안전망
                        vals = [round(v * 1e-6, 1) for v in vals]
                    if vals:
                        return vals[0], vals
    except Exception:
        pass
    return None, None


def find_perf(pages, p3):
    """
    '3. 집합투자기구의 운용실적' 이후 표.

    실제 줄 형태 (미래에셋 기준):
        투자신탁      -22.43  3.31  -9.25  4.14  3.95
        비교지수       -9.63  3.58  -6.94  1.77  1.17
        수익률 변동성  26.49  24.07 23.02  25.14 18.55
    운용사에 따라 종류별 행(종류A …)이 함께 나오기도 한다.
    """
    # 운용실적 헤딩을 찾는다. 표기가 여러 가지라 대안 패턴 필요.
    HEAD = [r"\d\s*[\.．]\s*집합투자기구의?\s*운용\s*실적",
            r"운용\s*실적\s*\(?\s*세전",
            r"연평균\s*수익률"]
    start = None
    for p in pages:
        if p["page"] < p3:
            continue
        t = flat(p["text"])
        if any(re.search(h, t) for h in HEAD):
            start = p["page"]
            break
    if start is None:
        start = p3          # 폴백 — 제3부 전체를 훑는다

    out = {"펀드수익률": None, "비교지수수익률": None,
           "수익률변동성": None, "매매회전율": None, "클래스수익률": []}

    NUMROW = r"((?:" + NUM + r"\s+){1,7}" + NUM + r")\s*$"
    DATE   = r"(?:\d{4}[-./]\d{1,2}[-./]\d{1,2}\s+)?"

    def nums(line):
        """줄 끝의 숫자 나열을 뽑는다. 없으면 None."""
        m = re.search(r"\s" + DATE + NUMROW, " " + line)
        if not m:
            return None
        v = [_f(x) for x in m.group(1).split()]
        v = [x for x in v if x is not None and -100 < x < 1000]
        return v or None

    for p in pages:
        if not (start <= p["page"] <= start + 14):
            continue
        lines = [l.strip() for l in p["text"].split("\n")]
        for i, L in enumerate(lines):
            if not L:
                continue
            key = re.sub(r"[\s()％%]", "", re.split(r"\s{1,}-?\d", L)[0])[:20]

            # ── 비교지수 / 변동성 / 회전율 ──
            if key.startswith("비교지수"):
                v = nums(L) or _read_below(lines, i)          # ★ 세로형 폴백
                if v and out["비교지수수익률"] is None:
                    out["비교지수수익률"] = v[:5]
                    # ★ 펀드 행은 비교지수 바로 위에 온다 (라벨이 제각각이라 위치로 잡는다)
                    #   세로형은 라벨+숫자 5줄이 위에 있어 탐색 폭을 12줄로 넓힌다
                    if out["펀드수익률"] is None:
                        for k in range(i - 1, max(-1, i - 12), -1):
                            pv = (nums(lines[k]) or _read_below(lines, k)) if k >= 0 else None
                            if pv and len(pv) == len(v):
                                out["펀드수익률"] = pv[:5]
                                break
                continue
            if key.startswith("수익률변동성"):
                v = nums(L) or _read_below(lines, i)          # ★ 세로형 폴백
                if v and out["수익률변동성"] is None:
                    out["수익률변동성"] = v[0]
                continue
            # ── 명시적 펀드 라벨 ──
            if key in ("투자신탁", "펀드", "이투자신탁"):
                v = nums(L) or _read_below(lines, i)          # ★ 세로형 폴백
                if v and out["펀드수익률"] is None:
                    out["펀드수익률"] = v[:5]
                continue

            # ── 클래스별 행 ──
            # ⚠️ 라틴 문자로 시작하는 모든 줄을 먹으면 지수명·지표명이 클래스가 된다
            #    "KOSPI 200 12.5 …" → class='KOSPI', 200이 수익률로 들어감
            m = CLASS_LABELED.match(L) or CLASS_BARE.match(L)
            if m and m.group(1).upper().replace("-", "") in NOT_CLASS:
                m = None
            if m:
                v = nums(L)
                if v and not any(r["class"] == m.group(1) for r in out["클래스수익률"]):
                    out["클래스수익률"].append({"class": m.group(1), "수익률": v[:5]})
    return out


def find_turnover(pages, p3):
    """
    매매회전율 — 운용실적 헤딩보다 **앞**(재무정보 구간)에 있다.
    "매매회전율 311 200 254"  ← 3개년, 최신이 맨 앞
    같은 미래에셋 주식형이라도 81% vs 530%로 6배 차이가 난다.
    위험등급으로 못 가르는 것을 가르는 지표.

    ⚠️ 같은 구간에 "매매회전율이란 주식매매의 빈번한 정도를 …" 설명 문장이 있다.
       길이 상한으로 배제하지 않으면 이 줄이 먼저 걸린다.
    """
    saw_label = False          # ★ 라벨을 봤는데 값이 없다 vs 라벨 자체가 없다
    for p in pages:
        if not (p3 <= p["page"] <= p3 + 14):
            continue
        lines = p["text"].split("\n")
        for i, line in enumerate(lines):
            L = line.strip()
            s = L.replace(" ", "")
            if not s.startswith("매매회전율") or len(s) > 60:
                continue
            saw_label = True
            m = re.match(r"^매매\s*회전율\s+((?:%s\s+)*%s)\s*$" % (NUM, NUM), L)
            if m:                                  # 가로형
                v = [_f(x) for x in m.group(1).split()]
            else:                                  # ★ 세로형 폴백
                v = _read_below(lines, i)
            v = [x for x in v if x is not None]
            if v:                      # ★ 0도 유효값이다 (실제로 '0 0 0'인 펀드가 있음)
                return v[0], v
    # 라벨은 있는데 값이 '-'/'_' → 문서가 명시한 '해당없음'
    # 라벨 자체가 없음 → 파싱 미검출이지 '해당없음'이 아니다
    return ("N/A", None) if saw_label else (None, None)


def main():
    six = load_json(os.path.join(TABLE_DIR, "six_axis.json"))
    smap = load_json(os.path.join(TABLE_DIR, "section_map.json"))
    global _DOCS
    _DOCS = {d["doc_id"]: d for d in load_json(os.path.join(TABLE_DIR, "docs.json"))}
    ok_aum = ok_perf = 0
    for rec in six:
        did = rec["doc_id"]
        dpath = os.path.join(DUMP_DIR, did + ".json")
        if not os.path.exists(dpath):
            continue
        pages = load_json(dpath)["pages"]
        p3 = smap.get(did, {}).get("sec", {}).get("제3부")
        if not p3:
            continue
        aum, trend = find_aum(pages, p3)
        if aum is None:
            # ★ 라벨·숫자 열이 완전히 분리된 문서는 좌표 재조립으로 잡는다
            dpath_pdf = (_DOCS.get(did) or {}).get("path")
            if dpath_pdf:
                aum, trend = find_aum_coord(dpath_pdf, p3)
        rec["시장잔고_백만원"], rec["시장잔고_추이"] = aum, trend
        perf = find_perf(pages, p3)
        rec.update(perf)
        to, to_tr = find_turnover(pages, p3)
        if to == "N/A":                    # 문서에 '_'로 표기된 경우
            rec["매매회전율"], rec["매매회전율_추이"] = None, None
            rec["매매회전율_해당없음"] = True
        else:
            rec["매매회전율"], rec["매매회전율_추이"] = to, to_tr
            # 채권형은 주식 매매 지표가 원래 없다 → 미검출이 아니라 '해당없음'
            nm = (rec.get("fund_name") or "") + (rec.get("상품분류") or "")
            # ⚠️ 주식형인데도 문서가 값을 미기재한 경우가 있다(9건 확인):
            #     "매매회전율___" / "매매회전율---" / 라벨 자체가 없음
            #    이것도 파싱 실패가 아니라 '해당없음'이다.
            # ★ 여기 오는 to is None 은 '라벨 자체 미검출' = 미확인이다.
            #   해당없음은 위 "N/A" 분기에서만 세운다.
            rec["매매회전율_해당없음"] = False
        if aum: ok_aum += 1
        if perf.get("펀드수익률") or perf.get("클래스수익률"): ok_perf += 1
    # 레이아웃이 깨진 문서 표시 — 정규식으로 복구 불가, 별도 처리 필요
    for rec in six:
        dp = os.path.join(DUMP_DIR, rec["doc_id"] + ".json")
        if rec.get("펀드수익률") or not os.path.exists(dp):
            continue
        pages = load_json(dp)["pages"]
        p3 = smap.get(rec["doc_id"], {}).get("sec", {}).get("제3부")
        if not p3:
            continue
        # 한 글자짜리 줄이 30% 이상이면 세로로 깨진 것
        for p in pages:
            if not (p3 <= p["page"] <= p3 + 14):
                continue
            ls = [l.strip() for l in p["text"].split("\n") if l.strip()]
            # ★ 세로형 표('5,711' 같은 줄)를 깨짐으로 오인하지 않도록 한 글자 줄만 센다
            if len(ls) > 40 and sum(1 for l in ls if len(l) <= 1) / len(ls) > 0.5:
                rec["레이아웃깨짐"] = True
                break

    save_json(six, os.path.join(TABLE_DIR, "six_axis.json"))
    n = len(six)
    ok_to = sum(1 for r in six if r.get("매매회전율") is not None)
    na    = sum(1 for r in six if r.get("매매회전율_해당없음"))
    broken = sum(1 for r in six if r.get("레이아웃깨짐"))
    print(f"시장잔고 {ok_aum}/{n}   수익률 {ok_perf}/{n}   "
          f"매매회전율 {ok_to}/{n-na}(채권형 {na} 제외)")
    if broken:
        print(f"  ⚠️ 레이아웃 깨짐 {broken}건 — 텍스트가 세로로 분해됨. 정규식 복구 불가")
    for r in six[:3]:
        print(f"  {(r['fund_name'] or '?')[:34]:<36} "
              f"AUM {r.get('시장잔고_백만원')}  펀드 {(r.get('펀드수익률') or ['?'])[0]}  "
              f"비교지수 {(r.get('비교지수수익률') or ['?'])[0]}  회전율 {r.get('매매회전율')}")

if __name__ == "__main__":
    main()