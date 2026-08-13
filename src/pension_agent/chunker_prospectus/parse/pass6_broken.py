"""
6패스 — 레이아웃이 깨진 문서 복구 (좌표 기반 재조립)

⚠️ 일부 PDF는 extract_text()가 텍스트를 **글자 단위로 세로 분해**한다.
   "-25.69"가 '-','2','5','.','6','9' 여섯 개 단어로 쪼개진다.
   정규식으로는 절대 복구할 수 없다.

   → extract_words()로 (x, y) 좌표를 얻어 **같은 y에 있는 글자를 x 순으로 이어붙인다.**

복구 결과 예 (하나IT코리아, 원래 수익률 None):
    펀드2007-05-03-25.69-8.08-7.354.744.71
    펀드ClassA2007-05-03-25.77-8.17-7.414.734.93
    펀드ClassC12007-05-03-26.35-8.88-8.123.934.09
→ 클래스별 수익률까지 전부 복구된다.

산출: six_axis.json 에 병합 (레이아웃깨짐 문서만)
"""
import os, re, sys, glob, collections, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import TABLE_DIR, load_json, save_json
warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

Y_TOL = 3          # 같은 행으로 볼 y 오차(pt)


def reassemble(page):
    """extract_words()를 y로 묶고 x 순으로 이어붙여 행을 복원한다."""
    rows = collections.defaultdict(list)
    for w in page.extract_words():
        rows[round(w["top"] / Y_TOL) * Y_TOL].append(w)
    out = []
    for y in sorted(rows):
        line = "".join(w["text"] for w in sorted(rows[y], key=lambda x: x["x0"]))
        if line.strip():
            out.append(line)
    return out


# 붙어버린 숫자열을 분리한다.
#   "-25.69-8.08-7.354.744.71" → [-25.69, -8.08, -7.35, 4.74, 4.71]
#   소수점 두 자리를 기준으로 자른다 (수익률 표기 관례)
NUM_SPLIT = re.compile(r"-?\d{1,3}\.\d{2}")

# ★ 수익률 데이터 행 판별 — '연평균수익률' 헤딩이 없는 문서를 위해 행 자체로 페이지를 고른다
PERF_ROW = re.compile(
    # ① "펀드2007-05-03-25.69…" / "비교지수9.44%6.87%…" / "펀드ClassA2016-04-184.21…"
    r"^(?:펀드|비교지수|참조지수)(?:Class)?[A-Za-z0-9\-]{0,8}"
    r"(?:\d{4}[-./]\d{2}[-./]\d{2})?-?\d{1,3}\.\d{2}"
    # ② 클래스 라벨이 행 머리인 문서:
    #    "수수료미징구-오프라인-퇴직연금(C-r)6.355.343.081.633.89"
    r"|^수수료[^()]{2,30}?\([A-Za-z][A-Za-z0-9\-]{0,6}\)-?\d{1,3}\.\d{2}")

# ⚠️ 라벨 뒤에 최초설정일이 붙는 문서가 있다: "비교지수2007-04-173.48…"
#    떼지 않으면 날짜의 '-17'이 첫 값에 붙어 -173.48이 된다.
DATE_PREFIX = re.compile(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}")
# 펀드명 라벨 뒤 어디에나 오는 설정일 제거용
DATE_ANY = re.compile(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}")


def split_nums(s):
    # ⚠️ '9.44%6.87%' 처럼 %가 붙는 문서가 있다. %는 구분자일 뿐이므로 제거한다.
    return [float(x) for x in NUM_SPLIT.findall(s.replace("%", ""))]


def strip_label(L, label):
    """라벨과 뒤따르는 설정일을 떼어낸 나머지를 돌려준다."""
    t = L[len(label):] if L.startswith(label) else L
    return DATE_PREFIX.sub("", t.lstrip("()%"))


def _looks_like_period(v):
    """
    ★ 결산기 연월 헤더 판별 — 수익률이 아니다.
      "24.05 23.05 22.05 20.05" 처럼 15~30 구간 값이 1~2씩 감소하면 연도 표기다.

    ⚠️ 라벨 없는 표에서 '첫 숫자행 = 펀드'로 가정하는 아래 폴백이 이걸 집는다.
       비어 있으면 "정보 없음"이라 답하지만, 24.05%를 넣으면 자신 있게 오답한다.
    """
    if not v or len(v) < 3:
        return False
    h = [x for x in v if 15 <= x <= 30]
    return len(h) >= 3 and all(round(h[i] - h[i + 1], 1) in (1.0, 2.0)
                               for i in range(len(h) - 1))


def parse_perf(lines):
    """복원된 행에서 펀드·클래스별 수익률과 변동성을 뽑는다."""
    out = {"펀드수익률": None, "수익률변동성": None, "클래스수익률": []}
    for i, L in enumerate(lines):
        # "펀드ClassC1 2007-05-03 -26.35-8.88-8.12 3.93 4.09"
        # ⚠️ 뒤 그룹이 전부 optional이라 백트래킹이 안 걸린다. 앞 그룹을 lazy로 두고
        #    lookahead로 '날짜 또는 소수점 두 자리 숫자' 앞에서 끊어야 'A2007-05'가 안 나온다.
        m = re.match(r"^펀드(?:Class)?([A-Za-z][A-Za-z0-9\-]{0,7}?)?"
                     r"(?=\d{4}-\d{2}-\d{2}|-?\d{1,3}\.\d{2})"
                     r"(\d{4}-\d{2}-\d{2})?(.+)$", L)
        if m:
            cls, _, tail = m.group(1), m.group(2), m.group(3)
            v = split_nums(tail)
            if not v:
                continue
            if _looks_like_period(v):
                continue
            if cls:
                if not any(r["class"] == cls for r in out["클래스수익률"]):
                    out["클래스수익률"].append({"class": cls, "수익률": v[:5]})
            elif out["펀드수익률"] is None:
                out["펀드수익률"] = v[:5]
            continue
        # ★ 클래스 라벨이 "수수료미징구-온라인-퇴직연금(C-re)6.455.45…" 형태인 문서
        m2 = re.match(r"^(수수료[^()]{2,30}?)\(([A-Za-z][A-Za-z0-9\-]{0,6})\)"
                      r"(-?\d{1,3}\.\d{2}.*)$", L)
        if m2:
            v = split_nums(m2.group(3))
            if v and not _looks_like_period(v):
                cls = m2.group(2)
                if not any(r["class"] == cls for r in out["클래스수익률"]):
                    out["클래스수익률"].append({"class": cls, "수익률": v[:5]})
                if out["펀드수익률"] is None:      # 첫 클래스를 대표값으로
                    out["펀드수익률"] = v[:5]
            continue

        if L.startswith("수익률변동성") and out["수익률변동성"] is None:
            v = split_nums(strip_label(L, "수익률변동성"))
            if v:
                out["수익률변동성"] = v[0]
        if L.startswith("비교지수") and not out.get("비교지수수익률"):
            v = split_nums(strip_label(L, "비교지수"))
            if v:
                out["비교지수수익률"] = v[:5]
                # ★ 펀드 행 라벨이 **펀드명 자체**인 문서가 있다:
                #   "Tomorrow장기우량증권투자신탁K-1호(채권)2006-09-054.856.87…"
                #   고정 패턴으로 못 잡으므로 위치로 잡는다 — 비교지수 바로 위 행에서
                #   같은 개수의 숫자가 나오면 그게 펀드 행이다(4패스와 동일한 논리).
                # ⚠️ 비교지수가 페이지 첫 줄이면 펀드 행은 이전 페이지에 있다 → 추측하지 않는다.
                if out["펀드수익률"] is None:
                    for k in range(i - 1, max(-1, i - 4), -1):
                        prev = lines[k].replace(" ", "")
                        if prev.startswith(("비교지수", "수익률변동성", "참조지수")):
                            continue
                        pv = split_nums(DATE_ANY.sub("", prev))
                        if pv and len(pv) == len(v) and not _looks_like_period(pv):
                            out["펀드수익률"] = pv[:5]
                            break

    # ── 라벨 없는 폴백 : **비활성** (철회) ─────────────────────────
    #   원래 "기간 헤더 다음 첫 숫자 행 = 펀드"로 가정해 값을 채웠으나,
    #   위치 가정이 깨지면 표 헤더를 수익률로 집는다. 실제 결과:
    #     1차: [24.05, 23.05, 22.05, 20.05]  ← 결산기 연월
    #     2차: [25.05, 25.05, 25.05, 25.05]  ← 기준일 반복 (필터 우회)
    #   3개 문서 전부 오검출이었고 정상 복구 사례는 0건이었다.
    #   → 라벨(펀드/ClassX/비교지수)이 있는 경우에만 복구한다.
    #     비면 '자료 미확인'으로 답할 수 있지만, 24.05%를 넣으면 자신 있게 오답한다.
    if out["펀드수익률"] is None:
        out["수익률_수동확인필요"] = True
    return out


def main():
    import pdfplumber
    six = load_json(os.path.join(TABLE_DIR, "six_axis.json"))
    docs = {d["doc_id"]: d for d in load_json(os.path.join(TABLE_DIR, "docs.json"))}
    smap = load_json(os.path.join(TABLE_DIR, "section_map.json"))

    # ★ 대상을 '레이아웃깨짐' 플래그가 아니라 **수익률이 비어 있는 모든 문서**로 넓힌다.
    #   깨짐 판정에 안 걸렸지만 4패스 텍스트 파싱이 실패한 문서가 다수였다(22/28).
    targets = [r for r in six if not r.get("펀드수익률")]
    print(f"수익률 미검출 {len(targets)}건 좌표 재조립 시도")
    fixed = 0
    for rec in targets:
        did = rec["doc_id"]
        p3 = smap.get(did, {}).get("sec", {}).get("제3부")
        if not p3 or did not in docs:
            continue
        with pdfplumber.open(docs[did]["path"]) as pdf:
            for i in range(p3, min(p3 + 16, len(pdf.pages)) + 1):
                lines = reassemble(pdf.pages[i - 1])
                # ⚠️ '연평균수익률' 헤딩이 없는 페이지에 표가 오는 문서가 많다.
                #    헤딩 대신 '펀드…'/'비교지수…' 데이터 행의 존재로 판단한다.
                if not any(PERF_ROW.match(l.replace(" ", "")) for l in lines):
                    continue
                perf = parse_perf(lines)
                if perf.get("펀드수익률") or perf.get("클래스수익률"):
                    for k, v in perf.items():
                        if v:
                            rec[k] = v
                    rec["복구_좌표재조립"] = True
                    fixed += 1
                    print(f"  ✅ {did}  펀드 {perf['펀드수익률']}  "
                          f"클래스 {len(perf['클래스수익률'])}개")
                    break
    # ★ 복구까지 실패한 문서에 플래그를 세운다.
    #   parse_perf 안에서 세우면 main의 병합 조건에 걸려 절대 rec에 붙지 않는다.
    #   pop은 재실행 시 낡은 플래그가 남는 것을 막는다.
    for rec in six:
        if not (rec.get("펀드수익률") or rec.get("클래스수익률")):
            rec["수익률_수동확인필요"] = True
        else:
            rec.pop("수익률_수동확인필요", None)

    save_json(six, os.path.join(TABLE_DIR, "six_axis.json"))
    n = len(six)
    ok = sum(1 for r in six if r.get("펀드수익률"))
    print(f"\n복구 {fixed}/{len(targets)}   →  수익률 최종 {ok}/{n}")


if __name__ == "__main__":
    main()