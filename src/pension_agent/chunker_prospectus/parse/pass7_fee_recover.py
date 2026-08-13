"""
7패스 — 보수 미검출 문서를 좌표 재조립으로 복구

6패스에서 검증된 기법을 보수 표에 적용한다.
extract_tables()도 실패하고 텍스트 줄 파싱도 실패한 문서에서,
좌표로 행을 복원하면 총보수 행이 살아난다:

    총보수0.25000.10000.20000.21250.45000.27500.3625-
    (동종유형총보수)0.34000.34000.2500-0.40000.2700--

⚠️ 보수는 **소수점 4자리** 표기다 (수익률은 2자리).
   자릿수를 잘못 가정하면 0.2500이 0.25/0.00으로 쪼개진다.

산출: six_axis.json 에 병합
"""
import os, re, sys, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import TABLE_DIR, load_json, save_json
from parse.pass3_extract import span
from parse.pass6_broken import reassemble
warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# ★ 보수 표기는 문서마다 소수 자릿수가 다르다
#     0.2500 0.1000 …  (4자리)
#     1.18   0.81   …  (2자리)  ← 붙으면 "1.180.811.11"
#   자릿수를 잘못 가정하면 1.18/0.81 을 1.180/0.811 로 오독한다.
#   → 4자리로 훑어 나머지가 없어야 4자리, 아니면 2자리로 판정한다.
FEE4 = re.compile(r"\d{1,2}\.\d{4}")
FEE2 = re.compile(r"\d{1,2}\.\d{2}")


def split_fees(s):
    """
    붙어버린 보수 숫자열을 분리한다.

    ⚠️ **소수 자릿수가 일정하지 않으면 원리적으로 분리 불가능하다.**
        "1.180.811.11" 은 [1.18, 0.81, 1.11] 일 수도 [1.180, 0.811, 1.11] 일 수도 있다.
        → 자릿수가 **일정한 경우에만** 분리하고, 혼용이면 실패 처리한다.
          잘못 쪼갠 값을 넣는 것보다 비우는 게 낫다 (오답 방지).
    """
    body = re.sub(r"^[^\d]*", "", s)
    body = re.sub(r"[-–—\s]+$", "", body)
    if not body:
        return []

    for digits in (4, 3, 2):
        pat = re.compile(r"\d{1,2}\.\d{%d}" % digits)
        vals = pat.findall(body)
        if len(vals) < 2:
            continue
        rest = pat.sub("", body)
        # 자릿수가 일정하면 숫자가 하나도 남지 않는다 ('-'만 남음)
        if not re.search(r"\d", rest):
            v = [float(x) for x in vals]
            if all(0 <= x <= 5 for x in v):     # 보수는 0~5% 범위
                return v
    return []            # 자릿수 혼용 → 분리 포기


def parse_fee_lines(lines):
    """
    복원된 행에서 클래스 헤더와 총보수 행을 찾아 매핑한다.
    표가 여러 개(클래스 그룹별)로 나뉘어 있으면 각각 처리한다.
    """
    results, head = [], None
    for L in lines:
        # 클래스 헤더 — 붙어있으므로 종류/Class 토큰을 찾아낸다
        cs = re.findall(r"종류\s*([A-Za-z][A-Za-z0-9\-]{0,6})", L)
        if not cs:
            cs = re.findall(r"Class\s*([A-Za-z][A-Za-z0-9\-]{0,6})", L)
        if len(cs) >= 2:
            head = cs
            continue
        # 총보수 행 — '동종유형'·'합성'·주석(주N))은 배제
        if re.match(r"^총\s*보수(?![ᆞ∙•·ㆍ])", L) and "동종" not in L and "합성" not in L:
            vals = split_fees(L)
            if head and vals:
                n = min(len(head), len(vals))
                results.append(list(zip(head[:n], vals[:n])))
                head = None
    return results


def main():
    import pdfplumber
    six = load_json(os.path.join(TABLE_DIR, "six_axis.json"))
    docs = {d["doc_id"]: d for d in load_json(os.path.join(TABLE_DIR, "docs.json"))}
    smap = load_json(os.path.join(TABLE_DIR, "section_map.json"))

    targets = [r for r in six if not r.get("보수")]
    print(f"보수 미검출 {len(targets)}건 — 좌표 재조립 시도")
    fixed = 0
    for rec in targets:
        did = rec["doc_id"]
        info = smap.get(did, {})
        if did not in docs or "13항보수" not in info.get("sec", {}):
            continue
        rng = span(info["sec"], "13항보수", info["pages"], 8)
        rows = []
        with pdfplumber.open(docs[did]["path"]) as pdf:
            # 헤더가 13항 시작 직전 페이지에 있는 경우가 있어 1페이지 앞부터 본다
            for i in range(max(1, rng[0] - 1), min(rng[1], len(pdf.pages)) + 1):
                for grp in parse_fee_lines(reassemble(pdf.pages[i - 1])):
                    for c, v in grp:
                        if not any(r["class"] == c for r in rows):
                            rows.append({"class": c, "총보수": v, "출처": "coord"})
        # ★ 헤더를 못 찾은 경우 — 클래스 헤더가 앞 페이지에 있거나 표 밖에 있다.
        #   이미 파싱된 클래스 목록으로 순서 매핑한다.
        #   클래스 목록조차 없으면 순서 인덱스(#1, #2 …)로 저장한다.
        #   값 자체는 정확하므로 버리지 않는다 — 클래스 매핑은 나중에 붙일 수 있다.
        if not rows:
            known = [c["class"] for c in (rec.get("클래스") or [])]
            best = None
            with pdfplumber.open(docs[did]["path"]) as pdf:
                for i in range(rng[0], min(rng[1], len(pdf.pages)) + 1):
                    for L in reassemble(pdf.pages[i - 1]):
                        if not re.match(r"^\(?총\s*보수(?![ᆞ∙•·ㆍ․])", L):
                            continue
                        if "동종" in L or "합성" in L:
                            continue
                        vals = split_fees(L)
                        if not vals:
                            continue
                        if known and len(vals) == len(known):   # 정확히 일치 → 확정
                            best = (vals, "coord-order")
                            break
                        # ★ 클래스 목록 자체가 없는 문서 — 순서만 보존해 살린다.
                        #   (known이 비면 위 조건은 절대 성립하지 않아 예전엔 죽은 코드였다)
                        if not known and best is None:
                            best = (vals, "coord-noclass")
                        # ★ 길이 불일치는 버린다.
                        #   앞에서부터 zip하면 클래스별 보수가 한 칸씩 어긋난 채
                        #   0~5% 정상 범위로 저장돼 범위검증을 그냥 통과한다.
                        #   잘못 매핑된 값을 넣느니 클래스명 미상(#1,#2…)으로 남기는 게 낫다.
                    if best and best[1] == "coord-order":
                        break
            if best:
                vals, src = best
                if known:
                    n = min(len(known), len(vals))
                    rows = [{"class": c, "총보수": v, "출처": src}
                            for c, v in zip(known[:n], vals[:n])]
                else:
                    # 클래스명 미상 — 순서만 보존한다
                    rows = [{"class": f"#{i+1}", "총보수": v,
                             "출처": "coord-noclass", "클래스명미상": True}
                            for i, v in enumerate(vals)]
        if rows:
            rec["보수"] = rows
            # ★ 클래스 매핑 없이 순서만 보존한 값 — 값 자체는 맞지만
            #   "C-P 총보수" 같은 질의에 잘못 대응되므로 펀드 레벨에 표시한다.
            #   범위검증(0~5%)은 통과하므로 여기서 세우지 않으면 조용히 오답이 된다.
            if any(r.get("클래스명미상") for r in rows):
                rec["보수_신뢰불가"] = True
            fixed += 1
            print(f"  ✅ {did}  {len(rows)}행  {rows[:3]}")

    save_json(six, os.path.join(TABLE_DIR, "six_axis.json"))
    ok = sum(1 for r in six if r.get("보수"))
    print(f"\n복구 {fixed}/{len(targets)}  →  보수 최종 {ok}/{len(six)}")


if __name__ == "__main__":
    main()