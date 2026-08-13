"""
2패스 — 섹션 지도 자동 생성 ★ 여기가 본작업

1패스 덤프를 읽어서 "이 문서의 13항은 27페이지, 제3부는 41페이지" 표를 만든다.
정규식은 한 번에 안 맞는다. 6개사 문서로 돌려보며 패턴을 늘려가는 게 이 주의 작업.

오검출원 4가지 — 전부 여기서 막는다:
  ① 목차          → is_toc()로 페이지 통째 제외
  ② 본문 내 상호참조 → "상단 N줄 + 줄 시작" 조건
  ③ 고정 오프셋    → 쓰지 않음. 헤딩 매칭만
  ④ 섹션 경계 침범 → 3패스에서 헤딩 문자열 기준으로 자름

산출: data/tables/section_map.json
"""
import os, re, sys, glob, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import DUMP_DIR, TABLE_DIR, is_toc, load_json, save_json

# ── 헤딩 패턴 ★ 여기를 늘려가는 게 작업의 핵심 ──────────────
# 값이 리스트인 이유: 운용사마다 표기가 달라 대안 패턴이 필요하다.
HEADINGS = {
    "요약정보":   [r"^간이투자설명서", r"^.{0,6}요약정보"],
    # ⚠️ 표지(1p)에도 "1. 집합투자기구 명칭 ○○신탁"이 있다.
    #    표지는 뒤에 펀드명이 바로 붙는 반면, 본문 헤딩은 줄이 거기서 끝난다.
    "제1부1항":   [r"^1\s*[\.．]\s*집합투자기구\s*의?\s*명칭\s*$",
                   r"^1\s*[\.．]\s*집합투자기구\s*의?\s*명칭\s*\(종류형",
                   r"^1\s*[\.．]\s*집합투자기구\s*명칭\s*\(?종류형",
                   r"^1\s*[\.．]\s*집합투자기구의?\s*명칭\s*및?\s*종류",
                   r"^1\s*[\.．]\s*집합투자기구\s*명칭\s*$"],
    "가입자격":   [r"^2-1\s*[\.．]?\s*집합투자기구의?\s*가입자격",
                   r"^2\s*[\.．]\s*1\s*집합투자기구의?\s*가입자격"],
    "연혁":       [r"^2\s*[\.．]\s*집합투자기구의?\s*연혁"],
    # ★ 일부 문서는 가입자격을 "제2부 6. 집합투자기구의 구조"로 넘긴다
    "제2부6항구조": [r"^6\s*[\.．]\s*집합투자기구의?\s*구조\s*$",
                    r"^6\s*[\.．]\s*집합투자기구의?\s*구조"],
    "투자대상":   [r"^\d+\s*[\.．]\s*집합투자기구의?\s*투자대상"],
    "투자전략":   [r"^\d+\s*[\.．]\s*집합투자기구의?\s*투자전략"],
    "투자위험":   [r"^\d+\s*[\.．]\s*집합투자기구의?\s*투자위험"],
    "11항매입환매": [r"^11\s*[\.．]\s*매입", r"^\d+\s*[\.．]\s*매입[,\s]*환매"],
    "13항보수":   [r"^13\s*[\.．]\s*보수", r"^\d+\s*[\.．]\s*보수\s*및\s*수수료"],
    "14항과세":   [r"^14\s*[\.．]\s*이익", r"^\d+\s*[\.．]\s*이익\s*배분\s*및\s*과세"],
    "별첨1":      [r"^\[?\s*제\s*2\s*부\s*별\s*첨\s*1", r"^별\s*첨\s*1"],
    "제3부":      [r"^제\s*3\s*부"],
    "제4부":      [r"^제\s*4\s*부"],
    "제5부":      [r"^제\s*5\s*부"],
}

# 헤딩이 페이지 상단에만 있지 않다. 문서에 따라 30번째 줄에 나오기도 한다.
# 대신 "줄 전체가 헤딩" 조건(뒤에 긴 문장이 안 붙음)으로 상호참조를 배제한다.
MAX_HEAD_LEN = 45      # 헤딩 줄 최대 길이. 이보다 길면 본문 문장으로 본다

def find_sections(pages):
    toc = {p["page"] for p in pages if is_toc(p["text"])}
    result, first_seen = {}, {}
    for p in pages:
        if p["page"] in toc:
            continue
        # ★ 표지(1p) 제외 — 표지에도 "1. 집합투자기구 명칭 ○○신탁"이 있어
        #   제1부1항이 1p로 오검출된다(9건 확인). 1p가 본문 제1부인 문서는 없다.
        if p["page"] == 1:
            continue
        lines = [l.strip() for l in p["text"].split("\n")
                 if l.strip() and len(l.strip()) <= MAX_HEAD_LEN]
        for name, pats in HEADINGS.items():
            if name in result:
                continue                       # 첫 등장만
            for pat in pats:
                if any(re.match(pat, l) for l in lines):
                    result[name] = p["page"]
                    break

    # ★ 순서 검증 — 제N부는 페이지 순서대로 나와야 한다.
    #   목차 오검출로 제5부가 제1부보다 앞서면(12p vs 15p) span()이 다음 섹션 직전까지
    #   범위를 자르는 구조상 앞뒤 섹션 범위가 전부 어긋난다. 역전된 항목은 버린다.
    ORDER = ["제1부1항", "제2부6항구조", "제3부", "제4부", "제5부"]
    prev = 0
    for k in ORDER:
        if k not in result:
            continue
        if result[k] < prev:
            del result[k]
        else:
            prev = result[k]
    return result, sorted(toc)

def main():
    files = sorted(glob.glob(os.path.join(DUMP_DIR, "*.json")))
    if not files:
        print("[!] 덤프가 없습니다. pass1을 먼저 실행하세요.")
        return
    out = {}
    for f in files:
        d = load_json(f)
        sec, toc = find_sections(d["pages"])
        out[d["doc_id"]] = {"pages": len(d["pages"]), "toc": toc, "sec": sec}
    save_json(out, os.path.join(TABLE_DIR, "section_map.json"))
    report(out)

def report(out):
    """★ 검증 — 검출률과 페이지 분포를 본다."""
    n = len(out)
    print(f"\n{'섹션':<14}{'검출':>10}{'중앙값':>8}{'범위':>12}   이상")
    print("-" * 62)
    for name in HEADINGS:
        v = [(k, m["sec"][name]) for k, m in out.items() if name in m["sec"]]
        if not v:
            print(f"{name:<14}{0:>6}/{n:<3}     —")
            continue
        pages = [p for _, p in v]
        med = statistics.median(pages)
        # 중앙값의 1/3 미만이면 목차 오검출 의심
        odd = [k for k, p in v if med >= 12 and p < med / 3]
        flag = f"⚠️ {odd[:3]}" if odd else ""
        print(f"{name:<14}{len(v):>6}/{n:<3}{med:>8.0f}{f'{min(pages)}~{max(pages)}':>12}   {flag}")
    print("\n판정 기준: 검출률 90% 미만이거나 ⚠️가 뜨면 패턴을 보강한다.")

if __name__ == "__main__":
    main()