# -*- coding: utf-8 -*-
"""
refine_evidence.py — 수작업 보정 45건의 근거를 필드별로 정리한다.

입력
  data/tables/manual_override.json     보정값
  data/tables/override_evidence.json   1차 탐색 결과(페이지 힌트) — 없어도 동작
  data/dump/{doc_id}.json              텍스트 레이어 (인용문은 여기서 뽑는다)
  data/tables/section_map.json

왜 다시 뽑나
  1차 탐색은 reassemble() 좌표 재조립본에서 문자열을 잡아 페이지는 맞지만
  인용문이 깨져 있다("라인-개인연금금저축계좌를통하여가입한----").
  답변에 붙일 문장은 텍스트 레이어에서 다시 뽑아야 한다.

원칙
  · 필드별로 따로 잡는다. 연금저축과 퇴직연금은 근거 문장이 다르다.
  · 값이 False인 건은 '부재'가 근거라 인용할 문장이 없다 → 근거불가로 표시.
  · 값과 근거 방향이 어긋나거나 못 찾으면 확인필요=True. 지우지 않고 남긴다.

산출: data/tables/override_evidence.json  (덮어씀, 기존은 .bak 로 백업)
"""
import os, re, json, glob, shutil, sys

BASE = ""          # 예: "/content/pension"


def find_base():
    if BASE:
        return BASE
    for root in ["/content", ".", "/content/drive/MyDrive"]:
        hits = glob.glob(os.path.join(root, "**", "data", "tables",
                                      "manual_override.json"), recursive=True)
        if hits:
            p = sorted(hits, key=len)[0]
            return os.path.dirname(os.path.dirname(os.path.dirname(p)))
    raise SystemExit("[!] manual_override.json 을 못 찾았습니다. BASE 를 지정하세요.")


B = find_base()
TABLE = os.path.join(B, "data", "tables")
DUMP = os.path.join(B, "data", "dump")
sys.path.insert(0, os.path.join(B, "src"))
from parse.pass3_extract import span                      # noqa: E402

EV_PATH = os.path.join(TABLE, "override_evidence.json")
ov = json.load(open(os.path.join(TABLE, "manual_override.json"), encoding="utf-8"))
smap = json.load(open(os.path.join(TABLE, "section_map.json"), encoding="utf-8"))
hint = {}
if os.path.exists(EV_PATH):
    try:
        hint = json.load(open(EV_PATH, encoding="utf-8"))
    except Exception:
        hint = {}

# ── 필드별 근거 키워드 ────────────────────────────────
KEYS = {
    "연금저축_가능": re.compile(r"연금저축계좌|개인연금|연금저축"),
    "퇴직연금_가능": re.compile(r"퇴직연금|개인형퇴직연금|IRP|개인퇴직계좌|퇴직급여"),
    "위험등급":     re.compile(r"\d\s*등급"),
    "위험라벨":     re.compile(r"\d\s*등급"),
    "운용사":       re.compile(r"자산운용|투자신탁운용|투신운용"),
    "상품분류":     re.compile(r"증권\s*\(|투자신탁\s*,|종류\s*및\s*형태"),
    "비교지수":     re.compile(r"비교지수|참조지수|벤치마크"),
    "수익률_해당없음": re.compile(r"연평균수익률|해당사항\s*없음|신규\s*설정"),
    "자산유형":     re.compile(r"증권\s*\(|투자신탁\s*,|종류\s*및\s*형태"),
    "펀드수익률":    re.compile(r"연평균수익률|운용실적|수익률"),
    "비교지수수익률":  re.compile(r"비교지수|참조지수|연평균수익률"),
    "수익률변동성":   re.compile(r"변동성|VaR"),
    "수익률_대표클래스": re.compile(r"연평균수익률|종류|Class"),
}
SEARCH_SEC = [("13항보수", 8), ("제2부6항구조", 6), ("11항매입환매", 6),
              ("제1부1항", 3), ("요약정보", 4)]

# ★ 사용자가 원문을 직접 확인해 기록한 9건 — 탐색보다 우선한다
MANUAL = {
    "R2_KR5144420020": {"운용사": (1, "NH-Amundi 자산운용주식회사")},
    "R2_KR5144420081": {"운용사": (1, "NH-Amundi 자산운용주식회사")},
    "R2_KR5144420091": {"운용사": (1, "NH-Amundi 자산운용주식회사")},
    "R2_KR514X450008": {"운용사": (1, "㈜브이아이피자산운용")},
    "R2_KR5156450026": {"상품분류": (11, "가. 집합투자기구의 종류 및 형태: 증권(주식형), 투자신탁")},
    "R2_KR5118420006": {"위험등급": (4, "5등급(낮은위험)"), "위험라벨": (4, "5등급(낮은위험)")},
    "R2_KR5118420062": {"위험등급": (4, "6등급(매우낮은위험)"), "위험라벨": (4, "6등급(매우낮은위험)")},
    "R2_KR5144450095": {"비교지수": (6, "(주1) 참조지수: KOSPI")},
    "R2_KR5127450117": {"펀드수익률": (37, "A 수수료선취-오프라인 81.86 41.22 27.24 18.77 9.23")},
    "R2_KR5152420028": {"펀드수익률": (37, "수수료선취-오프라인(A) 4.98 3.63 2.79 1.80 1.37")},
    "R2_KR5147430065": {"수익률_해당없음": (42, "가. 연평균수익률(세전기준): 신규설정으로 해당사항없음")},
}
NO_EVIDENCE = {                       # 부재가 근거라 인용문이 없는 건
    "R2_KR514X450008": ["수익률_해당없음"],
    "R2_KR5153420339": ["수익률_해당없음"],
}

BROKEN = re.compile(r"-{3,}|_{3,}")   # 좌표 재조립 흔적


def readable(line):
    """사람이 읽을 수 있는 줄인가 — 인용 가능성 판정."""
    s = line.strip()
    if not (10 <= len(s) <= 120):
        return False
    if BROKEN.search(s):
        return False
    han = len(re.findall(r"[가-힣]", s))
    return han >= 6 and han / len(s) >= 0.3


def pages_of(did):
    p = os.path.join(DUMP, did + ".json")
    if not os.path.exists(p):
        return {}
    return {x["page"]: x["text"] for x in json.load(open(p, encoding="utf-8"))["pages"]}


def search(did, field, by_page):
    """(page, quote, section) 또는 None. 힌트 페이지 → 섹션 범위 → 전체 순."""
    pat = KEYS.get(field)
    if not pat:
        return None
    order = []
    h = (hint.get(did) or {})
    hp = h.get("페이지") if isinstance(h, dict) else None
    if isinstance(hp, int):
        order.append((hp, h.get("섹션") or "힌트"))
    sec = smap.get(did, {}).get("sec", {})
    npages = smap.get(did, {}).get("pages", max(by_page) if by_page else 0)
    for key, smax in SEARCH_SEC:
        rng = span(sec, key, npages, smax)
        if rng:
            order += [(p, key) for p in range(rng[0], rng[1] + 1)]
    order += [(p, "전체") for p in sorted(by_page)]

    seen = set()
    for pg, secname in order:
        if pg in seen or pg not in by_page:
            continue
        seen.add(pg)
        for line in by_page[pg].split("\n"):
            if pat.search(line) and readable(line):
                return pg, re.sub(r"\s+", " ", line.strip()), secname
    return None


out, stat = {}, {"ok": 0, "manual": 0, "확인필요": 0, "근거불가": 0}
for did, patch in ov.items():
    by_page = pages_of(did)
    rec = {}
    for field, value in patch.items():
        if field.startswith("_") or field.endswith("_사유"):
            continue

        if did in MANUAL and field in MANUAL[did]:      # ① 사람이 확인한 값
            pg, q = MANUAL[did][field]
            rec[field] = {"page": pg, "quote": q, "section": "수기확인",
                          "확인필요": False}
            stat["manual"] += 1
            continue

        if field in NO_EVIDENCE.get(did, []) or value is False:
            rec[field] = {"page": None, "quote": None, "section": None,
                          "확인필요": False, "근거불가": True,
                          "사유": "값이 '없음'이라 부재가 근거 — 인용할 문장 없음"}
            stat["근거불가"] += 1
            continue

        hit = search(did, field, by_page)               # ② 텍스트 레이어 탐색
        if hit:
            pg, q, secname = hit
            rec[field] = {"page": pg, "quote": q, "section": secname,
                          "확인필요": False}
            stat["ok"] += 1
        else:
            rec[field] = {"page": None, "quote": None, "section": None,
                          "확인필요": True,
                          "사유": "근거 문장 미발견 — 원문 확인 필요"}
            stat["확인필요"] += 1
    if rec:
        out[did] = rec

if os.path.exists(EV_PATH) and not os.path.exists(EV_PATH + ".bak"):
    shutil.copy2(EV_PATH, EV_PATH + ".bak")
json.dump(out, open(EV_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

n = sum(len(v) for v in out.values())
print(f"문서 {len(out)}개 / 필드 {n}건 → {EV_PATH}")
print(f"  탐색 성공 {stat['ok']}  수기확인 {stat['manual']}  "
      f"근거불가(값 없음) {stat['근거불가']}  확인필요 {stat['확인필요']}")

need = [(d, f) for d, v in out.items() for f, e in v.items() if e.get("확인필요")]
if need:
    print(f"\n⚠️ 원문 확인이 필요한 {len(need)}건:")
    for d, f in need[:20]:
        print(f"   {d}  {f}")

print("\n--- 샘플 5건 ---")
for d, v in list(out.items())[:5]:
    for f, e in v.items():
        tag = "근거불가" if e.get("근거불가") else ("확인필요" if e["확인필요"] else f"p.{e['page']}")
        print(f"  {d}  {f:<14} {tag:<8} {(e['quote'] or '')[:58]}")
