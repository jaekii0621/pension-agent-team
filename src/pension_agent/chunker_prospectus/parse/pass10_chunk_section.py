"""
10패스 — 섹션 원문 청크 생성

서술이 답인 부분(투자대상·투자전략·투자위험·매입환매·별첨)을 원문 그대로 청킹한다.
숫자·필드는 펀드 카드와 클래스 카드가 담당하고, 여기는 **인용될 문장**을 담는다.

⚠️ 포함/배제 판단
  포함: 투자대상 · 투자전략 · 투자위험 · 11항 매입환매 · 별첨1
  배제:
    · 제3부(719p) — 수익률·시장잔고·회전율은 이미 필드로 뽑아 펀드 카드에 들어갔다.
      span이 9페이지 상한에 걸린 문서가 58건이라 재무제표 숫자 나열이 대량 유입된다.
    · 제4부(469p) — 판매회사명·전화번호 나열. 검색 가치 대비 양이 너무 많다.
    · 13항 보수(342p) — 클래스 카드가 같은 내용을 표로 담고 있다(중복).
    · 연혁 — 과거 위험등급이 섞여 있어 오염원.
    · 12항 · 제5부 · 14항 과세 — 안내문서 docs가 세제 정본(8/6 Q&A: 상충 시 docs 우선).

⚠️ 분할 규칙
  ① 페이지 경계를 넘지 않는다. `page`가 int 단일 필드라 청크가 여러 페이지에
     걸치면 근거 표시가 부정확해진다(평가지표: 근거 문서 표시 필수).
  ② 페이지 안에서 800자를 넘으면 **문단 경계**로 자른다. 문장 중간에서 자르면
     인용이 깨진다.
  ③ 표로 보이는 블록(숫자·구분자 비율이 높은 줄 뭉치)은 쪼개지 않는다.
  ④ 너무 짧은 조각(<120자)은 앞 청크에 붙인다.

산출: data/chunks/section_chunks.jsonl
"""
import os, re, re, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import DUMP_DIR, TABLE_DIR, load_json, make_embed_text
from parse.pass2_sections import HEADINGS, MAX_HEAD_LEN
from parse.pass3_extract import span

CHUNK_DIR = os.path.join(os.path.dirname(TABLE_DIR), "chunks")

# (섹션키, span_max, 표시명)
# ★ 계약 INTERFACES.md §1-3의 "✅ 포함/정본" 목록을 따른다.
#   ⚠️ 카드에 값이 있다고 원문을 빼면 안 된다. 카드는 총보수만 담고 있어서
#      "환매수수료 있나요?" 같은 질의에 답할 근거가 사라졌다(13항 누락).
KEEP = [
    ("요약정보",     4, "요약정보"),          # 계약 ✅ 포함 — OPTIONAL
    ("제1부1항",     3, "제1부 클래스"),       # 계약 ✅ 정본 — 클래스 목록·협회코드
    ("투자대상",     5, "투자대상"),
    ("투자전략",     5, "투자전략"),
    ("투자위험",     5, "투자위험"),
    ("11항매입환매", 6, "매입·환매"),
    ("13항보수",     8, "보수·수수료"),        # 계약 ✅ 정본 — 환매·판매수수료 원문
    ("별첨1",        4, "별첨1"),
    ("제3부",        8, "재무·운용실적"),      # 계약 ✅ 포함 — 수익률 원문·기준일
    ("제4부",        6, "관련회사"),           # 계약 ✅ 포함(1·3항) — OPTIONAL
]

# 계약에 있으나 질의 세트에 대응 문항이 없고 청크만 2,060개 늘어나는 둘은
# 기본적으로 뺀다. 필요해지면 이 집합을 비우면 된다.
_OPTIONAL = {"요약정보", "제4부"}
# ★ 요약정보·제4부도 계약 §1-3의 "✅ 포함" 항목이라 기본 포함으로 바꿨다.
#   빼려면 SKIP_OPTIONAL=1 로 실행한다.
if os.environ.get("SKIP_OPTIONAL") == "1":
    KEEP = [k for k in KEEP if k[0] not in _OPTIONAL]

# 계약이 ❌로 지정해 넣지 않는 것:
#   연혁(과거 등급·보수 오염원) / 12항 기준가격산정 / 14항 과세(distractor) /
#   제5부(법정 문구) / 엄브렐러 전환 대상 목록(다른 펀드 정보)

MAX_LEN = 850            # 산문 조각 상한 (508토큰 ≈ 870자 실측 기준)
TABLE_MAX = 600          # 표 조각 상한 — 숫자·구분자는 토큰을 더 먹는다
MIN_LEN = 120



# ===== 헤딩 오프셋 절단 =====================================================
BOUNDARY_ONLY = {
    "12항기준가격": [
        r"^12\s*[\.．]\s*(?:집합투자증권의?\s*)?기준\s*가격",
        r"^\d+\s*[\.．]\s*집합투자증권의?\s*기준가격\s*산정",
        r"^\d+\s*[\.．]\s*기준가격\s*(?:의\s*)?산정",
    ],
    "엄브렐러전환": [
        r"^\d+\s*[\.．]\s*전환\s*(?:형|가능)",
        r"^전환\s*가능한?\s*집합투자기구",
    ],
}

_ALL = {}
for _k, _v in list(HEADINGS.items()) + list(BOUNDARY_ONLY.items()):
    _ALL[_k] = [re.compile(p) for p in _v]

STATS = {}


def _iter_heading_lines(text):
    pos = 0
    for line in text.split("\n"):
        s = line.strip()
        if s and len(s) <= MAX_HEAD_LEN:
            yield pos, s
        pos += len(line) + 1


def heading_offset(text, key):
    pats = _ALL.get(key)
    if not pats or not text:
        return None
    for off, line in _iter_heading_lines(text):
        if any(p.match(line) for p in pats):
            return off
    return None


def next_boundary(by_page, start_pg, start_off, last_pg, self_key, toc):
    best = None
    for pg in range(start_pg, last_pg + 1):
        if pg in toc:
            continue
        for off, line in _iter_heading_lines(by_page.get(pg) or ""):
            if pg == start_pg and start_off is not None and off <= start_off:
                continue
            for key, pats in _ALL.items():
                if key == self_key:
                    continue
                if any(p.match(line) for p in pats):
                    if best is None or (pg, off) < best:
                        best = (pg, off)
                    break
            if best and best[0] == pg:
                break
        if best:
            break
    return best if best else (last_pg, None)


def page_slices(sec, key, npages, smax, by_page, toc):
    rng = span(sec, key, npages, smax)
    if not rng:
        return []
    a, b = rng
    # 문서 앞머리 섹션은 헤딩 정규식이 본문 줄에 오탐하기 쉬워 페이지 처음부터 쓴다
    HEAD_FROM_PAGE = {"요약정보"}
    if key in HEAD_FROM_PAGE:
        start_off = 0
    else:
        start_off = heading_offset(by_page.get(a) or "", key)
    k = "head_ok" if start_off is not None else "head_miss"
    STATS[k] = STATS.get(k, 0) + 1
    if start_off is None:
        STATS.setdefault("miss_keys", set()).add(key)
        start_off = 0
    # ★ patch24 — 경계 탐색만 span 밖으로 넓힌다.
    #   span 의 smax 는 헤딩 미검출 시 폭주를 막는 안전장치인데, 그 상한이
    #   next_boundary 의 시야까지 막아 다음 섹션 시작 페이지의 '헤딩 앞부분'
    #   (= 이 섹션의 정당한 꼬리)이 통째로 버려졌다. VaR 20건이 이 경우다.
    #   실제 헤딩을 찾은 경우에만 확장분을 쓰므로 폭주 위험은 없다.
    LOOKAHEAD = 3
    end_pg, end_off = next_boundary(
        by_page, a, start_off, min(npages, b + LOOKAHEAD), key, toc)
    if end_off is None:
        end_pg, end_off = b, None          # 헤딩 미검출 → 기존 동작(smax 상한)
    elif end_pg > b:
        STATS["tail_recovered"] = STATS.get("tail_recovered", 0) + 1
    else:
        STATS["tail_trimmed"] = STATS.get("tail_trimmed", 0) + 1
    out = []
    for pg in range(a, end_pg + 1):
        if pg in toc:
            continue
        text = by_page.get(pg) or ""
        if not text:
            continue
        s = start_off if pg == a else 0
        e = end_off if (pg == end_pg and end_off is not None) else len(text)
        piece = text[s:e].strip()
        if piece:
            out.append((pg, piece))
    return out


def _long_ratio(t):
    ls = [l.strip() for l in (t or "").split("\n") if l.strip()]
    return sum(1 for l in ls if len(l) > 40) / max(len(ls), 1)


def resolve_sec(sec, by_page, toc):
    """section_map에 없는 섹션을 페이지 직접 스캔으로 보충한다.
    common.py는 건드리지 않으므로 pass2~5 결과에 영향이 없다."""
    out = dict(sec)
    for key, smax, label in KEEP:
        if key in out:
            continue
        pats = _ALL.get(key) or []
        for pg in sorted(by_page):
            if pg == 1:
                continue
            text = by_page[pg]
            if pg in toc and _long_ratio(text) <= 0.15:
                continue          # 긴 문장 없는 진짜 목차만 건너뛴다
            for off, line in _iter_heading_lines(text):
                if any(p.match(line) for p in pats):
                    out[key] = pg
                    break
            if key in out:
                break
    return out


BLEED_KEYS = ["12항기준가격", "14항과세", "연혁", "제5부", "엄브렐러전환"]


def check_bleed(chunks):
    hit = {k: 0 for k in BLEED_KEYS}
    for c in chunks:
        for off, line in _iter_heading_lines(c.get("text") or ""):
            for k in BLEED_KEYS:
                if any(p.match(line) for p in _ALL.get(k, [])):
                    hit[k] += 1
                    break
    return hit
# ===========================================================================

def is_table_block(text):
    """숫자·구분자가 많고 줄이 짧으면 표로 본다."""
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    short = sum(1 for l in lines if len(l.strip()) <= 30)
    numeric = sum(1 for l in lines if len(re.findall(r"[\d.,%-]", l)) > len(l) * 0.35)
    return short / len(lines) > 0.6 or numeric / len(lines) > 0.5


# ★ 표 조각마다 앞에 복제할 머리줄을 고른다.
#   ⚠️ 길이 상한이 없으면 잘린 본문 문장("…이란 집합투자기구 수익률이")이 머리로
#      뽑혀 모든 조각에 복제된다. 짧은 줄만 머리로 인정한다.
COLHEAD = re.compile(r"^(구\s*분|종\s*류|클래스|항\s*목)\b|구\s*분\s")


def _table_head(lines):
    head = []
    for l in lines[:3]:
        if len(l) <= 45 and len(re.findall(r"\d", l)) <= len(l) * 0.25:
            head.append(l)
        else:
            break
    head = head[:2]
    for l in lines[:6]:                      # 열 머리행(구분/종류…)이 있으면 함께
        if l not in head and COLHEAD.search(l) and len(l) <= 120:
            head.append(l)
            break
    return head


def split_table(text):
    """
    ★ 표를 줄 단위로 나누되 **머리줄을 각 조각에 복제**한다.

    ⚠️ 예전 규칙은 '표는 통째로'였다. 헤더를 잃지 않으려는 의도는 맞지만
       75줄짜리 표가 한 청크가 되고, 임베딩(512토큰)에서 뒤가 통째로 잘려
       인덱싱됐다(전체 글자의 16%가 검색 불가). 머리를 복제하면 두 목적이 다 된다.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if len(text) <= TABLE_MAX or len(lines) < 2:
        return [text]
    head = _table_head(lines)
    body = lines[len(head):]
    if not body:
        return [text]
    hd = "\n".join(head)
    room = TABLE_MAX - len(hd) - 12
    if room < 150:                           # 머리가 너무 길면 머리 없이 자른다
        hd, room = "", TABLE_MAX

    groups, cur, clen = [], [], 0
    for l in body:
        if cur and clen + len(l) + 1 > room:
            groups.append(cur)
            cur, clen = [], 0
        cur.append(l)
        clen += len(l) + 1
    if cur:
        groups.append(cur)
    if len(groups) > 1 and sum(len(x) for x in groups[-1]) < MIN_LEN:
        last = groups.pop()                  # 꼬리 조각이 짧으면 앞에 붙인다
        groups[-1] += last

    out = []
    for i, g in enumerate(groups):
        pre = hd if i == 0 else ((hd + "  (표 계속)") if hd else "(표 계속)")
        out.append("\n".join(([pre] if pre else []) + g))
    return out


def split_sentences(p, limit):
    """한 문단이 상한을 넘으면 문장 경계로 나눈다. 문장 중간 절단을 피한다."""
    sents = re.split(r"(?<=다\.)\s+|(?<=\.)\s+(?=[가-힣A-Z])", p)
    res, cur = [], []
    for s in sents:
        if len(s) > limit:                   # 한 문장이 이미 상한 초과 — 강제 절단
            if cur:
                res.append(" ".join(cur))
                cur = []
            res.extend(s[i:i + limit] for i in range(0, len(s), limit))
            continue
        if cur and sum(len(x) + 1 for x in cur) + len(s) > limit:
            res.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        res.append(" ".join(cur))
    return [x for x in res if x.strip()]


def split_page(text):
    """페이지 텍스트를 조각으로 나눈다. 표는 머리 복제, 산문은 문단→문장 순."""
    text = text.strip()
    if not text:
        return []
    if is_table_block(text):
        return split_table(text)
    if len(text) <= MAX_LEN:
        return [text]

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) == 1:
        paras = [l.strip() for l in text.split("\n") if l.strip()]

    out, cur = [], ""
    for p in paras:
        if len(p) > MAX_LEN:                 # ★ 예전엔 그대로 뒀다 → 문장 분할
            if cur:
                out.append(cur)
                cur = ""
            out.extend(split_sentences(p, MAX_LEN))
            continue
        if len(cur) + len(p) + 1 > MAX_LEN and cur:
            out.append(cur)
            cur = p
        else:
            cur = (cur + "\n" + p).strip() if cur else p
    if cur:
        out.append(cur)

    merged = []
    for c in out:
        if merged and len(c) < MIN_LEN:
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)
    return merged




def build(rec, smap_entry, pages):
    """한 문서의 섹션 청크 목록."""
    sec = smap_entry.get("sec", {})
    npages = smap_entry.get("pages", len(pages))
    by_page = {p["page"]: p["text"] for p in pages}
    toc = {p for p in (smap_entry.get("toc") or [])
           if _long_ratio(by_page.get(p, "")) <= 0.15}
    sec = resolve_sec(sec, by_page, toc)
    out = []
    # ★ 섹션 카드가 #c001(펀드)·#c002(클래스)를 쓰므로 본문은 #c010부터
    seq = {}

    for key, smax, label in KEEP:
        for pg, txt in page_slices(sec, key, npages, smax, by_page, toc):
            if len(txt) < MIN_LEN:
                continue
            for piece in split_page(txt):
                if len(piece) < MIN_LEN:
                    continue
                # ★ 제1부 1항은 다음 섹션 헤딩이 없는 문서에서 3항(모집예정금액)·
                #   5항(운용전문인력)까지 밀고 나간다. 계약이 지정한 건 "클래스 목록"뿐이라
                #   협회코드나 종류 표기가 없는 조각은 버린다.
                if key == "제1부1항" and not re.search(
                        r"종류\s*[A-Za-z]|\(Class\)|펀드\s*코드|[A-Z]{1,2}\d{3,4}", piece):
                    continue
                seq[pg] = seq.get(pg, 9) + 1
                out.append({
                    # ── 필수 7 ──
                    "chunk_id": f"{rec['doc_id']}#p{pg}#c{seq[pg]:03d}",
                    "doc_id": rec["doc_id"],
                    "source": "상품",
                    "page": pg,
                    "section": label,
                    "text": piece,
                    "parse_method": "text",
                    "confidence": 1.0,          # 계약 §2-1
                    "table_ref": None,
                    # ★ 인덱싱 입력. text는 인용용이라 자르지 않는다.
                    "embed_text": make_embed_text(
                        f"{rec.get('fund_name') or ''} · {label}", piece),
                    # ── 조건부 ──
                    "chunk_type": "section",
                    "fund_code": rec.get("fund_codes") or None,
                    "fund_name": rec.get("fund_name"),
                    "위험등급": rec.get("위험등급"),
                    "자산유형": rec.get("자산유형"),
                    "연금저축_가능": rec.get("연금저축_가능"),
                    "퇴직연금_가능": rec.get("퇴직연금_가능"),
                })
    return out


def main():
    six = load_json(os.path.join(TABLE_DIR, "six_axis.json"))
    smap = load_json(os.path.join(TABLE_DIR, "section_map.json"))
    os.makedirs(CHUNK_DIR, exist_ok=True)

    out = []
    for rec in six:
        did = rec["doc_id"]
        dp = os.path.join(DUMP_DIR, did + ".json")
        if did not in smap or not os.path.exists(dp):
            continue
        pages = load_json(dp)["pages"]
        out.extend(build(rec, smap[did], pages))

    path = os.path.join(CHUNK_DIR, "section_chunks.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    from collections import Counter
    ID = re.compile(r"^.+#p\d+#c\d{3}$")
    lens = [len(c["text"]) for c in out]
    ids = [c["chunk_id"] for c in out]
    print(f"섹션 청크 {len(out)}개 → {path}")
    print(f"  chunk_id 위반 {sum(1 for i in ids if not ID.match(i))} / "
          f"중복 {len(ids) - len(set(ids))}")
    print(f"  길이 min {min(lens)} / 중앙 {sorted(lens)[len(lens)//2]} / max {max(lens)}")
    print(f"  {MAX_LEN}자 초과 {sum(1 for x in lens if x > MAX_LEN)}개")
    print(f"  문서 {len({c['doc_id'] for c in out})}/92")
    print("\n섹션별:", dict(Counter(c["section"] for c in out)))
    _miss = STATS.pop("miss_keys", set())
    print("\n[절단 진단]", STATS, " 헤딩 미검출:", sorted(_miss) if _miss else "없음")
    print("[배제 섹션 유입]", check_bleed(out), " <- 전부 0이어야 정상")

    print("\n--- 샘플 ---")
    print(out[0]["chunk_id"], "|", out[0]["section"])
    print(out[0]["text"][:400])


if __name__ == "__main__":
    main()
