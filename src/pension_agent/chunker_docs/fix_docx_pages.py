"""
장르B/D(docx 원본) 청크들의 page 필드를, Word로 변환해둔 PDF와 텍스트 매칭해서 채워넣는 스크립트.

사전 준비:
1. data/raw/ 폴더에 각 docx와 같은 이름의 pdf가 있어야 함 (예: doc20.docx 옆에 doc20.pdf)
   -> Word에서 "다른 이름으로 저장 -> PDF"로 만든 바로 그 파일
2. data/chunks/chunks_genre_b.jsonl, chunks_genre_d.jsonl 이 이미 있어야 함 (기존 page=1 상태 그대로 둬도 됨,
   이 스크립트가 읽어서 page/chunk_id를 고친 버전으로 덮어씀)

동작 방식:
- 청크는 원래 만들어진 순서 그대로(문서 안에서 위->아래 순서)라는 전제 하에,
  각 청크의 매칭용 텍스트가 변환된 PDF의 몇 페이지에 처음 등장하는지 순서대로 찾아나감.
- 순서를 보장하기 위해, 항상 "직전 청크가 매칭된 페이지"부터 검색 시작 (거꾸로 안 감).
- 못 찾으면 직전 페이지값을 그대로 이어받음(같은 페이지에 있다고 간주) + page_unverified 표시.

⚠️ 설계 노트 (스키마 확장 없이 해결한 방법):
  table_row/calc_box_text 청크의 text는 "헤더: 값" 형태로 재구성돼 있어서 원본 PDF에
  그 문자열 그대로 없다 — 그래서 text 그대로 앵커 매칭하면 거의 항상 실패한다.
  이걸 고치려고 "매칭용 필드를 청크에 새로 저장"하는 방법도 시도해봤는데, 굳이
  최종 데이터 스키마를 늘릴 필요가 없었다: build_chunks_genre_b.py가 이미
  "docx table -> grid" 변환 로직(resolve_table_grid 등)을 갖고 있고, data/raw/에
  원본 docx도 그대로 있으니, 이 스크립트가 build_chunks_genre_b를 그대로 import해서
  **원본 docx를 한 번 더 읽어 매칭용 정보를 그때그때 계산**하면 된다. 청크 만들 때와
  똑같은 순서로 순회하므로(python-docx는 항상 같은 순서로 읽음), jsonl에 저장된
  청크들과 1:1로 zip해서 대응시킬 수 있다. 이 매칭용 정보는 이 스크립트 실행 중에만
  잠깐 쓰고 버려지며, 최종 jsonl에는 절대 저장되지 않는다 (스키마 그대로 유지).
"""
import json
import re
import os
import sys
import pdfplumber

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "..", "data", "raw"))
CHUNKS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "..", "data", "chunks"))

sys.path.insert(0, BASE_DIR)
import build_chunks_genre_b as genb  # noqa: E402  (원본 docx -> grid 변환 로직 재사용)

# 장르B: 전부 docx
GENRE_B_DOCX_IDS = ["doc20", "doc23", "doc25", "doc26", "doc36", "doc38", "doc39",
                     "doc40", "doc41", "doc42", "doc43", "doc44", "doc45", "doc51", "doc52"]

# 장르D 중 docx인 것만 (doc34는 xlsx라 이 스크립트 대상 아님 - 별도 처리)
GENRE_D_DOCX_IDS = ["doc55", "doc6", "doc35"]

TABLE_ORIGINS = ("table_row", "calc_box_text")


def normalize(text):
    """공백/줄바꿈 차이를 무시하고 비교하기 위한 정규화"""
    return re.sub(r'\s+', '', text or "")


def load_page_texts(doc_id):
    pdf_path = os.path.join(RAW_DIR, f"{doc_id}.pdf")
    if not os.path.exists(pdf_path):
        print(f"  ⚠️ {doc_id}.pdf 없음 (data/raw/ 확인 필요) - 건너뜀")
        return None
    with pdfplumber.open(pdf_path) as pdf:
        return [normalize(p.extract_text() or "") for p in pdf.pages]


def load_page_tables(doc_id):
    """find_tables()로 표를 구조화해서 페이지별로 뽑는다 (장르A와 동일한 방식).
    격자선 위치 기반이라 셀이 세로 병합돼 있어도 순서가 뒤섞이지 않는다.
    문자열 매칭이 실패했을 때의 안전망으로 쓴다 — 표 대 표(셀 리스트 대 셀 리스트) 비교."""
    pdf_path = os.path.join(RAW_DIR, f"{doc_id}.pdf")
    if not os.path.exists(pdf_path):
        return None
    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for page in pdf.pages:
            page_tables = []
            for t in page.find_tables():
                data = t.extract()
                rows = [[normalize(cell) for cell in row if cell] for row in data]
                page_tables.append(rows)
            pages.append(page_tables)
        return pages


def build_match_info(doc_id):
    """⚠️ 핵심: 원본 docx를 build_chunks_genre_b의 로직으로 다시 한번 읽어서,
    표/계산박스 청크(생성 순서 그대로)에 대응하는 "매칭용 텍스트"와 "매칭용 셀
    리스트"를 이 스크립트 실행 중에만 임시로 계산한다. 저장은 안 함 — 스키마 그대로 유지."""
    genb.RAW_DIR = RAW_DIR  # fix_docx_pages.py 쪽 data/raw 경로를 그대로 씀
    doc = genb.Document(os.path.join(RAW_DIR, f"{doc_id}.docx"))

    match_infos = []  # [(anchor_text, cell_values), ...] — 표/계산박스 청크와 같은 순서
    table_idx = 0
    for kind, block in genb.iter_block_items(doc):
        if kind == "paragraph":
            continue
        elif kind == "table":
            table_idx += 1
            ttype = genb.table_type(doc_id, table_idx)
            if ttype == "flowchart":
                continue
            grid = genb.resolve_table_grid(block)
            if ttype == "calc_box":
                text, raw_anchor, cell_rows = genb.calc_box_to_text(grid)
                if text:
                    flat_cells = [c for row in cell_rows for c in row]
                    match_infos.append((raw_anchor, flat_cells))
            else:
                for rt, raw_anchor, cell_values in genb.normal_table_rows_to_texts(grid):
                    match_infos.append((raw_anchor, cell_values))
    return match_infos


def anchor_source(chunk, match_info):
    """페이지 매칭에 쓸 텍스트를 고른다.
    표/계산박스 origin이면 방금 docx에서 다시 뽑은 매칭 텍스트, 아니면 text 그대로."""
    if match_info is not None:
        return match_info[0]
    return chunk.get("text") or ""


def find_page(anchor_text, page_texts, start_page_idx):
    """매칭용 텍스트 앞부분이 어느 페이지(1-based)에 있는지, start_page_idx(0-based)부터 순서대로 검색"""
    norm = normalize(anchor_text)
    for anchor_len in (40, 25, 15):
        anchor = norm[:anchor_len]
        if not anchor:
            continue
        for p_idx in range(start_page_idx, len(page_texts)):
            if anchor in page_texts[p_idx]:
                return p_idx + 1
    return None


def find_page_structural(match_cells, page_tables, start_page_idx):
    """문자열 매칭이 실패했을 때 쓰는 안전망.
    ⚠️ 수정 (doc55 실측 반영): 처음엔 "한 행(row) 안에 셀 값이 다 있는지"로 봤는데,
    셀 안 텍스트가 길게 줄바꿈되는 표는 Word→PDF 변환 시 줄바꿈마다 얇은 경계선이
    추가로 생겨서 find_tables()가 논리적으로 한 행인 걸 물리적으로 여러 행으로
    쪼개 인식한다 (doc55 "부담금 유형" 표에서 실측 확인: 36행×7열로 조각남).
    그래서 "행 단위"가 아니라 "페이지 전체 표 셀 값을 원래 순서 그대로 이어붙인
    문자열" 안에서 찾는 방식으로 완화한다 — 줄바꿈으로 쪼개졌던 조각들이
    이어붙이면 다시 원문 그대로 복원되기 때문에 순서 보존이 핵심이다."""
    if not match_cells or not page_tables:
        return None
    cells = [normalize(c) for c in match_cells if c and len(normalize(c)) >= 2]
    if not cells:
        return None

    best_page, best_score = None, 0
    for p_idx in range(start_page_idx, len(page_tables)):
        # ⚠️ set이 아니라 리스트로 순서 보존한 채 이어붙인다 (핵심)
        flat_parts = []
        for table_rows in page_tables[p_idx]:
            for row in table_rows:
                for cell in row:
                    if cell:
                        flat_parts.append(cell)
        flat_text = "".join(flat_parts)
        score = sum(1 for c in cells if c in flat_text)
        if score > best_score:
            best_score, best_page = score, p_idx + 1
    if best_page is not None and best_score >= max(1, (len(cells) + 1) // 2):
        return best_page
    return None


_SENT_SPLIT = re.compile(
    r'(?<=다\.)\s+|(?<=요\.)\s+|(?<=\.)\s+(?=[가-힣A-Z])|(?=\d{1,2}\.\s[가-힣])'
)


def split_into_sentences(text):
    """문장 경계에서 나눈다 (민서님 pass10_chunk_section.py의 split_sentences와
    동일한 원칙 — 단어/문장 중간 절단을 피한다)."""
    parts = _SENT_SPLIT.split(text)
    return [p for p in parts if p.strip()]


def split_paragraph_across_pages(text, head_page, page_texts):
    """⚠️ 신규 (doc38/40/45/doc42 실측으로 발견된 문제 해결):
    build_chunks_genre_b.py는 docx를 읽을 때 페이지 개념이 아예 없어서, 원본에서
    이미 페이지 경계를 걸치고 있는 문단이 통째로 하나의 청크에 담기는 경우가 있다
    (예: doc42 "운용기간 중 발생한 이익과 손실이..." 문장이 실제로는 2페이지인데
    직전 문장이 1페이지라서 청크 전체가 page:1로 찍힘. 문자열 머리(anchor) 매칭만
    확인하고 꼬리는 확인 안 해서 조용히 틀린 값이 저장되고 있었음).

    문장 단위로 순회하면서, 각 문장이 head_page 페이지 텍스트에 있는지 확인.
    없고 그 뒤 페이지 텍스트에 있으면, 거기서 쪼개서 새 그룹을 시작한다.
    반환: (groups, unresolved)
      - groups: [(text, page), (text, page), ...] — 걸치지 않으면 길이 1.
      - unresolved: 문장 하나가 통째로 페이지 경계를 걸쳐서(예: doc43 "이를 | 예방하려면"
        처럼 단어 중간에서 페이지가 갈려버려서) 앞/뒤 어느 페이지에서도 못 찾은 문장이
        하나라도 있었으면 True. 텍스트는 그대로 유지하되(억지로 쪼개지 않음), 호출부에서
        이 값을 보고 page_unverified 플래그만 붙인다 — "이 청크의 페이지 번호는 100%
        확신할 수 없다"는 신호를 남기기 위함.
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return [(text, head_page)], False

    groups = []
    cur_page = head_page
    cur_sentences = []
    pending_short = []  # ⚠️ 신규: 판단을 유보한 짧은 문장(조항번호 등).
                         # 다음으로 "정상 판단된" 문장이 속한 페이지에 함께 묶는다.
    unresolved = False
    for s in sentences:
        ns = normalize(s)
        if not ns:
            cur_sentences.append(s)
            continue
        if len(ns) < 4:
            # 너무 짧아서 자체 검색은 신뢰 못함(페이지 어디든 우연히 매칭될 수 있음).
            # 검색하지 않고 유보만 해뒀다가, 뒤따르는 문장이 결정되는 페이지를 그대로 따라간다.
            pending_short.append(s)
            continue

        head_idx = cur_page - 1
        if 0 <= head_idx < len(page_texts) and ns in page_texts[head_idx]:
            cur_sentences.extend(pending_short)
            pending_short = []
            cur_sentences.append(s)
            continue

        found_page = None
        for p_idx in range(head_idx + 1, len(page_texts)):
            if ns in page_texts[p_idx]:
                found_page = p_idx + 1
                break
        if found_page is not None:
            if cur_sentences:
                groups.append((" ".join(cur_sentences), cur_page))
            cur_page = found_page
            cur_sentences = pending_short + [s]  # 유보했던 짧은 문장도 새 페이지로 함께 이동
            pending_short = []
        else:
            unresolved = True
            cur_sentences.extend(pending_short)
            pending_short = []
            cur_sentences.append(s)

    if pending_short:
        # 문서/청크가 짧은 문장으로 끝나는 드문 경우 -> 마지막 그룹에 붙인다
        cur_sentences.extend(pending_short)
    if cur_sentences:
        groups.append((" ".join(cur_sentences), cur_page))
    return groups, unresolved


def process_file(jsonl_path, docx_doc_ids):
    if not os.path.exists(jsonl_path):
        print(f"❌ {jsonl_path} 없음")
        return

    with open(jsonl_path, encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]

    by_doc = {}
    for c in all_chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)

    updated_count = 0
    unmatched_count = 0
    skipped_docs = []

    for doc_id in docx_doc_ids:
        if doc_id not in by_doc:
            continue
        page_texts = load_page_texts(doc_id)
        if page_texts is None:
            skipped_docs.append(doc_id)
            continue
        page_tables = load_page_tables(doc_id)

        chunks = by_doc[doc_id]

        # ⚠️ 표/계산박스 청크에만 순서대로 매칭 정보를 대응시킨다.
        table_match_infos = build_match_info(doc_id)
        table_chunk_indices = [i for i, c in enumerate(chunks) if c.get("origin") in TABLE_ORIGINS]
        match_by_chunk_idx = {}
        if len(table_chunk_indices) != len(table_match_infos):
            print(f"  ⚠️ {doc_id}: 표 청크 개수({len(table_chunk_indices)})와 docx 재파싱 결과"
                  f"({len(table_match_infos)})가 안 맞음 — 구조 매칭 없이 진행")
        else:
            for idx, info in zip(table_chunk_indices, table_match_infos):
                match_by_chunk_idx[idx] = info

        last_page = 1
        chunk_no_by_page = {}
        doc_unmatched = 0
        doc_structural = 0
        doc_split = 0
        doc_unresolved_sentence = 0  # ⚠️ 문장이 페이지를 걸치는데 어디서도 못 찾은 경우(카운트용)
        expanded_chunks = []  # 페이지 걸침 감지로 쪼개진 결과까지 포함한 최종 리스트
        for i, c in enumerate(chunks):
            match_info = match_by_chunk_idx.get(i)
            start_idx = max(0, last_page - 1)
            page = find_page(anchor_source(c, match_info), page_texts, start_idx)
            if page is None and match_info is not None:
                page = find_page_structural(match_info[1], page_tables, start_idx)
                if page is not None:
                    doc_structural += 1
            if page is None:
                page = last_page
                doc_unmatched += 1
                c["page_unverified"] = True
            else:
                c.pop("page_unverified", None)
            last_page = page

            # ⚠️ origin이 paragraph인 청크만, 꼬리가 같은 페이지에 있는지
            # 확인해서 안 맞으면 문장 단위로 쪼갠다 (doc38/40/45/42 실측 문제 해결).
            if c.get("origin") == "paragraph" and not c.get("page_unverified"):
                groups, unresolved = split_paragraph_across_pages(c["text"], page, page_texts)
                if unresolved:
                    doc_unresolved_sentence += 1
            else:
                groups, unresolved = [(c["text"], page)], False

            if len(groups) > 1:
                doc_split += 1
                for gi, (gtext, gpage) in enumerate(groups):
                    nc = dict(c)
                    nc["text"] = gtext if gi == 0 else f"(단락 계속 — {c.get('section') or c['doc_id']})\n{gtext}"
                    nc["page"] = gpage
                    if gi == len(groups) - 1 and unresolved:
                        # ⚠️ 문장 하나가 페이지를 걸쳐서 못 찾은 케이스는, 그 문장이 속한
                        # 마지막 그룹의 페이지 번호를 100% 확신할 수 없다는 표시를 남긴다.
                        nc["page_unverified"] = True
                    chunk_no_by_page[gpage] = chunk_no_by_page.get(gpage, 0) + 1
                    nc["chunk_id"] = f"{doc_id}#p{gpage}#c{chunk_no_by_page[gpage]:03d}"
                    expanded_chunks.append(nc)
                last_page = groups[-1][1]
            else:
                c["page"] = page
                if unresolved:
                    c["page_unverified"] = True
                chunk_no_by_page[page] = chunk_no_by_page.get(page, 0) + 1
                c["chunk_id"] = f"{doc_id}#p{page}#c{chunk_no_by_page[page]:03d}"
                expanded_chunks.append(c)

            updated_count += 1

        # ⚠️ 이 문서의 원래 청크들을 쪼개진 결과로 교체
        by_doc[doc_id] = expanded_chunks
        unmatched_count += doc_unmatched
        extra = f", 구조매칭으로 복구 {doc_structural}건" if doc_structural else ""
        extra2 = f", 페이지 걸침 감지로 {doc_split}건 분리" if doc_split else ""
        extra3 = f", ⚠️문장이 페이지 걸쳐서 확신 못한 것 {doc_unresolved_sentence}건(page_unverified)" if doc_unresolved_sentence else ""
        status = "✅" if doc_unmatched == 0 else f"⚠️ (그래도 매칭 실패 {doc_unmatched}건, 직전 페이지로 대체됨)"
        print(f"  {status} {doc_id}: {len(chunks)}개 청크 페이지 매칭 완료 (총 {len(page_texts)}페이지){extra}{extra2}{extra3}")

    # ⚠️ all_chunks를 by_doc 기준으로 재구성 (쪼개진 결과 반영, 원래 순서 유지)
    all_chunks = []
    for did in by_doc:
        all_chunks.extend(by_doc[did])

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n{os.path.basename(jsonl_path)}: {updated_count}개 청크 업데이트 (매칭 실패로 직전 페이지 대체 {unmatched_count}건)")
    if skipped_docs:
        print(f"  ⚠️ PDF 없어서 건너뛴 문서: {skipped_docs}")

print("BASE_DIR   :", BASE_DIR)
print("RAW_DIR    :", RAW_DIR, "| 존재함?", os.path.exists(RAW_DIR))
print("CHUNKS_DIR :", CHUNKS_DIR, "| 존재함?", os.path.exists(CHUNKS_DIR))

if __name__ == "__main__":
    print("=== 장르B 처리 ===")
    process_file(os.path.join(CHUNKS_DIR, "chunks_genre_b.jsonl"), GENRE_B_DOCX_IDS)

    print("\n=== 장르D 처리 (docx만, xlsx인 doc34 제외) ===")
    process_file(os.path.join(CHUNKS_DIR, "chunks_genre_d.jsonl"), GENRE_D_DOCX_IDS)

    print("\n완료!")
