from docx import Document
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
import openpyxl
import os
import re
import json
from chunk_genre_d_schema import ChunkD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw")
RAW_DIR = os.path.normpath(RAW_DIR)

MIN_CHUNK_LEN = 300
MAX_CHUNK_LEN = 800


# ---------- docx 공통 (장르B에서 검증된 병합 인식 로직 재사용) ----------

def iter_block_items(doc):
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            yield ('paragraph', Paragraph(child, doc))
        elif tag == 'tbl':
            yield ('table', Table(child, doc))


def get_real_row_cells(row):
    cells = []
    for tc in row._tr.tc_lst:
        cell = _Cell(tc, row.table)
        text = cell.text.strip()
        tcPr = tc.tcPr
        grid_span = 1
        vmerge = None
        if tcPr is not None:
            gs = tcPr.find(qn('w:gridSpan'))
            if gs is not None:
                grid_span = int(gs.get(qn('w:val')))
            vm = tcPr.find(qn('w:vMerge'))
            if vm is not None:
                val = vm.get(qn('w:val'))
                vmerge = val or 'continue'
        cells.append({"text": text, "colspan": grid_span, "vmerge": vmerge})
    return cells


def resolve_table_grid(table):
    raw_rows = [get_real_row_cells(row) for row in table.rows]
    expanded_rows = []
    for row in raw_rows:
        expanded = []
        for cell in row:
            for _ in range(cell["colspan"]):
                expanded.append({"text": cell["text"], "vmerge": cell["vmerge"]})
        expanded_rows.append(expanded)

    n_cols = max((len(r) for r in expanded_rows), default=0)
    grid = []
    col_last_value = [None] * n_cols
    for row in expanded_rows:
        resolved_row = []
        for col_idx in range(n_cols):
            if col_idx < len(row):
                cell = row[col_idx]
                if cell["vmerge"] == "continue":
                    value = col_last_value[col_idx] or ""
                else:
                    value = cell["text"]
                    col_last_value[col_idx] = value
            else:
                value = ""
            resolved_row.append(value)
        grid.append(resolved_row)
    return grid


def normal_table_rows_to_texts(grid):
    if len(grid) < 2:
        return [" / ".join(c for c in grid[0] if c)] if grid else []
    header, *data_rows = grid
    texts = []
    for row in data_rows:
        pairs = []
        for h, v in zip(header, row):
            if not v:
                continue
            pairs.append(f"{h}: {v}" if h else v)
        if pairs:
            texts.append(", ".join(pairs))
    return texts


def looks_like_heading(text):
    text = text.strip()
    if not text:
        return False
    if re.match(r'^\[.+\]$', text):          # [납입], [지급] 같은 대괄호 제목
        return True
    if re.match(r'^[가-힣]\.\s', text):       # 가. 나. 다.
        return True
    if re.match(r'^\d+\)\s', text):           # 1) 2) 3)
        return True
    if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]', text):
        return True
    if re.match(r'^\d+\.\s*[가-힣]', text) and len(text) <= 30:  # "1. 부담금 종류"
        return True
    if len(text) <= 20 and not text.endswith(('.', '다', '요', ')')):
        return True
    return False


def process_docx(doc_id, source):
    path = os.path.join(RAW_DIR, f"{doc_id}.docx")
    doc = Document(path)

    chunks = []
    buffer = []
    buffer_heading = None

    def flush_buffer():
        nonlocal buffer, buffer_heading
        if buffer:
            text = " ".join(buffer).strip()
            if text:
                chunks.append((text, "paragraph", buffer_heading))
        buffer = []

    for kind, block in iter_block_items(doc):
        if kind == "paragraph":
            text = block.text.strip()
            if not text:
                continue
            if looks_like_heading(text) and buffer and sum(len(b) for b in buffer) >= MIN_CHUNK_LEN:
                flush_buffer()
                buffer_heading = text
            elif looks_like_heading(text) and not buffer:
                buffer_heading = text
            buffer.append(text)
            if sum(len(b) for b in buffer) >= MAX_CHUNK_LEN:
                flush_buffer()

        elif kind == "table":
            grid = resolve_table_grid(block)
            flush_buffer()
            row_texts = normal_table_rows_to_texts(grid)
            for rt in row_texts:
                chunks.append((rt, "table_row", buffer_heading))

    flush_buffer()
    return chunks


# ---------- xlsx (doc34) ----------

def process_xlsx_doc34():
    path = os.path.join(RAW_DIR, "doc34.xlsx")
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Sheet1"]

    rows = list(ws.iter_rows(values_only=True))
    # 헤더 행("불가사유코드","불가사유") 찾기
    header_idx = None
    for i, row in enumerate(rows):
        if row[1] == "불가사유코드":
            header_idx = i
            break

    chunks = []
    for row in rows[header_idx + 1:]:
        code = (row[1] or "").strip()
        desc = (row[2] or "").strip()
        if not code:
            continue
        text = f"[실물이전 불가사유 {code}] {desc}"
        chunks.append((text, "table_row", "실물이전 불가사유 코드표"))
    return chunks


if __name__ == "__main__":
    all_chunks = []

    doc_specs = [
        ("doc55", "docx", "절차"),
        ("doc6", "docx", "절차"),
        ("doc35", "docx", "절차"),
    ]

    for doc_id, file_type, source in doc_specs:
        doc_chunks = process_docx(doc_id, source)
        for i, (text, origin, section) in enumerate(doc_chunks, start=1):
            c = ChunkD(
                chunk_id=f"{doc_id}#p1#c{i:03d}",
                doc_id=doc_id,
                source=source,
                page=1,
                text=text,
                parse_method="text",
                confidence=1.0,
                section=(section[:40] if section else None),
                origin=origin,
                file_type=file_type,
            )
            all_chunks.append(c)

    # doc34 (xlsx)
    doc34_chunks = process_xlsx_doc34()
    for i, (text, origin, section) in enumerate(doc34_chunks, start=1):
        c = ChunkD(
            chunk_id=f"doc34#p1#c{i:03d}",
            doc_id="doc34",
            source="절차",
            page=1,
            text=text,
            parse_method="text",
            confidence=1.0,
            section=section,
            origin=origin,
            file_type="xlsx",
        )
        all_chunks.append(c)

    OUTPUT_PATH = os.path.join(BASE_DIR, "..", "..", "..", "data", "chunks", "chunks_genre_d.jsonl")
    OUTPUT_PATH = os.path.normpath(OUTPUT_PATH)
        
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    print(f"총 {len(all_chunks)}개 청크 생성 → {OUTPUT_PATH}")
    by_doc = {}
    for c in all_chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    for doc_id, cs in by_doc.items():
        origins = {}
        for c in cs:
            origins[c.origin] = origins.get(c.origin, 0) + 1
        print(f"{doc_id}: 총 {len(cs)}개 | {origins}")
