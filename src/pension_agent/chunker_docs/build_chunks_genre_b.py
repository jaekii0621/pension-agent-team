from docx import Document
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
import os
import re
import json
from chunk_genre_b_schema import ChunkB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = "/mnt/user-data/uploads"  # 로컬에서는 data/raw로 바꿔서 쓰면 됨

DOC_IDS = ["doc20", "doc23", "doc25", "doc26", "doc36", "doc38", "doc39",
           "doc40", "doc41", "doc42", "doc43", "doc44", "doc45", "doc51", "doc52"]

# 표가 아니라 "흐름도"라서, 표 대신 이미 본문에 있는 서술 문장을 그대로 쓰기로 한 케이스
# (doc_id, 표 순번(1부터)) -> 스킵. 대신 문단이 순서상 그대로 청크가 됨
FLOWCHART_TABLES = {("doc23", 2)}

# 표가 아니라 "계산식 강조박스"라서, 행 단위 대신 계산 흐름을 문장으로 풀어써야 하는 케이스
CALC_BOX_TABLES = {("doc39", 1), ("doc52", 2)}

MIN_CHUNK_LEN = 300
MAX_CHUNK_LEN = 800


def iter_block_items(doc):
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            yield ('paragraph', Paragraph(child, doc))
        elif tag == 'tbl':
            yield ('table', Table(child, doc))


def get_real_row_cells(row):
    """병합 정보(gridSpan/vMerge)를 포함해서, 실제 존재하는 셀만 반환"""
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
    """
    병합 태그(gridSpan/vMerge)를 실제 값으로 채워서 완전한 사각 그리드로 복원.
    - gridSpan: 값을 span 칸 수만큼 반복
    - vMerge continue: 같은 열의 바로 위 restart 값을 그대로 사용
    """
    raw_rows = [get_real_row_cells(row) for row in table.rows]

    # 각 행을 gridSpan 기준으로 확장(값 반복)해서 열 개수를 맞춤
    expanded_rows = []
    for row in raw_rows:
        expanded = []
        for cell in row:
            for _ in range(cell["colspan"]):
                expanded.append({"text": cell["text"], "vmerge": cell["vmerge"]})
        expanded_rows.append(expanded)

    n_cols = max(len(r) for r in expanded_rows) if expanded_rows else 0
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


def table_type(doc_id, table_idx):
    if (doc_id, table_idx) in FLOWCHART_TABLES:
        return "flowchart"
    if (doc_id, table_idx) in CALC_BOX_TABLES:
        return "calc_box"
    return "normal"


def calc_box_to_text(grid):
    """계산식 강조박스를, 순서를 나타내는 기호(-, =)를 살려서 문장으로 풀어씀"""
    lines = []
    for row in grid:
        row = [c for c in row if c]
        if not row:
            continue
        if len(row) == 1:
            lines.append(row[0])
        elif row[0] in ("-", "=", "+"):
            op = {"-": "차감", "=": "결과", "+": "합산"}[row[0]]
            lines.append(f"({op}) {' '.join(row[1:])}")
        else:
            lines.append(" ".join(row))
    return " → ".join(lines)


def normal_table_rows_to_texts(grid):
    """일반 표: 헤더 행 기준으로, 데이터 행 하나당 문장 하나씩 생성"""
    if len(grid) < 2:
        return [" / ".join(grid[0])] if grid else []
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
    """새 절 제목처럼 보이는 짧은 문단인지 판단(청크 분리 기준점)"""
    text = text.strip()
    if not text:
        return False
    if len(text) <= 20 and not text.endswith(('.', '다', '요', ')')):
        return True
    if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]', text):
        return True
    if re.match(r'^\d+\.\s*[가-힣]', text) and len(text) <= 30:
        return True
    return False


def process_doc(doc_id):
    path = os.path.join(RAW_DIR, f"{doc_id}.docx")
    doc = Document(path)

    chunks = []  # list of (text, origin, section)
    buffer = []
    buffer_heading = None

    def flush_buffer():
        nonlocal buffer, buffer_heading
        if buffer:
            text = " ".join(buffer).strip()
            if text:
                chunks.append((text, "paragraph", buffer_heading))
        buffer = []

    table_idx = 0
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
            table_idx += 1
            ttype = table_type(doc_id, table_idx)

            if ttype == "flowchart":
                continue  # 표 내용 스킵. 이미 본문 서술 문장이 문단으로 따로 청크됨

            grid = resolve_table_grid(block)

            if ttype == "calc_box":
                flush_buffer()
                text = calc_box_to_text(grid)
                if text:
                    chunks.append((text, "calc_box_text", buffer_heading))
            else:
                flush_buffer()
                row_texts = normal_table_rows_to_texts(grid)
                for rt in row_texts:
                    chunks.append((rt, "table_row", buffer_heading))

    flush_buffer()
    return chunks


if __name__ == "__main__":
    all_chunks = []

    for doc_id in DOC_IDS:
        doc_chunks = process_doc(doc_id)
        for i, (text, origin, section) in enumerate(doc_chunks, start=1):
            c = ChunkB(
                chunk_id=f"{doc_id}#p1#c{i:03d}",
                doc_id=doc_id,
                source="세제",
                page=1,
                text=text,
                parse_method="text",
                confidence=1.0,
                section=(section[:40] if section else None),
                origin=origin,
            )
            all_chunks.append(c)

    OUTPUT_PATH = os.path.join(BASE_DIR, "chunks_genre_b.jsonl")
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