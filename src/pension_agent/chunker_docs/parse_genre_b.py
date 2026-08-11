from docx import Document
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw")
RAW_DIR = os.path.normpath(RAW_DIR)

DOC_IDS = ["doc20", "doc23", "doc25", "doc26", "doc36", "doc38", "doc39",
           "doc40", "doc41", "doc42", "doc43", "doc44", "doc45", "doc51", "doc52"]

OUTPUT_PATH = os.path.join(BASE_DIR, "output_genre_b.txt")


def get_doc_path(doc_id):
    docx_path = os.path.join(RAW_DIR, f"{doc_id}.docx")
    pdf_path = os.path.join(RAW_DIR, f"{doc_id}.pdf")
    if os.path.exists(docx_path):
        return docx_path, "docx"
    elif os.path.exists(pdf_path):
        return pdf_path, "pdf"
    else:
        return None, None


def iter_block_items(doc):
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            yield ('paragraph', Paragraph(child, doc))
        elif tag == 'tbl':
            yield ('table', Table(child, doc))


def get_real_row_cells(row):
    """
    row.cells는 병합된 셀을 grid 칸 수만큼 복제해서 돌려준다 (가로/세로 병합 둘 다).
    그래서 row._tr.tc_lst로 '진짜 존재하는 셀'만 순회하고,
    gridSpan(가로 병합 칸 수)/vMerge(세로 병합 여부) 정보를 같이 붙여서 반환한다.
    빈 문자열도 삭제하지 않고 그대로 유지해 열 개수(칸 정렬)를 보존한다.
    """
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
                vmerge = val or 'continue'  # w:vMerge에 val 속성이 없으면 'continue'가 기본값

        cells.append({"text": text, "colspan": grid_span, "vmerge": vmerge})
    return cells


def format_cell(cell_info):
    """셀 텍스트에 병합 정보를 사람이 읽기 쉬운 태그로 붙여서 출력용 문자열 생성"""
    text = cell_info["text"]
    tags = []
    if cell_info["colspan"] > 1:
        tags.append(f"가로병합:{cell_info['colspan']}칸")
    if cell_info["vmerge"] == "restart":
        tags.append("세로병합 시작")
    elif cell_info["vmerge"] == "continue":
        tags.append("세로병합 이어짐(윗셀과 동일값)")

    if not tags:
        return text
    return f"{text} [{', '.join(tags)}]"


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    for doc_id in DOC_IDS:
        file_path, file_type = get_doc_path(doc_id)

        if file_path is None:
            f.write(f"\n### {doc_id}: 파일 없음 ###\n")
            continue

        f.write(f"\n{'#'*10} {doc_id} ({file_type}) {'#'*10}\n")

        if file_type != "docx":
            f.write(f"### {doc_id}: pdf는 parse_genre_a.py 참고 ###\n")
            continue

        doc = Document(file_path)
        para_idx = 0
        table_idx = 0

        for kind, block in iter_block_items(doc):
            if kind == 'paragraph':
                if block.text.strip():
                    f.write(f"[문단{para_idx}] {block.text}\n")
                para_idx += 1
            elif kind == 'table':
                table_idx += 1
                f.write(f"\n>>> [표{table_idx}] ({len(block.rows)}행 x {len(block.columns)}열) <<<\n")
                for row in block.rows:
                    real_cells = get_real_row_cells(row)
                    formatted = [format_cell(c) for c in real_cells]
                    f.write(f"{formatted}\n")
                f.write(">>> [표 끝] <<<\n\n")

print(f"완료! {OUTPUT_PATH}")