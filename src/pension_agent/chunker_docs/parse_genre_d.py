from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pptx import Presentation
import openpyxl
import pdfplumber
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw")
RAW_DIR = os.path.normpath(RAW_DIR)

# 장르D - doc22는 OCR 대상이라 제외
DOC_IDS = ["doc55", "doc6", "doc35", "doc57", "doc58", "doc53", "doc33", "doc34"]

OUTPUT_PATH = os.path.join(BASE_DIR, "output_genre_d.txt")


def get_doc_path(doc_id):
    """확장자별로 순서대로 찾아서, 있는 것과 그 형식을 반환"""
    for ext in ["docx", "pdf", "pptx", "xlsx"]:
        path = os.path.join(RAW_DIR, f"{doc_id}.{ext}")
        if os.path.exists(path):
            return path, ext
    return None, None


def clean_table_row(row_cells):
    return [cell for cell in row_cells if cell not in (None, '')]


# ---------- docx 처리 ----------
def iter_block_items(doc):
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            yield ('paragraph', Paragraph(child, doc))
        elif tag == 'tbl':
            yield ('table', Table(child, doc))


def parse_docx(file_path, f):
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
                cells = [cell.text.strip() for cell in row.cells]
                f.write(f"{clean_table_row(cells)}\n")
            f.write(">>> [표 끝] <<<\n\n")


# ---------- pdf 처리 ----------
def parse_pdf(file_path, f):
    with pdfplumber.open(file_path) as pdf:
        f.write(f"총 페이지 수: {len(pdf.pages)}\n")
        for page_num, page in enumerate(pdf.pages, start=1):
            f.write(f"\n{'='*20} 페이지 {page_num} {'='*20}\n")
            text = page.extract_text()
            f.write("--- 텍스트 ---\n")
            f.write(f"{text}\n")

            found_tables = page.find_tables()
            real_tables = []
            for t in found_tables:
                table_data = t.extract()
                if len(table_data) > 0 and len(table_data[0]) >= 2:
                    real_tables.append(t)

            f.write(f"\n--- 표 {len(real_tables)}개 발견 ---\n")
            for i, t in enumerate(real_tables):
                f.write(f"표 {i+1}: top={t.bbox[1]:.1f}\n")
                table_data = t.extract()
                for row in table_data:
                    f.write(f"{clean_table_row(row)}\n")


# ---------- pptx 처리 (신규) ----------
def parse_pptx(file_path, f):
    prs = Presentation(file_path)
    f.write(f"총 슬라이드 수: {len(prs.slides)}\n")

    for slide_num, slide in enumerate(prs.slides, start=1):
        f.write(f"\n{'='*20} 슬라이드 {slide_num} {'='*20}\n")
        table_idx = 0

        for shape in slide.shapes:
            # 슬라이드 안의 텍스트 상자
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    f.write(f"[텍스트] {text}\n")

            # 슬라이드 안의 표
            if shape.has_table:
                table_idx += 1
                tbl = shape.table
                f.write(f"\n>>> [표{table_idx}] ({len(tbl.rows)}행 x {len(tbl.columns)}열) <<<\n")
                for row in tbl.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    f.write(f"{clean_table_row(cells)}\n")
                f.write(">>> [표 끝] <<<\n\n")


# ---------- xlsx 처리 (신규) ----------
def parse_xlsx(file_path, f):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    f.write(f"시트 목록: {wb.sheetnames}\n")

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        f.write(f"\n{'='*20} 시트: {sheet_name} ({ws.max_row}행 x {ws.max_column}열) {'='*20}\n")

        for row in ws.iter_rows(values_only=True):
            # 완전히 빈 행은 건너뜀
            cleaned = [cell for cell in row if cell not in (None, '')]
            if cleaned:
                f.write(f"{cleaned}\n")


# ---------- 메인 루프 ----------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    for doc_id in DOC_IDS:
        file_path, file_type = get_doc_path(doc_id)

        if file_path is None:
            f.write(f"\n### {doc_id}: 파일 없음 ###\n")
            continue

        f.write(f"\n{'#'*10} {doc_id} ({file_type}) {'#'*10}\n")

        # 파일 형식에 따라 알맞은 함수로 분기
        if file_type == "docx":
            parse_docx(file_path, f)
        elif file_type == "pdf":
            parse_pdf(file_path, f)
        elif file_type == "pptx":
            parse_pptx(file_path, f)
        elif file_type == "xlsx":
            parse_xlsx(file_path, f)

print(f"완료! {OUTPUT_PATH} 파일을 확인하세요.")