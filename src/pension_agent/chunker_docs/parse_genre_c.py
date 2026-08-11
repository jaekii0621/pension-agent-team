import pdfplumber
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw")
RAW_DIR = os.path.normpath(RAW_DIR)

DOC_IDS = ["doc46", "doc47", "doc48", "doc49", "doc50"]

OUTPUT_PATH = os.path.join(BASE_DIR, "output_genre_c.txt")

COMMON_INTRO_MARKERS = [
    "중도인출",
    "퇴직연금제도는 사용자가 퇴직하는 근로자에게",
    "제도별로는 DC(확정기여형)제도",
]

# 이 단어가 나오는 줄부터는 "서류 세부 목록" 구간으로 보고,
# 표 데이터와 중복되므로 본문에서는 버림
DETAIL_SECTION_MARKER = "신청양식"


def clean_table_row(row):
    return [cell for cell in row if cell not in (None, '')]


def split_common_intro(text):
    """텍스트를 '공통 도입부'와 '나머지(사유별)'로 나눔"""
    lines = text.split('\n')
    intro_lines = []
    body_start_idx = 0

    for i, line in enumerate(lines):
        intro_lines.append(line)
        if COMMON_INTRO_MARKERS[2] in line:
            body_start_idx = i + 1
            break

    intro_text = '\n'.join(intro_lines)
    body_lines = lines[body_start_idx:]
    return intro_text, body_lines


def split_lead_from_detail(body_lines):
    """
    사유별 본문을, '서두 설명(살릴 부분)'과 '서류 세부 목록(버릴 부분, 표와 중복)'으로 나눔.
    DETAIL_SECTION_MARKER가 처음 나오는 줄부터를 detail로 취급.
    """
    lead_lines = []
    for i, line in enumerate(body_lines):
        if DETAIL_SECTION_MARKER in line:
            # 이 줄부터는 detail이므로 lead에 포함하지 않고 멈춤
            return '\n'.join(lead_lines)
        lead_lines.append(line)
    # 마커를 못 찾았으면 전체를 lead로 (안전장치)
    return '\n'.join(lead_lines)


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    for doc_id in DOC_IDS:
        pdf_path = os.path.join(RAW_DIR, f"{doc_id}.pdf")

        if not os.path.exists(pdf_path):
            f.write(f"\n### {doc_id}: 파일 없음 ###\n")
            continue

        f.write(f"\n{'#'*10} {doc_id} {'#'*10}\n")

        with pdfplumber.open(pdf_path) as pdf:
            f.write(f"총 페이지 수: {len(pdf.pages)}\n")

            for page_num, page in enumerate(pdf.pages, start=1):
                f.write(f"\n{'='*20} 페이지 {page_num} {'='*20}\n")

                text = page.extract_text()

                if page_num == 1:
                    intro, body_lines = split_common_intro(text)
                    lead_text = split_lead_from_detail(body_lines)

                    f.write("--- [공통 도입부] ---\n")
                    f.write(f"{intro}\n")
                    f.write("\n--- [사유별 서두 설명 - 이것만 본문으로 살림] ---\n")
                    f.write(f"{lead_text}\n")
                    f.write("\n--- [세부 서류 목록은 본문에서 제외 - 아래 표 데이터로 대체됨] ---\n")
                else:
                    f.write("--- 텍스트 ---\n")
                    f.write(f"{text}\n")

                found_tables = page.find_tables()
                real_tables = []
                for t in found_tables:
                    table_data = t.extract()
                    if len(table_data) > 0 and len(table_data[0]) >= 2:
                        real_tables.append(t)

                f.write(f"\n--- 표 {len(real_tables)}개 발견 (이 데이터가 세부 서류 목록 역할) ---\n")
                for i, t in enumerate(real_tables):
                    f.write(f"표 {i+1}: top={t.bbox[1]:.1f}\n")
                    table_data = t.extract()
                    for row in table_data:
                        f.write(f"{clean_table_row(row)}\n")

print(f"완료! {OUTPUT_PATH} 파일을 확인하세요.")