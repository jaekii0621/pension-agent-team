import pdfplumber
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw")
RAW_DIR = os.path.normpath(RAW_DIR)

# Q&A 패턴(○질문/•답변)으로 확인된 문서만 직접 지정
# doc11, doc19는 다른 구조라 별도 파서에서 처리 예정
DOC_IDS = ["doc10", "doc12", "doc13", "doc14", "doc15", "doc16", "doc17", "doc18"]

OUTPUT_PATH = os.path.join(BASE_DIR, "output.txt")

# 문제B(표 분절 병합) 판단 기준
BOTTOM_MARGIN = 80   # 이 값 이내로 페이지 하단에 붙어있으면 "페이지 끝까지 이어짐" 후보
TOP_MARGIN = 80       # 이 값 이내로 페이지 상단에 붙어있으면 "페이지 시작부터 이어짐" 후보
X_TOLERANCE = 5       # 좌우 경계(x0, x1)가 이 값 이내로 같으면 "같은 표" 로 판단


def clean_table_row(row):
    return [cell for cell in row if cell not in (None, '')]


def extract_real_tables(page):
    """2열 이상인 진짜 표만 추려서, bbox와 raw 데이터를 함께 반환"""
    found = page.find_tables()
    tables = []
    for t in found:
        data = t.extract()
        if len(data) > 0 and len(data[0]) >= 2:
            tables.append({"bbox": t.bbox, "data": data})
    return tables


def try_merge_across_pages(prev_page_tables, prev_page_height, curr_page_tables):
    """
    직전 페이지의 마지막 표 vs 현재 페이지의 첫 표를 비교해서,
    페이지 경계에서 분절된 같은 표인지 판단하고 병합한다.
    병합되면 curr_page_tables에서 첫 표를 제거하고, prev_page_tables의 마지막 표에 데이터를 이어붙인다.
    반환값: (병합됐는지 여부, 병합 로그 메시지 또는 None)
    """
    if not prev_page_tables or not curr_page_tables:
        return False, None

    last_prev = prev_page_tables[-1]
    first_curr = curr_page_tables[0]

    prev_x0, prev_top, prev_x1, prev_bottom = last_prev["bbox"]
    curr_x0, curr_top, curr_x1, curr_bottom = first_curr["bbox"]

    near_bottom = (prev_page_height - prev_bottom) < BOTTOM_MARGIN
    near_top = curr_top < TOP_MARGIN
    same_left = abs(prev_x0 - curr_x0) < X_TOLERANCE
    same_right = abs(prev_x1 - curr_x1) < X_TOLERANCE

    if near_bottom and near_top and same_left and same_right:
        # 병합: 직전 표 데이터 뒤에 현재 표 데이터를 이어붙임
        last_prev["data"] = last_prev["data"] + first_curr["data"]
        last_prev["merged_across_page"] = True
        curr_page_tables.pop(0)
        return True, (
            f"    [문제B 자동 병합] 직전 페이지 마지막 표(bottom={prev_bottom:.1f}) + "
            f"이번 페이지 첫 표(top={curr_top:.1f})를 같은 표로 판단해 병합함 "
            f"(좌우 경계 x0={curr_x0:.1f}, x1={curr_x1:.1f} 일치)"
        )
    return False, None


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    for doc_id in DOC_IDS:
        pdf_path = os.path.join(RAW_DIR, f"{doc_id}.pdf")

        if not os.path.exists(pdf_path):
            f.write(f"\n### {doc_id}: 파일 없음 (건너뜀) ###\n")
            continue

        f.write(f"\n{'#'*10} {doc_id} {'#'*10}\n")

        with pdfplumber.open(pdf_path) as pdf:
            f.write(f"총 페이지 수: {len(pdf.pages)}\n")

            # 1단계: 모든 페이지의 텍스트/표/질문위치를 먼저 수집
            pages_info = []
            for page in pdf.pages:
                text = page.extract_text()
                real_tables = extract_real_tables(page)
                words = page.extract_words()
                question_positions = [w['top'] for w in words if w['text'] == '○']
                pages_info.append({
                    "height": page.height,
                    "text": text,
                    "tables": real_tables,
                    "question_positions": question_positions,
                })

            # 2단계: 문제B - 페이지 경계에서 분절된 표 병합
            for i in range(1, len(pages_info)):
                merged, log_msg = try_merge_across_pages(
                    pages_info[i - 1]["tables"],
                    pages_info[i - 1]["height"],
                    pages_info[i]["tables"],
                )
                if merged:
                    pages_info[i]["merge_log"] = log_msg

            # 3단계: 문제A - 이전 페이지 마지막 질문을 이어받아 매칭 + 출력
            question_serial = 0          # 문서 전체 기준 질문 누적 번호
            carry_over_serial = None     # 이전 페이지 마지막 질문의 전체 누적 번호

            for page_num, info in enumerate(pages_info, start=1):
                f.write(f"\n{'='*20} 페이지 {page_num} {'='*20}\n")

                if info.get("merge_log"):
                    f.write(info["merge_log"] + "\n")

                f.write("--- 텍스트 ---\n")
                f.write(f"{info['text']}\n")

                real_tables = info["tables"]
                f.write(f"\n--- 표 {len(real_tables)}개 발견 ---\n")
                for i, t in enumerate(real_tables):
                    tag = " [페이지 경계 병합됨]" if t.get("merged_across_page") else ""
                    f.write(f"표 {i+1}: top={t['bbox'][1]:.1f}{tag}\n")
                    for row in t["data"]:
                        f.write(f"{clean_table_row(row)}\n")

                question_positions = info["question_positions"]
                # 이번 페이지 질문들의 전체 누적 번호 목록
                this_page_serials = []
                for q_top in question_positions:
                    question_serial += 1
                    this_page_serials.append(question_serial)

                f.write(f"\n--- '○' 질문 위치: {question_positions} (전체 누적 {this_page_serials}) ---\n")

                f.write("--- 표-질문 매칭 ---\n")
                for i, t in enumerate(real_tables):
                    table_top = t["bbox"][1]
                    matched_serial = None
                    for q_idx, q_top in enumerate(question_positions):
                        if q_top < table_top:
                            matched_serial = this_page_serials[q_idx]

                    if matched_serial is not None:
                        f.write(f"표 {i+1} → 전체 질문 {matched_serial}번째에 속함 (같은 페이지)\n")
                    elif carry_over_serial is not None:
                        # 문제A 수정: 이 페이지에 표보다 앞선 질문이 없으면 이전 페이지 마지막 질문으로 매칭
                        f.write(
                            f"표 {i+1} → 전체 질문 {carry_over_serial}번째에 속함 "
                            f"[문제A 수정: 이전 페이지 마지막 질문으로 이어받음]\n"
                        )
                    else:
                        f.write(f"표 {i+1}은 매칭할 질문 없음 (문서 첫 질문보다 앞에 위치)\n")

                # 다음 페이지를 위해 carry-over 갱신 (이번 페이지에 질문이 있었을 때만 갱신)
                if this_page_serials:
                    carry_over_serial = this_page_serials[-1]

print(f"완료! {OUTPUT_PATH} 파일을 확인하세요.")