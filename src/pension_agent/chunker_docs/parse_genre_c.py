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

# "필요서류" 이후, 표가 시작되기 전까지가 "사유별 서두 설명"
# (마커 이후 텍스트를 통째로 버리지 않고, 표의 실제 bbox 위쪽 경계까지만 사용)
DETAIL_SECTION_MARKER = "신청양식"


def extract_real_tables(page):
    """⚠️ 수정: len(data[0]) >= 2 필터 제거.
    doc46/48/49에서 확인된 문제 - 라벨행/내용행이 분리된 1칸짜리 행 구조의 표는
    첫 행만 보는 필터에 걸려 표 전체가 통째로 유실됐음(genre_a와 동일 필터였으나
    genre_a는 bbox 기반 텍스트 스트림이라 유실되어도 텍스트로 살아남는 안전망이 있었음).
    C는 안전망이 없으므로, 표는 "존재하면 무조건 인정"으로 완화."""
    found = page.find_tables()
    tables = []
    for t in found:
        data = t.extract()
        if len(data) > 0:
            cleaned = [[c for c in row if c not in (None, '')] for row in data]
            tables.append({"bbox": t.bbox, "data": cleaned})
    return tables


def rows_to_items(table_data):
    """표 행 -> (라벨, 내용) 쌍 리스트.
    - 2칸 이상 행: [라벨, 내용...] 그대로 사용
    - 1칸 행: 다음 행도 1칸이면 (현재=라벨, 다음=내용)으로 짝짓기
    - 1칸인데 못 짝지으면(각주 등 단독행): 직전 항목 내용에 이어붙임
    """
    items = []
    i = 0
    while i < len(table_data):
        row = table_data[i]
        if len(row) >= 2:
            items.append([row[0], " ".join(row[1:])])
            i += 1
        elif len(row) == 1:
            if i + 1 < len(table_data) and len(table_data[i + 1]) == 1:
                items.append([row[0], table_data[i + 1][0]])
                i += 2
            else:
                if items:
                    items[-1][1] += " " + row[0]
                else:
                    items.append([None, row[0]])
                i += 1
        else:
            i += 1
    return items


def extract_footer_text(page, real_tables):
    """⚠️ 신규: 표들 중 가장 아래(bottom)에 있는 표보다 더 아래에 위치한 텍스트를 캡처.
    doc46/50에서 확인된 문제 - '※ 서류심사시...' 공통 각주, doc50의 법령 서식 각주가
    표 bbox 밖(아래)에 있어서 기존 로직(공통도입부/사유별서두/표) 어디에도 안 걸리고
    유실됐음. A의 bbox 비교 방식을 그대로 차용 - 마지막 표 bottom보다 top이 큰
    단어만 모아서 순서대로 이어붙임."""
    if not real_tables:
        return ""
    last_bottom = max(t["bbox"][3] for t in real_tables)
    words = page.extract_words()
    # ⚠️ 여백을 +2로 뒀더니 표 바로 아래 딱 붙은 줄(예: doc50의 첫 각주 줄, top이
    # last_bottom보다 겨우 1.9pt 큼)이 걸러지는 문제 발견 → 여백 제거, bottom 기준 그대로 비교
    footer_words = [w for w in words if w["top"] >= last_bottom]
    if not footer_words:
        return ""
    footer_words.sort(key=lambda w: (round(w["top"] / 3), w["x0"]))
    # 줄바꿈 보존: top이 크게 바뀌는 지점마다 줄바꿈
    lines, cur_line, cur_top = [], [], None
    for w in footer_words:
        if cur_top is not None and abs(w["top"] - cur_top) > 5:
            lines.append(" ".join(cur_line))
            cur_line = []
        cur_line.append(w["text"])
        cur_top = w["top"]
    if cur_line:
        lines.append(" ".join(cur_line))
    return "\n".join(lines)


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
    """사유별 본문을 '서두 설명(살릴 부분)'과 '서류 세부 목록(표 데이터로 대체)'으로 나눔.
    ⚠️ 이 함수 자체는 유지 - "신청양식" 마커 이전 텍스트(서두 설명)는 표에 없는
    유일한 정보라 여전히 필요함. 문제는 이 이후였고, 그 이후는 이제 표 추출이
    완전해졌으니 안전하게 버려도 됨 (표1+표2+표3... 전부 인정되므로)."""
    lead_lines = []
    for i, line in enumerate(body_lines):
        if DETAIL_SECTION_MARKER in line:
            return '\n'.join(lead_lines)
        lead_lines.append(line)
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

                real_tables = extract_real_tables(page)

                f.write(f"\n--- 표 {len(real_tables)}개 발견 (이 데이터가 세부 서류 목록 역할) ---\n")
                for i, t in enumerate(real_tables):
                    f.write(f"표 {i+1}: top={t['bbox'][1]:.1f}\n")
                    items = rows_to_items(t["data"])
                    for label, content in items:
                        label_disp = (label or "").replace("\n", " ")
                        f.write(f"  [{label_disp}] {content}\n")

                footer_text = extract_footer_text(page, real_tables)
                if footer_text:
                    f.write("\n--- [표 아래 각주/추가텍스트] ---\n")
                    f.write(f"{footer_text}\n")

print(f"완료! {OUTPUT_PATH} 파일을 확인하세요.")
