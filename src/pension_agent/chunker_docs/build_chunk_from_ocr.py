"""
CLOVA OCR 응답(raw json)을 실제 Chunk로 변환.
doc57로 먼저 검증하고, 나중에 doc58/53/33/22/이미지PDF18개에도 그대로 재사용할 수 있게
'문서 하나 처리하는 함수' 형태로 설계함.
"""
import json
import re
import os
import sys
from schema import Chunk

MIN_CHUNK_LEN = 300
MAX_CHUNK_LEN = 800


def looks_like_heading(text):
    """새 절 제목처럼 보이는 짧은 줄인지 판단 (장르D 청킹 때 쓴 규칙 재사용)"""
    text = text.strip()
    if not text:
        return False
    if re.match(r'^[가-힣]\.\s', text) or re.match(r'^[가-힣]\.$', text):
        return True
    if re.match(r'^\d+\)\s', text):
        return True
    if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]', text):
        return True
    if len(text) <= 20 and not text.endswith(('.', '다', '요', ')')):
        return True
    return False


def point_in_bbox(x, y, bbox):
    """점(x,y)이 bbox(사각형) 안에 있는지 확인"""
    xs = [v["x"] for v in bbox["vertices"]]
    ys = [v["y"] for v in bbox["vertices"]]
    return min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys)


import re


def is_rotated(bbox, tolerance=8):
    """
    글자 박스가 삐딱하게(회전되어) 있는지 확인.
    정상 글자는 좌표 4개가 거의 직사각형을 이루지만, OCR 특성상 1~수픽셀 오차는 항상 있음.
    진짜 워터마크(대각선)는 수십 픽셀 이상 차이나므로, tolerance 픽셀 이내 차이는 "같은 값"으로 봐줌.
    """
    xs = sorted(v["x"] for v in bbox["vertices"])
    ys = sorted(v["y"] for v in bbox["vertices"])

    def cluster_count(values):
        clusters = 1
        for i in range(1, len(values)):
            if values[i] - values[i - 1] > tolerance:
                clusters += 1
        return clusters

    return cluster_count(xs) > 2 or cluster_count(ys) > 2


def is_stray_number(text):
    """외로운 1~3자리 숫자(예: '46', '14') - 페이지번호/워터마크로 추정되는 노이즈"""
    return bool(re.fullmatch(r"\d{1,3}", text.strip()))


def is_noise(field_like):
    text = field_like.get("inferText", "")
    bbox = field_like["boundingPoly"]
    return is_rotated(bbox) or is_stray_number(text)


def cell_text(cell):
    """
    표 셀 하나의 텍스트를 꺼냄.
    표 안에서는 '회전된(워터마크)' 것만 제외하고, 외로운 숫자 필터는 안 씀
    -> "100분의 40" 같은 진짜 퍼센트 수치가 짧은 숫자라는 이유로 삭제되는 걸 방지하기 위함
    """
    if "text" in cell:  # 테스트용 단순화 데이터
        return cell["text"]
    parts = []
    for line in cell.get("cellTextLines", []):
        words = [w["inferText"] for w in line.get("cellWords", []) if not is_rotated(w["boundingPoly"])]
        if words:
            parts.append(" ".join(words))
    return " ".join(parts)


def build_table_grid(table):
    """
    cells 배열(rowIndex/columnIndex + rowSpan/columnSpan 포함)을 2차원 리스트로 재구성.
    rowSpan/columnSpan이 있으면 그 범위 전체에 같은 값을 채워 넣어서
    (예: '채무증권'이 5개 행에 걸쳐 병합돼 있으면 5개 행 전부에 '채무증권'이 들어가게)
    나중에 헤더-데이터 짝짓기 할 때 분류 정보가 안 사라지게 함.
    """
    cells = table["cells"]
    n_rows = max(c["rowIndex"] + c.get("rowSpan", 1) for c in cells)
    n_cols = max(c["columnIndex"] + c.get("columnSpan", 1) for c in cells)
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for c in cells:
        text = cell_text(c)
        if not text:
            continue
        for r in range(c["rowIndex"], c["rowIndex"] + c.get("rowSpan", 1)):
            for col in range(c["columnIndex"], c["columnIndex"] + c.get("columnSpan", 1)):
                grid[r][col] = text
    return grid


def table_rows_to_texts(grid):
    """장르B/D에서 쓴 것과 같은 방식: 헤더 행 기준 데이터 행 텍스트화"""
    if len(grid) < 2:
        return [" / ".join(grid[0])] if grid else []

    n_cols = len(grid[0])
    if n_cols == 1:
        # 1열짜리 표 = 법령 조문처럼 순서대로 나열된 문단인 경우가 많음
        # "헤더-데이터" 관계가 아니라 각 행이 독립된 문장이므로 그냥 그대로 반환
        return [row[0] for row in grid if row[0]]

    header, *data_rows = grid
    texts = []
    for row in data_rows:
        seen_pairs = set()
        pairs = []
        for h, v in zip(header, row):
            if not v:
                continue
            key = (h, v)
            if key in seen_pairs:
                continue  # 병합된 칸이 여러 열로 복제돼서 완전히 똑같은 짝이 반복되는 경우만 제거
            seen_pairs.add(key)
            pairs.append(f"{h}: {v}" if h else v)
        if pairs:
            texts.append(", ".join(pairs))
    return texts


def build_chunks_from_ocr(json_path, doc_id, source="절차"):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    all_chunks_raw = []

    # 페이지가 여러 장이면 images 리스트에 여러 개 들어있음 - 전부 순회
    for image_result in data["images"]:
        page_num = image_result.get("realPageNumber", 1)
        tables = image_result.get("tables", [])
        fields = image_result.get("fields", [])

        # 1) 실제 '셀' 하나하나의 bbox 목록 (표 전체의 큰 바깥 경계가 아니라!)
        #    표 바깥경계로 판단하면, 표 근처에 있지만 어느 셀에도 안 속한 글자가
        #    "표 영역이라 본문에서 제외" + "어느 셀 데이터도 아님" 이라는 사각지대에 빠져서 통째로 사라짐
        cell_bboxes = []
        for t in tables:
            for c in t["cells"]:
                cell_bboxes.append(c["boundingPoly"])

        def in_any_table(field):
            v = field["boundingPoly"]["vertices"][0]  # 좌상단 좌표로 판단
            return any(point_in_bbox(v["x"], v["y"], bbox) for bbox in cell_bboxes)

        # 2) 표에 안 속한 글자 + 노이즈(워터마크/외톨이 숫자) 아닌 글자만 모아서, lineBreak 기준으로 줄 단위 재구성
        lines = []
        current_line = []
        for field in fields:
            if in_any_table(field) or is_noise(field):
                continue
            current_line.append(field["inferText"])
            if field.get("lineBreak"):
                lines.append(" ".join(current_line))
                current_line = []
        if current_line:
            lines.append(" ".join(current_line))

        # 3) 줄들을 문단 청크로 묶기
        buffer = []
        buffer_heading = None

        def flush():
            nonlocal buffer, buffer_heading
            if buffer:
                text = " ".join(buffer).strip()
                if text:
                    all_chunks_raw.append((text, "paragraph", buffer_heading, page_num))
            buffer = []

        for line in lines:
            if looks_like_heading(line) and buffer and sum(len(b) for b in buffer) >= MIN_CHUNK_LEN:
                flush()
                buffer_heading = line
            elif looks_like_heading(line) and not buffer:
                buffer_heading = line
            buffer.append(line)
            if sum(len(b) for b in buffer) >= MAX_CHUNK_LEN:
                flush()
        flush()

        # 4) 표는 별도로 행 단위 청크화 (표마다, 페이지마다 전부)
        for t_idx, table in enumerate(tables, start=1):
            grid = build_table_grid(table)
            row_texts = table_rows_to_texts(grid)
            for rt in row_texts:
                all_chunks_raw.append((rt, "table_row", f"표{t_idx}", page_num))

    # 5) 실제 Chunk 객체로 변환 (페이지별로 번호 다시 매김)
    result = []
    chunk_no_by_page = {}
    for text, origin, section, page_num in all_chunks_raw:
        chunk_no_by_page[page_num] = chunk_no_by_page.get(page_num, 0) + 1
        c = Chunk(
            chunk_id=f"{doc_id}#p{page_num}#c{chunk_no_by_page[page_num]:03d}",
            doc_id=doc_id,
            source=source,
            page=page_num,
            text=text,
            parse_method="ocr",
            confidence=0.95,
            section=section[:40] if section else None,
        )
        result.append(c)
    return result


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "ocr_raw_response.json"
    doc_id = sys.argv[2] if len(sys.argv) > 2 else "doc57"

    chunks = build_chunks_from_ocr(json_path, doc_id=doc_id, source="절차")

    OUTPUT_PATH = f"chunks_{doc_id}_ocr.jsonl"
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    print(f"총 {len(chunks)}개 청크 생성 → {OUTPUT_PATH}\n")
    for c in chunks:
        print(f"[{c.chunk_id}] section={c.section}")
        print(f"  {c.text[:150]}")
        print()
