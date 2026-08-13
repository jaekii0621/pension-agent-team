import pdfplumber
import re
import os
import json
from chunk_genre_a_schema import ChunkA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw")
RAW_DIR = os.path.normpath(RAW_DIR)

DOC_IDS = ["doc10", "doc12", "doc13", "doc14", "doc15", "doc16", "doc17", "doc18"]

BOTTOM_MARGIN = 80
TOP_MARGIN = 80
X_TOLERANCE = 5


def extract_real_tables(page):
    found = page.find_tables()
    tables = []
    for t in found:
        data = t.extract()
        if len(data) > 0 and len(data[0]) >= 2:
            cleaned = [[c for c in row if c not in (None, '')] for row in data]
            tables.append({"bbox": t.bbox, "data": cleaned})
    return tables


def try_merge_across_pages(prev_tables, prev_height, curr_tables):
    if not prev_tables or not curr_tables:
        return
    last_prev, first_curr = prev_tables[-1], curr_tables[0]
    px0, ptop, px1, pbottom = last_prev["bbox"]
    cx0, ctop, cx1, cbottom = first_curr["bbox"]
    if (prev_height - pbottom) < BOTTOM_MARGIN and ctop < TOP_MARGIN \
            and abs(px0 - cx0) < X_TOLERANCE and abs(px1 - cx1) < X_TOLERANCE:
        last_prev["data"] += first_curr["data"]
        curr_tables.pop(0)
        # ⚠️ 참고: 병합된 표는 두 페이지에 걸쳐있으므로, 정확한 단일 page 값이 없음.
        # 이 스크립트에서는 "표가 시작된 페이지(직전 페이지)"를 기준으로 삼음.


def table_to_text(table):
    """표 데이터를 사람이 읽는 문장처럼 풀어서 표현"""
    header, *rows = table["data"]
    lines = []
    for row in rows:
        if len(row) == len(header):
            pair = ", ".join(f"{h}: {v}" for h, v in zip(header, row))
        else:
            pair = " / ".join(row)
        lines.append(pair)
    return "; ".join(lines)


def get_clean_words_in_order(page, exclude_tables):
    """표 영역(top~bottom)에 속한 단어는 제외하고, 나머지 단어를 읽는 순서(top→x0)로 정렬해서 반환"""
    words = page.extract_words()
    ranges = [(t["bbox"][1], t["bbox"][3]) for t in exclude_tables]

    def in_table(w):
        return any(top - 2 <= w["top"] <= bottom + 2 for top, bottom in ranges)

    kept = [w for w in words if not in_table(w)]
    kept.sort(key=lambda w: (round(w["top"] / 3), w["x0"]))  # 같은 줄끼리 top 오차 흡수, 줄 내에서는 좌->우
    return kept


class QAParser:
    """페이지가 넘어가도 상태(현재 어떤 질문을 파싱 중인지)를 유지하는 파서.
    ⚠️ 수정: 각 질문(q)이 "시작된 페이지 번호"를 함께 기록하도록 확장함.
    """

    def __init__(self):
        self.mode = "before_qlist"
        self.q_words = []
        self.a_words = []
        self.in_answer = False
        self.qa_pairs = []       # (q_text, a_text, start_page)
        self.appendix_words = []
        self.appendix_start_page = None
        self.current_q_start_page = None

    def feed(self, tokens, page_num):
        for tok in tokens:
            if tok == "첨부":
                self.mode = "appendix"
                if self.appendix_start_page is None:
                    self.appendix_start_page = page_num
                continue

            if self.mode == "appendix":
                self.appendix_words.append(tok)
                continue

            if self.mode == "before_qlist":
                if tok.startswith("■"):
                    self.mode = "qlist"
                continue

            if self.mode == "qlist":
                if tok.startswith("○"):
                    self.mode = "qa"
                else:
                    continue

            if self.mode == "qa":
                if tok.startswith("○"):
                    if self.q_words or self.a_words:
                        self.qa_pairs.append((
                            " ".join(self.q_words),
                            " ".join(self.a_words),
                            self.current_q_start_page,
                        ))
                    self.q_words = [tok.lstrip("○")] if len(tok) > 1 else []
                    self.a_words = []
                    self.in_answer = False
                    self.current_q_start_page = page_num  # 새 질문이 시작된 페이지 기록
                elif tok.startswith("•"):
                    self.in_answer = True
                    rest = tok.lstrip("•")
                    if rest:
                        self.a_words.append(rest)
                else:
                    if self.in_answer:
                        self.a_words.append(tok)
                    else:
                        self.q_words.append(tok)
                        if self.current_q_start_page is None:
                            self.current_q_start_page = page_num

    def finalize(self):
        if self.q_words or self.a_words:
            self.qa_pairs.append((
                " ".join(self.q_words),
                " ".join(self.a_words),
                self.current_q_start_page,
            ))
            self.q_words, self.a_words = [], []
        return self.qa_pairs, " ".join(self.appendix_words), self.appendix_start_page


KNOWN_LAW_PREFIXES = ["근로자퇴직급여", "민사집행법", "근로기준법"]


def trim_stray_prefix(text):
    """원본 PDF에서 앞 문장 꼬리와 법령명이 띄어쓰기 없이 붙어버린 경우, 법령명 시작 지점부터 잘라냄"""
    earliest = None
    for prefix in KNOWN_LAW_PREFIXES:
        idx = text.find(prefix)
        if idx > 0 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest:
        return text[earliest:]
    return text


def split_appendix_articles(appendix_text):
    """첨부 법령 원문을, '(법령명) 제N조(제목)' 패턴 기준으로 조문 단위 청크로 분리"""
    pattern = r'(?<![가-힣])(?:[가-힣]+\s+)?[가-힣]+법(?:\s*시행령|\s*시행규칙)?\s*제\d+조(?:의\d+)?\([^)]+\)'
    matches = list(re.finditer(pattern, appendix_text))
    if not matches:
        return [appendix_text.strip()] if appendix_text.strip() else []

    parts = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(appendix_text)
        piece = appendix_text[start:end].strip()
        piece = trim_stray_prefix(piece)
        if piece:
            parts.append(piece)
    return parts


def process_doc(doc_id):
    pdf_path = os.path.join(RAW_DIR, f"{doc_id}.pdf")
    with pdfplumber.open(pdf_path) as pdf:
        pages_tables = []
        for page in pdf.pages:
            pages_tables.append(extract_real_tables(page))
        for i in range(1, len(pages_tables)):
            try_merge_across_pages(pages_tables[i - 1], pdf.pages[i - 1].height, pages_tables[i])

        parser = QAParser()
        pending_tables_per_page = []  # (target_idx, is_current, tables, table_page_num)

        for page_num, page in enumerate(pdf.pages, start=1):
            tables = pages_tables[page_num - 1]
            words = get_clean_words_in_order(page, tables)
            tokens = [w["text"] for w in words]
            parser.feed(tokens, page_num)  # ⚠️ page_num 전달하도록 수정

            if tables:
                if parser.qa_pairs:
                    target_idx = len(parser.qa_pairs) - 1
                    is_current = False
                elif parser.q_words or parser.a_words:
                    target_idx = None
                    is_current = True
                else:
                    target_idx = None
                    is_current = False
                pending_tables_per_page.append((target_idx, is_current, tables, page_num))

        qa_pairs, appendix_text, appendix_page = parser.finalize()

        # (q, a, start_page) 3-tuple 리스트로 관리
        qa_list = list(qa_pairs)
        for target_idx, is_current, tables, table_page_num in pending_tables_per_page:
            if is_current:
                idx = 0
            else:
                idx = target_idx
            if idx is None or idx < 0 or idx >= len(qa_list):
                continue
            q, a, start_page = qa_list[idx]
            for t in tables:
                a += " " + table_to_text(t)
            qa_list[idx] = (q, a, start_page)

        articles = split_appendix_articles(appendix_text)
        # appendix_page가 None이면(문서에 '첨부' 자체가 없으면) 1로 폴백
        return qa_list, articles, (appendix_page or 1)


if __name__ == "__main__":
    all_chunks = []

    for doc_id in DOC_IDS:
        qa_list, articles, appendix_page = process_doc(doc_id)
        chunk_no_by_page = {}

        for q, a, start_page in qa_list:
            page = start_page or 1  # 혹시 모를 None 방어
            chunk_no_by_page[page] = chunk_no_by_page.get(page, 0) + 1
            text = f"Q: {q}\nA: {a.strip()}"
            c = ChunkA(
                chunk_id=f"{doc_id}#p{page}#c{chunk_no_by_page[page]:03d}",
                doc_id=doc_id,
                source="제도",
                page=page,
                text=text,
                parse_method="text",
                confidence=1.0,
                section=q[:40],
                distractor=False,
            )
            all_chunks.append(c)

        for art in articles:
            page = appendix_page
            chunk_no_by_page[page] = chunk_no_by_page.get(page, 0) + 1
            c = ChunkA(
                chunk_id=f"{doc_id}#p{page}#c{chunk_no_by_page[page]:03d}",
                doc_id=doc_id,
                source="제도",
                page=page,
                text=art,
                parse_method="text",
                confidence=1.0,
                section="첨부-관련법령",
                distractor=True,
            )
            all_chunks.append(c)

    OUTPUT_PATH = os.path.join(BASE_DIR, "chunks_genre_a.jsonl")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    print(f"총 {len(all_chunks)}개 청크 생성 → {OUTPUT_PATH}")
    for c in all_chunks:
        print(f"{c.chunk_id} | distractor={c.distractor} | section={c.section[:30]:30s} | 글자수={len(c.text)}")
