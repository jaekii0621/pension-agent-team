import pdfplumber
import re
import os
import json
from chunk_genre_a_schema import ChunkA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw")
RAW_DIR = os.path.normpath(RAW_DIR)

DOC_IDS = ["doc10", "doc12", "doc13", "doc14", "doc15", "doc16", "doc17", "doc18"]

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


def table_to_text(table, continued=False):
    """표 데이터를 사람이 읽는 문장처럼 풀어서 표현.
    ⚠️ 수정: 더 이상 페이지를 넘어 표를 병합하지 않는다(민서님 section_chunk 방식과 통일).
    표가 다음 페이지로 이어지는 경우, 이어지는 조각의 맨 앞에 헤더 행을 복제해서
    문맥이 끊기지 않게 한다 — 원본 pass10_chunk_section.py의 _table_head()/split_table() 패턴 차용.
    """
    header, *rows = table["data"]
    lines = []
    for row in rows:
        if len(row) == len(header):
            pair = ", ".join(f"{h}: {v}" for h, v in zip(header, row))
        else:
            pair = " / ".join(row)
        lines.append(pair)
    body = "; ".join(lines)
    if continued:
        head = ", ".join(header) if header else ""
        prefix = f"[{head} — 표 계속] " if head else "[표 계속] "
        return prefix + body
    return body


def get_ordered_stream(page, tables):
    """단어와 표를 하나의 스트림으로 합쳐서, 실제 읽는 순서(top 좌표) 그대로 반환.
    ⚠️ 수정 (표 오귀속 버그 해결): 이전에는 단어를 다 읽은 "뒤에" 표를 어느 질문에
    붙일지 판단해서, 한 페이지 안에 질문 경계가 여러 개 있으면(예: Q6→Q7) 표가
    전부 마지막에 열린 질문한테 쏠려버렸다. 표도 자기 위치(top 좌표)를 갖고 있으므로
    단어와 같은 좌표축으로 정렬해 하나의 스트림으로 합치면, 파서가 "이 표를 만난
    시점에 어떤 질문이 열려있었는지"를 정확히 알 수 있다.
    반환: [("word", 텍스트), ...] 와 [("table", 표객체), ...] 가 섞인 리스트, top 순서대로.
    """
    words = page.extract_words()
    ranges = [(t["bbox"][1], t["bbox"][3]) for t in tables]

    def in_table(w):
        return any(top - 2 <= w["top"] <= bottom + 2 for top, bottom in ranges)

    kept = [w for w in words if not in_table(w)]
    items = [(w["top"], round(w["top"] / 3), w["x0"], "word", w["text"]) for w in kept]
    for t in tables:
        # 표는 자기 bbox의 top 위치에서 "발생"한 것으로 취급 — 표가 시작되는 지점의
        # 읽기 순서에 끼워넣는다.
        items.append((t["bbox"][1], round(t["bbox"][1] / 3), t["bbox"][0], "table", t))
    items.sort(key=lambda x: (x[1], x[2]))
    return [(kind, val) for _, _, _, kind, val in items]


def group_by_page(word_page_list):
    """(단어, 페이지) 리스트를 페이지가 바뀌는 지점마다 잘라서
    [(text, page), (text, page), ...] 형태로 묶어준다.
    부록/답변에 공통으로 쓰는 헬퍼 — 페이지 추적 로직을 한 곳에 모아둠."""
    groups = []
    cur_words, cur_page = [], None
    for w, p in word_page_list:
        if cur_page is None:
            cur_page = p
        if p != cur_page:
            groups.append((" ".join(cur_words), cur_page))
            cur_words, cur_page = [], p
        cur_words.append(w)
    if cur_words:
        groups.append((" ".join(cur_words), cur_page))
    return groups


class QAParser:
    """페이지가 넘어가도 상태(현재 어떤 질문을 파싱 중인지)를 유지하는 파서.

    ⚠️ 수정 (3차): 부록뿐 아니라 질문(Q)·답변(A) 토큰도 전부 (단어, 페이지) 쌍으로
    저장한다. 답변이 다음 페이지로 넘어가는 경우(패턴3)를 잡으려면, "답변이 시작된
    페이지" 하나만으로는 부족하고 답변을 이루는 단어 하나하나가 실제 몇 페이지에서
    왔는지가 필요하기 때문. 부록에서 쓴 것과 동일한 (토큰, 페이지) 방식을 그대로 재사용.
    """

    def __init__(self):
        self.mode = "before_qlist"
        self.q_tokens = []   # [(word, page), ...] 현재 진행 중인 질문
        self.a_tokens = []   # [(word, page), ...] 현재 진행 중인 답변
        self.in_answer = False
        self.qa_raw_pairs = []       # [(q_tokens, a_tokens, tables), ...]
        self.appendix_tokens = []    # [(token, page_num), ...]
        # ⚠️ 수정 (4차, 표 오귀속 버그 해결): 표를 "지금 열려있는 질문"에 즉시 붙인다.
        # 이전엔 페이지를 다 읽은 뒤에 표 target을 판단해서, 한 페이지 안에 질문
        # 경계가 여러 번 있으면(Q6→Q7처럼) 표가 전부 마지막 질문한테 쏠렸다.
        # feed()가 단어/표가 섞인 위치순 스트림을 받으므로, 표를 만난 "그 순간"
        # 열려있는 질문에 바로 담아두면 이 문제가 근본적으로 사라진다.
        self.cur_tables = []

    def feed(self, stream, page_num):
        """stream: [("word", 텍스트) | ("table", 표객체), ...] — 읽는 순서(top 좌표) 그대로."""
        for kind, val in stream:
            if kind == "table":
                if self.mode == "qa":
                    self.cur_tables.append((val, page_num))
                continue

            tok = val
            if tok == "첨부":
                self.mode = "appendix"
                continue

            if self.mode == "appendix":
                self.appendix_tokens.append((tok, page_num))
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
                    if self.q_tokens or self.a_tokens:
                        self.qa_raw_pairs.append((self.q_tokens, self.a_tokens, self.cur_tables))
                    rest = tok.lstrip("○")
                    self.q_tokens = [(rest, page_num)] if rest else []
                    self.a_tokens = []
                    self.cur_tables = []
                    self.in_answer = False
                elif tok.startswith("•"):
                    self.in_answer = True
                    rest = tok.lstrip("•")
                    if rest:
                        self.a_tokens.append((rest, page_num))
                else:
                    if self.in_answer:
                        self.a_tokens.append((tok, page_num))
                    else:
                        self.q_tokens.append((tok, page_num))

    def finalize(self):
        if self.q_tokens or self.a_tokens:
            self.qa_raw_pairs.append((self.q_tokens, self.a_tokens, self.cur_tables))
            self.q_tokens, self.a_tokens, self.cur_tables = [], [], []

        qa_pairs = []  # (q_text, a_text, q_page, extra_a_pieces, tables)
        for q_toks, a_toks, tables in self.qa_raw_pairs:
            q_groups = group_by_page(q_toks)
            q_text = " ".join(t for t, _ in q_groups)
            q_page = q_groups[0][1] if q_groups else None

            a_groups = group_by_page(a_toks)
            if not a_groups:
                qa_pairs.append((q_text, "", q_page, [], tables))
                continue
            first_text, first_page = a_groups[0]
            extra = a_groups[1:]  # 답변이 페이지를 더 넘어간 나머지 조각들
            qa_pairs.append((q_text, first_text, q_page, extra, tables))

        appendix_text = " ".join(t for t, _ in self.appendix_tokens)
        return qa_pairs, appendix_text, self.appendix_tokens


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


def split_appendix_articles(appendix_text, appendix_tokens):
    """첨부 법령 원문을 '(법령명) 제N조(제목)' 패턴 기준으로 조문 단위로 분리.

    ⚠️ 수정 (패턴2 버그 해결): 이제 각 조문마다 실제로 몇 페이지에 걸쳐 있었는지
    appendix_tokens의 (토큰, 페이지) 매핑을 통해 계산한다.
    조문 하나가 한 페이지 안에 들어가면 그대로 청크 하나.
    조문 하나가 페이지 경계를 걸치면(예: 3p 끝~4p 시작), 민서님 표 분할 방식과
    동일한 원칙으로 페이지별로 쪼개서 청크 여러 개로 만든다 — 절대 잘못된
    단일 페이지 번호로 뭉치지 않는다.

    반환: [(text, page), ...]  -- 이미 페이지별로 쪼개진 상태이므로 항상 단일 page
    """
    if not appendix_tokens:
        return []

    # 토큰 → 그 토�acky 텍스트 내 시작 문자 offset 매핑 (appendix_text는 " ".join(tokens) 결과)
    token_starts = []
    pos = 0
    for tok, pg in appendix_tokens:
        token_starts.append((pos, pg))
        pos += len(tok) + 1  # +1 은 join할 때 붙는 공백

    def pages_in_span(start, end):
        return [pg for offset, pg in token_starts if start <= offset < end]

    pattern = r'(?<![가-힣])(?:[가-힣]+\s+)?[가-힣]+법(?:\s*시행령|\s*시행규칙)?\s*제\d+조(?:의\d+)?\([^)]+\)'
    matches = list(re.finditer(pattern, appendix_text))
    if not matches:
        piece = trim_stray_prefix(appendix_text.strip())
        if not piece:
            return []
        page = appendix_tokens[0][1]
        return [(piece, page)]

    out = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(appendix_text)
        piece = trim_stray_prefix(appendix_text[start:end].strip())
        if not piece:
            continue

        pages = pages_in_span(start, end)
        if not pages:
            continue
        unique_pages = sorted(set(pages))

        if len(unique_pages) == 1:
            # 한 페이지 안에 완전히 들어가는 정상 케이스
            out.append((piece, unique_pages[0]))
            continue

        # ⚠️ 조문이 페이지 경계를 걸침 — 페이지별로 쪼갠다 (병합해서 단일 page로 우기지 않는다)
        # 조문 텍스트 안에서 "이 문자가 몇 페이지 토큰에서 왔는지" 어절 단위로 재구성
        words = piece.split(" ")
        # piece 시작이 원본 offset 'start'와 정확히 일치하지 않을 수 있어(strip 때문에),
        # 안전하게 다시 원문 slice 기준으로 어절-페이지를 매핑한다
        sub_tokens = [(off, pg) for off, pg in token_starts if start <= off < end]
        # sub_tokens 순서대로 어절을 페이지별로 묶는다
        groups = []
        cur_page = None
        cur_words = []
        tok_idx = 0
        for tok, pg in appendix_tokens:
            off = token_starts[tok_idx][0]
            tok_idx += 1
            if off < start or off >= end:
                continue
            if cur_page is None:
                cur_page = pg
            if pg != cur_page:
                groups.append((cur_words, cur_page))
                cur_words, cur_page = [], pg
            cur_words.append(tok)
        if cur_words:
            groups.append((cur_words, cur_page))

        for gi, (gwords, gpage) in enumerate(groups):
            gtext = " ".join(gwords)
            if gi > 0:
                gtext = "(조문 계속) " + gtext
            if gtext.strip():
                out.append((gtext.strip(), gpage))

    return out


def process_doc(doc_id):
    pdf_path = os.path.join(RAW_DIR, f"{doc_id}.pdf")
    with pdfplumber.open(pdf_path) as pdf:
        pages_tables = []
        for page in pdf.pages:
            pages_tables.append(extract_real_tables(page))
        # ⚠️ 수정: try_merge_across_pages() 호출 제거.
        # 표는 더 이상 페이지를 넘어 병합하지 않는다 — pdfplumber가 페이지별로
        # 인식한 조각을 그대로 각자의 페이지에 남겨둔다 (민서님 pass10 방식과 통일).

        parser = QAParser()

        for page_num, page in enumerate(pdf.pages, start=1):
            tables = pages_tables[page_num - 1]
            # ⚠️ 수정 (표 오귀속 버그 해결): 단어와 표를 하나의 위치순 스트림으로 합쳐서
            # 넘긴다. 표를 "만난 시점"에 열려있는 질문이 자동으로 정해지므로,
            # 한 페이지 안에 질문 경계가 여러 번 있어도(Q6→Q7처럼) 더 이상 헷갈리지 않는다.
            stream = get_ordered_stream(page, tables)
            parser.feed(stream, page_num)

        qa_pairs, appendix_text, appendix_tokens = parser.finalize()
        # qa_pairs 항목: (q_text, a_text_first_page, q_page, extra_a_pieces, tables)
        #   tables: [(표객체, 그 표가 실제 있던 페이지), ...] — 이 질문에 속한 표들

        qa_list = []
        extra_table_chunks = []  # (section_text, page, text)
        for q_text, a_text, q_page, extra_a, tables in qa_pairs:
            a = a_text
            # ⚠️ 표가 실제로 있던 페이지가 질문 페이지와 같으면 답변에 자연스럽게 녹여 넣고,
            # 다르면(표가 다음 페이지로 넘어간 경우) 별도의 "표 계속" 청크로 분리한다.
            for t, table_page_num in tables:
                if q_page is not None and table_page_num != q_page:
                    extra_table_chunks.append(
                        (q_text, table_page_num, table_to_text(t, continued=True))
                    )
                else:
                    a += " " + table_to_text(t)
            qa_list.append((q_text, a, q_page, extra_a))

        articles = split_appendix_articles(appendix_text, appendix_tokens)
        return qa_list, articles, extra_table_chunks


if __name__ == "__main__":
    all_chunks = []

    for doc_id in DOC_IDS:
        qa_list, articles, extra_table_chunks = process_doc(doc_id)
        chunk_no_by_page = {}

        for q, a, q_page, extra_a in qa_list:
            page = q_page or 1  # 혹시 모를 None 방어
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

            # ⚠️ 새로 추가 (패턴3 해결): 답변이 다음 페이지(들)로 넘어간 부분은
            # 억지로 앞 청크에 붙이지 않고, 실제 페이지에 별도 청크로 만든다.
            for extra_text, extra_page in extra_a:
                if not extra_text.strip():
                    continue
                chunk_no_by_page[extra_page] = chunk_no_by_page.get(extra_page, 0) + 1
                cont_text = f"(답변 계속 — {q[:40]})\n{extra_text.strip()}"
                c2 = ChunkA(
                    chunk_id=f"{doc_id}#p{extra_page}#c{chunk_no_by_page[extra_page]:03d}",
                    doc_id=doc_id,
                    source="제도",
                    page=extra_page,
                    text=cont_text,
                    parse_method="text",
                    confidence=1.0,
                    section=q[:40],
                    distractor=False,
                )
                all_chunks.append(c2)

        # ⚠️ 새로 추가: 페이지 경계를 걸친 표 조각들을 별도 청크로.
        # 원래 질문과 같은 section 라벨을 붙여 검색 시 맥락이 이어지도록 함.
        for q_text, page, table_text in extra_table_chunks:
            chunk_no_by_page[page] = chunk_no_by_page.get(page, 0) + 1
            text = f"[{q_text[:40]} — 관련 표 계속]\n{table_text}"
            c = ChunkA(
                chunk_id=f"{doc_id}#p{page}#c{chunk_no_by_page[page]:03d}",
                doc_id=doc_id,
                source="제도",
                page=page,
                text=text,
                parse_method="text",
                confidence=1.0,
                section=q_text[:40],
                distractor=False,
            )
            all_chunks.append(c)

        for art_text, page in articles:
            chunk_no_by_page[page] = chunk_no_by_page.get(page, 0) + 1
            c = ChunkA(
                chunk_id=f"{doc_id}#p{page}#c{chunk_no_by_page[page]:03d}",
                doc_id=doc_id,
                source="제도",
                page=page,
                text=art_text,
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
