import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunk_genre_c_schema import ChunkC
from genre_c_data import COMMON_INTRO, COMMON_OUTRO, DOCS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

all_chunks = []

for doc_id, info in DOCS.items():
    사유 = info["사유"]
    chunk_no = 1

    # "신청양식" 항목은 5개 문서 전부 내용이 짧고 거의 동일(양식명만 있음) → 별도 청크 대신 개요에 합침
    items = info["items"]
    form_item = next((it for it in items if it["label"] == "신청양식"), None)
    detail_items = [it for it in items if it["label"] != "신청양식"]

    # c001: 공통 도입부 + 사유별 서두설명 + 신청양식 안내를 하나로 합쳐서 "개요" 청크로
    intro_text = f"{COMMON_INTRO}\n\n{info['lead']}"
    if form_item:
        intro_text += f"\n\n[신청양식] {form_item['content']}"

    c = ChunkC(
        chunk_id=f"{doc_id}#p1#c{chunk_no:03d}",
        doc_id=doc_id,
        source="절차",
        page=1,
        text=intro_text,
        parse_method="text",
        confidence=1.0,
        section="개요",
        사유=사유,
        하위유형=None,
    )
    all_chunks.append(c)
    chunk_no += 1

    # c002~: 세부 서류 항목(신청양식 제외) 하나당 청크 하나
    for item in detail_items:
        item_text = f"[{사유} 중도인출 - {item['label']}]\n{item['content']}"
        c = ChunkC(
            chunk_id=f"{doc_id}#p1#c{chunk_no:03d}",
            doc_id=doc_id,
            source="절차",
            page=1,
            text=item_text,
            parse_method="text",
            confidence=1.0,
            section=item["label"],
            사유=사유,
            하위유형=item["하위유형"],
        )
        all_chunks.append(c)
        chunk_no += 1

    # ⚠️ 신규: 표 bbox 밖에 있어서 이번에 새로 발견된 공통 유의사항 3줄.
    # "개요"에 억지로 붙이면 신청기준/서류 안내랑 성격이 안 맞아서(질의응답 시 잡음 유발),
    # A의 "첨부-관련법령" 패턴처럼 별도 청크로 분리 — 유의사항만 물어보는 질의에 정확히 매칭되도록.
    outro_text = f"[{사유} 중도인출 - 유의사항]\n{COMMON_OUTRO}"
    c_outro = ChunkC(
        chunk_id=f"{doc_id}#p1#c{chunk_no:03d}",
        doc_id=doc_id,
        source="절차",
        page=1,
        text=outro_text,
        parse_method="text",
        confidence=1.0,
        section="유의사항",
        사유=사유,
        하위유형=None,
    )
    all_chunks.append(c_outro)
    chunk_no += 1

# 결과 저장 — B/D와 동일하게 data/chunks/ 로 통일 
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "..", "..", "data", "chunks", "chunks_genre_c.jsonl")
OUTPUT_PATH = os.path.normpath(OUTPUT_PATH)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for c in all_chunks:
        f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

print(f"총 {len(all_chunks)}개 청크 생성 완료 → {OUTPUT_PATH}")
print()
for c in all_chunks:
    print(f"{c.chunk_id} | 사유={c.사유} | 하위유형={c.하위유형} | section={c.section} | 글자수={len(c.text)}")
