from schema import Chunk

# 일부러 짧은 텍스트로 테스트 (경고 뜨는지 확인용)
c = Chunk(
    chunk_id="doc10#p1#c001",
    doc_id="doc10",
    source="제도",
    page=1,
    text="가입대상은 DB형과 DC형에 따라 다릅니다. " * 10,
    parse_method="text",
    confidence=1.0,
    section="가입대상"
)
print(c.to_dict())

# 일부러 잘못된 source 넣어서 에러 뜨는지 확인
try:
    bad = Chunk(
        chunk_id="doc10#p1#c002",
        doc_id="doc10",
        source="잘못된값",  # 이건 VALID_SOURCES에 없는 값
        page=1,
        text="테스트 텍스트 " * 30,
        parse_method="text",
        confidence=1.0,
    )
except ValueError as e:
    print(f"✅ 검증 성공: {e}")