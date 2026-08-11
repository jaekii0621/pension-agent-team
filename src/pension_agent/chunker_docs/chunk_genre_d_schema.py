"""
장르D 안전문서(doc55, doc6, doc35, doc34) 전용 Chunk 확장.
장르B의 ChunkB와 거의 동일한 목적(origin 추적)이지만, xlsx 코드표(doc34)도 다루므로
문서 형식(file_type)도 같이 기록해서 나중에 소스별로 구분하기 쉽게 함.
"""
from dataclasses import dataclass
from schema import Chunk


@dataclass
class ChunkD(Chunk):
    origin: str = "paragraph"   # "paragraph" | "table_row"
    file_type: str = "docx"     # "docx" | "xlsx"
