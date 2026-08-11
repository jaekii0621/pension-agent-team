"""
장르B(세제 해설) 전용 Chunk 확장.
청크가 원본에서 어떤 형태였는지(문단/표 행/흐름도 대체/계산박스 대체) 추적하기 위한 필드 추가.
디버깅 및 나중에 "이 청크 왜 이렇게 생겼지?" 확인할 때 유용.
"""
from dataclasses import dataclass
from schema import Chunk


@dataclass
class ChunkB(Chunk):
    origin: str = "paragraph"  # "paragraph" | "table_row" | "flowchart_narrative" | "calc_box_text"
