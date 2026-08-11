"""
장르C(중도인출) 전용 Chunk 확장.
schema.py의 Chunk를 상속해서 '사유', '하위유형' 커스텀 필드만 추가함.
공통 7필드 + section 검증 로직은 부모 클래스(Chunk)의 __post_init__을 그대로 재사용.
"""
from dataclasses import dataclass
from typing import Optional
from schema import Chunk


@dataclass
class ChunkC(Chunk):
    사유: str = ""                      # 예: "요양", "파산개인회생", "전세보증금", "주택구입", "재난피해"
    하위유형: Optional[str] = None       # doc49(주택구입), doc50(재난피해)만 사용. 나머지는 None

    def __post_init__(self):
        super().__post_init__()  # 부모의 검증(source, parse_method, text 길이 체크) 그대로 실행
        if not self.사유:
            raise ValueError(f"{self.chunk_id}: '사유' 필드는 필수예요. 장르C 문서는 반드시 사유가 있어야 해요.")
