from dataclasses import dataclass, asdict
from typing import Optional

# source 필드에 들어갈 수 있는 값 4개 (INTERFACES.md 기준)
VALID_SOURCES = {"제도", "세제", "상품", "절차"}

# parse_method 필드에 들어갈 수 있는 값 3개
VALID_PARSE_METHODS = {"text", "ocr", "manual"}


@dataclass
class Chunk:
    """청크 하나를 표현하는 클래스. 공통 7개 필수 필드 + section(선택)."""

    chunk_id: str          # 예: "doc10#p1#c001"
    doc_id: str             # 예: "doc10"
    source: str              # "제도" / "세제" / "상품" / "절차" 중 하나
    page: int                # 1부터 시작
    text: str                # 실제 내용 (300~800자 권장)
    parse_method: str        # "text" / "ocr" / "manual"
    confidence: float        # text/manual=1.0, ocr=엔진 신뢰도
    section: Optional[str] = None   # 없으면 None (→ json에서는 null로 저장됨)

    def __post_init__(self):
        # 여기서 값이 규칙에 맞는지 자동으로 검사해요.
        # 잘못된 값이 들어오면 여기서 바로 에러가 나서, 나중에 문제 찾기 편해요.
        if self.source not in VALID_SOURCES:
            raise ValueError(f"source는 {VALID_SOURCES} 중 하나여야 해요. 받은 값: {self.source}")
        if self.parse_method not in VALID_PARSE_METHODS:
            raise ValueError(f"parse_method는 {VALID_PARSE_METHODS} 중 하나여야 해요. 받은 값: {self.parse_method}")
        if not (300 <= len(self.text) <= 800):
            # 권장 사항이라 에러는 아니고 경고만 (강제로 막지는 않음)
            print(f"⚠️ {self.chunk_id}: text 길이가 {len(self.text)}자 — 권장 범위(300~800자) 벗어남")

    def to_dict(self) -> dict:
        """jsonl 파일에 저장하기 위해 딕셔너리로 변환"""
        return asdict(self)