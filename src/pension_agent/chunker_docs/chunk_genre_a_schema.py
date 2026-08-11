"""
장르A(제도 Q&A) 전용 Chunk 확장.
Q&A 본문 청크와, 첨부된 관련법령 조문 청크를 구분하기 위해 'distractor' 필드를 추가함.
- distractor=False: 질문-답변 본문 (실제 검색 근거로 우선 사용)
- distractor=True: 첨부 관련법령·조문 (근거 보강용이지만, 질문과 무관하게 낚시성으로 매칭될 수 있어 태깅해둠)
"""
from dataclasses import dataclass
from schema import Chunk


@dataclass
class ChunkA(Chunk):
    distractor: bool = False
