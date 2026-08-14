"""인덱스 계층.

    from index import get_index, load_corpus, to_citation

선행모델의 vector_store.py / retriever.py / embeddings.py 를 합친 자리다.
서버 프로세스가 아니라 **인메모리 인덱스**인 점이 가장 큰 차이 (이유는 store.py 상단 참고).
"""

from .loader import (                                    # noqa: F401
    corpus_stats,
    format_chunk_for_llm,
    fund_names,
    load_corpus,
    searchable_text,
    to_citation,
)
from .store import PensionIndex, get_index, reset_index  # noqa: F401

__all__ = [
    "load_corpus", "corpus_stats", "format_chunk_for_llm", "to_citation",
    "fund_names", "searchable_text",
    "PensionIndex", "get_index", "reset_index",
]
