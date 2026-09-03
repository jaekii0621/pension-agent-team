"""인덱스 계층.

    from index import get_index, load_corpus, to_citation

선행모델의 vector_store.py / retriever.py / embeddings.py 를 합친 자리다.
서버 프로세스가 아니라 **인메모리 인덱스**인 점이 가장 큰 차이 (이유는 store.py 상단 참고).
"""

from . import artifact                                   # noqa: F401
from .loader import (                                    # noqa: F401
    chunk_by_id,
    chunk_text_for_values,
    corpus_stats,
    format_chunk_for_llm,
    fund_core,
    fund_discriminators,
    fund_names,
    fund_variant,
    is_table_chunk,
    is_table_continuation,
    load_corpus,
    match_known_fund,
    match_known_funds,
    searchable_text,
    sibling_funds,
    table_neighbor_ids,
    to_citation,
)
from .store import (                                     # noqa: F401
    PensionIndex,
    bm25_top1,
    get_index,
    product_signal,
    reset_index,
)

__all__ = [
    "load_corpus", "corpus_stats", "format_chunk_for_llm", "to_citation",
    "fund_names", "match_known_fund", "match_known_funds",
    "fund_core", "fund_variant", "sibling_funds", "fund_discriminators",
    "chunk_by_id", "chunk_text_for_values",
    "is_table_continuation", "table_neighbor_ids",
    "searchable_text", "is_table_chunk",
    "PensionIndex", "get_index", "reset_index",
    "bm25_top1", "product_signal",
    "artifact",
]
