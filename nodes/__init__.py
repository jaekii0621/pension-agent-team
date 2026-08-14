"""노드(작업대) 모음.

선행모델과 같은 규칙을 따른다.
  · 노드는 `state` 를 받아 **바꿀 키만** dict 로 반환하는 평범한 파이썬 함수다.
  · 노드 안에서 다음 노드를 부르지 않는다. 연결은 workflows/ 가 한다.
  · 모든 노드는 `trace` 키를 함께 반환해 실행 경로를 남긴다.

에이전트 종류와 파일 대응
  질문 분리   → question_decomposer.py
  라우터      → router.py
  검색/채점   → document_retriever.py, document_grader.py
  도메인 전문 → answer_generator.py (제도/세제/절차/상품 프롬프트로 분기)
  검증        → verifier.py (환각 + 품질)
  계산기      → calculator.py + pension_tools.py
  전략        → strategy.py
  합성        → combiner.py
"""

from .answer_generator import generate_answer, generate_web_answer          # noqa: F401
from .calculator import calc_agent                                          # noqa: F401
from .combiner import combine_answer, out_of_scope_answer                   # noqa: F401
from .document_grader import grade_documents                                # noqa: F401
from .document_retriever import retrieve_documents                          # noqa: F401
from .query_analyzer import analyze_query                                   # noqa: F401
from .question_decomposer import (                                          # noqa: F401
    check_sufficiency,
    decompose_question,
    decompose_residual,
)
from .question_rewriter import rewrite_question                             # noqa: F401
from .router import route_query                                             # noqa: F401
from .safety_checker import check_safety                                    # noqa: F401
from .strategy import strategy_agent                                        # noqa: F401
from .verifier import verify_answer                                         # noqa: F401
from .web_searcher import search_web                                        # noqa: F401

__all__ = [
    "check_safety", "analyze_query", "route_query",
    "decompose_question", "decompose_residual", "check_sufficiency",
    "retrieve_documents", "grade_documents",
    "generate_answer", "generate_web_answer",
    "verify_answer", "rewrite_question", "search_web",
    "calc_agent", "strategy_agent",
    "combine_answer", "out_of_scope_answer",
]
