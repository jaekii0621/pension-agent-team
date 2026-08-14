"""성능 평가 계층.

지금 프로젝트의 목적이 **배포가 아니라 성능 측정**이므로 이 폴더가 사실상 주인공이다.

    python -m eval.runner --retrieval-only    # 1단계: 검색이 정답을 가져오나 (LLM 0회)
    python -m eval.runner --no-llm-judge      # 2단계: 파이프라인 경로·비용 점검
    python -m eval.runner --tag full          # 3단계: 정확도까지 전부

측정 순서를 이렇게 잡는 이유: 검색이 정답 청크를 못 가져오면 그 뒤 단계를 아무리
고쳐도 정답이 안 나온다. **아래 단계부터 확정하고 올라간다.**
"""

from .metrics import aggregate_scores, retrieval_metrics, source_accuracy  # noqa: F401
from .runner import eval_case, eval_retrieval_only, load_goldset           # noqa: F401

__all__ = ["load_goldset", "eval_case", "eval_retrieval_only",
           "retrieval_metrics", "source_accuracy", "aggregate_scores"]
