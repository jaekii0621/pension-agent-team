"""워크플로(공장 배치도).

    from workflows import pension_agent, run

  · rag_core.py   : 서브질문 1개를 푸는 자기교정 RAG 루프
  · main_graph.py : 최상위 — 안전성 → 분석 → 라우팅 → 분해 → 병렬 RAG → 합성

선행모델은 실험용 대안 아키텍처를 20여 개 남겨 뒀지만(naive_rag, contregen,
generator_critic, series_parallel …), 우리는 채택본 하나만 둔다.
지금은 성능을 **측정**하는 단계이므로 경로가 하나여야 측정값의 의미가 분명하다.
구성요소 비교는 config.WORKFLOW_SETTINGS 스위치로 한다.
"""

from .main_graph import pension_agent, run          # noqa: F401
from .rag_core import rag_core                      # noqa: F401

__all__ = ["pension_agent", "run", "rag_core"]
