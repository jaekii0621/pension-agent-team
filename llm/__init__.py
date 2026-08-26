"""LLM 계층 진입점.

노드 코드는 항상 이렇게만 쓴다:

    from llm import llm, embedder

공급자를 바꾸더라도 이 파일과 clova_client.py 만 손대면 된다
(선행모델이 `from llm import llm` 한 줄로 4개 모델을 감춘 것과 같은 발상).

■ 드라이런(대역 LLM)을 제거했다
  예전에는 키가 없으면 규칙 기반 `OfflineLLM` 으로 조용히 전환됐다. 배선 검증에는
  쓸모가 있었지만 대가가 컸다.
    · 답변이 자리표시자라 품질을 전혀 대변하지 못하는데
    · 기록 파일(eval/results)에는 실제 실행과 **똑같은 모양**으로 남는다
    · 토큰이 0 이고 안전검사가 정규식이라 결과가 체계적으로 낙관적이다
  섞이면 구분할 방법이 없고, 그런 기록으로 판단하면 틀린 결론에 도달한다.

  배선만 확인하고 싶다면 LLM 없이 돌아가는 경로가 따로 있다.
      python run_cli.py --check              설정·코퍼스·검색 단독 점검
      python -m index.build_index --stats    코퍼스 구성만 출력

■ 키가 없으면 어떻게 되나
  import 는 성공한다(점검 명령이 키 없이도 돌아야 하므로). 실제 호출 시점에
  `LLMError` 로 명확히 실패한다 — 조용히 다른 것으로 대체되지 않는다.
"""

import logging

import config

from .clova_client import (                                      # noqa: F401
    CallLog,
    ClovaLLM,
    LLMBudgetExceeded,
    LLMError,
)
from .embeddings import (                                        # noqa: F401
    ClovaEmbedder,
    E5Embedder,
    EmbeddingError,
    get_embedder,
)

_log = logging.getLogger(__name__)

if not config.CLOVA_API_KEY:
    _log.warning(
        "CLOVA_API_KEY 가 없습니다 — LLM 호출은 실패합니다.\n"
        "  · 파이프라인 실행에는 키가 필요합니다 (.env 또는 환경변수)\n"
        "  · 키 없이 확인 가능한 것: `python run_cli.py --check` (검색·코퍼스 점검)"
    )

llm = ClovaLLM()


def embedder() -> E5Embedder:
    """지연 초기화 — 모델 로드가 수 초~수십 초 걸리므로 실제로 쓸 때 만든다."""
    return get_embedder()


__all__ = ["llm", "embedder", "ClovaLLM", "E5Embedder", "ClovaEmbedder",
           "LLMError", "LLMBudgetExceeded", "EmbeddingError", "CallLog"]
