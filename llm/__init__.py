"""LLM 계층 진입점.

노드 코드는 항상 이렇게만 쓴다:

    from llm import llm, embedder

공급자를 바꾸더라도 이 파일과 clova_client.py 만 손대면 된다
(선행모델이 `from llm import llm` 한 줄로 4개 모델을 감춘 것과 같은 발상).
"""

import logging

import config

from .clova_client import ClovaLLM, CallLog, LLMError          # noqa: F401
from .embeddings import (                                       # noqa: F401
    ClovaEmbedder,
    E5Embedder,
    EmbeddingError,
    get_embedder,
)
from .offline import OfflineLLM                                 # noqa: F401

_log = logging.getLogger(__name__)

# 키가 없거나 PENSION_DRY_RUN=true 면 드라이런으로 자동 전환한다.
# → 배선·분기·병합은 공짜로 검증하고, 답변 품질만 실제 키로 측정한다.
#
# ⚠️ 이 전환이 **조용히** 일어나면 안 된다. 드라이런에서는
#   · 토큰 사용량이 항상 0 이고 (네트워크 호출이 없으므로)
#   · 답변이 근거를 옮겨 적은 자리표시자이며
#   · 안전검사·질의분석이 정규식이라 오타에 그대로 뚫린다 ("퇴집급여" → 범위 밖)
#   토큰 0 만 보고 "왜 0이지?" 하게 만들면 출력 설계가 잘못된 것이다.
DRY_RUN_ACTIVE = bool(config.DRY_RUN or not config.CLOVA_API_KEY)

if DRY_RUN_ACTIVE:
    _why = ("PENSION_DRY_RUN=true" if config.DRY_RUN else "CLOVA_API_KEY 없음")
    _log.warning(
        "드라이런 모드로 실행합니다 (%s) — LLM 을 호출하지 않습니다.\n"
        "  · 토큰 사용량은 항상 0 으로 표시됩니다\n"
        "  · 답변은 규칙 기반 자리표시자입니다 (품질 측정 불가)\n"
        "  · 배선·검색·필터·인용 번호 매핑은 그대로 검증됩니다\n"
        "  실제 CLOVA 로 돌리려면 CLOVA_API_KEY 를 설정한 뒤 다시 import 하세요.",
        _why,
    )
    llm = OfflineLLM()
else:
    llm = ClovaLLM()


def dry_run_banner() -> str:
    """CLI·노트북이 출력 앞에 붙일 한 줄 경고. 실제 모드면 빈 문자열."""
    if not DRY_RUN_ACTIVE:
        return ""
    why = "PENSION_DRY_RUN=true" if config.DRY_RUN else "CLOVA_API_KEY 없음"
    return (f"⚠️  드라이런 ({why}) — 아래 답변은 규칙 기반 자리표시자이고 "
            f"토큰은 0으로 표시됩니다.")


def embedder() -> E5Embedder:
    """지연 초기화 — 모델 로드가 수 초 걸리므로 실제로 쓸 때 만든다."""
    return get_embedder()


__all__ = ["llm", "embedder", "ClovaLLM", "E5Embedder", "ClovaEmbedder",
           "LLMError", "EmbeddingError", "CallLog",
           "DRY_RUN_ACTIVE", "dry_run_banner"]
