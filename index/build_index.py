"""임베딩 사전 계산 스크립트.

    python -m index.build_index --probe    # 모델이 로드되는지 1건으로 확인
    python -m index.build_index            # 전 청크 임베딩 → cache/*.npy
    python -m index.build_index --stats    # 코퍼스 구성 확인 (모델 불필요)

선행모델은 Pathway 서버가 문서를 감시하며 자동 색인했지만, 우리 코퍼스는 고정이므로
**한 번 만들어 두고 재사용**한다.

임베딩을 로컬 모델(multilingual-e5-large)로 바꾸면서 과금은 사라졌지만, 캐시는
그대로 둔다 — 3,853건 재계산에 드는 시간을 아끼고, 청크를 하나만 고쳐도 그 청크만
다시 계산되기 때문이다.
"""

from __future__ import annotations

import argparse
import json
import sys

import config
from llm.embeddings import get_embedder

from .loader import corpus_stats, load_corpus, searchable_text

try:
    import numpy as np
except ImportError:
    np = None


def cmd_stats() -> int:
    s = corpus_stats()
    print(json.dumps(s, ensure_ascii=False, indent=2))
    return 0


def cmd_probe() -> int:
    r = get_embedder().probe()
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if not r["ok"]:
        print("\n[해결 순서]")
        print("  1) pip install sentence-transformers")
        print("  2) 첫 실행은 모델을 내려받으므로 네트워크가 필요하다 (약 2.2GB)")
        print("  3) GPU 를 쓰려면 .env 에 PENSION_EMBED_DEVICE=cuda")
        return 1
    return 0


def cmd_build() -> int:
    if np is None:
        print("numpy 가 필요합니다: pip install numpy")
        return 1

    emb = get_embedder()
    if not emb.enabled:
        print("임베딩 비활성 상태입니다 (pip install sentence-transformers).")
        print("BM25 단독으로도 파이프라인은 동작합니다.")
        return 1

    chunks = load_corpus()
    texts = [searchable_text(c) for c in chunks]
    print(f"{len(texts)}개 청크 임베딩 시작 (모델={config.EMBED_MODEL})")

    # ⚠️ 문서 쪽은 반드시 embed_passages 로 — "passage: " 접두사가 붙어야 한다.
    mat = emb.embed_passages(texts)
    if mat is None:
        print("임베딩 실패")
        return 1

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.CACHE_DIR / f"vectors_{config.EMBED_SLUG}.npy", mat)
    (config.CACHE_DIR / f"vectors_{config.EMBED_SLUG}.ids.txt").write_text(
        "\n".join(c["chunk_id"] for c in chunks), encoding="utf-8")
    print(f"완료: {mat.shape} → {config.CACHE_DIR}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="연금 에이전트 인덱스 빌더")
    ap.add_argument("--probe", action="store_true", help="임베딩 모델 로드 확인")
    ap.add_argument("--stats", action="store_true", help="코퍼스 통계 (모델 불필요)")
    a = ap.parse_args()

    if a.stats:
        return cmd_stats()
    if a.probe:
        return cmd_probe()
    return cmd_build()


if __name__ == "__main__":
    sys.exit(main())
