"""[빌드 단계] 코퍼스 → 벡터 아티팩트.

    python -m index.build_index            # 아티팩트 생성 (최신이면 건너뜀)
    python -m index.build_index --force    # 최신이어도 다시 만든다
    python -m index.build_index --status   # 아티팩트 상태만 확인 (모델 불필요)
    python -m index.build_index --stats    # 코퍼스 구성만 확인 (모델 불필요)
    python -m index.build_index --probe    # 임베딩 모델이 로드되는지 1건으로 확인

■ 이 파일이 유일하게 문서를 임베딩하는 곳이다
  서빙 단계(get_index → 검색)는 여기서 만든 .npy 만 읽고, 문서를 다시 인코딩하지
  않는다. 무거운 일(모델 2.2GB 로드 + 7,939건 인코딩)을 한 번만 하기 위한 분리다.

  한 번 만들어 두면 다음부터는 아티팩트 폴더만 있으면 된다. 코랩처럼 세션이
  날아가는 환경에서는 `cache/index_*/` 폴더를 드라이브에 복사해 두고 재사용하면
  임베딩을 다시 돌릴 필요가 없다.

■ 증분 재계산
  텍스트 SHA-1 → 벡터 캐시(emb_*.jsonl)가 있어서, 청크 몇 개만 바뀌면 그것만
  다시 인코딩한다. 이 캐시는 **빌드 단계 전용**이다(서빙 단계는 건드리지 않는다).
"""

from __future__ import annotations

import argparse
import json
import sys

import config
from llm.embeddings import get_embedder

from . import artifact
from .loader import corpus_stats, load_corpus, searchable_text

try:
    import numpy as np
except ImportError:
    np = None


# ==========================================================================
def cmd_stats() -> int:
    print(json.dumps(corpus_stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_status() -> int:
    """아티팩트가 지금 코퍼스와 맞는지 확인한다. 모델을 로드하지 않는다."""
    info = artifact.describe()
    print(json.dumps(info, ensure_ascii=False, indent=2))

    if not info["exists"]:
        print("\n→ 아티팩트가 없습니다. `python -m index.build_index` 로 만드세요.")
        return 1

    chunks = load_corpus()
    fp = artifact.corpus_fingerprint(
        [c["chunk_id"] for c in chunks],
        [searchable_text(c) for c in chunks],
    )
    if fp == info.get("fingerprint"):
        print(f"\n✓ 현재 코퍼스({len(chunks)}청크)와 일치합니다. 그대로 쓰면 됩니다.")
        return 0

    print(f"\n✗ 현재 코퍼스({len(chunks)}청크)와 맞지 않습니다.")
    print(f"    아티팩트 지문: {info.get('fingerprint', '')[:16]}…")
    print(f"    현재   지문: {fp[:16]}…")
    print("  → `python -m index.build_index` 로 다시 만드세요.")
    return 1


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


# ==========================================================================
def cmd_build(force: bool = False) -> int:
    if np is None:
        print("numpy 가 필요합니다: pip install numpy")
        return 1

    chunks = load_corpus()
    texts = [searchable_text(c) for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    fp = artifact.corpus_fingerprint(ids, texts)

    # 이미 최신이면 모델을 로드조차 하지 않는다 (수십 초를 아낀다).
    if not force and artifact.exists():
        info = artifact.describe()
        if info.get("fingerprint") == fp:
            print(f"이미 최신입니다 — {info['n_vectors']}개 벡터, "
                  f"{info['size_mb']}MB, 빌드 {info['built_at']}")
            print(f"  {info['path']}")
            print("  다시 만들려면 --force")
            return 0
        print("코퍼스가 바뀌었습니다 → 다시 만듭니다.")

    emb = get_embedder()
    if not emb.enabled:
        print("임베딩 모델을 쓸 수 없습니다 (sentence-transformers 미설치 또는 "
              "메모리 부족).")
        print("  자세한 원인은 위 로그를 확인하세요. "
              "`python -m index.build_index --probe` 로도 진단할 수 있습니다.")
        return 1

    print(f"{len(texts)}개 청크 임베딩 시작 (모델={config.EMBED_MODEL})")
    # ⚠️ 문서 쪽은 반드시 embed_passages 로 — "passage: " 접두사가 붙어야 한다.
    mat = emb.embed_passages(texts)
    if mat is None:
        print("임베딩 실패")
        return 1

    d = artifact.save(mat, ids, fp, n_chunks=len(chunks))
    info = artifact.describe()
    print(f"\n완료 — {info['n_vectors']}개 벡터 ({info['embed_dim']}차원), "
          f"{info['size_mb']}MB")
    print(f"  {d}")
    print("\n이 폴더를 보관해 두면 다음 실행부터는 임베딩을 다시 하지 않습니다.")
    return 0


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="[빌드] 코퍼스를 벡터 아티팩트로 만든다")
    ap.add_argument("--force", action="store_true",
                    help="최신이어도 다시 만든다")
    ap.add_argument("--status", action="store_true",
                    help="아티팩트가 코퍼스와 맞는지 확인 (모델 불필요)")
    ap.add_argument("--stats", action="store_true",
                    help="코퍼스 구성 확인 (모델 불필요)")
    ap.add_argument("--probe", action="store_true",
                    help="임베딩 모델 로드 확인")
    a = ap.parse_args()

    if a.stats:
        return cmd_stats()
    if a.status:
        return cmd_status()
    if a.probe:
        return cmd_probe()
    return cmd_build(force=a.force)


if __name__ == "__main__":
    sys.exit(main())
