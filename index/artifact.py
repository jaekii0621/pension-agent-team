"""벡터 아티팩트 — **만드는 단계**와 **쓰는 단계**를 가르는 경계.

■ 왜 나눴나
  임베딩은 비싸고(7,939청크 × e5-large), 검색은 싸다(수십 ms). 그런데 예전 구조는
  둘이 붙어 있어서, 파이프라인을 한 번 돌릴 때마다 임베딩 계층이 통째로 딸려 왔다.

      · 문서 임베딩 캐시(emb_*.jsonl, 7,939건)를 **프로세스 시작 때 전부** 파이썬
        리스트로 메모리에 올렸다. 정작 검색은 .npy 행렬만 쓰는데도.
      · 청크 텍스트가 바뀌어도 chunk_id 만 같으면 낡은 벡터를 그대로 썼다.
        (id 목록만 대조했기 때문이다 — 조용히 틀린 벡터로 검색하게 된다)

  그래서 산출물을 **아티팩트**로 못 박고, 두 단계를 분리했다.

      [빌드]  코퍼스 → 임베딩 모델 로드 → 인코딩 → 아티팩트 저장   (무겁다, 가끔)
                python -m index.build_index
      [서빙]  아티팩트 로드 → 검색                                  (가볍다, 매번)
                get_index()

  서빙 단계는 **문서를 절대 다시 인코딩하지 않는다.** 질의 1건을 인코딩할 때만
  모델이 필요하고, 그것도 실제로 벡터 검색을 할 때 지연 로딩된다.

■ 아티팩트 구조 (한 폴더로 묶어 통째로 복사·공유할 수 있게)

      cache/index_<모델슬러그>/
        vectors.npy    (N, dim) float16, L2 정규화 완료
        ids.txt        행 순서대로 chunk_id
        meta.json      지문·모델·차원·빌드 시각

■ 지문(fingerprint)이 핵심이다
  "이 아티팩트가 지금 코퍼스와 맞는가" 를 한 값으로 판정한다.
  chunk_id **와 실제 임베딩 대상 텍스트**를 함께 해시하므로,
    · 청크가 늘거나 줄면          → 달라진다
    · 청크 본문이 수정되면        → 달라진다  (예전엔 못 잡던 경우)
    · 임베딩 모델·접두사가 바뀌면 → 달라진다
  맞지 않으면 **조용히 넘어가지 않고 실패시킨다.** 낡은 벡터로 검색하면
  증상이 "검색이 좀 이상하다" 뿐이라 끝까지 발견되지 않는다.

■ float16 으로 저장하는 이유
  e5 벡터는 L2 정규화돼 있어 값이 [-1, 1] 이고, 코사인 순위를 매기는 데 fp16 정밀도면
  충분하다. 디스크·전송량이 절반이 된다 (7,939 × 1024: 32MB → 16MB).
  로드 시 float32 로 되돌려 내적 속도는 그대로 유지한다.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import config

log = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:                                     # numpy 없으면 벡터검색 비활성
    np = None                                           # type: ignore

FORMAT_VERSION = 1


# ==========================================================================
def artifact_dir() -> Path:
    """이 임베딩 설정에 해당하는 아티팩트 폴더."""
    return Path(config.CACHE_DIR) / f"index_{config.EMBED_SLUG}"


def corpus_fingerprint(chunk_ids: Sequence[str], texts: Sequence[str]) -> str:
    """코퍼스 + 임베딩 설정의 지문.

    임베딩 대상 텍스트까지 해시에 넣는 것이 요점이다. chunk_id 만 보면
    "본문은 고쳤는데 id 는 그대로" 인 경우를 놓치고, 낡은 벡터로 조용히 검색하게 된다.
    """
    h = hashlib.sha1()
    # 임베딩 결과를 바꾸는 설정은 전부 지문에 포함한다.
    h.update(f"v{FORMAT_VERSION}|{config.EMBED_MODEL}|{config.EMBED_DIM}|"
             f"{config.EMBED_PASSAGE_PREFIX}|{config.EMBED_MAX_SEQ}".encode())
    for cid, txt in zip(chunk_ids, texts):
        h.update(cid.encode("utf-8"))
        h.update(b"\x00")
        h.update((txt or "").encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


# ==========================================================================
@dataclass
class VectorArtifact:
    """빌드 산출물. 서빙 단계는 이것만 읽는다."""
    vectors: "np.ndarray"           # (N, dim) float32, L2 정규화 완료
    ids: List[str]
    meta: dict

    @property
    def fingerprint(self) -> str:
        return self.meta.get("fingerprint", "")

    def __len__(self) -> int:
        return len(self.ids)


class ArtifactError(RuntimeError):
    """아티팩트가 없거나 코퍼스와 맞지 않는다."""


# ==========================================================================
def save(vectors, ids: Sequence[str], fingerprint: str, *,
         n_chunks: int) -> Path:
    """빌드 단계에서 호출. 아티팩트 폴더에 3개 파일을 쓴다."""
    if np is None:
        raise ArtifactError("numpy 가 필요합니다: pip install numpy")

    d = artifact_dir()
    d.mkdir(parents=True, exist_ok=True)

    mat = np.asarray(vectors, dtype="float32")
    # 저장 전에 정규화를 확정해 둔다 — 서빙 단계가 이 작업을 반복하지 않게.
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    np.save(d / "vectors.npy", mat.astype("float16"))
    (d / "ids.txt").write_text("\n".join(ids), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({
        "format_version": FORMAT_VERSION,
        "fingerprint": fingerprint,
        "embed_model": config.EMBED_MODEL,
        "embed_dim": int(mat.shape[1]),
        "passage_prefix": config.EMBED_PASSAGE_PREFIX,
        "query_prefix": config.EMBED_QUERY_PREFIX,
        "max_seq": config.EMBED_MAX_SEQ,
        "n_vectors": int(mat.shape[0]),
        "n_chunks": n_chunks,
        "normalized": True,
        "dtype": "float16",
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("아티팩트 저장: %s (%d × %d)", d, mat.shape[0], mat.shape[1])
    return d


def exists() -> bool:
    d = artifact_dir()
    return all((d / f).exists() for f in ("vectors.npy", "ids.txt", "meta.json"))


def load(expected_fingerprint: Optional[str] = None) -> VectorArtifact:
    """서빙 단계에서 호출. 지문이 안 맞으면 **예외로 실패한다.**"""
    if np is None:
        raise ArtifactError("numpy 가 필요합니다: pip install numpy")

    d = artifact_dir()
    if not exists():
        raise ArtifactError(
            f"벡터 아티팩트가 없습니다: {d}\n"
            f"  → `python -m index.build_index` 로 먼저 만드세요.\n"
            f"  (BM25 단독으로 돌리려면 PENSION_REQUIRE_VECTORS=false)"
        )

    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format_version") != FORMAT_VERSION:
        raise ArtifactError(
            f"아티팩트 형식 버전이 다릅니다 "
            f"(파일 {meta.get('format_version')} ≠ 코드 {FORMAT_VERSION}).\n"
            f"  → `python -m index.build_index --force` 로 다시 만드세요."
        )

    if expected_fingerprint and meta.get("fingerprint") != expected_fingerprint:
        raise ArtifactError(
            "벡터 아티팩트가 현재 코퍼스와 맞지 않습니다 (지문 불일치).\n"
            f"  아티팩트: {meta.get('fingerprint', '')[:16]}…  "
            f"({meta.get('n_vectors')}건, {meta.get('built_at')})\n"
            f"  현재코퍼스: {expected_fingerprint[:16]}…\n"
            "  청크가 늘거나 줄었거나, 본문이 수정됐거나, 임베딩 설정이 바뀌었습니다.\n"
            "  → `python -m index.build_index` 로 다시 만드세요.\n"
            "  (이대로 두면 의미 검색이 낡은 벡터로 조용히 잘못 동작합니다)"
        )

    # mmap 으로 읽어 로드를 즉시 끝낸다. 실제 사용 시점에 float32 로 승격.
    mat = np.load(d / "vectors.npy", mmap_mode="r")
    ids = (d / "ids.txt").read_text(encoding="utf-8").splitlines()
    if len(ids) != mat.shape[0]:
        raise ArtifactError(
            f"아티팩트가 손상됐습니다 (ids {len(ids)}건 ≠ vectors {mat.shape[0]}행).\n"
            f"  → `python -m index.build_index --force`"
        )

    return VectorArtifact(vectors=np.asarray(mat, dtype="float32"),
                          ids=ids, meta=meta)


def describe() -> dict:
    """아티팩트 상태 요약 (점검용). 없으면 exists=False."""
    d = artifact_dir()
    if not exists():
        return {"exists": False, "path": str(d)}
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    size_mb = sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / (1024 * 1024)
    return {"exists": True, "path": str(d), "size_mb": round(size_mb, 1), **meta}
