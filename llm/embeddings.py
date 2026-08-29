"""임베딩 — `intfloat/multilingual-e5-large` (로컬 실행).

■ 왜 CLOVA clir-sts-dolphin 에서 바꿨나
  1) **과금·레이트리밋이 사라진다.** CLOVA 임베딩은 한 번에 문장 하나만 받아서
     3,853청크를 넣으려면 3,853회 유료 호출 + sleep 이었다. 청킹을 손볼 때마다
     전체 재임베딩이 필요한데, 그 비용이 실험 횟수를 제한했다.
  2) **검색 특화 모델이다.** e5 계열은 (query, passage) 쌍 대조학습으로 만든
     retrieval 모델이다. STS(문장 유사도) 모델은 "두 문장이 같은 뜻인가"를 재도록
     학습돼서, 짧은 질문 ↔ 긴 문서 매칭에는 결이 맞지 않는다.
  3) 1024차원이라 기존 캐시 구조(.npy)를 그대로 쓴다.

■ ⚠️ E5 는 접두사가 필수다
      질의  → "query: 퇴직연금 중도인출 사유"
      문서  → "passage: 중도인출 ■ 중도인출 - 근로자퇴직급여보장법 …"
  비대칭 검색(짧은 질의 ↔ 긴 문서)을 학습할 때 이 접두사로 역할을 구분했기 때문에,
  빼면 두 벡터가 같은 공간에 놓이지 않아 성능이 눈에 띄게 떨어진다.
  → `embed_query()` / `embed_passages()` 로 진입점을 아예 나눠 두어
    호출부에서 접두사를 빠뜨릴 수 없게 했다.

■ 캐시
  텍스트 SHA-1 → 벡터를 jsonl 로 남긴다. 청크 하나만 고쳐도 그 청크만 다시 계산한다.
  (로컬 모델이라 비용은 없지만, 3,853건 재계산에 드는 시간을 아낀다)

■ 설치
      pip install sentence-transformers
  미설치·로드 실패 시 `enabled=False` 로 떨어지고 **BM25 단독으로 계속 동작한다.**
  검색이 멈추는 것보다 낫다.

■ ⚠️ 메모리 가드 — 이게 없으면 프로세스가 세그폴트로 죽는다
  e5-large 는 fp32 가중치만 2.2GB 다. 여유 메모리가 부족한 상태에서 로드하면
  파이썬 예외가 아니라 **네이티브 세그폴트(exit 139)** 로 프로세스가 통째로 죽는다.
  try/except 로는 못 잡는다. 그래서 **로드 전에 여유 메모리를 재고**, 모자라면
  아예 시도하지 않고 BM25 단독으로 떨어진다.

  실측(이 개발 PC): 총 3.9GB / 여유 0.3GB → e5-large 적재 불가.
  대안은 `.env` 에서 모델만 바꾸면 된다(코드 수정 없음).
      intfloat/multilingual-e5-base    ~1.1GB, 768차원
      intfloat/multilingual-e5-small   ~0.47GB, 384차원
  ※ 차원이 바뀌면 PENSION_EMBED_DIM 도 함께 바꾸고 인덱스를 다시 만들어야 한다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import config

log = logging.getLogger(__name__)

# 모델별 대략적인 fp32 적재 메모리(MB). 가드 판정에만 쓰는 어림값이다.
_MODEL_MB = {
    "multilingual-e5-large": 2400,
    "multilingual-e5-base": 1200,
    "multilingual-e5-small": 550,
}
_SAFETY_MB = 400          # 토크나이저·활성값·파이썬 런타임 여유분


def _free_memory_mb() -> Optional[float]:
    """여유 물리 메모리(MB). 못 재면 None (그 경우 가드를 건너뛴다)."""
    try:
        import psutil                                   # 있으면 가장 정확하다
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        pass
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullAvailPhys / (1024 * 1024)
    except Exception:
        pass
    return None


def _required_mb(model_name: str) -> int:
    slug = model_name.split("/")[-1]
    for key, mb in _MODEL_MB.items():
        if key in slug:
            return mb
    # 목록에 없는 모델: 이름에 large/base/small 이 있으면 그걸로 어림잡는다
    if re.search(r"large", slug, re.I):
        return 2400
    if re.search(r"small|mini", slug, re.I):
        return 550
    return 1200

try:
    import numpy as np
except ImportError:                                    # numpy 없으면 벡터검색 비활성
    np = None                                          # type: ignore


class EmbeddingError(RuntimeError):
    pass


# ==========================================================================
class E5Embedder:
    """multilingual-e5-large 임베딩 + 디스크 캐시."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.model_name = config.EMBED_MODEL
        self.dim = config.EMBED_DIM
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, List[float]] = {}
        self._model = None                 # 지연 로딩 (모델 로드가 수 초 걸린다)
        self._load_failed = False
        self._cache_loaded = False         # 디스크 캐시는 **빌드 단계에서만** 올린다

    # ------------------------------------------------------------ 단계 분리
    #
    # 문서 캐시(emb_*.jsonl)를 생성자에서 통째로 올리던 것을 지연 로딩으로 바꿨다.
    #
    #   예전: 프로세스가 뜰 때마다 7,939건을 파이썬 리스트 dict 으로 메모리에 적재
    #         (수백 MB). 정작 서빙 단계는 이 캐시를 한 번도 안 쓴다 — 검색은 빌드
    #         단계가 만든 .npy 행렬만 보기 때문이다.
    #   지금: `embed_passages()`(빌드 단계)가 처음 불릴 때만 올린다.
    #         질의 임베딩은 프로세스 안 메모리 캐시만 쓰고 디스크를 건드리지 않는다.

    # ---------------------------------------------------------------- 모델
    @property
    def enabled(self) -> bool:
        """모델을 실제로 쓸 수 있는가. 실패했으면 두 번 다시 시도하지 않는다."""
        if self._load_failed or np is None:
            return False
        return self._ensure_model() is not None

    def _ensure_model(self):
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.warning(
                "sentence-transformers 가 없어 벡터검색을 끕니다 (BM25 단독으로 동작). "
                "설치: pip install sentence-transformers"
            )
            self._load_failed = True
            return None

        # ── 메모리 가드 ─────────────────────────────────────────────
        # 모자란 상태로 로드하면 예외가 아니라 세그폴트로 프로세스가 죽는다.
        # try/except 로 못 잡으므로 **로드 자체를 시도하지 않는다.**
        if config.EMBED_MEMORY_GUARD:
            free = _free_memory_mb()
            need = _required_mb(self.model_name) + _SAFETY_MB
            if free is not None and free < need:
                log.warning(
                    "여유 메모리 부족으로 임베딩을 끕니다 (BM25 단독으로 동작).\n"
                    "  모델=%s  필요≈%dMB  여유=%dMB\n"
                    "  해결책 ① 다른 프로그램을 닫고 다시 실행\n"
                    "        ② .env 에서 더 작은 모델로 교체 (차원도 함께 변경)\n"
                    "             PENSION_EMBED_MODEL=intfloat/multilingual-e5-base\n"
                    "             PENSION_EMBED_DIM=768\n"
                    "        ③ 가드를 끄고 강행: PENSION_EMBED_MEMORY_GUARD=false",
                    self.model_name, need, free,
                )
                self._load_failed = True
                return None

        try:
            kwargs = {}
            if config.EMBED_DEVICE:
                kwargs["device"] = config.EMBED_DEVICE
            log.info("임베딩 모델 로딩: %s", self.model_name)
            m = SentenceTransformer(self.model_name, **kwargs)
            m.max_seq_length = config.EMBED_MAX_SEQ
            self._model = m
            return m
        except Exception as e:
            log.warning("임베딩 모델 로드 실패(%s) → BM25 단독으로 진행합니다.", e)
            self._load_failed = True
            return None

    # ---------------------------------------------------------------- 캐시
    @property
    def _cache_file(self) -> Path:
        return self.cache_dir / f"emb_{config.EMBED_SLUG}.jsonl"

    def _load_cache(self) -> None:
        """디스크 캐시를 메모리로 올린다. **빌드 단계에서만** 호출된다."""
        if self._cache_loaded:
            return
        self._cache_loaded = True
        f = self._cache_file
        if not f.exists():
            return
        n = 0
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                self._mem[rec["h"]] = rec["v"]
                n += 1
            except Exception:
                continue
        log.info("임베딩 캐시 %d건 로드", n)

    def _append_cache(self, pairs: List[tuple[str, List[float]]]) -> None:
        if not pairs:
            return
        with self._cache_file.open("a", encoding="utf-8") as fp:
            for h, vec in pairs:
                fp.write(json.dumps({"h": h, "v": vec}) + "\n")

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------- 계산
    def _encode(self, prefixed: List[str], *,
                use_disk_cache: bool) -> List[List[float]]:
        """배치 인코딩한다. 입력은 **접두사가 이미 붙은** 문자열.

        `use_disk_cache` 가 두 단계를 가른다.
          True  (빌드) : 디스크 캐시를 읽고 새로 계산한 것을 덧붙인다.
                         청크 몇 개만 바뀌었을 때 그것만 다시 계산하기 위한 것.
          False (서빙) : 프로세스 안 메모리 캐시만 쓴다. 질의는 매번 다르고
                         재사용되지 않아서, 디스크에 쌓아 봐야 파일만 커진다.
        """
        # ⚠️ 모델 로드는 **캐시 미스가 있을 때만** 한다 (2026-08-29).
        #    예전에는 맨 앞에서 무조건 _ensure_model() 을 불렀다. 캐시가 100%
        #    맞아떨어지는 재빌드에서도 2.2GB 를 올리고 시작한 셈이다.
        if use_disk_cache:
            self._load_cache()

        hashes = [self._hash(t) for t in prefixed]
        todo_idx = [i for i, h in enumerate(hashes) if h not in self._mem]

        if todo_idx:
            model = self._ensure_model()
            if model is None:
                raise EmbeddingError(
                    f"임베딩 모델을 사용할 수 없고 캐시에도 없습니다 "
                    f"({len(todo_idx)}/{len(prefixed)}건 미스).")
            texts = [prefixed[i] for i in todo_idx]
            vecs = model.encode(
                texts,
                batch_size=config.EMBED_BATCH_SIZE,
                normalize_embeddings=True,     # e5 는 코사인 유사도를 전제로 쓴다
                show_progress_bar=len(texts) > 200,
                convert_to_numpy=True,
            )
            fresh: List[tuple[str, List[float]]] = []
            for i, v in zip(todo_idx, vecs):
                lst = [float(x) for x in v]
                self._mem[hashes[i]] = lst
                fresh.append((hashes[i], lst))
            if use_disk_cache:
                self._append_cache(fresh)

        return [self._mem[h] for h in hashes]

    # ---------------------------------------------------------------- 공개
    def embed_query(self, text: str) -> Optional[List[float]]:
        """[서빙] 질의 임베딩. `query: ` 접두사를 붙인다.

        서빙 단계에서 모델이 필요한 곳은 **여기 하나뿐이다.** 문서는 빌드 단계가
        이미 인코딩해 아티팩트로 저장해 뒀다.
        """
        if not self.enabled:
            return None
        text = (text or "").strip()
        if not text:
            return None
        try:
            return self._encode([config.EMBED_QUERY_PREFIX + text],
                                use_disk_cache=False)[0]
        except Exception as e:
            log.warning("질의 임베딩 실패(%s) → BM25 단독", e)
            return None

    def embed_passages(self, texts: List[str], *,
                       progress: bool = True) -> Optional["np.ndarray"]:
        """[빌드] 문서 임베딩. `passage: ` 접두사를 붙인다. (N, dim) 행렬."""
        if not self.enabled or np is None:
            return None
        prefixed = [config.EMBED_PASSAGE_PREFIX + (t or "").strip() for t in texts]
        try:
            if progress:
                print(f"  임베딩 {len(prefixed)}건 (모델={self.model_name})")
            return np.array(self._encode(prefixed, use_disk_cache=True),
                            dtype="float32")
        except Exception as e:
            log.warning("문서 임베딩 실패: %s", e)
            return None

    # --- 하위 호환 (기존 호출부가 embed_one / embed_many 를 쓴다) --------
    def embed_one(self, text: str) -> Optional[List[float]]:
        return self.embed_query(text)

    def embed_many(self, texts: List[str], *, progress: bool = True):
        return self.embed_passages(texts, progress=progress)

    def probe(self) -> dict:
        """모델이 살아 있는지 1건으로 확인한다."""
        try:
            v = self.embed_query("퇴직연금 세액공제 한도")
            if v is None:
                return {"ok": False, "model": self.model_name,
                        "error": "sentence-transformers 미설치 또는 모델 로드 실패"}
            return {"ok": True, "model": self.model_name, "dim": len(v),
                    "cache": str(self._cache_file)}
        except Exception as e:
            return {"ok": False, "model": self.model_name, "error": str(e)}


# --------------------------------------------------------------------------
_EMBEDDER: Optional[E5Embedder] = None


def get_embedder() -> E5Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = E5Embedder()
    return _EMBEDDER


# 예전 이름으로 import 하던 곳을 위한 별칭
ClovaEmbedder = E5Embedder
