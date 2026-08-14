"""검색 엔진 — 하이브리드(BM25 + 벡터) + scope 필터 + 출처 신뢰도 가중.

선행모델의 `vector_store.py`(Pathway 서버) + `retriever.py`(클라이언트) 를 합친 자리다.

■ 왜 Pathway 서버를 쓰지 않는가
  선행모델은 "PDF 폴더를 감시하다가 새 파일이 들어오면 자동 재색인"이 필요해서
  Pathway 서버 + 이중화(fast/slow) 구조를 썼다. 우리는
    · 문서가 고정되어 있고(파싱 완료),
    · 지금 단계가 서버 배포가 아니라 **성능 측정**이며,
    · 요구사항에서 이중화·시맨틱캐시를 제외했다.
  → 인메모리 인덱스가 정답이다. 3,853 × 1024차원 = 15MB 남짓, 검색 1회 수십 ms.

■ 하이브리드인 이유 (선행모델도 같은 결론)
  · BM25(어휘) : "환산급여공제", "C-P2e" 같은 **정확한 도메인 용어·코드**에 강하다.
  · 벡터(의미) : "돈을 미리 빼 쓸 수 있나요" ↔ "중도인출" 처럼 **말이 다른 경우**에 강하다.
  가중치는 0.35 : 0.65 (BM25 : 벡터). multilingual-e5-large 가 검색 목적으로 학습된
  모델이라 의미 매칭 쪽에 더 실었다.

■ 필터는 scope 하나다 (2026-08 개편)
  예전 `_passes()` 는 source/계좌유형/금융주제/금융목적/asset_type/risk_grade/
  연금저축_가능/퇴직연금_가능 8축을 전부 AND 로 걸었다. 실측 결과 이 구조가
  정답을 지우고 있었다.

      "연금저축 세액공제 받으며 담을 채권형 펀드 추천"
        → 금융주제=[세액공제] → fund_card 92장 중 0장 통과
        → 전체 1건만 남는데, 1건이라도 나오면 완화 루프가 안 돈다(if hits: break)
        → ISA 전환 FAQ 1장을 근거로 펀드를 추천하는 **조용한 오답**

  선행모델이 topics 를 필터에서 뺀 것과 같은 이유로, 우리도 축을 하나로 줄였다.

      scope="doc"  → tier=="doc" 인 청크만        (사실조회·절차·계산)
      scope="all"  → 전부                          (추천)
      fund_name    → 질문에 펀드가 명시됐을 때만 (선택)

  나머지는 전부 **점수 가중**으로 내렸다. 가중은 후보를 지우지 않으므로
  잘못 걸려도 순위가 밀릴 뿐 정답이 사라지지 않는다.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import config
from utils import tokenize_ko

from .loader import load_corpus, searchable_text

log = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None                                          # type: ignore


# ==========================================================================
# BM25 (의존성 없이 직접 구현 — 이 규모에선 이게 가장 단순하고 빠르다)
# ==========================================================================
class BM25:
    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf: List[Counter] = [Counter(d) for d in docs_tokens]
        df: Counter = Counter()
        for d in docs_tokens:
            df.update(set(d))
        self.idf: Dict[str, float] = {
            t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()
        }
        # 역색인: 후보 문서만 훑기 위해
        self.inverted: Dict[str, List[int]] = defaultdict(list)
        for i, d in enumerate(docs_tokens):
            for t in set(d):
                self.inverted[t].append(i)

    def scores(self, query_tokens: Sequence[str],
               candidates: Optional[Sequence[int]] = None) -> Dict[int, float]:
        """후보 문서 index → 점수. candidates 를 주면 그 안에서만 계산한다."""
        cand = set(candidates) if candidates is not None else None
        out: Dict[int, float] = defaultdict(float)
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.inverted.get(t, ()):
                if cand is not None and i not in cand:
                    continue
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                out[i] += idf * (f * (self.k1 + 1)) / (denom or 1)
        return out


# ==========================================================================
# 인덱스
# ==========================================================================
class PensionIndex:
    """청크 리스트 위에 BM25 + (선택)벡터 인덱스를 얹은 것."""

    def __init__(self, chunks: List[dict]):
        self.chunks = chunks
        self._search_text = [searchable_text(c) for c in chunks]
        self.bm25 = BM25([tokenize_ko(t) for t in self._search_text])
        self.vectors = None                    # (N, dim) — build_index 로 채운다
        log.info("인덱스 구축: %d 청크", len(chunks))

    # 하위 호환 (build_index 등이 부른다)
    @staticmethod
    def _searchable(c: dict) -> str:
        return searchable_text(c)

    # ------------------------------------------------------------- 벡터
    def attach_vectors(self, mat) -> None:
        if np is None or mat is None:
            return
        if len(mat) != len(self.chunks):
            log.warning("벡터 개수(%d)와 청크 수(%d)가 달라 벡터검색을 끕니다.",
                        len(mat), len(self.chunks))
            return
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = mat / norms                       # 미리 정규화 → 내적 = 코사인
        log.info("벡터 인덱스 연결: %s", self.vectors.shape)

    # ------------------------------------------------------------- 필터
    def _passes(self, c: dict, f: Dict[str, Any]) -> bool:
        """scope 필터. 축이 하나뿐이라 여기서 정답이 사라질 여지가 거의 없다."""
        if not f:
            return True

        # ① scope — 유일한 필수 축
        scope = f.get("scope")
        if scope == "doc" and c["tier"] != "doc":
            return False

        # ② fund_name — 질문에 특정 펀드가 명시됐을 때만.
        #    선행모델의 company_name 과 같은 성격이다(1문서=1펀드, 100% 확정).
        #    ⚠️ doc 문서에는 펀드명이 없다. 그것까지 배제하면 "이 펀드 IRP로 살 수
        #    있어?" 같은 질문에서 제도 문서가 전멸하므로, **펀드명을 가진 청크만**
        #    불일치로 배제한다.
        names = f.get("fund_name")
        if names:
            fn = (c.get("fund") or {}).get("name")
            if fn and not any(n in fn or fn in n for n in names):
                return False

        return True

    def candidates(self, filters: Dict[str, Any]) -> List[int]:
        return [i for i, c in enumerate(self.chunks) if self._passes(c, filters)]

    # ------------------------------------------------------------- 가중
    @staticmethod
    def _weights(c: dict, 질문유형: str) -> tuple[float, List[str]]:
        """검색 점수에 곱할 도메인 가중치. **후보를 지우지 않는다.**"""
        reasons: List[str] = []
        w = 1.0

        # 출처 신뢰도 (doc 문서 > 투자설명서)
        trust = config.TIER_TRUST.get(c["tier"], 1.0)
        if trust != 1.0:
            w *= trust
            reasons.append(f"출처가중 x{trust}")

        # distractor 감점
        if c.get("distractor"):
            w *= config.DISTRACTOR_PENALTY
            reasons.append(f"distractor x{config.DISTRACTOR_PENALTY}")

        # 파싱 신뢰도 (class_cards 의 좌표 파싱 등)
        if config.USE_CONFIDENCE_WEIGHT:
            conf = c.get("confidence", 1.0)
            if conf < 1.0:
                w *= conf
                reasons.append(f"파싱신뢰도 x{conf}")

        # 표 청크 가산 — 계산·사실조회에서는 세율표·한도표가 정답인 경우가 많다
        if 질문유형 in ("계산", "사실조회"):
            sec = str(c.get("section") or "")
            if sec.startswith(config.TABLE_SECTION_PREFIXES):
                w *= config.TABLE_BOOST_ON_CALC
                reasons.append(f"표 x{config.TABLE_BOOST_ON_CALC}")

        return w, reasons

    # ------------------------------------------------------------- 검색
    def search(self, query: str, *, top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None,
               query_vector: Optional[List[float]] = None,
               질문유형: str = "사실조회") -> List[dict]:
        """하이브리드 검색. 반환: [{chunk, score, bm25, vec, reason}, ...]"""
        filters = filters or {}
        cand = self.candidates(filters)
        if not cand:
            return []

        # --- 1) BM25 ---
        bm = self.bm25.scores(tokenize_ko(query), cand)
        bm_max = max(bm.values()) if bm else 0.0

        # --- 2) 벡터 ---
        vec: Dict[int, float] = {}
        if self.vectors is not None and query_vector is not None and np is not None:
            q = np.array(query_vector, dtype="float32")
            n = float(np.linalg.norm(q)) or 1.0
            sims = self.vectors[cand] @ (q / n)
            vec = {idx: float(s) for idx, s in zip(cand, sims)}
        vec_max = max(vec.values()) if vec else 0.0

        # --- 3) 결합 + 도메인 가중 ---
        use_vec = bool(vec)
        w_bm = config.BM25_WEIGHT if use_vec else 1.0
        w_vec = config.VEC_WEIGHT if use_vec else 0.0

        scored: List[dict] = []
        for i in cand:
            b = (bm.get(i, 0.0) / bm_max) if bm_max else 0.0
            v = (max(vec.get(i, 0.0), 0.0) / vec_max) if vec_max else 0.0
            base = w_bm * b + w_vec * v
            if base <= 0:
                continue

            c = self.chunks[i]
            w, reasons = self._weights(c, 질문유형)
            scored.append({
                "chunk": c,
                "score": base * w,
                "bm25": b,
                "vec": v,
                "reason": ", ".join(reasons),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ==========================================================================
# 싱글턴
# ==========================================================================
_INDEX: Optional[PensionIndex] = None


def get_index() -> PensionIndex:
    """인덱스를 만들고 캐시된 임베딩이 있으면 붙인다."""
    global _INDEX
    if _INDEX is None:
        _INDEX = PensionIndex(load_corpus())
        _attach_cached_vectors(_INDEX)
    return _INDEX


def _attach_cached_vectors(index: PensionIndex) -> None:
    """`python -m index.build_index` 가 만들어 둔 .npy 를 붙인다."""
    if np is None:
        return
    path = config.CACHE_DIR / f"vectors_{config.EMBED_SLUG}.npy"
    ids_path = config.CACHE_DIR / f"vectors_{config.EMBED_SLUG}.ids.txt"
    if not (path.exists() and ids_path.exists()):
        log.info("벡터 캐시 없음 → BM25 단독 검색 "
                 "(`python -m index.build_index` 로 생성)")
        return
    try:
        mat = np.load(path)
        ids = ids_path.read_text(encoding="utf-8").splitlines()
        # 청크 순서가 바뀌었을 수 있으므로 chunk_id 로 재정렬
        pos = {cid: i for i, cid in enumerate(ids)}
        order = [pos.get(c["chunk_id"], -1) for c in index.chunks]
        if any(o < 0 for o in order):
            log.warning("벡터 캐시와 청크가 불일치 → 재생성이 필요합니다 "
                        "(`python -m index.build_index`).")
            return
        index.attach_vectors(mat[order])
    except Exception as e:                              # 캐시 손상은 치명적이지 않다
        log.warning("벡터 캐시 로드 실패(%s) → BM25 단독", e)


def reset_index() -> None:
    global _INDEX
    _INDEX = None
    load_corpus.cache_clear()
