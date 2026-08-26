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

■ 코퍼스를 지우는 필터가 **하나도 없다** (2026-08-16 재개편)

  ▸ 1세대: 8축 하드필터(source/계좌유형/금융주제/…)를 AND 로 걸었다.
      "연금저축 세액공제 받으며 담을 채권형 펀드 추천"
        → 금융주제=[세액공제] → fund_card 92장 중 0장 통과 → 조용한 오답
  ▸ 2세대: 축을 scope 하나로 줄였다(추천이면 all, 아니면 doc).
      나아졌지만 병이 같다 — **LLM 판정 1회의 실패 비용이 코퍼스 절반**이다.
        · "미래에셋장기성장포커스 총보수 얼마?" 는 추천이 아닌데 답은 투자설명서에만
          있다. 추천=false 로 찍히면 정답 후보가 0건이 된다.
        · "세액공제 한도 채우고 그 돈으로 살 채권형 펀드도" 는 두 코퍼스가 동시에
          필요한데, 어느 쪽으로 찍어도 절반이 사라진다.
        · 판정이 실행마다 흔들리면 성능 측정 자체가 무의미해진다.

  ▸ 3세대(지금): **자리를 나눠 준다.**

      product_affinity 0.0~1.0 → top-k 안에서 투자설명서가 가져갈 자리 비율
        · 규정문서 몫 = top_k - 투자설명서 몫
        · 양쪽 모두 후보가 있으면 최소 MIN_STRATUM_QUOTA 자리는 보장
        · 판정이 틀려도 **배분이 기울 뿐 정답이 사라지지 않는다**

      이렇게 하면 "코퍼스 비율(투자설명서 90%)에 눌려 규정문서가 안 보이는" 원래
      문제도 그대로 해결된다. 하드필터는 그 문제를 풀려고 도입한 것인데, 정답을
      지우지 않고도 같은 목적을 달성할 수 있다.

  남은 조건 둘은 성격이 다르다.
      사유      : 소프트 필터. 결과가 모자라면 **자동으로 풀린다**(완화 사다리 0단계).
      fund_name : 유일한 배제형. 1문서=1펀드라 100% 확정값이고, 펀드명이 없는
                  규정문서 청크는 배제하지 않는다.

■ 한 문서가 top-k 를 독식하지 못하게 한다
  섹션 원문이 7,156건(전체의 90%)이라, 상한이 없으면 추천 질의의 top-24 가 특정
  펀드 하나의 원문 조각으로 도배된다. 비교할 펀드가 후보에 없으면 추천은 실패고,
  다른 펀드 근거가 섞이면 근거 완전성이 깨진다.
"""

from __future__ import annotations

import logging
import re
import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import config
from utils import tokenize_ko

from .loader import is_table_chunk, load_corpus, searchable_text

log = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None                                          # type: ignore


def _affinity_of(filters: Dict[str, Any]) -> float:
    """필터에서 상품 관여도(0.0~1.0)를 읽는다.

    · `product_affinity` 가 있으면 그것을 쓴다 (현행)
    · 없고 옛 `scope` 만 있으면 근사값으로 변환한다 (하위 호환)
        scope="doc" → 0.15 처럼 **낮은 값**이지 0 이 아니다. 0 으로 두면 옛 하드필터와
        같아져서, 이 개편이 없애려던 실패(정답 코퍼스 통째 유실)가 되돌아온다.
    """
    if filters:
        a = filters.get("product_affinity")
        if isinstance(a, (int, float)):
            return max(0.0, min(1.0, float(a)))
        scope = filters.get("scope")
        if scope == "all":
            return config.PRODUCT_INTENT_LEVELS["상품"]
        if scope == "doc":
            return config.PRODUCT_INTENT_LEVELS["제도"]
    return config.DEFAULT_PRODUCT_INTENT


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
        """빌드 단계가 만든 행렬을 붙인다. **이미 L2 정규화된 상태**로 들어온다."""
        if np is None or mat is None:
            return
        if len(mat) != len(self.chunks):
            log.warning("벡터 개수(%d)와 청크 수(%d)가 달라 벡터검색을 끕니다.",
                        len(mat), len(self.chunks))
            return
        self.vectors = mat                               # 정규화 완료 → 내적 = 코사인
        log.info("벡터 인덱스 연결: %s", self.vectors.shape)

    # ------------------------------------------------------------- 필터
    def _passes(self, c: dict, f: Dict[str, Any]) -> bool:
        """배제형 조건. **tier(코퍼스)는 여기서 절대 거르지 않는다.**

        코퍼스 배분은 필터가 아니라 계층 쿼터(_select_stratified)가 담당한다.
        여기 남은 것은 "값이 확정된 축" 둘뿐이다.
        """
        if not f:
            return True

        # ① 사유 — 장르C(중도인출) 전용 소프트 필터
        #    5개 문서(doc46~50)는 도입부 문장이 동일해서 검색으로 구분되지 않는다.
        #    ⚠️ `사유` 값을 **가진 청크만** 대조한다. 값이 없는 청크(제도·세제·상품
        #    전부)를 배제하면 "주택 구입 중도인출 세금은?" 같은 질문에서 세제 문서가
        #    통째로 사라진다. 사유는 장르C 안에서 형제를 가르는 용도지, 코퍼스를
        #    좁히는 용도가 아니다.
        want = f.get("사유")
        if want:
            got = c.get("사유")
            if got and got not in want:
                return False

        # ② fund_name — 질문에 특정 펀드가 명시됐을 때만.
        #    선행모델의 company_name 과 같은 성격이다(1문서=1펀드, 100% 확정).
        #    ⚠️ 규정문서에는 펀드명이 없다. 그것까지 배제하면 "이 펀드 IRP로 살 수
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
    def _weights(c: dict, 질문유형: str, affinity: float) -> tuple[float, List[str]]:
        """검색 점수에 곱할 도메인 가중치. **후보를 지우지 않는다.**"""
        reasons: List[str] = []
        w = 1.0

        # 출처 신뢰도 (규정문서 > 투자설명서)
        trust = config.TIER_TRUST.get(c["tier"], 1.0)
        if trust != 1.0:
            w *= trust
            reasons.append(f"출처가중 x{trust}")

        # distractor 감점 (제도 파일의 '첨부-관련법령')
        if c.get("distractor"):
            w *= config.DISTRACTOR_PENALTY
            reasons.append(f"distractor x{config.DISTRACTOR_PENALTY}")

        # ⚠️ confidence 는 곱하지 않는다 — 상수(1.0/0.95)라 품질 신호가 아니다.
        #    켜면 "OCR 로 읽은 문서 일괄 5% 감점" 이 된다. config 주석 참고.
        if config.USE_CONFIDENCE_WEIGHT:
            conf = c.get("confidence", 1.0)
            if conf < 1.0:
                w *= conf
                reasons.append(f"파싱신뢰도 x{conf}")

        # 표·계산박스 가산 — 계산·사실조회에서는 세율표·한도표가 정답인 경우가 많다.
        # origin(genre_b/d)과 섹션명 둘 중 하나만 맞아도 표로 본다.
        if 질문유형 in ("계산", "사실조회") and is_table_chunk(c):
            w *= config.TABLE_BOOST_ON_CALC
            reasons.append(f"표 x{config.TABLE_BOOST_ON_CALC}")

        # 카드 가산 — 상품 질문일수록 세게. 섹션 원문 7,156 : 카드 183 이라
        # 가중이 없으면 "이 펀드 총보수" 의 정본인 카드가 구조적으로 밀린다.
        if c.get("chunk_type") in ("fund_card", "class_card") and config.CARD_BOOST:
            m = 1.0 + config.CARD_BOOST * affinity
            w *= m
            reasons.append(f"카드 x{m:.2f}")

        return w, reasons

    # ------------------------------------------------------------- 계층 쿼터
    @staticmethod
    def _quotas(top_k: int, affinity: float,
                n_doc: int, n_pros: int) -> tuple[int, int]:
        """top-k 자리를 규정문서 / 투자설명서로 나눈다.

        어느 쪽도 0 이 되지 않게 하는 것이 핵심이다 — 0 은 곧 하드필터고,
        하드필터의 실패 비용은 코퍼스 절반이다.
        """
        if n_doc == 0:
            return 0, top_k
        if n_pros == 0:
            return top_k, 0

        q_pros = int(round(top_k * affinity))
        q_doc = top_k - q_pros

        # 양쪽 모두 후보가 있으면 최소 자리를 보장한다.
        floor = min(config.MIN_STRATUM_QUOTA, top_k // 2)
        if q_pros < floor:
            q_pros, q_doc = floor, top_k - floor
        elif q_doc < floor:
            q_doc, q_pros = floor, top_k - floor

        # 후보가 쿼터보다 적으면 남는 자리는 반대편에 넘긴다(자리를 버리지 않는다).
        if q_doc > n_doc:
            q_pros += q_doc - n_doc
            q_doc = n_doc
        if q_pros > n_pros:
            q_doc += q_pros - n_pros
            q_pros = n_pros
        return q_doc, min(q_pros, n_pros)

    @staticmethod
    def _take_prospectus(pool: List[dict], limit: int, per_doc: int) -> List[dict]:
        """투자설명서 몫을 **카드와 섹션 원문으로 나눠서** 뽑는다 (2026-08-18 신설).

        ■ 왜 필요한가 — 계층 쿼터가 한 단계 모자랐다

          MIN_STRATUM_QUOTA 는 doc(규정문서) vs prospectus(투자설명서) **사이**만
          조정한다. 투자설명서 안에서 카드 183건과 섹션 원문 7,156건이 자리를
          다투는 것은 아무도 조정하지 않았다.

          그 결과 카드가 항상 이겼다. 카드는 짧고 펀드명·수치가 빽빽해서
          BM25 와 코사인 양쪽에서 유리하고, 거기에 CARD_BOOST 까지 곱해진다.

          실측(78문항, 검색 결과 1,920건):
              카드   46.1%   (코퍼스 비중  2.3%)
              섹션    8.6%   (코퍼스 비중 90.1%)

          카드에 없고 섹션에만 있는 항목 — 환매수수료, 기준가격 산정, 관련회사,
          자산보관회사 — 은 구조적으로 도달 불가가 됐다(P-F4·P-G0a·P-G0b).

        ■ CARD_BOOST 를 내리는 것만으로는 안 된다

          배수는 최대 x1.43 으로 그리 크지 않았다. 0.1 로 낮춰도 카드가 여전히
          자리를 다 가져간다 — 점수 자체로 이기기 때문이다.
          **자리를 따로 떼어 주는 것**이 유일하게 확실한 방법이다.

        ■ 카드 자리를 0 으로 만들지 않는다

          섹션 몫 = min(MIN_SECTION_QUOTA, limit // 2) 로 상한을 건다.
          "이 펀드 총보수 얼마?" 처럼 **카드가 정본인** 질문을 망치면 안 되므로,
          섹션이 절반을 넘게 가져가는 일은 없다.
              limit 6 -> 섹션 2 + 카드 4
              limit 3 -> 섹션 1 + 카드 2
              limit 1 -> 섹션 0 + 카드 1  (자리가 하나뿐이면 점수대로)

          남는 자리는 서로 넘긴다 — 한쪽 후보가 부족하다고 자리를 버리지 않는다.
        """
        if limit <= 0 or not pool:
            return []

        cards = [h for h in pool
                 if h["chunk"].get("chunk_type") in ("fund_card", "class_card")]
        sections = [h for h in pool
                    if h["chunk"].get("chunk_type") not in ("fund_card", "class_card")]

        # 한쪽이 비면 쿼터를 걸 이유가 없다.
        if not cards or not sections:
            return PensionIndex._take(pool, limit, per_doc)

        q_sec = min(config.MIN_SECTION_QUOTA, limit // 2)
        if q_sec <= 0:
            return PensionIndex._take(pool, limit, per_doc)

        picked = PensionIndex._take(sections, q_sec, per_doc)
        rest = limit - len(picked)                # 섹션 후보가 모자라면 카드가 더 가져간다
        chosen = {id(h) for h in picked}
        remainder = [h for h in pool if id(h) not in chosen]
        picked += PensionIndex._take(remainder, rest, per_doc)
        return picked

    @staticmethod
    def _take(pool: List[dict], limit: int, per_doc: int) -> List[dict]:
        """점수 순으로 뽑되 한 문서(doc_id)가 per_doc 개를 넘지 않게 한다."""
        out: List[dict] = []
        used: Dict[str, int] = {}
        overflow: List[dict] = []
        for h in pool:
            if len(out) >= limit:
                break
            d = h["chunk"]["doc_id"]
            if used.get(d, 0) >= per_doc:
                overflow.append(h)
                continue
            used[d] = used.get(d, 0) + 1
            out.append(h)
        # 상한 때문에 자리가 남으면 밀어냈던 것으로 채운다 (자리를 비우지 않는다)
        for h in overflow:
            if len(out) >= limit:
                break
            out.append(h)
        return out

    # ------------------------------------------------------------- 검색
    def search(self, query: str, *, top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None,
               query_vector: Optional[List[float]] = None,
               질문유형: str = "사실조회") -> List[dict]:
        """하이브리드 검색 + 계층 쿼터. 반환: [{chunk, score, bm25, vec, reason}, ...]"""
        filters = filters or {}
        affinity = _affinity_of(filters)
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
            w, reasons = self._weights(c, 질문유형, affinity)
            scored.append({
                "chunk": c,
                "score": base * w,
                "bm25": b,
                "vec": v,
                "reason": ", ".join(reasons),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        # --- 4) 계층 쿼터로 자리를 나눈다 (하드필터의 대체물) ---
        doc_pool = [h for h in scored if h["chunk"]["tier"] == "doc"]
        pros_pool = [h for h in scored if h["chunk"]["tier"] != "doc"]
        q_doc, q_pros = self._quotas(top_k, affinity, len(doc_pool), len(pros_pool))

        # 펀드를 지목한 질문이면 그 문서 하나를 깊게 파야 하므로 문서당 상한을 푼다.
        per_doc = (config.MAX_CHUNKS_PER_DOC_FOCUSED if filters.get("fund_name")
                   else config.MAX_CHUNKS_PER_DOC)

        # 투자설명서 몫은 카드/섹션 쿼터를 한 번 더 적용해서 뽑는다.
        # (규정문서 쪽은 chunk_type 이 qa/tax/procedure/withdrawal 로 이미 고르게
        #  섞여 들어와서 추가 쿼터가 필요 없다 — 실측 분포로 확인했다)
        picked = self._take(doc_pool, q_doc, per_doc) + \
            self._take_prospectus(pros_pool, q_pros, per_doc)
        picked.sort(key=lambda x: x["score"], reverse=True)
        return picked[:top_k]


# ==========================================================================
# 신호 — "코퍼스가 이 질문에 대해 할 말이 있는가" (LLM 호출 0회)
# ==========================================================================
def bm25_top1(question: str, *,
              tiers: Optional[Sequence[str]] = None,
              chunk_types: Optional[Sequence[str]] = None) -> float:
    """지정한 부분 코퍼스에 대한 raw BM25 top-1 점수.

    ■ 왜 부분 코퍼스로 재나
      전체를 대상으로 하면 변별력이 없다. 투자설명서 7,904건은 어떤 질문에도
      뭔가 걸리기 때문이다. **무엇을 묻는 질문인지** 를 판정하려면 그 판정에
      해당하는 청크만 봐야 한다.

        범위 판정   tiers=("doc",)              → 규정문서에 할 말이 있나
        되묻기 판정 chunk_types=("withdrawal",) → 중도인출 축 질문이 맞나

    예외는 0 이 아니라 inf 를 돌려준다 — 인덱스 문제로 질문을 막으면 안 된다.
    """
    try:
        idx = get_index()
        cand = [
            i for i, c in enumerate(idx.chunks)
            if (tiers is None or c.get("tier") in tiers)
            and (chunk_types is None or c.get("chunk_type") in chunk_types)
        ]
        if not cand:
            return 0.0
        scores = idx.bm25.scores(tokenize_ko(question), cand)
        return max(scores.values()) if scores else 0.0
    except Exception as e:
        log.warning("BM25 신호 조회 실패(통과 처리): %s", e)
        return float("inf")


# ── 상품 질문 신호 ──────────────────────────────────────────────────────
#
# ■ 무엇을 판정하나
#   "이 질문이 개별 상품(펀드)에 관한 것인가" 를 LLM 없이 정한다.
#   펀드명 목록 대조는 **이름이 나온 질문만** 잡는데, 상품 질문의 상당수는
#   이름 없이 개념만 묻는다("채권형으로 하나 골라주세요, 보수 싼 걸로").
#
# ■ 어휘 목록을 쓰지 않는다 — 한 번 그렇게 만들었다가 걷어냈다
#
#   처음에는 상품 어휘 목록(총보수·위험등급·클래스·채권형…)을 손으로 만들었다.
#   그런데 그 목록은 **실패한 질의 문항을 보고 거기 나온 단어를 고른 것**이었다.
#   즉 평가 세트에 맞춰 만든 값이라, 대회 본 질문에는 통한다는 보장이 없다.
#
#   실제로 코퍼스로 검증해 보니 근거가 없었다. 투자설명서/규정문서 출현빈도
#   lift 를 재 보면:
#       위험등급·선취판매수수료·혼합형·초단기채·TDF·ETF·추천 → 빈도 미달(측정 불가)
#       채권형 5.1 · 주식형 2.9 · 펀드 1.7 · 상품 1.5        → 변별력 거의 없음
#   손으로 고른 20개 중 절반이 코퍼스에서 뽑히지 않는 단어였다.
#
# ■ 대신 **두 코퍼스에 직접 물어본다**
#
#       비율 = BM25(질문 | 투자설명서) / BM25(질문 | 규정문서)
#
#   비율 1.0 은 "두 코퍼스가 동점" 이라는 **자명한 기준**이다.
#   튜닝한 숫자가 아니라 정의상 중립점이라, 문항을 보고 정한 값이 아니다.
#   코퍼스가 바뀌면 자동으로 따라간다는 것도 어휘 목록에 없는 성질이다.
#
#   실측(78문항, BM25 단독):
#       제도 트랙 37건 → 상품 판정 **0건**   (오탐 없음)
#       상품 트랙 41건 → 상품 판정 26건      (놓침 15건)
#
#   한쪽으로만 정확하다. 제도 질문을 상품으로 오판하지 않으므로 자리 배분을
#   올리는 용도로 안전하고, 놓친 15건은 기존 기본값(0.35)으로 떨어질 뿐
#   더 나빠지지 않는다. **recall 을 올리려고 어휘를 덧붙이면 다시 과적합이 된다.**
def product_signal(question: str) -> tuple[bool, float]:
    """이 질문이 개별 상품을 묻는가 — (판정, 투자설명서/규정문서 BM25 비율)."""
    try:
        sp = bm25_top1(question, tiers=("prospectus",))
        sd = bm25_top1(question, tiers=("doc",))
        if sd <= 0:
            return (sp > 0), float("inf") if sp > 0 else 0.0
        ratio = sp / sd
        return ratio >= config.PRODUCT_SIGNAL_RATIO, ratio
    except Exception:
        return False, 0.0


# ==========================================================================
# 싱글턴
# ==========================================================================
_INDEX: Optional[PensionIndex] = None


def get_index() -> PensionIndex:
    """[서빙 단계] 인덱스를 만들고 **미리 만들어 둔 벡터 아티팩트**를 붙인다.

    여기서 문서를 임베딩하지 않는다. 무거운 일은 빌드 단계가 이미 끝냈다
    (`python -m index.build_index`). 이 함수가 하는 일은 .npy 를 읽어 붙이는 것뿐이라
    임베딩 모델을 로드하지 않는다 — 모델은 **질의를 인코딩할 때** 처음 로드된다.
    """
    global _INDEX
    if _INDEX is None:
        _INDEX = PensionIndex(load_corpus())
        _attach_artifact(_INDEX)
    return _INDEX


def _attach_artifact(index: PensionIndex) -> None:
    """빌드 단계 산출물을 붙인다.

    ⚠️ **낡은 아티팩트는 예외로 죽인다.**

    예전에는 벡터가 청크와 안 맞으면 경고 한 줄 남기고 조용히 리턴했다. 그러면
    self.vectors 가 None 으로 남아 **의미 검색(가중치 0.65)이 통째로 꺼진 채**
    BM25 단독으로 계속 돈다. 증상이 로그 한 줄뿐이라 그대로 데모를 돌리게 된다.

    구분이 중요하다.
      · 아티팩트가 **없다**  → 선택일 수 있다. PENSION_REQUIRE_VECTORS 가 정한다.
      · 아티팩트가 **낡았다** → 버그다. 재생성 전까지 진행시키지 않는다.
    """
    if np is None:
        return

    from . import artifact

    if not artifact.exists():
        msg = ("벡터 아티팩트가 없습니다 → 의미 검색이 꺼진 채 BM25 단독으로 "
               "동작합니다.\n  → `python -m index.build_index` 로 만드세요.")
        if config.REQUIRE_VECTORS:
            raise RuntimeError(
                msg + "\n  (BM25 단독으로 일부러 돌리려면 "
                      "PENSION_REQUIRE_VECTORS=false)")
        log.warning(msg)
        return

    # 지문 대조 — 청크 증감뿐 아니라 **본문 수정**까지 잡는다.
    fp = artifact.corpus_fingerprint(
        [c["chunk_id"] for c in index.chunks],
        [searchable_text(c) for c in index.chunks],
    )
    art = artifact.load(expected_fingerprint=fp)         # 안 맞으면 ArtifactError

    # 지문이 같아도 행 순서까지 같다는 보장은 없으므로 chunk_id 로 정렬해 붙인다.
    pos = {cid: i for i, cid in enumerate(art.ids)}
    order = [pos[c["chunk_id"]] for c in index.chunks]
    index.attach_vectors(art.vectors[order])
    log.info("벡터 아티팩트 연결: %d건 (빌드 %s)",
             len(art), art.meta.get("built_at", "?"))


def reset_index() -> None:
    global _INDEX
    _INDEX = None
    load_corpus.cache_clear()
