"""검색 노드 — 서브질문 하나에 대해 근거 청크를 가져온다.

■ 선행모델 대응
  FinSight 의 `nodes/document_retriever.py`(454줄) 자리다. 가져온 발상 세 가지:

  1) **필터 점진 완화 (metadata_fallback)**
     처음엔 좁게 검색하고, 부족하면 조건을 푼다. 선행모델은 필터를 순서 있는
     리스트로 두고 재시도마다 `metadata_filters.pop()` 했고, 한도를 넘으면
     `metadata_filters = []` 로 전면 해제했다. 마지막에는 무필터 재검색까지 한다.

         if len(docs) == 0:
             docs += retriever_helper(retriever, question, k, "")

     우리는 축이 scope 하나뿐이라 완화도 한 단계다.  doc → all → 무필터.

  2) **질문유형별 검색 전략 분기**
       · 추천  → top-k 24 로 넓게 (요구사항)
       · 계산·사실조회 → 표 청크 가산 (store.py 에서 처리)
       · 그 외 → 기본 top-k 5

  3) **다중 질의 검색 (HyDE 포함)**
     선행모델은 재작성 질의와 HyDE 문서를 `xxxxxxxxxx` 로 이어붙여 넘기고 여기서
     split 했다. 우리는 State 필드(`hyde_doc`)로 명시적으로 받아 두 갈래로 검색한다.
     결과를 chunk_id 기준으로 합치고(중복 제거) 점수 순으로 자른다.

     ⚠️ HyDE 문서는 **검색 질의로만** 쓴다. 답변 근거로는 절대 넘기지 않는다
       (LLM 이 지어낸 글이므로 근거로 쓰면 그게 환각이다).

■ 완화 조건을 '0건'에서 '임계 미달'로 바꿨다 (2026-08)
  예전 루프는 `if hits: break` 였다. 1건만 나와도 멈춘다는 뜻인데, 이게 실제로
  가장 나쁜 실패를 만들었다.

      "연금저축 세액공제 받으며 담을 채권형 펀드 추천"
        → 옛 필터로 후보 1건(ISA 전환 FAQ) → 완화 루프 미발동
        → 그 1건을 근거로 펀드를 추천하는 **조용한 오답**

  0건이 아니라서 폴백이 안 도는 상황을 막으려면 임계값이 필요하다.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import config
from index import get_index, to_citation
from llm import embedder
from state import RAGState
from utils import Timer, log_message, step, truncate

# 완화 순서 — scope 를 넓히는 것이 전부다.
#   level 0 : 서브질문이 정한 scope 그대로
#   level 1 : scope="all" (투자설명서까지 개방)
#   level 2 : fund_name 까지 해제
#   level -1: 필터 전면 해제 (선행모델의 무필터 재검색에 대응)
RELAX_STEPS = ["scope", "fund_name"]


def _relax(filters: Dict[str, Any], level: int) -> Dict[str, Any]:
    """level 단계만큼 조건을 푼다. level=0 이면 원본 그대로."""
    f = deepcopy(filters or {})
    if level >= 1:
        f["scope"] = "all"
    if level >= 2:
        f["fund_name"] = []
    return f


def _top_k_for(질문유형: str, filters: Dict[str, Any]) -> int:
    """질문유형에 따른 검색 폭 결정.

    추천은 "정답 1개를 찾는" 문제가 아니라 "후보군을 모아 조건으로 거르는" 문제다.
    top-5 만 보면 위험등급·자산유형 조건에 맞는 상품이 아예 후보에 없을 수 있다.
    실측: 골드셋 G011 이 top-5 에서 recall 0.00 → top-24 에서 1.00.
    """
    if 질문유형 == "추천" or (filters or {}).get("scope") == "all":
        return config.NUM_DOCS_RECOMMEND
    return config.NUM_DOCS_TO_RETRIEVE


def _embed(text: str) -> Optional[List[float]]:
    """임베딩 실패는 치명적이지 않다 — BM25 로 계속 간다."""
    try:
        return embedder().embed_query(text)
    except Exception as e:
        log_message(f"임베딩 실패 → BM25 단독으로 진행: {e}", "warning")
        return None


def _merge_hits(*hit_lists: List[dict]) -> List[dict]:
    """여러 질의의 검색 결과를 chunk_id 기준으로 합친다 (같은 청크는 최고점 유지)."""
    best: Dict[str, dict] = {}
    for hits in hit_lists:
        for h in hits:
            cid = h["chunk"]["chunk_id"]
            if cid not in best or h["score"] > best[cid]["score"]:
                best[cid] = h
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def retrieve_documents(state: RAGState) -> dict:
    """필터를 걸어 검색하고, 결과가 부족하면 scope 를 넓혀 재시도한다."""
    question = state["question"]
    질문유형 = state.get("질문유형", "사실조회")
    base_filters = state.get("filters", {}) or {}
    hyde_doc = (state.get("hyde_doc") or "").strip()
    qid = state.get("qid", "")

    index = get_index()
    top_k = _top_k_for(질문유형, base_filters)

    # 질의별 임베딩 (HyDE 가 있으면 그것도)
    qvec = _embed(question)
    hvec = _embed(hyde_doc) if hyde_doc else None

    def _search(f: Dict[str, Any]) -> List[dict]:
        hits = index.search(question, top_k=top_k, filters=f,
                            query_vector=qvec, 질문유형=질문유형)
        if hyde_doc:
            hyde_hits = index.search(hyde_doc, top_k=top_k, filters=f,
                                     query_vector=hvec, 질문유형=질문유형)
            hits = _merge_hits(hits, hyde_hits)[:top_k]
        return hits

    # 완화를 발동시키는 임계값. "0건이 아니면 통과"로 두면 1건짜리 오답을 그냥 쓴다.
    need = min(config.DOCS_RELEVANCE_THRESHOLD + 1, top_k)

    with Timer() as t:
        hits: List[dict] = []
        used_level = 0
        max_level = len(RELAX_STEPS) if config.WORKFLOW_SETTINGS["metadata_filtering"] else 0

        for level in range(max_level + 1):
            found = _search(_relax(base_filters, level))
            if len(found) > len(hits):
                hits, used_level = found, level
            if len(hits) >= need:
                break

        # 그래도 부족하면 필터 전면 해제 (선행모델의 무필터 재검색)
        if len(hits) < config.DOCS_RELEVANCE_THRESHOLD:
            found = _search({})
            if len(found) > len(hits):
                hits, used_level = found, -1

    applied = _relax(base_filters, used_level) if used_level >= 0 else {}
    docs = [h["chunk"] for h in hits]

    relax_note = (
        "필터 없음(전체검색)" if used_level < 0
        else ("완화 없음" if used_level == 0 else f"{used_level}단계 완화")
    )
    dist: Dict[str, int] = {}
    for d in docs:
        key = f"{d['tier']}/{d['chunk_type']}"
        dist[key] = dist.get(key, 0) + 1

    summary = (f"{len(docs)}건 (top_k={top_k}, scope={applied.get('scope', '-')}, "
               f"{relax_note}) 분포={dist}")
    if hyde_doc:
        summary += " | HyDE 병합검색"

    return {
        "documents": docs,
        "applied_filters": applied,
        "filter_retries": state.get("filter_retries", 0) + (1 if used_level != 0 else 0),
        "trace": step("retrieve", qid=qid, summary=summary,
                      detail={"scores": [round(h["score"], 3) for h in hits[:5]],
                              "filters": applied,
                              "hyde": truncate(hyde_doc, 120) if hyde_doc else None},
                      ms=t.ms),
    }


def documents_to_citations(docs: List[dict]) -> List[dict]:
    """검색 결과를 인용 레코드로."""
    return [to_citation(d) for d in docs]
