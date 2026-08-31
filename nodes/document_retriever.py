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

# 완화 순서 — **소프트 필터를 순서대로 푼다.**
#
#   level 0 : 서브질문이 정한 조건 그대로 (사유 + 펀드명)
#   level 1 : 사유 해제      ← 사유 추정이 빗나갔거나 장르C 밖에 답이 있는 경우
#   level 2 : 펀드명까지 해제
#   level -1: 전면 해제 (선행모델의 무필터 재검색에 대응)
#
# ★ 코퍼스(tier)는 완화 대상이 아니다. 애초에 좁힌 적이 없기 때문이다 —
#   규정문서/투자설명서 배분은 계층 쿼터가 top-k 안에서 처리한다(store.py).
#   그래서 "추천 판정이 틀려서 정답 코퍼스를 통째로 못 봤다" 는 실패 자체가 없다.
#
# 사유를 **하드 필터가 아니라 완화 사다리 0단계**에 둔 것이 핵심이다.
# 예전 8축 태그 필터가 망한 이유는 필터 자체가 아니라 **완화가 안 돌아서**였다
# (`if hits: break` — 1건만 나와도 멈췄다). 사유는 결과가 모자라면 자동으로 풀린다.
RELAX_STEPS = ["사유", "fund_name"]


def _relax(filters: Dict[str, Any], level: int) -> Dict[str, Any]:
    """level 단계만큼 조건을 푼다. level=0 이면 원본 그대로."""
    f = deepcopy(filters or {})
    if level >= 1:
        f.pop("사유", None)
    if level >= 2:
        f["fund_name"] = []
    return f


def _top_k_for(질문유형: str, filters: Dict[str, Any]) -> int:
    """질문유형에 따른 검색 폭 결정.

    추천·비교는 "정답 1개를 찾는" 문제가 아니라 "후보군을 모아 조건으로 거르는"
    문제다. top-5 만 보면 위험등급·자산유형 조건에 맞는 상품이 아예 후보에 없을 수
    있다. 실측: 골드셋 G011 이 top-5 에서 recall 0.00 → top-24 에서 1.00.

    상품 관여도가 높으면 넓힌다 — 계층 쿼터가 자리를 나누므로, top_k 가 넓어져도
    규정문서 자리가 줄지 않는다(예전 scope="all" 과 결정적으로 다른 점).
    """
    from index.store import _affinity_of

    if 질문유형 == "추천" or _affinity_of(filters or {}) >= 0.6:
        return config.NUM_DOCS_RECOMMEND
    return config.NUM_DOCS_TO_RETRIEVE


def _embed(text: str) -> Optional[List[float]]:
    """임베딩 실패는 치명적이지 않다 — BM25 로 계속 간다."""
    try:
        return embedder().embed_query(text)
    except Exception as e:
        log_message(f"임베딩 실패 → BM25 단독으로 진행: {e}", "warning")
        return None


def _ensure_named_fund_cards(docs: List[dict], names: List[str]) -> tuple[List[dict], List[str]]:
    """질문이 **지목한** 펀드의 카드가 근거에 없으면 채워 넣는다 (2026-08-31).

    ■ 왜 필요한가 (9차 이슈 3)

      검색은 점수 순으로 자리를 채우므로, 형제펀드 셋을 비교하는 질문에서
      한 펀드의 카드가 자리 싸움에 밀려 빠질 수 있다. 그러면 답변은 그 펀드의
      총보수·수익률·등급을 **받지 못한 채** 표를 채워야 한다.

      실측 P-B1: 국공채·주식 카드만 들어오고 채권 카드가 빠졌다. 답변은 빈 칸을
      두는 대신 국공채 값(0.544%/5.91%)을 채권 행에 그대로 복사했다.
      카드가 있었으면 5.74% 라고 적을 수 있었다.

    ■ 왜 안전한가
      · 붙이는 대상이 **질문이 이름을 댄 펀드의 카드**뿐이다. 검색 상위 몇 건을
        무조건 붙이던 옛 폴백(P-A1 오출처)과 성격이 다르다.
      · 이미 들어온 카드는 건드리지 않는다. 없는 것만 채운다.
      · 인용은 여전히 값 대조·모델 판단을 거친다 — 카드가 근거 목록에 있다고
        자동으로 출처가 되지는 않는다.
      · 상한이 config.MAX_SIBLING_FUNDS 라 근거 목록이 부풀지 않는다.

    반환: (보강된 문서 목록, 새로 채운 펀드명)
    """
    if not (config.ENSURE_NAMED_FUND_CARDS and names):
        return docs, []

    have = {(d.get("fund") or {}).get("name")
            for d in docs if d.get("chunk_type") == "fund_card"}
    missing = [n for n in names if n and n not in have]
    if not missing:
        return docs, []

    seen = {d.get("chunk_id") for d in docs}
    added: List[str] = []
    try:
        chunks = get_index().chunks
    except Exception as e:                      # 인덱스 문제로 검색을 막지 않는다
        log_message(f"펀드카드 보강 건너뜀: {e}", "warning")
        return docs, []

    for c in chunks:
        if len(added) >= config.MAX_SIBLING_FUNDS:
            break
        if c.get("chunk_type") != "fund_card" or c.get("chunk_id") in seen:
            continue
        fn = (c.get("fund") or {}).get("name")
        if fn in missing and fn not in added:
            docs.append(c)
            seen.add(c.get("chunk_id"))
            added.append(fn)
    return docs, added


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

    # 질문이 이름을 댄 펀드의 카드는 자리 싸움에 밀려도 반드시 넣는다.
    # (완화 단계와 무관하게 **원래 필터의** 펀드명을 기준으로 본다 — 완화는
    #  후보를 넓히려고 조건을 푼 것이지, 질문이 그 펀드를 안 말한 게 아니다.)
    docs, filled = _ensure_named_fund_cards(
        docs, list(base_filters.get("fund_name") or []))

    relax_note = (
        "필터 없음(전체검색)" if used_level < 0
        else ("완화 없음" if used_level == 0
              else f"{used_level}단계 완화({'·'.join(RELAX_STEPS[:used_level])} 해제)")
    )
    dist: Dict[str, int] = {}
    for d in docs:
        key = f"{d['tier']}/{d['chunk_type']}"
        dist[key] = dist.get(key, 0) + 1

    from index.store import _affinity_of
    summary = (f"{len(docs)}건 (top_k={top_k}, 상품관여도={_affinity_of(applied):.2f}, "
               f"{relax_note}) 분포={dist}")
    if filled:
        summary += f" | 지목펀드 카드 보강 {len(filled)}건"
    if applied.get("사유"):
        summary += f" | 사유={applied['사유']}"
    if hyde_doc:
        summary += " | HyDE 병합검색"

    return {
        "documents": docs,
        "applied_filters": applied,
        "filter_retries": state.get("filter_retries", 0) + (1 if used_level != 0 else 0),
        # chunk_type 분포를 **문자열 요약이 아니라 dict 로** 남긴다 (2026-08-18).
        #
        # 예전에는 summary 문자열 안에만 있어서, 분석하려면 정규식으로 다시
        # 파싱해야 했다. 실패 문항을 볼 때 이 분포가 없으면
        #   "검색부터 실패인가(원하는 chunk_type 이 아예 안 들어왔다)"
        #   "검색은 됐는데 인용을 안 한 건가(들어왔는데 안 썼다)"
        # 를 구분할 수 없다. 두 경우는 고칠 곳이 완전히 다르다.
        "trace": step("retrieve", qid=qid, summary=summary,
                      detail={"scores": [round(h["score"], 3) for h in hits[:5]],
                              "filters": applied,
                              "chunk_type_dist": dict(dist),
                              "n_docs": len(docs),
                              "top_k": top_k,
                              "hyde": truncate(hyde_doc, 120) if hyde_doc else None},
                      ms=t.ms),
    }


def documents_to_citations(docs: List[dict]) -> List[dict]:
    """검색 결과를 인용 레코드로."""
    return [to_citation(d) for d in docs]
