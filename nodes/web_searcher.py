"""네이버 웹 검색 폴백.

■ 선행모델 대응
  FinSight 는 Tavily → Google 스크래핑 → Bing 순의 3단 폴백을 썼다.
  요구사항에 따라 우리는 **네이버 검색 API** 하나만 쓴다.

■ 언제 호출되는가
  내부 문서(494청크)에서 근거를 못 찾고, 질문 재작성 재시도까지 소진했을 때만.
  즉 **최후의 수단**이다. 이 경로를 탄 답변은 신뢰도가 낮으므로
  · evidence_ok = False 로 표시하고
  · 답변 첫 줄에 경고 문구를 강제한다(prompts.web_answer_system)
  · 성능 평가에서 '문서 근거 실패' 로 따로 집계한다

■ 검색 품질 보정
  연금 질문에 그냥 검색하면 광고·블로그가 상위를 차지한다. 그래서
  1) 질의에 도메인 키워드를 덧붙이고
  2) 신뢰 도메인(정부·공단·금감원)을 우대하며
  3) HTML 태그(<b> 등)를 제거한다
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import List

import config
from state import RAGState
from utils import Timer, log_message, step

# 신뢰 도메인 — 검색 결과 정렬 시 가산점
TRUSTED_DOMAINS = [
    "go.kr", "or.kr", "moel.go.kr", "nps.or.kr", "fss.or.kr",
    "nts.go.kr", "kcomwel.or.kr", "pension.fss.or.kr", "law.go.kr",
]

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    s = _TAG_RE.sub("", s or "")
    return (s.replace("&quot;", '"').replace("&amp;", "&")
             .replace("&lt;", "<").replace("&gt;", ">").strip())


def _trust_score(url: str) -> int:
    return 1 if any(d in url for d in TRUSTED_DOMAINS) else 0


def naver_search(query: str, k: int = 5) -> List[dict]:
    """네이버 웹문서 검색. 키가 없으면 빈 리스트."""
    if not (config.NAVER_CLIENT_ID and config.NAVER_CLIENT_SECRET):
        log_message("네이버 검색 키 없음 → 웹 폴백 건너뜀", "warning")
        return []

    params = urllib.parse.urlencode({"query": query, "display": max(k * 2, 10)})
    req = urllib.request.Request(
        f"{config.NAVER_SEARCH_URL}?{params}",
        headers={
            "X-Naver-Client-Id": config.NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": config.NAVER_CLIENT_SECRET,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        log_message(f"네이버 검색 HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}",
                    "warning")
        return []
    except Exception as e:
        log_message(f"네이버 검색 실패: {e}", "warning")
        return []

    items = data.get("items", [])
    out = [{
        "title": _clean(it.get("title")),
        "text": _clean(it.get("description")),
        "url": it.get("link", ""),
        "section": _clean(it.get("title")),
        "source": "웹검색",
        "trust": _trust_score(it.get("link", "")),
    } for it in items]

    out.sort(key=lambda x: x["trust"], reverse=True)     # 공공기관 우선
    return out[:k]


def search_web(state: RAGState) -> dict:
    """웹 검색 결과를 documents 자리에 넣어 generate_web_answer 로 넘긴다."""
    question = state["question"]
    qid = state.get("qid", "")

    if not config.WORKFLOW_SETTINGS["web_search_fallback"]:
        return {"documents": [], "web_used": False,
                "trace": step("search_web", qid=qid, summary="비활성(설정)")}

    # 도메인 키워드를 붙여 잡음을 줄인다
    q = question if "연금" in question else f"{question} 퇴직연금"

    with Timer() as t:
        results = naver_search(q, config.NUM_WEB_RESULTS)

    n_trust = sum(r["trust"] for r in results)
    return {
        "documents": results,
        "web_used": True,
        "trace": step("search_web", qid=qid,
                      summary=f"네이버 검색 {len(results)}건 (공공기관 {n_trust}건)",
                      ms=t.ms),
    }
