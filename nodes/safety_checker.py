"""안전성 검사 노드 — 파이프라인의 첫 관문.

선행모델의 `nodes/safety_checker.py` 대응. 하는 일이 두 가지다.
  1) 위법·유해 요청 차단
  2) **도메인 범위 이탈 판정** (선행모델엔 없던 것)

2)를 여기서 하는 이유: 연금과 무관한 질문에 대해 검색·분해·생성을 다 돌린 뒤
"모르겠습니다" 하면 LLM 호출 5~10회를 낭비한다. 첫 노드에서 1회로 끝낸다.

──────────────────────────────────────────────────────────────────────────
■ 범위 판정의 오탐 비용은 미탐 비용보다 훨씬 크다 (2026-08 보강)

이 노드는 **1번 노드**이고, 여기서 막히면 **복구 경로가 없다.**

    오탐(정상 질문을 막음)  → 파이프라인 전체가 죽는다. 사용자는 답을 못 받는다.
    미탐(엉뚱한 질문 통과)  → LLM 몇 회 낭비 + "제공된 자료로는 확인되지 않습니다"
                              로 우아하게 끝난다.

비용이 극단적으로 비대칭이므로 **의심스러우면 통과**시켜야 한다.

실제로 오탐이 났다:

    "퇴집급여를 은행에 가서 직접 받으려면 어떻게 해?"
      → '퇴직'이 아니라 '퇴집'(오타) → 범위 밖 판정 → 1번 노드에서 차단
      → 정작 doc57 에 「퇴직급여 직접청구 프로세스」가 있는데 답을 못 냈다

■ 해결: 키워드 목록이 아니라 **코퍼스에 직접 물어본다**

키워드 화이트리스트는 두 방향으로 다 샌다.
  · 오타 한 글자에 뚫린다 ("퇴집급여")
  · 도메인 어휘가 없는 정상 질문을 못 잡는다 ("돈 미리 빼 쓸 수 있나요")

그래서 **BM25 로 한 번 검색해 본다.** LLM 호출 0회, 수 ms.
"우리 문서가 이 질문에 대해 할 말이 있는가"를 데이터에게 직접 묻는 것이라,
손으로 관리하는 목록보다 정확하고 코퍼스가 바뀌어도 자동으로 따라간다.

실측 (raw BM25 top-1 점수, doc 코퍼스 174건 대상):

    도메인 안                                          점수
      "퇴집급여를 은행에 가서 직접 받으려면?"           39.8   ← 오타에도 잡힌다
      "IRP 중도인출 사유가 뭐야"                        48.6
      "만기 ISA 연금계좌로 옮기기"                      47.9
      "돈 미리 빼 쓸 수 있나요"                         31.1   ← 도메인 어휘 0개인데 잡힌다
      "회사가 규약 바꾸려면 직원 동의 필요해?"          29.3
    도메인 밖
      "오늘 서울 날씨 어때?"                            12.4
      "삼성전자 주가 전망 알려줘"                        6.5
      "파이썬으로 웹크롤러 만드는 법"                    3.0
      "안녕하세요 반갑습니다"                            1.7

임계값 20 이면 깨끗하게 갈린다.

⚠️ 이 보정은 **`out_of_scope` 에만** 적용한다. `safe=false`(위법·유해·개인정보)는
   코퍼스와 무관하게 반드시 막아야 하므로 절대 덮어쓰지 않는다.
   (예: "주민등록번호 알려줘" 는 BM25 점수가 31 로 높지만 safe=false 로 걸린다)
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

import config
from llm import llm
from prompt import prompts
from state import OverallState
from utils import Timer, step, tokenize_ko

log = logging.getLogger(__name__)


class SafetyCheck(BaseModel):
    safe: bool = Field(description="위법·유해 요청이 아니면 true")
    out_of_scope: bool = Field(description="연금 도메인 밖이면 true")
    reason: str = Field(default="", description="판정 이유 한 문장")
    safe_question: str = Field(description="개인정보를 마스킹한 질문. 문제없으면 원문 그대로")


def corpus_signal(question: str) -> float:
    """코퍼스가 이 질문에 대해 할 말이 있는가 — raw BM25 top-1 점수.

    LLM 을 부르지 않는다. 인덱스가 아직 없으면 만들지 않고 0.0 을 돌려준다
    (범위 판정 때문에 3,853청크 인덱싱을 유발하면 안 된다 — 다만 이 파이프라인은
    어차피 곧 검색을 하므로 실제로는 여기서 만들어도 손해가 없다).
    """
    try:
        from index import get_index

        idx = get_index()
        # 규정문서만 대상으로 본다. 투자설명서 3,679건은 어떤 질문에도 뭔가
        # 걸리는 편이라 범위 판정의 변별력을 떨어뜨린다.
        cand = [i for i, c in enumerate(idx.chunks) if c["tier"] == "doc"]
        if not cand:
            cand = list(range(len(idx.chunks)))
        scores = idx.bm25.scores(tokenize_ko(question), cand)
        return max(scores.values()) if scores else 0.0
    except Exception as e:                       # 인덱스 문제로 질문을 막으면 안 된다
        log.warning("범위 판정용 코퍼스 조회 실패(통과 처리): %s", e)
        return float("inf")


def check_safety(state: OverallState) -> dict:
    """질문의 안전성·범위를 판정한다."""
    question = state["question"]

    if not config.WORKFLOW_SETTINGS["check_safety"]:
        return {"safe": True, "trace": step("check_safety", summary="비활성(설정)")}

    with Timer() as t:
        res = llm.structured(
            prompts.safety_system,
            f"질문: {question}",
            SafetyCheck,
            model=config.MODEL_PLAN,
            stage="safety",
        )

    out_of_scope = res.out_of_scope
    override = ""

    # ── 범위 밖 판정 뒤집기 (safe=false 는 건드리지 않는다) ──────────────
    if res.safe and out_of_scope:
        sig = corpus_signal(question)
        if sig >= config.SCOPE_MIN_BM25:
            out_of_scope = False
            override = f" [보정: 코퍼스 신호 {sig:.1f} ≥ {config.SCOPE_MIN_BM25} → 통과]"
            log.info("범위 밖 판정을 뒤집었습니다 (BM25 %.1f): %s", sig, question[:60])

    verdict = "통과"
    if not res.safe:
        verdict = "차단(유해)"
    elif out_of_scope:
        verdict = "차단(범위밖)"

    return {
        "safe": res.safe and not out_of_scope,
        "safety_reason": res.reason,
        "question": res.safe_question or question,
        "llm_calls": 1,
        "trace": step("check_safety",
                      summary=f"{verdict} — {res.reason}{override}", ms=t.ms),
    }
