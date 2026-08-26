"""슬롯 해소 노드 — 답이 갈리는 값을 확정하거나, 못 하면 **되묻는다**(역질문).

■ 왜 이 노드가 필요한가 — 검색으로는 풀 수 없는 문제가 하나 있다

  장르C(중도인출, doc46~50)는 **5개 문서의 도입부 문장이 동일하다.**
  "중도인출 되나요?" 로 검색하면 다섯이 거의 같은 유사도로 올라온다.
  BM25 도 벡터도 이걸 가르지 못한다 — 텍스트가 실제로 같기 때문이다.
  검색 기법을 아무리 손봐도 안 되고, **질문 쪽에서 사유를 확정해야** 풀린다.

  사유가 틀리면 조용히 틀린다는 점이 특히 나쁘다. 주택구입으로 물었는데
  전세보증금 서류를 근거로 그럴듯한 답이 나가면, 사용자는 틀린 줄 모른다.

■ 왜 키워드 규칙이 아니라 LLM 인가

  키워드 매칭은 두 방향으로 다 샌다.
    · "집 사려고요" / "이사 가는데 목돈이" → 어느 키워드에도 안 걸린다
    · "전세 사는데 병원비가 급해요"        → '전세'와 '병원' 둘 다 걸려 오판한다
  사유 판정은 **어휘 매칭이 아니라 의도 이해**다. 그래서 LLM 에 맡긴다.
  대신 출력은 데이터에 실재하는 5종으로 강제하고(사후 검증), 목록 밖 값은 버린다 —
  없는 값으로 필터를 걸면 후보가 0건이 되어 파이프라인이 헛돈다.

■ 못 뽑으면 되묻는다 (역질문)

  추측해서 진행하는 것보다 한 번 되묻는 편이 정확하다. 중도인출은 사유마다
  요건·서류·횟수·과세가 전부 다르므로, 사유 없는 답변은 **어떻게 써도 불완전**하다.

  ⚠️ 다만 되묻기는 남용하면 그 자체가 감점이다("정보한계 대응" 이 아니라 불친절).
  그래서 발동 조건을 좁게 건다.
    · 사유가 답을 가르는 질문일 때만 (LLM 이 판정)
    · 그리고 사유를 **하나도** 확정하지 못했을 때만
  질문에서 사유를 추론할 수 있으면("집 사려고요" → 주택구입) 되묻지 않고 진행한다.

■ 이 노드는 필터를 '거는' 것이 아니라 '제안'한다
  여기서 채운 `filters["사유"]` 는 하드 필터가 아니다. 검색 결과가 모자라면
  document_retriever 의 완화 사다리 1단계에서 **자동으로 풀린다.**
  즉 사유 판정이 빗나가도 정답이 사라지지 않는다.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

import config
from llm import llm
from prompt import prompts
from state import OverallState
from utils import Timer, step


class SlotResolution(BaseModel):
    # 좁고 사실적인 질문 — '사유_필요' 보다 판정이 쉽다.
    # 둘 다 true 일 때만 되묻는다 (resolve_slots 의 게이트 주석 참고).
    중도인출_질문: bool = Field(
        default=True,
        description="이 질문이 적립금을 **중도에 꺼내는 것**(중도인출·중간정산)을 "
                    "묻고 있는가. 세액공제·제도구조·수령방법·상품 질문이면 false")
    사유_필요: bool = Field(
        default=False,
        description="사유(중도인출/중간정산 목적)에 따라 답이 달라지는 질문이면 true")
    사유: List[str] = Field(
        default_factory=list,
        description="요양 | 파산개인회생 | 전세보증금 | 주택구입 | 재난피해 중에서. "
                    "질문에서 확정할 수 없으면 빈 배열")
    하위유형: str = Field(default="", description="주택구입의 매매/신축/분양 등. 없으면 빈 문자열")

    # ── 두 번째 축: 계좌유형 (2026-08-20) ────────────────────────────
    # 코퍼스가 계좌유형별로 다른 규정을 명시한다(config.ACCOUNT_VALUES 주석 참고).
    #   doc48 "DC 는 재직 중 1회, 개인형IRP 는 횟수제한 없음"
    #   doc46 "의료비 12.5% 초과 (개인형IRP 는 무관)"
    # 모르면 어느 쪽으로 답해도 절반이 틀린다.
    계좌유형_필요: bool = Field(
        default=False,
        description="계좌유형(DB/DC/IRP/연금저축)에 따라 **답이 달라지는** 질문이면 true. "
                    "요건·횟수·서류가 계좌유형별로 다른 경우다. "
                    "계좌유형과 무관한 질문이면 false")
    계좌유형: str = Field(
        default="",
        description="질문에서 확정되는 계좌유형. DB | DC | IRP | 연금저축 중 하나. "
                    "질문에 없으면 빈 문자열. 추측하지 마라")
    근거: str = Field(default="", description="그렇게 판단한 이유 한 문장")
    역질문: str = Field(
        default="",
        description="사유가 필요한데 확정할 수 없을 때 사용자에게 되물을 한 문장")


# 사유가 확정되지 않았을 때의 기본 역질문.
# LLM 이 역질문을 못 만들었을 때만 쓰는 안전망이다.
_DEFAULT_CLARIFY = (
    "중도인출은 **사유에 따라 요건·필요서류·인출 횟수·세율이 모두 다릅니다.**\n"
    "어떤 사유에 해당하시는지 알려주시면 정확히 안내드리겠습니다.\n\n"
    "- 본인 또는 부양가족의 6개월 이상 요양\n"
    "- 파산 선고 또는 개인회생절차 개시\n"
    "- 무주택자의 주거 목적 전세보증금(임차보증금)\n"
    "- 무주택자의 주택 구입\n"
    "- 천재지변 등 재난 피해"
)


# 계좌유형을 못 정했을 때의 기본 역질문. LLM 이 못 만들었을 때만 쓴다.
_DEFAULT_ACCOUNT_CLARIFY = (
    "가입하신 **계좌 유형에 따라 요건과 횟수가 다릅니다.**\n"
    "어느 쪽인지 알려주시면 정확히 안내드리겠습니다.\n\n"
    "- DB (확정급여형) — 회사가 운용, 원칙적으로 중도인출 불가\n"
    "- DC (확정기여형) — 근로자가 운용, 사유별 요건·횟수 제한 있음\n"
    "- 개인형 IRP — 사유별 요건이 다르고 횟수 제한이 없는 경우가 있음\n"
    "- 연금저축 — 사유와 무관하게 인출 가능(세제만 달라짐)"
)


def _clean(values: List[str]) -> List[str]:
    """LLM 출력을 데이터에 실재하는 값으로 제한한다.

    목록 밖 값("의료비", "주거")으로 필터를 걸면 매칭되는 청크가 0건이 된다.
    프롬프트 주입(사전 방어)과 이 사후 검증을 **둘 다** 둔다 — query_analyzer 의
    펀드명 처리와 같은 방어다.
    """
    known = set(config.REASON_VALUES)
    out: List[str] = []
    for v in values or []:
        v = (v or "").strip()
        if v in known and v not in out:
            out.append(v)
    return out


def _account_in_question(question: str) -> str:
    """질문에 계좌유형이 이미 적혀 있으면 그것. LLM 없이 목록 대조.

    사용자가 "IRP에서 …" 라고 이미 밝혔는데 계좌유형을 되묻는 것은 그 자체로
    감점이다. LLM 추출이 놓쳐도 여기서 받친다 — 표기 변형 목록 대조라 오탐이 없다.
    (query_analyzer 의 펀드명 목록 대조와 같은 방어다)
    """
    q = (question or "").upper().replace(" ", "")
    for canon, aliases in config.ACCOUNT_ALIASES.items():
        for a in aliases:
            if a.upper().replace(" ", "") in q:
                return canon
    return ""


def resolve_slots(state: OverallState) -> dict:
    """사유·계좌유형을 확정하거나 역질문을 준비한다.

    ■ 두 축은 성격이 다르다
        사유     — **필터 축**. 확정되면 filters 로 넘겨 검색을 좁힌다.
        계좌유형 — **답변 축**. 청크에 메타데이터가 없어 필터로 못 쓴다.
                   되묻기 판정에만 쓰고, 확정되면 slots 에만 기록한다.
    """
    question = state["question"]
    filters = dict(state.get("filters", {}) or {})

    if not config.WORKFLOW_SETTINGS.get("resolve_slots", True):
        return {"trace": step("resolve_slots", summary="비활성(설정)")}

    with Timer() as t:
        res = llm.structured(
            prompts.slot_filler_system,
            f"질문: {question}",
            SlotResolution,
            model=config.MODEL_PLAN,
            stage="slots",
        )

    사유 = _clean(res.사유)
    needs = bool(res.사유_필요)

    # 계좌유형 — LLM 추출을 목록 대조가 받친다 (사용자가 이미 밝혔으면 안 묻는다)
    계좌유형 = (res.계좌유형 or "").strip()
    if 계좌유형 not in config.ACCOUNT_VALUES:
        계좌유형 = ""
    계좌유형 = 계좌유형 or _account_in_question(question)
    acct_needs = bool(res.계좌유형_필요) and not 계좌유형

    slots = {
        "사유": 사유,
        "하위유형": res.하위유형 or "",
        "계좌유형": 계좌유형,

        # ── 판정 **원값**을 함께 남긴다 (2026-08-23) ──────────────────
        #
        # 되묻기가 발동하지 않았을 때 원인이 두 가지로 갈린다.
        #   · 모델이 "계좌유형은 상관없다"고 판정했다  → 프롬프트 문제
        #   · 판정은 필요라고 했는데 코드가 못 걸렀다   → 분기 문제
        # 합성된 결과(clarified)만 보면 이 둘이 구분되지 않아서, 4차·5차
        # 두 번 모두 "역질문 0/4"의 원인을 좁히지 못했다. 원값을 남긴다.
        "사유_필요": needs,
        "계좌유형_필요": bool(res.계좌유형_필요),
    }

    # 사유가 확정됐으면 소프트 필터로 넘긴다.
    # ⚠️ **반환은 아직 하지 않는다** — 아래 계좌유형 검사를 먼저 통과해야 한다.
    if 사유:
        filters["사유"] = 사유

    # ── 계좌유형이 갈리는데 확정 못 했다 → 되묻는다 ──────────────────
    #
    # ⚠️ 순서가 중요하다 (2026-08-23 수정)
    #
    #   예전에는 `if 사유:` 가 여기보다 **먼저** 반환해 버렸다. 그래서 사유가
    #   잡히는 순간 계좌유형 검사에 도달하지 못했다.
    #
    #   실측(5차 78문항): 역질문이 정답인 4문항 중 2건이 정확히 여기서 막혔다.
    #       D-A1  사유=['요양']      계좌유형='' → 되묻지 않음
    #       D-A2  사유=['전세보증금']  계좌유형='' → 되묻지 않음
    #
    #   아래 주석이 되묻기 근거로 든 문항이 바로 D-A1 인데, 정작 그 문항이
    #   코드에 도달하지 못하고 있었다. 두 축은 독립이므로 순서를 뒤집는다 —
    #   사유를 알아도 계좌유형이 갈리면 답의 절반이 틀린다.
    #
    #   실측 D-A1 "의료비가 12.5% 안 되는데 중도인출 되나요?"
    #     doc46 "…12.5% 초과하는 경우 신청가능 (개인형IRP 는 무관)"
    #   계좌유형을 모르면 어느 쪽으로 답해도 절반이 틀린다.
    #
    #   사유 축과 달리 중도인출 게이트를 걸지 않는다 — 계좌유형은 중도인출뿐
    #   아니라 부담금·수령방법·세제에서도 답을 가르기 때문이다. 대신 위에서
    #   **질문에 계좌유형이 이미 있으면 묻지 않는** 결정론적 가드를 걸어 뒀다.
    if acct_needs and config.CLARIFY_MISSING_SLOTS:
        ask = (res.역질문 or "").strip() or _DEFAULT_ACCOUNT_CLARIFY
        return {
            "needs_clarification": True,
            "clarification": ask,
            "slots": slots,
            "llm_calls": 1,
            "trace": step("resolve_slots",
                          summary=f"계좌유형 미확정 → 역질문 — {res.근거}", ms=t.ms),
        }

    # ── 사유는 확정됐고 계좌유형은 물을 필요가 없다 → 그대로 진행 ────
    if 사유:
        return {
            "filters": filters,
            "slots": slots,
            "needs_clarification": False,
            "llm_calls": 1,
            "trace": step("resolve_slots",
                          summary=f"사유={사유}"
                                  + (f", 하위유형={res.하위유형}" if res.하위유형 else "")
                                  + (f", 계좌={계좌유형}" if 계좌유형 else "")
                                  + f" — {res.근거}",
                          ms=t.ms),
        }

    # ── 되묻기 게이트 — 두 판정이 **일치할 때만** 되묻는다 (2026-08-20) ──
    #
    # ■ 무엇이 잘못됐나
    #   실측 4차 실행에서 되묻기가 기대와 하나도 안 맞았다.
    #       역질문이 정답인 문항: P-A1, P-B2, D-A1, D-A2
    #       실제로 되물은 문항  : P-G3, D-A5, D-E1      ← 교집합 0건
    #
    #   되물은 셋은 전부 **중도인출 질문이 아니다**(세제 2건, 제도 1건).
    #   P-G3 의 역질문이 "명퇴수당을 연금계좌에 **입금하신** 이유가 어떻게 되시나요?"
    #   였던 게 결정적이다 — 모델이 '사유' 를 "중도인출 사유" 가 아니라 일반적인
    #   "이유" 로 읽었다. 프롬프트에 "세액공제 … 전부 false" 라고 이미 적혀 있는데도
    #   그랬으므로 문구를 더 붙이는 것으로는 안 고쳐진다.
    #
    # ■ 왜 임계값이 아니라 이중 판정인가
    #   처음에는 중도인출 청크에 BM25 를 걸어 임계값으로 거르려 했다. 그런데 그
    #   임계값은 **평가 문항 사이 간격을 보고 고른 숫자**였다. 코퍼스 기준으로
    #   재 보니 근거가 없었다 — 비-중도인출 청크의 84%가 통과하는 값이었다.
    #   평가 세트에 맞춘 상수는 대회 본 질문에서 그대로 무너진다.
    #
    #   대신 **모델에게 좁은 사실 질문을 하나 더 던지고 둘 다 참일 때만** 되묻는다.
    #     사유_필요     : "사유에 따라 답이 달라지나"        (추상적 — 오판이 잦다)
    #     중도인출_질문 : "적립금을 중도에 꺼내는 걸 묻는가"  (구체적 — 판정이 쉽다)
    #   튜닝할 숫자가 없고 어느 문항에도 맞춰져 있지 않다. 같은 LLM 호출 안에서
    #   필드가 하나 늘 뿐이라 비용도 0 이다.
    if needs and not bool(getattr(res, "중도인출_질문", True)):
        return {
            "needs_clarification": False,
            "slots": slots,
            "llm_calls": 1,
            "trace": step("resolve_slots",
                          summary="사유 미확정이나 중도인출 질문이 아님 → 되묻지 않고 통과",
                          ms=t.ms),
        }

    if needs and config.CLARIFY_MISSING_SLOTS:
        ask = (res.역질문 or "").strip() or _DEFAULT_CLARIFY
        return {
            "needs_clarification": True,
            "clarification": ask,
            "slots": slots,
            "llm_calls": 1,
            "trace": step("resolve_slots",
                          summary=f"사유 미확정 → 역질문 — {res.근거}", ms=t.ms),
        }

    # ── 사유와 무관한 질문 → 그대로 진행 ─────────────────────────────
    return {
        "needs_clarification": False,
        "slots": slots,
        "llm_calls": 1,
        "trace": step("resolve_slots",
                      summary="사유 축 없음 → 통과"
                              + (" (되묻기 비활성)" if needs else ""),
                      ms=t.ms),
    }


def clarify_answer(state: OverallState) -> dict:
    """역질문을 최종 답변으로 내보낸다 (LLM 호출 없음).

    되묻기는 **실패가 아니라 정답**인 경우가 있다. 정보가 부족한 상태에서 단정한
    답을 내는 것이 오히려 감점이므로, 이 경로를 정상 종료로 취급한다.
    """
    return {
        "final_answer": state.get("clarification") or _DEFAULT_CLARIFY,
        "citations": [],
        "trace": step("clarify", summary="역질문 반환 (정보 부족)"),
    }
