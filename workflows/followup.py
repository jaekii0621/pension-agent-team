"""되묻기 이어받기 — 사용자의 답을 **원 질문에 붙여** 다시 태운다 (2026-08-27 신설).

■ 무엇이 문제였나

  되묻기는 `clarify_answer` 에서 그래프가 **끝난다.** 역질문 문장이 최종 답변으로
  나가고 거기서 종료다. 그런데 사용자가 "IRP요" 라고 답하면 그 입력은
  **완전히 새로운 질문**으로 처음부터 다시 돈다 — 앞에서 무엇을 물었는지,
  무엇을 되물었는지 파이프라인이 하나도 모른다.

      1턴  "의료비가 연봉의 12.5%가 안 되는데 중도인출 되나요?"
           → "DB / DC / 개인형IRP / 연금저축 중 어느 쪽인가요?"
      2턴  "IRP요"
           → 분해도 검색도 이 두 글자만 보고 돈다. 답이 나올 수가 없다.

  `run()` 이 질문 문자열 하나만 받고, `session_id` 는 토큰 계측용 라벨일 뿐
  아무것도 저장하지 않기 때문이다.

■ 어떻게 고치나 — 파이프라인은 그대로 두고 **질문을 합친다**

  되묻기로 끝나면 (원 질문, 역질문) 을 세션에 적어 둔다. 다음 입력이 그 답으로
  보이면 원 질문에 붙여 **하나의 자족적인 문장**으로 만들어 다시 태운다.

      "개인형IRP 기준으로 답해 주세요. 의료비가 연봉의 12.5%가 안 되는데
       중도인출 되나요?"

  그래프·노드·프롬프트를 건드리지 않는다. 분해기가 요구하는
  "서브질문은 그것만 읽어도 뜻이 통해야 한다" 를 입력 단계에서 맞춰 주는 셈이다.

■ LLM 을 쓰지 않는다

  합성은 별칭 목록 대조로 한다(`config.ACCOUNT_ALIASES`, `config.REASON_VALUES`).
  되묻기가 묻는 것이 계좌유형 아니면 사유 둘뿐이라 규칙으로 충분하고,
  여기에 LLM 을 한 번 더 태우면 되묻기 경로만 비용이 늘어난다.

■ 평가 실행에는 영향이 없다

  `eval.run_queryset` 는 문항마다 다른 `session_id`(`qs-<qid>`) 를 쓴다.
  세션이 다르면 이어받을 것도 없으므로 78문항 배치는 지금과 똑같이 돈다.
  대회 평가 시스템도 1요청 1응답이라 마찬가지다 — 이 기능은 CLI·노트북·
  실제 서비스처럼 **사람이 다음 턴을 칠 수 있는 곳**에서만 작동한다.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import config

# session_id → {"question", "clarification", "ts"}
_PENDING: Dict[str, dict] = {}

_MAX_SESSIONS = 500          # 서버에서 오래 돌 때 메모리가 늘지 않게
_REPLY_MAX_LEN = 80          # 이보다 길면 무조건 새 질문
_REPLY_SLOT_LEN = 25         # 슬롯 어휘가 있을 때 허용 길이
_REPLY_PLAIN_LEN = 15        # 슬롯 어휘가 없을 때 ("회사가 운용하는 거요")


def _sweep() -> None:
    """만료된 항목을 버린다. 저장하거나 꺼낼 때마다 한 번씩."""
    now = time.time()
    dead = [k for k, v in _PENDING.items()
            if now - v["ts"] > config.CLARIFY_FOLLOWUP_TTL]
    for k in dead:
        _PENDING.pop(k, None)
    if len(_PENDING) > _MAX_SESSIONS:
        for k, _ in sorted(_PENDING.items(), key=lambda kv: kv[1]["ts"])[:-_MAX_SESSIONS]:
            _PENDING.pop(k, None)


def remember(session_id: str, question: str, clarification: str) -> None:
    """되묻기로 끝났다 — 다음 입력을 기다린다."""
    if not (session_id and question):
        return
    _sweep()
    _PENDING[session_id] = {
        "question": question,
        "clarification": clarification or "",
        "ts": time.time(),
    }


def take(session_id: str) -> Optional[dict]:
    """대기 중인 되묻기를 꺼낸다(꺼내면 지운다)."""
    if not session_id:
        return None
    _sweep()
    return _PENDING.pop(session_id, None)


def clear(session_id: str) -> None:
    _PENDING.pop(session_id, None)


def pending(session_id: str) -> Optional[dict]:
    """지우지 않고 들여다본다(디버깅·서버 상태 확인용)."""
    _sweep()
    return _PENDING.get(session_id)


# ---------------------------------------------------------------- 슬롯 인식
def account_in(text: str) -> str:
    """문장에 적힌 계좌유형. `slot_filler._accounts_in_question` 과 같은 대조."""
    t = (text or "").upper().replace(" ", "")
    for canon, aliases in config.ACCOUNT_ALIASES.items():
        for a in aliases:
            if a.upper().replace(" ", "") in t:
                return canon
    return ""


# 사유는 코퍼스 값(요양/파산개인회생/…) 자체보다 사용자가 쓰는 말이 다양하다.
# 되묻기 선택지에 쓰인 표현을 그대로 받는 정도로만 본다.
_REASON_WORDS = {
    "요양": ("요양", "치료", "의료비", "병원", "입원"),
    "파산개인회생": ("파산", "개인회생", "회생"),
    "전세보증금": ("전세", "월세", "임차", "보증금"),
    "주택구입": ("주택구입", "집 사", "집사", "매매", "신축", "분양", "내 집", "내집"),
    "재난피해": ("재난", "천재지변", "침수", "화재", "태풍", "수해"),
}


def reason_in(text: str) -> str:
    # ⚠️ 띄어쓰기를 지우고 맞춘다. "주택 구입 때문에요" 는 "주택구입" 을 포함하지
    #    않아서, 공백을 그대로 두면 사용자가 사유를 답해도 못 잡는다.
    t = (text or "").replace(" ", "")
    for canon, words in _REASON_WORDS.items():
        for w in words:
            if w.replace(" ", "") in t:
                return canon
    return ""


def looks_like_reply(text: str) -> bool:
    """이 입력이 방금 한 되묻기에 대한 **답**으로 보이는가.

    새 질문을 답으로 오인해 엉뚱하게 합치면 원 질문이 오염된다. 그래서 좁게 본다.
      · 길면 새 질문이다 (되묻기 답은 "IRP요" 처럼 짧다)
      · 물음표로 끝나면 새 질문이다
      · 슬롯 어휘가 들어 있으면 조금 길어도 답으로 본다
    """
    t = (text or "").strip()
    if not t or len(t) > _REPLY_MAX_LEN:
        return False
    if t.endswith("?") or t.endswith("까요") or t.endswith("나요"):
        return False
    # 슬롯 어휘가 있어도 길면 새 질문이다.
    #   "연금저축 펀드 추천해 주시고 보수도 비교해서 알려주세요" 는 '연금저축' 이
    #   들어 있지만 되묻기의 답이 아니라 새 요청이다.
    if account_in(t) or reason_in(t):
        return len(t) <= _REPLY_SLOT_LEN
    return len(t) <= _REPLY_PLAIN_LEN


def merge(original: str, reply: str) -> str:
    """원 질문 + 사용자의 답 → 자족적인 한 문장.

    원 질문을 **그대로 남긴다.** 사용자의 답으로 갈아치우면 무엇을 묻는지가
    사라지고, 요약해 다시 쓰면 검색 키워드가 날아간다.
    """
    original = (original or "").strip()
    reply = (reply or "").strip()
    if not reply:
        return original

    acct = account_in(reply)
    reason = reason_in(reply)

    if acct and acct not in original:
        head = f"{acct} 기준으로 답해 주세요."
    elif reason and reason not in original:
        head = f"{reason} 사유입니다."
    else:
        head = f"추가로 알려드립니다: {reply}"

    return f"{head} {original}"
