"""CLOVA Studio (HyperCLOVA X) 호출 클라이언트.

선행모델(FinSight)의 `llm/custom_llm.py` 에 대응한다. 역할이 같다:
  **온 시스템에서 LLM 을 부르는 통로를 이 파일 하나로 만든다.**
  노드 코드는 `from llm import llm` 만 하고, 어떤 모델이 실제로 답했는지 모른다.

선행모델과 다른 점
  · 선행모델은 OpenAI→Anthropic→Mistral→Llama 4단 폴백을 썼다.
    우리는 CLOVA 단일 공급자이므로, 폴백 대신 **재시도 + 구조화 출력 복구**에 집중한다.
  · LangChain 을 쓰지 않고 urllib 로 직접 호출한다. 의존성이 줄고,
    요청/응답 형태가 코드에 그대로 드러나 스펙이 바뀌었을 때 대응점이 명확하다.

──────────────────────────────────────────────────────────────────────────
■ 단계별 모델 분리 (2026-08)

CLOVA Studio Chat Completions v3 가 제공하는 모델은 둘이다.
    HCX-005        고성능(비전 포함)
    HCX-DASH-002   경량

우리 호출은 성격이 두 갈래로 갈린다.
    (A) 판정/분류형 — 안전성, 분해, 추천여부, 문서채점, 재작성, HyDE
        출력이 짧고 정답 공간이 좁다. 호출 수는 많다.  → HCX-DASH-002
    (B) 생성/검증형 — 답변 생성, 환각·품질 검증, 전략, 최종 합성
        출력이 길고 사실 정확도가 곧 시스템 품질이다. → HCX-005

질문 1건당 호출 10~15회 중 8~10회가 (A)에 해당하므로, 이 분리만으로 비용이
크게 준다. 모델 배정은 config.MODEL_* 에 있고 노드는 그 상수만 참조한다.

■ 토큰 계측
  1) **사후 집계(기본)** — 응답의 result.usage(promptTokens/completionTokens/
     totalTokens)를 CallLog 에 그대로 쌓는다. 추가 호출이 없어 공짜다.
  2) **사전 계산(옵션)** — CLOVA Tokenizer API
        POST /v1/api-tools/chat-tokenize/{modelName}
        요청 {"messages":[{"role","content"}]}  응답 {"result":{"messages":[{"count"}]}}
     프롬프트를 고쳤을 때 입력 토큰이 얼마나 늘지 **호출 전에** 재는 용도다.
     PENSION_PRECOUNT_TOKENS=true 로 켠다.

  질문 1건당 토큰은 `mark_question()` 으로 구간을 끊어 집계한다
  (`usage_by_question()` → 질문별 in/out/total/호출수/단계별 내역).
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar, get_args, get_origin

from pydantic import BaseModel, ValidationError

import config

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMBudgetExceeded(LLMError):
    """질문 하나가 하드 상한(config.LLM_CALL_HARD_CAP)을 넘었다.

    LLMError 를 상속하므로 기존 예외 처리에 그대로 걸린다 — 새로 잡을 필요 없다.
    """


# --------------------------------------------------------------------------
@dataclass
class CallLog:
    """호출 1건 기록. 성능 측정 단계라 토큰·지연을 전부 남긴다."""
    stage: str
    model: str
    in_tok: int
    out_tok: int
    ms: int
    ok: bool = True
    note: str = ""
    qkey: str = ""          # 어느 질문에 속한 호출인가 (질문당 토큰 집계용)
    pre_tok: int = 0        # 토크나이저 API 로 미리 잰 입력 토큰 (옵션)

    @property
    def total_tok(self) -> int:
        return self.in_tok + self.out_tok


# --------------------------------------------------------------------------
class ClovaLLM:
    """CLOVA Studio Chat Completions v3 래퍼."""

    def __init__(self) -> None:
        self.key = config.CLOVA_API_KEY
        self.base = config.CLOVA_BASE.rstrip("/")
        self.calls: List[CallLog] = []
        # 구조화 파싱이 실패해 구제 경로로 빠진 기록 (2026-08-25)
        #   인용 0건의 상당수가 여기서 나온다 — 파싱이 실패하면 citations
        #   배열이 통째로 비기 때문이다. 로그로만 남기면 결과 분석에서
        #   원인을 못 좁히므로 기록으로 남긴다.
        self.salvages: List[dict] = []
        self.enabled = bool(self.key)
        self._qkey = ""                       # 현재 처리 중인 질문 라벨
        if not self.enabled:
            log.warning(
                "CLOVA_API_KEY 가 비어 있습니다 — LLM 호출은 실패합니다. "
                ".env 에 키를 넣으세요. 키 없이 확인 가능한 것: `python run_cli.py --check` (코퍼스·벡터·검색 점검)."
            )

    # ---- CLOVA 스펙 대응 지점 -------------------------------------------
    def _endpoint(self, model: str) -> str:
        """OpenAI 와 달리 모델명이 **경로**에 들어간다."""
        return f"{self.base}/v3/chat-completions/{model}"

    def _tokenize_endpoint(self, model: str) -> str:
        return f"{self.base}{config.TOKENIZE_PATH}/{model}"

    def _headers(self, request_id: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": request_id,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }

    @staticmethod
    def _parse(data: Dict[str, Any]) -> tuple[str, int, int]:
        """응답 JSON → (본문, 입력토큰, 출력토큰)."""
        status = data.get("status", {})
        code = status.get("code")
        if code not in (None, "20000"):
            raise LLMError(f"CLOVA 오류 {code}: {status.get('message')}")
        result = data.get("result") or {}
        msg = (result.get("message") or {}).get("content", "")
        usage = result.get("usage") or {}
        return msg, usage.get("promptTokens", 0), usage.get("completionTokens", 0)
    # ---------------------------------------------------------------------

    def _request(self, url: str, payload: dict, *, timeout: float,
                 retries: int) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        rid = uuid.uuid4().hex

        last: Optional[Exception] = None
        for attempt in range(retries):
            req = urllib.request.Request(
                url, data=body, headers=self._headers(rid), method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "ignore")[:400]
                if e.code in (401, 403):
                    raise LLMError(f"CLOVA 인증 실패({e.code}). API 키/권한 확인: {detail}") from e

                if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    # 429(레이트리밋)는 5xx 와 성격이 다르다.
                    #
                    # 5xx 는 서버가 잠깐 흔들린 것이라 곧바로 다시 걸면 대개 성공한다.
                    # 429 는 **내가 너무 빨리 보내서** 난 것이라, 짧게 기다렸다 다시
                    # 보내면 같은 이유로 또 걸린다. 선형 대기(1.5s, 3.0s)로는
                    # 레이트리밋 창을 못 넘긴다 — 실측 1차 실행에서 2건이 여기서 죽었다.
                    #
                    # 그래서 429 는 지수 백오프로 늘리고, 서버가 `Retry-After` 로
                    # 정확한 대기 시간을 알려 주면 그 값을 우선한다.
                    if e.code == 429:
                        wait = _retry_after(e) or min(
                            config.LLM_BACKOFF_BASE * (2 ** attempt),
                            config.LLM_BACKOFF_MAX)
                        # 여러 요청이 같은 시각에 깨어나 다시 몰리는 것을 막는다
                        wait += random.uniform(0, 0.4 * wait)
                        log.warning("레이트리밋(429) → %.1f초 대기 후 재시도 "
                                    "(%d/%d)", wait, attempt + 1, retries)
                    else:
                        wait = 1.5 * (attempt + 1)
                    time.sleep(wait)
                    last = e
                    continue

                if e.code == 429:
                    raise LLMError(
                        f"CLOVA 레이트리밋(429) — 재시도 {retries}회 모두 소진.\n"
                        f"  요청 간격을 늘리거나(PENSION_REQUEST_GAP), "
                        f"재시도 한도를 올리세요(CLOVA_MAX_RETRY). {detail}") from e
                raise LLMError(f"CLOVA HTTP {e.code}: {detail}") from e
            except Exception as e:                      # 네트워크 오류
                last = e
                if attempt < retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
        raise LLMError(f"CLOVA 요청 실패: {last}")

    def _post(self, model: str, messages: List[dict], max_tokens: int,
              temperature: float) -> Dict[str, Any]:
        if not self.enabled:
            raise LLMError("CLOVA_API_KEY 가 설정되지 않았습니다.")
        self._guard_budget()
        payload = {
            "messages": messages,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": 0.8,
            "repetitionPenalty": 1.1,
            "includeAiFilters": True,
        }
        return self._request(self._endpoint(model), payload,
                             timeout=config.LLM_TIMEOUT, retries=config.LLM_MAX_RETRY)

    # ---------------------------------------------------------------- 토크나이저
    def count_tokens(self, messages: List[dict], *,
                     model: Optional[str] = None) -> Optional[int]:
        """CLOVA Tokenizer API 로 입력 토큰 수를 **호출 전에** 잰다.

        POST /v1/api-tools/chat-tokenize/{modelName}
        실패해도 파이프라인을 막지 않는다 — 계측은 부가 기능이다.
        """
        if not self.enabled:
            return None
        model = model or config.MODEL_PLAN
        try:
            data = self._request(self._tokenize_endpoint(model),
                                 {"messages": messages},
                                 timeout=15.0, retries=1)
            status = data.get("status", {})
            if status.get("code") not in (None, "20000"):
                log.debug("토크나이저 오류 %s: %s", status.get("code"), status.get("message"))
                return None
            msgs = (data.get("result") or {}).get("messages") or []
            return sum(int(m.get("count", 0) or 0) for m in msgs)
        except Exception as e:
            log.debug("토크나이저 호출 실패(무시): %s", e)
            return None

    def _precount(self, messages: List[dict], model: str) -> int:
        if not config.PRECOUNT_TOKENS:
            return 0
        return self.count_tokens(messages, model=model) or 0

    # ---------------------------------------------------------------- 공개 API
    def chat(self, system: str, user: str, *, model: Optional[str] = None,
             max_tokens: int = 1200, temperature: Optional[float] = None,
             stage: str = "") -> str:
        """자유 텍스트 응답."""
        model = model or config.MODEL_ANSWER
        temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        pre = self._precount(messages, model)

        t0 = time.time()
        data = self._post(model, messages, max_tokens, temperature)
        text, i, o = self._parse(data)
        self._log(stage, model, i, o, t0, pre)
        return text

    def structured(self, system: str, user: str, schema: Type[T], *,
                   model: Optional[str] = None, max_tokens: int = 1600,
                   stage: str = "") -> T:
        """Pydantic 모델로 구조화된 응답을 받는다.

        선행모델이 `llm.with_structured_output(PydanticModel)` 로 하던 것을
        CLOVA 용으로 직접 구현한 것. **에이전트가 성립하는 이유가 이 함수다** —
        LLM 출력을 코드가 if 문으로 쓸 수 있게 만들어 주기 때문.

        전략: 프롬프트로 JSON 강제 → 파싱 실패 시 1회 '고쳐 달라'고 재요청
             (선행모델의 OutputFixingParser 와 같은 발상).
             복구 호출은 형식 교정일 뿐이므로 **항상 경량 모델**로 보낸다.
        """
        model = model or config.MODEL_PLAN
        json_schema = _schema_for_prompt(schema)
        keys = ", ".join(json_schema.get("properties", {}).keys())
        strict_system = (
            f"{system}\n\n"
            "[출력 형식 — 반드시 지켜라]\n"
            "설명·인사말 없이 JSON 객체 **하나만** 출력한다.\n"
            "마크다운 코드블록(```)을 쓰지 않는다. 모든 문자열은 한국어로 쓴다.\n"
            # ★ 이 한 줄이 없으면 모델이 스키마 이름으로 한 겹 더 감싼다.
            #   실측: {"document_grade": {"relevant": "no", ...}} → 파싱 5건 전부 실패
            f"최상위 키는 정확히 [{keys}] 뿐이다. "
            "스키마 이름이나 다른 이름으로 한 번 더 감싸지 마라.\n"
            f"스키마: {json.dumps(json_schema, ensure_ascii=False)}"
        )
        messages = [{"role": "system", "content": strict_system},
                    {"role": "user", "content": user}]
        pre = self._precount(messages, model)

        t0 = time.time()
        data = self._post(model, messages, max_tokens, 0.0)
        text, i, o = self._parse(data)
        self._log(stage, model, i, o, t0, pre)

        obj = _try_parse(text, schema)
        if obj is not None:
            return obj

        # ── 1회 복구 시도 (형식 교정 → 경량 모델로 충분하다) ─────────────
        # 복구는 호출·토큰을 2배로 쓴다. 자주 돌면 프롬프트가 잘못된 것이므로
        # 조용히 넘어가지 말고 로그를 남긴다 (실측: 채점 5건 × 2회 = 10회 낭비).
        # ── 복구 호출이 무의미한 경우를 먼저 거른다 (2026-08-20) ──────
        #
        #   모델이 JSON 을 **아예 시도하지 않고** 표·산문을 그대로 낸 경우가 있다.
        #       "| 펀드명 | 최근 1년 매매회전율 |\n| --- | --- |"
        #       "요청하신 지시는 따를 수 없어, 확인된 근거를 기준으로 …"
        #
        #   이건 문법 오류가 아니라 형식 무시다. "형식만 고쳐라"고 다시 물으면
        #   모델은 답변을 처음부터 다시 쓰고(출력 토큰 2배) 대개 또 같은 모양을 낸다.
        #   실측(2차 실행): answer:recommend 에서 복구 호출 후 재실패가 반복됐다.
        #
        #   중괄호가 하나도 없으면 고칠 JSON 자체가 없다는 뜻이므로,
        #   자유 서술형 스키마에 한해 **호출 없이** 원문을 본문 필드에 담는다.
        if "{" not in (text or "") and _free_text_field(schema):
            obj = _salvage_text(text, schema)
            if obj is not None:
                log.warning("JSON 없이 본문만 반환 → 복구 호출 생략하고 원문 사용 "
                            "(%s / %s)", schema.__name__, stage or "-")
                self._salvage("json_없음", schema, stage)
                return obj

        log.warning("구조화 파싱 실패 → 복구 호출 (%s / %s): %s",
                    schema.__name__, stage or "-", (text or "")[:120])
        self._salvage("복구호출", schema, stage)

        repair_user = (
            "아래 출력이 JSON 스키마를 만족하지 않는다. 내용은 그대로 두고 "
            "형식만 고쳐서 JSON 객체 하나만 다시 출력하라.\n"
            f"최상위 키는 정확히 [{keys}] 뿐이어야 한다. 다른 이름으로 감싸지 마라.\n\n"
            f"[스키마]\n{json.dumps(json_schema, ensure_ascii=False)}\n\n"
            f"[잘못된 출력]\n{text[:2000]}"
        )
        t0 = time.time()
        data = self._post(config.MODEL_PLAN,
                          [{"role": "system", "content": "너는 JSON 형식 교정기다."},
                           {"role": "user", "content": repair_user}],
                          max_tokens, 0.0)
        text2, i2, o2 = self._parse(data)
        self._log(stage + ":repair", config.MODEL_PLAN, i2, o2, t0, 0)
        obj = _try_parse(text2, schema)
        if obj is not None:
            return obj

        # ── 최후 폴백 — 형식 실패로 **내용까지 버리지는 않는다** ──────────
        # 모델이 답을 제대로 써 놓고 JSON 모양만 틀린 경우가 있다. 여기서 예외를
        # 던지면 그 질문은 통째로 죽는다(실측 1차 실행: 6건이 이렇게 유실).
        # 자유 서술형 스키마(주 필드가 긴 str 하나)에 한해 원문을 담아 돌려준다.
        # 판정·분류형 스키마는 임의로 채우면 잘못된 판정이 되므로 그대로 실패시킨다.
        obj = _salvage_text(text, schema)
        if obj is not None:
            log.warning("구조화 파싱 최종 실패 → 원문을 %s 에 담아 진행 (%s)",
                        _free_text_field(schema), schema.__name__)
            self._salvage("최종실패", schema, stage)
            return obj

        raise LLMError(f"구조화 응답 파싱 실패({schema.__name__}): {text[:300]}")

    # ------------------------------------------------- 구조화 파싱 구제 기록
    def _salvage(self, kind: str, schema, stage: str) -> None:
        """구제 경로로 빠진 사실을 남긴다.

        kind: json_없음 | 복구호출 | 최종실패
          json_없음 — 모델이 JSON 을 아예 시도하지 않음. 복구 호출 없이 원문 사용
          복구호출  — 형식만 고쳐 달라고 1회 더 부름 (토큰 2배)
          최종실패  — 복구도 실패해 원문을 본문 필드에 담음. **citations 는 빈다**
        """
        self.salvages.append({
            "qkey": self._qkey, "kind": kind,
            "schema": getattr(schema, "__name__", str(schema)),
            "stage": stage or "-",
        })

    def salvage_events(self, qkey: Optional[str] = None) -> List[dict]:
        """질문 하나가 겪은 구제 경로 목록. qkey 를 안 주면 전체."""
        if qkey is None:
            return list(self.salvages)
        return [s for s in self.salvages if s["qkey"] == qkey]

    # ---------------------------------------------------------------- 계측
    def _log(self, stage: str, model: str, i: int, o: int,
             t0: float, pre: int) -> None:
        self.calls.append(CallLog(
            stage=stage, model=model, in_tok=i, out_tok=o,
            ms=int((time.time() - t0) * 1000), qkey=self._qkey, pre_tok=pre,
        ))

    # ---------------------------------------------------------------- 예산
    #
    # ■ 예전에는 예산이 **한 곳에서만** 검사됐다 (2026-08-18 수정)
    #
    #   config.LLM_CALL_BUDGET(=40) 을 보는 코드는 edges/decomposition.py 의
    #   should_repeat() 하나뿐이었다. 그것도 "RRR 라운드를 한 번 더 돌까?" 를
    #   정하는 순간에만 본다.
    #
    #   그런데 호출을 실제로 태우는 곳은 서브질문 fan-out **안쪽**이다.
    #   서브질문 N개가 각자 재검색·재작성·재생성 루프를 도는데, 그 루프는
    #   예산을 전혀 보지 않았다. 라운드가 끝나고 나서야 "예산 소진" 판정이 난다.
    #
    #   실측: 예산 40 인데 문항당 LLM 호출 **중앙 29 / 최대 95**.
    #         상한이 사실상 안 걸리고 있었다.
    #
    # ■ 지금은 3중으로 건다
    #
    #   ① calls_for_question()  질문당 호출 수를 실시간으로 센다 (여기)
    #   ② over_budget()         재시도 분기(edges/*)가 이걸 보고 재시도를 포기한다
    #   ③ _guard_budget()       하드 상한을 넘으면 LLMBudgetExceeded 로 즉시 끊는다
    #
    #   ①② 는 **부드러운 제동**이다 — 답변은 나오되 품질 향상 시도를 그만둔다.
    #   ③ 은 폭주 방지용 최후 수단이라 여유 있게(예산의 1.5배) 잡는다.
    def calls_for_question(self, qkey: Optional[str] = None) -> int:
        """지금 처리 중인 질문이 여기까지 쓴 LLM 호출 수."""
        k = self._qkey if qkey is None else qkey
        return sum(1 for c in self.calls if c.qkey == k)

    def over_budget(self, qkey: Optional[str] = None) -> bool:
        """소프트 예산 초과 — 재시도·품질 향상을 그만둘 시점인가."""
        return self.calls_for_question(qkey) >= config.LLM_CALL_BUDGET

    def budget_note(self, qkey: Optional[str] = None) -> str:
        """로그용 한 줄. '29/40회' 형태."""
        return f"{self.calls_for_question(qkey)}/{config.LLM_CALL_BUDGET}회"

    def _guard_budget(self) -> None:
        """하드 상한. 넘으면 호출 자체를 거부한다."""
        used = self.calls_for_question()
        if used >= config.LLM_CALL_HARD_CAP:
            raise LLMBudgetExceeded(
                f"LLM 호출 하드 상한 초과: 질문 '{self._qkey or '-'}' 가 "
                f"{used}회 호출했습니다 (상한 {config.LLM_CALL_HARD_CAP}). "
                f"PENSION_LLM_HARD_CAP 으로 조정할 수 있습니다.")

    def mark_question(self, qkey: str) -> None:
        """이후의 호출을 이 질문에 귀속시킨다.

        eval/runner 나 CLI 가 질문을 시작할 때 부른다. 질문 1건당 토큰을
        따로 뽑으려면 어디서 어디까지가 한 질문인지 표시가 필요하다.
        """
        self._qkey = qkey

    def usage_summary(self) -> dict:
        """지금까지의 호출 비용 요약. 성능 측정 리포트에 넣는다."""
        return {
            "calls": len(self.calls),
            "in_tokens": sum(c.in_tok for c in self.calls),
            "out_tokens": sum(c.out_tok for c in self.calls),
            "total_tokens": sum(c.total_tok for c in self.calls),
            "total_ms": sum(c.ms for c in self.calls),
            "by_stage": _group(self.calls, lambda c: c.stage or "-"),
            "by_model": _group(self.calls, lambda c: c.model),
        }

    def usage_for(self, qkey: str) -> dict:
        """질문 1건의 토큰 사용량."""
        calls = [c for c in self.calls if c.qkey == qkey]
        return _usage_block(calls)

    def usage_by_question(self) -> dict:
        """질문별 토큰 사용량 전체."""
        out: Dict[str, dict] = {}
        for c in self.calls:
            out.setdefault(c.qkey or "-", {"_calls": []})["_calls"].append(c)
        return {k: _usage_block(v["_calls"]) for k, v in out.items()}

    def reset_usage(self) -> None:
        self.calls.clear()
        self.salvages.clear()
        self._qkey = ""


# --------------------------------------------------------------------------
def _usage_block(calls: List[CallLog]) -> dict:
    return {
        "calls": len(calls),
        "in_tokens": sum(c.in_tok for c in calls),
        "out_tokens": sum(c.out_tok for c in calls),
        "total_tokens": sum(c.total_tok for c in calls),
        "precounted_in_tokens": sum(c.pre_tok for c in calls),
        "ms": sum(c.ms for c in calls),
        "by_stage": _group(calls, lambda c: c.stage or "-"),
        "by_model": _group(calls, lambda c: c.model),
    }


def _group(calls: List[CallLog], key) -> dict:
    out: Dict[str, dict] = {}
    for c in calls:
        d = out.setdefault(key(c), {"n": 0, "in": 0, "out": 0, "total": 0, "ms": 0})
        d["n"] += 1
        d["in"] += c.in_tok
        d["out"] += c.out_tok
        d["total"] += c.total_tok
        d["ms"] += c.ms
    return out


def _schema_for_prompt(schema: Type[T]) -> dict:
    """프롬프트에 넣을 JSON 스키마. **title 을 뺀다.**

    `model_json_schema()` 는 `"title": "DocumentGrade"` 를 포함하는데, 모델이
    이걸 "그 이름을 최상위 키로 쓰라"는 뜻으로 읽는 일이 잦다.
        실측: {"document_grade": {"relevant": "no", ...}}  ← 5건 전부 이 형태
    스키마 이름은 모델이 알 필요가 없으므로 제거한다.
    """
    js = schema.model_json_schema()
    js.pop("title", None)
    for prop in (js.get("properties") or {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    return js


def _envelope_keys(schema: Type[T]) -> set:
    """모델이 껍데기로 쓸 법한 이름들 (스키마명의 변형)."""
    n = schema.__name__
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", n).lower()
    return {n, n.lower(), snake, snake.replace("_", ""), "result", "output",
            "response", "data", "answer"}


# 따옴표 없는 키:  {sufficient: true}  →  {"sufficient": true}
#   `{` 또는 `,` 뒤에 오는 식별자만 잡는다. 문자열 값 안의 콜론은 건드리지 않는다.
_UNQUOTED_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_$][A-Za-z0-9_$\-]*)(\s*:)')
_TRAILING_COMMA_RE = re.compile(r',(\s*[}\]])')
_PY_LITERAL_RE = re.compile(r'\b(True|False|None)\b')
_PY_MAP = {"True": "true", "False": "false", "None": "null"}


def _close_truncated(s: str) -> Optional[str]:
    """중간에 잘린 JSON 을 닫아 준다.

    ■ 왜 필요한가
      max_tokens 에 걸리거나 스트리밍이 끊기면 JSON 이 **열린 채로** 끝난다.

          {"addresses_question": true, "score": 0.6, "missing": "조건에 대한 답변 부족
                                                                              ↑ 여기서 끝

      내용은 사실상 다 왔는데 닫는 따옴표·중괄호가 없어서 파싱만 실패한다.
      실측(2차 실행): QualityCheck 가 이 형태로 2번 실패했고, 그때마다 복구 호출이
      나갔다. 닫아 주기만 하면 되는 것을 LLM 에게 다시 물은 셈이다.

    ■ 원칙
      **닫기만 한다.** 값을 채우거나 지어내지 않는다. 잘린 마지막 값은 잘린 그대로
      두고(그래도 앞부분 정보는 살아 있다), 열린 괄호만 순서대로 닫는다.
      실제로 안 잘렸으면 None 을 돌려준다.
    """
    stack: List[str] = []
    in_str = esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()

    if not in_str and not stack:
        return None                       # 안 잘렸다 — 다른 이유로 실패한 것

    out = s
    if in_str:
        out += '"'                        # 열린 문자열을 닫는다
    out = out.rstrip()
    if out.endswith(","):
        out = out[:-1]                    # `{"a": 1,` 같은 꼬리 정리
    elif out.endswith(":"):
        out += " null"                    # 키만 있고 값이 안 온 경우
    for ch in reversed(stack):
        out += "}" if ch == "{" else "]"
    return out


def _repair_json(s: str) -> Optional[dict]:
    """**거의 맞는** JSON 을 고쳐서 다시 파싱한다.

    ■ 왜 필요한가 — 파싱 실패 1건 = 호출 2배

      구조화 응답이 파싱에 실패하면 "형식을 고쳐 달라"고 LLM 을 한 번 더 부른다.
      즉 실패 한 건마다 호출과 토큰이 두 배가 되고, 그 복구도 실패하면 결과를 잃는다.

      그런데 실측을 보면 실패의 상당수가 **의미는 완벽하고 문법만 살짝 어긋난** 것이다.

          {sufficient: true, missing: ""}      ← 키에 따옴표가 없다
          {"a": 1,}                            ← 후행 콤마
          {"ok": True}                         ← 파이썬 리터럴

      차단 11건 재실행에서 Sufficiency 스키마가 **7번** 이 형태로 실패했다.
      LLM 을 다시 부를 이유가 전혀 없는 것들이라, 문자열 교정으로 먼저 처리한다.

    ■ 원칙 — 값은 절대 건드리지 않는다
      키 따옴표, 후행 콤마, 파이썬 리터럴만 고친다. 홑따옴표 문자열 변환은
      하지 않는다 — 한국어 본문에 아포스트로피가 들어 있으면 값을 망가뜨린다.
      고쳐서도 안 되면 None 을 돌려주고 원래대로 복구 호출로 넘어간다.
    """
    fixed = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', s)
    fixed = _TRAILING_COMMA_RE.sub(r'\1', fixed)
    fixed = _PY_LITERAL_RE.sub(lambda m: _PY_MAP[m.group(1)], fixed)

    for cand in (fixed, _close_truncated(fixed)):
        if not cand or cand == s:
            continue
        try:
            out = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(out, dict):
            return out
    return None


# 스키마를 그대로 반향한 껍데기 키. 이 안쪽에 내용이 들어 있는 경우가 있다.
#   실측: {"$defs": {"DecomposedItem": {"question": "...", ...}}}
_SCHEMA_ECHO_KEYS = {"$defs", "definitions", "properties", "schema", "result",
                     "output", "response", "data"}


def _unwrap(raw: dict, depth: int = 3):
    """한 겹씩 벗기며 나오는 dict 후보를 전부 내놓는다 (얕은 것부터).

    예전에는 한 겹만 벗겼다. 그런데 모델이 스키마를 통째로 반향하면
    `{"$defs": {"DecomposedItem": {...}}}` 처럼 **두 겹**이 씌워진다.
    """
    yield raw
    cur = raw
    for _ in range(depth):
        if not isinstance(cur, dict) or len(cur) != 1:
            return
        inner = next(iter(cur.values()))
        if not isinstance(inner, dict):
            return
        yield inner
        cur = inner


def _try_parse(text: str, schema: Type[T]) -> Optional[T]:
    """LLM 출력 문자열에서 JSON 을 뽑아 Pydantic 으로 검증한다.

    세 가지 흔한 오염을 벗겨낸다.
      1. 마크다운 코드펜스  ```json … ```
      2. 앞뒤 잡담          "네, 결과입니다: { … } 도움이 되었길"
      3. **한 겹 더 감싼 껍데기**  {"document_grade": { … }}
         ← 이걸 못 벗기면 복구 호출(토큰 2배)이 나가고, 그것도 실패하면
           채점 자체가 무력화된다. 실측에서 문서 채점 5건이 전부 이 형태로
           실패해 '탈락 0건'(= 채점 기능 off)이 됐다.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S).strip()
    if not s.startswith("{"):
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b > a:
            s = s[a:b + 1]

    try:
        raw = json.loads(s)
    except json.JSONDecodeError:
        # 문법만 어긋난 경우를 문자열 교정으로 살린다 (LLM 재호출 없이).
        raw = _repair_json(s)
        if raw is None:
            return None
        log.debug("JSON 문법 교정으로 파싱 성공 (%s)", schema.__name__)

    # ── 후보를 순서대로 시험한다 ────────────────────────────────────
    #
    #   ⚠️ "검증 통과 = 정답" 이 아니다. 모델이 **스키마를 반향**하면
    #      ({"$defs": {...}}) 알려진 키가 하나도 없는데도 검증은 통과한다 —
    #      필드에 기본값이 있기 때문이다. 그러면 "서브질문 0건" 같은 **빈 결과**가
    #      정상 응답인 척 흘러가고, 분해가 통째로 사라진다(실측 D-A2).
    #
    #      그래서 통과한 후보라도 _is_vacuous 면 보관만 하고 다음 후보를 계속 본다.
    #      끝까지 더 나은 것이 없으면 그때 보관해 둔 것을 돌려준다 —
    #      복구 호출(토큰 2배)을 아끼는 쪽이 낫고, 빈 결과에는 폴백 경로가 있다.
    best_empty: Optional[T] = None

    def _accept(cand) -> Optional[T]:
        nonlocal best_empty
        if cand is None:
            return None
        try:
            obj = schema.model_validate(cand)
        except ValidationError:
            return None
        if _is_vacuous(obj, cand if isinstance(cand, dict) else {}, schema):
            if best_empty is None:
                best_empty = obj
            return None
        return obj

    # ① 그대로
    got = _accept(raw)
    if got is not None:
        return got

    if not isinstance(raw, dict):
        return best_empty

    # ② 타입 어긋남 교정 — 값은 맞는데 **모양만** 다른 경우
    got = _accept(_coerce_shapes(raw, schema))
    if got is not None:
        return got

    # ③ 껍데기 벗기기 — 한 겹씩 들어가며 본다 (스키마 반향은 두 겹인 경우가 있다)
    for cand0 in _unwrap(raw):
        for cand in (cand0, _coerce_shapes(cand0, schema),
                     _wrap_as_item(cand0, schema)):
            got = _accept(cand)
            if got is not None:
                return got

    # ④ 이름이 스키마명 변형인 키를 찾아 본다 (키가 여러 개인 경우)
    for k, v in raw.items():
        if isinstance(v, dict) and k.lower() in _envelope_keys(schema):
            for cand in (v, _coerce_shapes(v, schema), _wrap_as_item(v, schema)):
                got = _accept(cand)
                if got is not None:
                    return got

    return best_empty


def _retry_after(e: "urllib.error.HTTPError") -> Optional[float]:
    """`Retry-After` 헤더를 초 단위로 읽는다. 서버가 알려 준 값이 가장 정확하다."""
    try:
        v = e.headers.get("Retry-After")
        if not v:
            return None
        return max(0.0, float(str(v).strip()))
    except Exception:                               # 날짜 형식 등은 무시
        return None


def _free_text_field(schema: Type[BaseModel]) -> Optional[str]:
    """이 스키마가 '자유 서술형'인가 — 그렇다면 본문이 들어갈 필드 이름.

    조건: 기본값이 없는 `str` 필드가 **정확히 하나**이고, 나머지는 전부 기본값이
    있다. 답변 생성처럼 "긴 글 하나 + 부가정보" 형태가 여기 해당한다.
    판정·분류형(필수 bool 이 있는 등)은 해당하지 않으므로 None 을 돌려준다.
    """
    required_str: List[str] = []
    for name, f in schema.model_fields.items():
        if f.is_required():
            if f.annotation is not str:
                return None                      # 필수 비-문자열 → 판정형이다
            required_str.append(name)
    return required_str[0] if len(required_str) == 1 else None


def _salvage_text(text: str, schema: Type[BaseModel]) -> Optional[BaseModel]:
    """파싱이 끝내 실패했을 때, 원문을 자유 서술형 필드에 담아 구제한다."""
    field_name = _free_text_field(schema)
    if not field_name:
        return None
    body = (text or "").strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.S).strip()
    if not body:
        return None
    try:
        return schema.model_validate({field_name: body})
    except ValidationError:
        return None


def _is_vacuous(obj: BaseModel, raw: dict, schema: Type[BaseModel]) -> bool:
    """검증은 통과했지만 **알맹이가 비었나** — 그리고 그게 수상한가.

    두 조건이 모두 맞을 때만 True 다.
      · 모든 필드가 기본값 그대로다 (빈 문자열·빈 리스트·None)
      · 그런데 입력 dict 에는 스키마에 없는 키가 있었다
        → 모델이 다른 모양으로 답했고, 기본값 덕분에 통과만 한 것이다

    두 번째 조건이 중요하다. 모델이 정말로 "서브질문 없음"을 뜻해서 빈 배열을
    보낸 경우({"sub_questions": []})는 정상이므로 건드리면 안 된다.
    """
    if not isinstance(raw, dict):
        return False
    known = set(schema.model_fields)
    if not (set(raw) - known):          # 모르는 키가 없다 → 정상 응답
        return False
    for name in known:
        v = getattr(obj, name, None)
        if v not in (None, "", [], {}, 0, False):
            return False
    return True


def _wrap_as_item(raw: dict, schema: Type[BaseModel]) -> Optional[dict]:
    """항목 하나를 통째로 내놓은 경우, 그것을 리스트에 담아 준다.

    실측: Decomposition(sub_questions: List[DecomposedItem]) 을 요구했는데
          모델이 DecomposedItem **하나**를 그대로 내놓았다
          ({"question": "...", "product_intent": "..."}).
          리스트 필드가 하나뿐인 스키마에서는 의도가 명확하므로 담아 준다.
    """
    list_fields = [(n, f) for n, f in schema.model_fields.items()
                   if get_origin(f.annotation) in (list, List)]
    if len(list_fields) != 1:
        return None
    name, field = list_fields[0]
    args = get_args(field.annotation)
    if not args:
        return None
    item_cls = args[0]
    if not (isinstance(item_cls, type) and issubclass(item_cls, BaseModel)):
        return None
    if not (set(raw) & set(item_cls.model_fields)):
        return None
    return {name: [raw]}


def _coerce_shapes(raw: dict, schema: Type[BaseModel]) -> dict:
    """선언된 타입에 맞게 **값의 모양만** 바꾼다. 내용은 건드리지 않는다.

    ■ 왜 필요한가
      LLM 은 내용을 맞게 만들어 놓고 모양을 자주 틀린다. 특히 답변에 표를 넣으라고
      지시하면 문자열 하나 대신 **행 배열**을 내놓는다. 지금까지는 그때마다 복구
      호출을 한 번 더 쓰고, 그것도 실패하면 **답변 전체를 버렸다.**

      실측(78문항 1차 실행): 파싱 실패 6건 중
        · `main_answer` 를 리스트로 반환      3건
        · `fund_name` 을 문자열로 반환        1건
      전부 "내용은 멀쩡한데 모양만 틀린" 경우였다.

    ■ 규칙 — 정보를 잃지 않는 변환만 한다
        str  자리에 list  → 줄바꿈으로 잇는다      (표·목록이 그대로 살아난다)
        str  자리에 dict  → JSON 문자열로 남긴다   (버리는 것보다 낫다)
        str  자리에 숫자·bool → str()
        list 자리에 스칼라 → [스칼라]              (단수/복수 혼동)
        list 자리에 None  → []

      값을 **해석하거나 지어내지 않는다.** 모양만 맞춰 준다.
    """
    out = dict(raw)
    for name, field in schema.model_fields.items():
        if name not in out:
            continue
        v = out[name]
        ann = field.annotation
        origin = get_origin(ann)

        # --- List[...] 자리 ---
        if origin in (list, List):
            if v is None:
                out[name] = []
            elif not isinstance(v, list):
                out[name] = [v]
            else:
                # List[모델] 자리에 **문자열 배열**을 준 경우.
                #   실측: {"sub_questions": ["질문1", "질문2"]}
                #         (DecomposedItem 객체 배열이어야 한다)
                # 항목의 첫 필드에 문자열을 넣어 준다 — 나머지는 기본값이 채운다.
                args = get_args(ann)
                item_cls = args[0] if args else None
                if (isinstance(item_cls, type) and issubclass(item_cls, BaseModel)
                        and any(isinstance(x, str) for x in v)):
                    key = next(iter(item_cls.model_fields))
                    out[name] = [{key: x} if isinstance(x, str) else x for x in v]
            continue

        # --- str 자리 ---
        if ann is str:
            if isinstance(v, list):
                parts = []
                for x in v:
                    parts.append(x if isinstance(x, str)
                                 else json.dumps(x, ensure_ascii=False))
                out[name] = "\n".join(parts)
            elif isinstance(v, dict):
                out[name] = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, (int, float, bool)):
                out[name] = str(v)
    return out
