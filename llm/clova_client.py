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
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

import config

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


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
        self.enabled = bool(self.key)
        self._qkey = ""                       # 현재 처리 중인 질문 라벨
        if not self.enabled:
            log.warning(
                "CLOVA_API_KEY 가 비어 있습니다 — LLM 호출은 실패합니다. "
                ".env 를 채우거나 offline 모드(llm/offline.py)로 파이프라인만 점검하세요."
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
                    time.sleep(1.5 * (attempt + 1))
                    last = e
                    continue
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
        log.warning("구조화 파싱 실패 → 복구 호출 (%s / %s): %s",
                    schema.__name__, stage or "-", (text or "")[:120])

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

        raise LLMError(f"구조화 응답 파싱 실패({schema.__name__}): {text[:300]}")

    # ---------------------------------------------------------------- 계측
    def _log(self, stage: str, model: str, i: int, o: int,
             t0: float, pre: int) -> None:
        self.calls.append(CallLog(
            stage=stage, model=model, in_tok=i, out_tok=o,
            ms=int((time.time() - t0) * 1000), qkey=self._qkey, pre_tok=pre,
        ))

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
        return None

    # ① 그대로 검증
    try:
        return schema.model_validate(raw)
    except ValidationError:
        pass

    if not isinstance(raw, dict):
        return None

    # ② 껍데기 벗기기 — 키가 하나뿐이고 그 값이 dict 면 안쪽을 본다
    if len(raw) == 1:
        inner = next(iter(raw.values()))
        if isinstance(inner, dict):
            try:
                return schema.model_validate(inner)
            except ValidationError:
                pass

    # ③ 이름이 스키마명 변형인 키를 찾아 본다 (키가 여러 개인 경우)
    for k, v in raw.items():
        if isinstance(v, dict) and k.lower() in _envelope_keys(schema):
            try:
                return schema.model_validate(v)
            except ValidationError:
                continue

    return None
