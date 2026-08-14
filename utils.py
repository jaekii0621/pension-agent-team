"""공용 유틸 — 로깅/트레이스/텍스트 처리.

■ 선행모델과의 차이 (요구사항: "디버깅·시각화는 단순화")
  선행모델은 노드마다 uuid 를 만들어 `log_tree` 라는 부모-자식 딕셔너리를 state 에
  누적하고, 별도 로그 서버(6969포트)로 전송해 웹 UI 에서 그래프를 그렸다.
  코드의 절반이 그 로깅이라 읽기가 힘들다.

  우리는 이렇게 단순화한다:
    · 각 노드는 `step(...)` 한 줄로 **평평한 리스트**에 이벤트를 하나 append 한다.
    · 실행이 끝나면 `render_trace()` 가 순서대로 출력한다. 서버도 uuid 도 없다.
    · 그래도 "무엇이 언제 몇 ms 걸렸고 무엇을 판단했는지"는 전부 남는다.
      → 성능 측정 단계에 필요한 건 이것뿐이다.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from typing import Any, Dict, List, Optional

import config

# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("pension")


def log_message(msg: str, level: str = "info") -> None:
    getattr(log, level, log.info)(msg)


# ==========================================================================
# 트레이스
# ==========================================================================
def step(node: str, *, summary: str = "", detail: Optional[Dict[str, Any]] = None,
         ms: int = 0, qid: str = "") -> List[dict]:
    """노드 1개의 실행 기록을 만든다.

    반환값을 그대로 state 의 `trace` 키에 넣으면 리듀서(operator.add)가 누적한다.

        return {"documents": docs, "trace": step("retrieve", summary=f"{len(docs)}건")}
    """
    if not config.WORKFLOW_SETTINGS.get("trace", True):
        return []
    ev = {
        "t": round(time.time(), 3),
        "node": node,
        "qid": qid,
        "summary": summary,
        "detail": detail or {},
        "ms": ms,
    }
    log_message(f"  ▸ {node}{f'[{qid}]' if qid else ''} — {summary}")
    return [ev]


class Timer:
    """with Timer() as t: ... → t.ms"""

    def __enter__(self) -> "Timer":
        self._t0 = time.time()
        self.ms = 0
        return self

    def __exit__(self, *exc) -> None:
        self.ms = int((time.time() - self._t0) * 1000)


def render_trace(trace: List[dict]) -> str:
    """실행 경로를 사람이 읽는 텍스트로."""
    if not trace:
        return "(트레이스 없음)"
    lines = ["┌─ 실행 경로 " + "─" * 50]
    t0 = trace[0]["t"]
    for i, ev in enumerate(trace, 1):
        rel = ev["t"] - t0
        qid = f" ({ev['qid']})" if ev.get("qid") else ""
        lines.append(f"│ {i:>2}. [{rel:6.2f}s] {ev['node']}{qid}")
        if ev.get("summary"):
            lines.append(f"│      └ {ev['summary']}")
    lines.append("└" + "─" * 62)
    return "\n".join(lines)


def render_mermaid(trace: List[dict]) -> str:
    """실행 경로를 mermaid 다이어그램 문자열로 (문서에 붙여넣기용).

    선행모델의 웹소켓 그래프 시각화를 대체하는 최소 구현이다.
    """
    if not trace:
        return "graph TD;\n  empty[트레이스 없음];"
    lines = ["graph TD;"]
    prev = None
    for i, ev in enumerate(trace):
        nid = f"n{i}"
        label = ev["node"] + (f"<br/>{ev['qid']}" if ev.get("qid") else "")
        lines.append(f'  {nid}["{label}"];')
        if prev is not None:
            lines.append(f"  {prev} --> {nid};")
        prev = nid
    return "\n".join(lines)


# ==========================================================================
# 텍스트 유틸
# ==========================================================================
_TOKEN_RE = re.compile(r"[가-힣]+|[a-zA-Z]+|\d+(?:[.,]\d+)?%?")


def tokenize_ko(text: str) -> List[str]:
    """한국어 친화 토크나이저 (형태소 분석기 의존성 없이).

    전략: 어절/영문/숫자 토큰 + **한글 2-gram**.
      "중도인출" 과 "중도 인출" 이 같은 2-gram(중도, 도인, 인출)을 공유하므로,
      띄어쓰기가 달라도 BM25 가 매칭된다. 형태소 분석기를 안 써도
      이 정도 코퍼스(494청크)에서는 충분히 잘 동작한다.
    """
    if not text:
        return []
    out: List[str] = []
    for tok in _TOKEN_RE.findall(text):
        out.append(tok.lower())
        if len(tok) > 1 and "가" <= tok[0] <= "힣":
            out.extend(tok[i:i + 2] for i in range(len(tok) - 1))
    return out


def truncate(text: str, n: int = 200) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "…"


def fmt_won(v: float) -> str:
    """숫자를 '1,234만원' 형태로. 계산기 결과 출력용."""
    v = float(v)
    if abs(v) >= 1_0000_0000:
        return f"{v / 1_0000_0000:,.2f}억원"
    if abs(v) >= 1_0000:
        return f"{v / 1_0000:,.0f}만원"
    return f"{v:,.0f}원"
