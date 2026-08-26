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
def _force_utf8_stdio() -> None:
    """Windows 콘솔(cp949)에서 한글·기호가 UnicodeEncodeError 로 죽는 것을 막는다.

    기본 인코딩이 cp949 라 "⚠", "→", "▸" 같은 기호를 print 하는 순간 프로세스가
    죽는다. 로그·트레이스 전부가 한국어라 사실상 모든 실행에서 터진다.
    (errors="replace" 까지 두는 이유: 재설정이 안 되는 환경에서도 죽지는 않게)
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


_force_utf8_stdio()

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
# 토큰 패턴 — **순서가 중요하다.** 앞에 있는 대안이 먼저 먹는다.
#
#  ① 영문/숫자 코드 : 하이픈으로 이어진 덩어리를 통째로 잡는다  (C-P2e, C-P-I, A-e)
#  ② 한글 어절
#  ③ 남은 영문
#  ④ 숫자(소수·퍼센트 포함)
_TOKEN_RE = re.compile(
    r"[A-Za-z]+\d*(?:-[A-Za-z0-9]+)+"     # ① C-P2e, C-P-I, A-e, C-P2
    r"|[가-힣]+"                           # ② 한글
    r"|[A-Za-z]+\d*"                       # ③ TDF, ETF, C1, KOSPI200
    r"|\d+(?:[.,]\d+)?%?"                  # ④ 16.5%, 900
)

# 코드 표기 변형을 하나로 묶기 위한 정규형: 하이픈·공백 제거 + 소문자
_CODE_STRIP_RE = re.compile(r"[-\s]+")


def tokenize_ko(text: str) -> List[str]:
    """한국어 친화 토크나이저 (형태소 분석기 의존성 없이).

    ■ 전략 1 — 한글 2-gram
      "중도인출" 과 "중도 인출" 이 같은 2-gram(중도, 도인, 인출)을 공유하므로,
      띄어쓰기가 달라도 BM25 가 매칭된다.

    ■ 전략 2 — 클래스 코드를 부수지 않는다 (2026-08 수정)
      예전 패턴은 `[a-zA-Z]+` 라서 하이픈에서 끊겼다. 실측하면 이랬다.

          'C-P2e' → ['c', 'p', '2', 'e']
          'C-e'   → ['c', 'e']          ← 'Ce' → ['ce'] 와 아예 다른 토큰

      즉 "C-P2e 로 살 수 있어?" 라고 물으면 BM25 가 알파벳 c/p/e 가 든 아무 청크나
      끌어왔다. "BM25 는 코드에 강하다"는 전제가 정반대로 깨져 있었던 것이다.

      운용사마다 표기가 24종으로 갈리는데(C-P/C-p, Ce/C-e/C-E) **데이터는 각 문서의
      정본이라 정규화하지 않는다**는 것이 코퍼스 쪽 방침이다. 그래서 질의·문서
      양쪽에 같은 토크나이저를 태워 여기서 묶는다.

          'C-P2e' → ['c-p2e', 'cp2e']    ← 원형 + 하이픈 제거 정규형
          'Ce'    → ['ce']
          'C-e'   → ['c-e', 'ce']        ← 정규형에서 'Ce' 와 만난다
    """
    if not text:
        return []
    out: List[str] = []
    for tok in _TOKEN_RE.findall(text):
        low = tok.lower()
        out.append(low)

        # 한글 어절 → 2-gram 확장
        if len(tok) > 1 and "가" <= tok[0] <= "힣":
            out.extend(tok[i:i + 2] for i in range(len(tok) - 1))
            continue

        # 코드 토큰 → 하이픈 제거 정규형을 함께 넣는다 (C-e ↔ Ce 를 잇는다)
        if "-" in low:
            norm = _CODE_STRIP_RE.sub("", low)
            if norm and norm != low:
                out.append(norm)
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
