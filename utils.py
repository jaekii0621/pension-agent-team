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


# ==========================================================================
# 마크다운 표 읽기 (2026-09-02 신설)
#
# 답변이 수치를 내놓는 자리는 거의 전부 마크다운 표다. 그런데 값 대조는 표를
# **글자 흐름**으로만 보고 있었다. 그래서 두 가지가 안 보였다.
#
#   ① 셀에 `%` 를 안 붙인 맨숫자 (10차 P-E5: `| 종류A | -23.62 | 1.79 | …`)
#      → 답변 쪽에서 값 토큰이 **하나도** 안 나와 `unsupported_values` 가
#        0으로 찍혔다. "다 검증됐다" 가 아니라 "볼 것이 없었다" 였다.
#        (9차 답변은 같은 표를 `-23.62%` 로 써서 20개가 잡혔다 — 지표가
#         답변의 표기 습관에 따라 켜졌다 꺼졌다 한 것이다)
#   ② 행 단위 구조 — 어느 값들이 한 행에 같이 있었나
#      → 두 표를 한 행 안에서 섞어도 값은 전부 '어딘가에' 있으므로 통과했다.
#
# ⚠️ 결과 파일(run_queryset)은 답변의 줄바꿈을 지우고 한 줄로 적는다. 저장된
#    답변으로 사후 검증을 돌리면 표가 통째로 한 줄이 되므로, 행 경계 `| |` 를
#    줄바꿈으로 되돌려 **실행 중이든 사후든 같은 결과**가 나오게 한다.
# ==========================================================================

# 한 행의 끝 `|` 과 다음 행의 시작 `|` 이 붙어 있는 자리.
_ROW_JOIN = re.compile(r"\|[ \t]+\|")
# 구분행 `|---|---|`
_SEP_ROW = re.compile(r"^[\s|:\-–—]+$")


def md_table_rows(text: str) -> List[List[str]]:
    """마크다운 표의 행들을 셀 목록으로 돌려준다 (구분행 제외).

    표가 아닌 줄은 버린다. 셀은 양끝 공백을 지운 문자열이다.
    """
    t = _ROW_JOIN.sub("|" + chr(10) + "|", text or "")
    rows: List[List[str]] = []
    for line in t.splitlines():
        line = line.strip()
        if line.count("|") < 3 or _SEP_ROW.match(line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2:
            rows.extend(_split_trailing_head(cells))
    return rows


def _split_trailing_head(cells: List[str]) -> List[List[str]]:
    """숫자 칸 뒤에 붙은 **다음 표의 머리행**을 떼어낸다.

    줄바꿈이 지워진 답변에서는 앞 표의 마지막 행과 다음 표의 제목·머리행이
    한 줄로 붙는다. 그대로 두면 두 표가 한 표로 읽혀 표 대조가 무력해진다.

        | 종류C4 | -23.62 | … | 33.33 |  *이 정보는… | 연도 | 20.01.01~ | …
                                          └─ 여기부터는 다음 표다
    """
    nums = [i for i, c in enumerate(cells) if md_cell_number(c)]
    if not nums:
        return [cells]
    tail = cells[nums[-1] + 1:]
    if len(tail) >= 2 and not any(md_cell_number(c) for c in tail):
        return [cells[:nums[-1] + 1], tail]
    return [cells]


# 숫자만 든 셀. 부호·천단위쉼표·꼬리 `%` 는 허용하되 **소수점이 있어야** 한다.
#   소수점을 요구하는 이유: 정수 칸은 금액·좌수·연도라 백분율로 읽으면 안 된다.
#   (투자설명서 실적표의 값은 전부 소수 1~2자리다)
_NUM_CELL = re.compile(r"^[▲▼△▽+\-−]?\s*(\d{1,3}(?:,\d{3})*\.\d{1,2})\s*%?$")


def md_cell_number(cell: str) -> str:
    """셀이 순수한 소수 하나면 그 값(부호·쉼표 제거)을, 아니면 빈 문자열."""
    m = _NUM_CELL.match((cell or "").strip())
    return m.group(1).replace(",", "") if m else ""


def truncate(text: str, n: int = 200) -> str:
    """**로그 한 줄**용. 줄바꿈을 지운다."""
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "…"


def clip(text: str, n: int) -> str:
    """길이만 자르고 **줄바꿈은 남긴다** (2026-09-02 신설).

    결과 파일의 `answer` 를 `truncate()` 로 적고 있었다. 그래서 저장된 답변은
    표가 통째로 한 줄이 된다 — 사람이 읽기 어렵고, 저장된 답변으로 표 대조를
    다시 돌리면 실행 중과 **다른 결과**가 나온다.
    잘라내는 것은 그대로 두되 구조는 보존한다.
    """
    text = text or ""
    return text if not n or len(text) <= n else text[:n] + "…"


def fmt_won(v: float) -> str:
    """숫자를 '1,234만원' 형태로. 계산기 결과 출력용."""
    v = float(v)
    if abs(v) >= 1_0000_0000:
        return f"{v / 1_0000_0000:,.2f}억원"
    if abs(v) >= 1_0000:
        return f"{v / 1_0000:,.0f}만원"
    return f"{v:,.0f}원"
