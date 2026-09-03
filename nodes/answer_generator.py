"""답변 생성 노드 — 도메인 전문 에이전트 4종이 여기서 갈린다.

■ 선행모델 대응
  FinSight 의 `nodes/answer_generator.py` 자리다. 핵심 기법 두 개를 그대로 가져왔다.

  1) **답변과 인용을 한 번의 호출로 동시에 받는다.**
     답을 만든 뒤 나중에 출처를 역추적하면 정확도가 크게 떨어진다.
     구조화 출력으로 {main_answer, citations[]} 를 함께 받는다.

  2) **근거를 번호로 넘긴다.** [1] [2] 형태로 본문에 인용 번호를 박게 하고,
     번호 ↔ 청크 매핑을 코드가 들고 있다가 최종 출처 목록을 만든다.

■ 우리 도메인의 추가 설계: 전문가별 프롬프트
  라우터/분해기가 정한 `expert`(제도/세제/절차/상품)에 따라 시스템 프롬프트를 바꾼다.
  같은 근거를 줘도 전문가마다 답변 형태가 달라야 하기 때문이다.
    · 절차 전문가 → 번호 매긴 순서 + 필요서류 목록
    · 세제 전문가 → 과세 시점 구분, 연금수령 vs 연금외수령 세율 차이
    · 상품 전문가 → 표 형식 + 투자권유 아님 고지 + 낮은 신뢰도 명시

■ 소스 신뢰도 반영 (요구사항)
  근거를 프롬프트에 넣을 때 doc 문서(제도/세제/절차)를 **먼저** 배치한다.
  LLM 은 앞쪽 컨텍스트에 더 큰 가중을 두는 경향이 있어서, 순서 자체가 신뢰도 신호가 된다.
  프롬프트 규칙(GROUNDING_RULES 4항)과 이중으로 건다.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from pydantic import BaseModel, Field

import config
from index import (chunk_text_for_values, format_chunk_for_llm, fund_core,
                   fund_discriminators, to_citation)
from llm import llm
from prompt import prompts
from state import RAGState
from utils import Timer, md_cell_number, md_table_rows, step, truncate

log = logging.getLogger(__name__)

# 본문에 박힌 인용 번호. combiner.CITE_RE 와 같은 규칙이다.
CITE_IN_TEXT = re.compile(r"\[(\d{1,2})\]")


class CitationRef(BaseModel):
    n: int = Field(description="근거 번호 (프롬프트에 표시된 [n])")
    quote: str = Field(default="", description="인용한 문장 일부")


class GeneratedAnswer(BaseModel):
    main_answer: str = Field(description="마크다운 형식 답변 본문")
    citations: List[CitationRef] = Field(default_factory=list)
    evidence_sufficient: bool = Field(
        default=True, description="근거만으로 답할 수 있었으면 true")


# --------------------------------------------------------------------------
def _order_by_trust(docs: List[dict]) -> List[dict]:
    """규정문서를 앞에, 투자설명서를 뒤에 배치한다.

    LLM 은 앞쪽 컨텍스트에 더 큰 가중을 둔다. 프롬프트 규칙(GROUNDING_RULES 3번)과
    배치 순서, 두 가지로 신뢰도 우선순위를 심는다.
    """
    return sorted(docs, key=lambda d: config.TIER_TRUST.get(d.get("tier", "doc"), 1.0),
                  reverse=True)


def _build_context(docs: List[dict]) -> tuple[str, List[dict]]:
    """근거 블록 문자열과 번호↔청크 매핑을 만든다."""
    ordered = _order_by_trust(docs)
    lines = [format_chunk_for_llm(c, idx=i + 1) for i, c in enumerate(ordered)]
    return "\n\n".join(lines), ordered


def _sibling_notice(docs: List[dict]) -> str:
    """근거에 **이름이 비슷한 형제 상품**이 섞여 있으면 그 사실을 못 박는다.

    ■ 왜 프롬프트 상수가 아니라 여기서 만드나 (2026-08-31, 9차 이슈 3)

      `answer_recommend_system` 에 이미 "근거가 같은 펀드의 것인지 확인한다" 는
      일반 규칙이 있다. 그런데 P-B1 에서 안 지켜졌다 — 세 펀드의 이름이
      "…투자신탁1호(국공채)" / "(채권)" / "(주식)" 로 **끝 두세 글자만** 다르다.
      일반 규칙만으로는 "이름이 비슷한 경우" 가 지금 눈앞의 일이라는 것을 모델이
      알아채지 못한다.

      그래서 **지금 이 근거 묶음에 실제로 있는 형제만** 이름·등급·자산유형과 함께
      나열한다. 형제가 없으면 빈 문자열이라 프롬프트가 그대로다 — 대부분의
      문항에는 아무 영향이 없다(코퍼스 92개 중 형제는 11개, 5개 군뿐이다).

    ■ 지시를 "섞지 마라" 가 아니라 "빈 칸은 비워 둬라" 로 적는 이유
      P-B1 답변은 채권 펀드 행에 국공채 펀드의 보수·수익률을 그대로 옮겨 적고
      "모든 펀드가 동일한 총보수와 수익률" 이라고 단정했다. 근거가 없을 때
      **빈 칸을 만드는 것이 허용된다**는 것을 모르면 채우는 쪽으로 기운다.
    """
    groups: dict = {}
    for ch in docs:
        f = ch.get("fund") or {}
        name = f.get("name")
        if name:
            groups.setdefault(fund_core(name), {})[name] = f

    lines = []
    for members in groups.values():
        if len(members) < 2:
            continue
        for name, f in sorted(members.items()):
            bits = [f"「{name}」"]
            if f.get("risk_grade"):
                bits.append(f"위험등급 {f['risk_grade']}")
            if f.get("asset_type"):
                bits.append(f"자산유형 {f['asset_type']}")
            lines.append("  · " + " / ".join(bits))

    if not lines:
        return ""
    return (
        "\n\n[주의 — 이름이 비슷한 **서로 다른 상품**이 근거에 섞여 있다]\n"
        + "\n".join(lines)
        + "\n  이들은 이름 끝 괄호(또는 호수)만 다른 **별개 펀드**다. "
          "보수·수익률·위험등급·자산유형이 서로 다를 수 있다.\n"
          "  · 한 펀드의 수치를 다른 펀드 자리에 옮겨 적지 마라. 값이 같은지 "
          "근거로 확인하지 않은 채 \"모두 동일하다\" 고 쓰지 마라.\n"
          "  · 어떤 펀드의 값이 근거에 없으면 그 칸은 \"자료 없음\" 으로 둔다. "
          "형제 펀드의 값으로 채우면 **틀린 답**이다.\n"
          "  · 펀드를 부를 때 변별자(괄호 안 표기)를 반드시 함께 쓴다."
    )


# ==========================================================================
# 값 대조 인용 부착 (2026-08-23 신설)
#
# ■ 무엇을 고치나
#   5차 78문항에서 인용 0건이 31건(40%)이었다. 그중 7건은 검색을 24~72청크나
#   해 놓고 **보수율·금액 같은 구체 수치를 인용 없이** 답했다.
#
#       P-A3  검색 24청크 → 인용 0 → "0.28%, 0.21%, 0.46%"
#       P-F3  검색 24청크 → 인용 0 → "22.43%, 9.63%, 52.95%"
#
#   저 수치가 카드에서 온 값인지 모델이 지어낸 값인지 **구분할 방법이 없다.**
#
# ■ 왜 프롬프트로 안 고치나
#   답변 프롬프트에 인용 규칙을 이미 넣었고 08-20 에 더 강화했다. 그래도
#   31건이 안 지켰다. 그리고 그 31건은 **본문에도 [n] 이 없어서** 기존
#   '본문 번호 복원' 경로로도 못 잡는다. 지시를 더 붙여서 될 문제가 아니다.
#
# ■ 삭제한 '상위 3건 폴백'과 무엇이 다른가
#   상위 3건 폴백 : 답변과 **무관하게** 검색 상위 3건을 붙였다 → 틀린 출처
#   값 대조       : 답변이 말한 **그 값이 실제로 들어 있는 청크만** 붙인다
#
#   "0.28%" 라고 답했으면 본문에 "0.28%" 가 있는 청크를 찾는다. 없으면 안 붙이고
#   대신 **근거 없는 값**으로 기록한다 — 그게 곧 환각 신호다.
#
# ■ LLM 호출 0회다. 문자열 대조라 비용이 없다.
# ==========================================================================

# 문서와 대조할 가치가 있는 값만 뽑는다.
#   HIGH — 이 값 하나가 일치하면 그 청크가 근거일 개연성이 높다
#   LOW  — 흔해서 혼자서는 부족하다 (기간·정수 백분율)
_TOK_HIGH = [
    re.compile(r"\d+\.\d+\s*%"),                     # 0.28% · 1.152%
    re.compile(r"\d{1,3}(?:,\d{3})+\s*원"),            # 264,000원
    re.compile(r"\d[\d,]*\s*만\s*원"),                 # 1,500만 원 · 900만원
    re.compile(r"\d[\d,]*\s*억\s*원"),                 # 1억원
]

# 클래스코드는 **이름표지 값이 아니다** — 별도 계층으로 뺀다 (2026-08-31).
#
# ■ 무엇이 잘못됐었나 (9차 이슈 2, P-E5)
#   예전엔 클래스코드가 HIGH 였고, 아래 부착 조건이 `if mh or len(ml) >= 2` 라
#   **클래스코드 하나만 겹쳐도 인용이 붙었다.**
#
#   실측 P-E5: 인용 3건 전부 HIGH 교집합이 `{C-e}` 하나뿐이었다.
#       R2_KR510902511M#p47#c010  HIGH={C-e}  LOW={5년}
#       R2_KR510902511M#p46#c011  HIGH={C-e}  LOW={5년}
#       R2_KR510902511M#p4#c011   HIGH={C-e}  LOW={}
#   답변이 클래스 이름 "C-e" 를 언급했다는 것만으로 세 청크가 근거로 붙었다.
#   답변이 그 청크의 표를 읽었다는 뜻이 전혀 아니다.
#
# ■ 왜 계층을 나누나 (아예 지우지 않고)
#   클래스코드는 **변별자로는 쓸모가 있다.** 답변이 C-P2e 를 말하고 청크에도
#   C-P2e 가 있으면, 같은 값을 가진 여러 후보 중 그 청크를 고를 근거가 된다.
#   그래서 부착 **문턱에는 안 세고**, 정렬 우선순위에만 쓴다.
#
#   ⚠️ 앞뒤 경계를 **둘 다** 요구한다 (2026-08-26).
#      예전 패턴 `C-[A-Za-z]{1,3}\d?` 는 뒷경계가 없어서 "C-P2e" 를 "C-P2" 로
#      잘랐고, 실제 클래스가 C-P2 인 **다른 펀드**(신영마라톤)와 문자열이 우연히
#      겹쳐 그 펀드가 P-A4 답변의 출처로 붙었다.
#      코퍼스 실측 클래스코드: C-P · C-Pe · C-e · C-P2 · C-P2e · C-P1e · C-Pe2 ·
#      C-P2I … 글자와 숫자가 섞이므로 통째로 잡아야 잘림이 안 생긴다.
_TOK_CLASS = [
    re.compile(r"(?<![A-Za-z0-9])C-[A-Za-z0-9]{1,4}(?![A-Za-z0-9])"),  # C-P · C-P2e
    re.compile(r"(?<![A-Za-z0-9])CRP\d?(?![A-Za-z0-9])"),             # CRP
]

# 등급은 표기가 두 가지라 **뽑은 뒤 한 모양으로 맞춘다** (2026-08-28).
#   답변  "위험등급은 2등급입니다"  → 2등급
#   카드  "위험등급 2등급"          → 위험등급2   ← 예전엔 이렇게 갈렸다
# 값이 같은데 교집합이 비어서 근거 있는 등급이 '근거 없는 값' 으로 잡혔다
# (7차 P-A1 의 '2등급'·'5등급' 이 이 경우로 보인다).
# 부분문자열 대조 때는 우연히 걸렸고, 토큰 교집합으로 바꾸면서 드러났다.
_TOK_GRADE = re.compile(r"(?:위험)?등급\s*([1-6])|([1-6])\s*등급")

# ── 단위를 머리에 선언한 표 (2026-08-31 신설) ─────────────────────────
#
# ■ 무엇이 안 보였나
#   `_TOK_HIGH` 의 백분율 패턴은 **`%` 기호를 반드시 요구한다.** 그런데
#   투자설명서 실적표는 기호를 셀마다 붙이지 않고 표 머리에 한 번 선언한다.
#
#       나. 연도별 수익률 추이(단위:%)
#       종류A-e   -23.33   36.13   -30.82   19.20   33.59
#
#   그래서 청크 쪽에서 값이 **하나도 추출되지 않았다.** 답변은 "-23.33%" 로
#   쓰므로 답변 쪽에서만 토큰이 나오고, 교집합이 비어 근거 있는 값이 전부
#   '근거 없는 값' 으로 잡힌다.
#
#   실측 (2026-09-02 재측정 — `scratchpad/audit_counts.py corpus` 로 재현):
#     대상       투자설명서 청크 7,904건 (section 7,721 + 펀드카드 92 + 클래스카드 91)
#     이 규칙에 걸리는 청크  263건 (3.3%) — 재무·운용실적 164 · 요약정보 99
#     그 안의 값 맨숫자 3,951개  대  `%` 표기 148개 (26.7배)
#
#   ⚠️ 이 자리에 있던 "412건 / 241:140 / 8,322개 대 423개" 는 재현되지 않는다.
#      같은 코퍼스에 어떤 정규식을 걸어도 412 도 241:140 도 안 나온다 —
#      리터럴 `단위:%` 102 · 괄호형 284 · 같은 줄에 단위와 % 423.
#      **논지는 그대로다**: 선언형 표 안에서 맨숫자가 `%` 표기를 22~27배로
#      압도하고, 걸린 청크의 62%가 재무·운용실적에 몰려 있다.
#
#   9차 `unsupported_values` 35개 중 16개(46%)가 이 경로의 오탐이었다
#   (P-E5 15 · P-E1 1). P-E5 는 "환각 15개" 로 기록됐지만 실제로는 답변이
#   인용한 p47 표에 **그대로 있는 값**이었다.
#
# ■ 왜 좁게 거나
#   맨숫자를 아무 데서나 값으로 읽으면 조항 번호·페이지·날짜까지 걸린다.
#   **표가 스스로 "이 숫자들은 %다" 라고 밝힌 경우에만** 연다.
_UNIT_PCT = re.compile(r"단위\s*[:：]?\s*\(?\s*%")
# 앞뒤로 숫자·소수점·% 가 붙지 않은 소수. 부호는 버린다(`%` 경로와 동일).
_BARE_DEC = re.compile(r"(?<![\d.,%])-?(\d{1,3}\.\d{1,2})(?![\d.]|\s*%)")
_TOK_LOW = [
    re.compile(r"(?<!\.)\b\d{1,3}\s*%"),               # 30%
    re.compile(r"(?<!\d)(?:19|20)\d{2}\s*년(?!차)"),    # 2024년 — 연도 축 주장
]

# ── 기간 라벨은 값이 아니다 (2026-09-02 신설, 10차 이슈) ──────────────
#
# ■ 무엇이 잘못됐었나
#   `\d+\s*년` · `\d+\s*개월` 이 LOW 값이었다. 그런데 이 코퍼스에서 "최근1년"
#   "3년" "12개월" 은 **표의 축 이름**이지 답변이 주장한 값이 아니다.
#   부착 조건이 `LOW 두 개 이상` 이라 기간 라벨 두 개만 겹쳐도 인용이 붙는다.
#
#   실측 10차 P-E5 — 인용 3건의 quote 가 전부 이 경로였다.
#       [1] p47  값 일치: 1년, 2년, 3년, 4년
#       [2] p48  값 일치: 1년, 2년, 3년, 4년   ← 「자산구성 현황」 페이지다.
#       [3] p46  값 일치: 1년, 2년, 3년, 5년      차트 축에 "최근1년…" 이 있어
#                                                근거로 붙었다
#   답변이 실제로 옮겨 적은 값(35.68 · 20.87 …)이 있는 p47#c011 은 인용에 없다.
#   **근거표시가 값과 무관하게 붙은 것**이라, 붙은 인용 수가 늘어도 근거가 아니다.
#
# ■ 그래서 계층을 하나 더 뺀다 — 클래스코드와 같은 취급
#   부착 문턱에 세지 않고 동점을 가르는 정렬에만 쓴다. `unsupported_values`
#   에도 넣지 않는다 — "3년" 이 근거에 없다는 건 오류가 아니라 표기 차이다
#   (10차 32개 중 7개가 `5년`·`3년`·`30개월` 같은 기간 라벨이었다).
#
#   연도(`2024년`)는 LOW 에 남긴다. 축 이름이 아니라 **주장**이고, 9차 P-E5 의
#   진짜 오류 5건이 정확히 이 토큰으로 잡혔다.
_TOK_PERIOD = re.compile(r"(?<![\d.])\d{1,2}\s*(?:년차|개월|년)(?![\d차])")

_AUTO_MAX = 3          # 자동 부착 상한 — 목록이 길어지면 오히려 근거가 흐려진다

# 질문이 **클래스 자체를 묻고 있는가**. 클래스코드를 근거로 셀지 가르는 신호다.
#   "C-P 총보수가 얼마죠?"                     → 코드가 질문에 있다
#   "연금저축으로 사려면 어느 클래스로 사야 해요?" → 코드는 없지만 클래스 질문이다
_CLASS_TOPIC = re.compile(r"클래스|종류형|클라스")


def _class_is_topic(question: str) -> bool:
    """클래스코드를 '값'으로 셀 수 있는 질문인가 (2026-08-31).

    ■ 왜 질문을 보나 — 9차 실측으로 규칙을 정했다

      "클래스코드만 겹치면 인용을 붙이지 않는다" 를 **무조건** 적용해 보니
      9차 값대조 인용 74건 중 26건(35%)이 날아갔다. 그런데 그중 14건이
      P-C 트랙 — **클래스가 질문의 주제인 문항들**이었다.

          P-C1  "미래에셋장기성장포커스 C-P 총보수가 얼마죠?"
          P-C6  "C-F 클래스는 기관 전용이라던데, 개인은 못 사나요?"
          P-C3  "…연금저축으로 사려면 어느 클래스로 사야 해요?"

      이 문항들에서 C-P 가 적힌 청크는 **정당한 근거다.** 사용자가 그 클래스를
      물었고 그 청크가 그 클래스를 설명한다. 여기서 인용을 떼면 근거표시 점수를
      P-C 트랙 전체에서 잃는다 — 고치려던 것보다 손해가 크다.

      반대로 P-E5 는 "최근 5년 수익률 추이" 를 물었다. 답변이 클래스 이름
      `C-e` 를 언급했다는 것만으로 수익률 표 청크가 근거로 붙었다. 질문은
      클래스와 무관하다.

    ■ 그래서 기준은 "코드가 답변에 나왔나" 가 아니라 "질문이 클래스를 묻나" 다
      코드가 질문에 직접 있거나(P-C1·C2·C5·C6), 질문이 클래스를 주제로
      부르면(P-C3) 근거로 센다. 둘 다 아니면 이름표로 본다.
    """
    if not question:
        return False
    if _CLASS_TOPIC.search(question):
        return True
    return any(p.search(question) for p in _TOK_CLASS)


def _fund_of(ch: dict) -> str:
    """청크의 펀드명. 청크는 `fund={"name": ...}` 구조다 (2026-08-27).

    ⚠️ `ch.get("fund_name")` 은 **항상 None 이었다.** `fund_name` 은
    `to_citation()` 이 만드는 **인용 레코드**의 키이지 청크의 키가 아니다.
    그래서 아래 두 곳이 조용히 죽어 있었다.
      · attach_citations_by_value 의 `named` — "답변이 특정한 펀드가 아니면
        버린다"는 정밀도 가드가 한 번도 발동하지 않았다.
      · _restored_cite_ok — 모든 청크를 '규정문서'로 보고 무조건 통과시켰다.
    P-A4 에서 신영마라톤이 계속 붙은 이유가 이것이다(6차 실행에서 확인).
    """
    f = ch.get("fund")
    if isinstance(f, dict):
        return f.get("name") or ""
    return ch.get("fund_name") or ""


def _fund_core(name: str) -> str:
    """'미래에셋장기성장포커스증권자투자신탁1호(주식)' -> '미래에셋장기성장포커스'

    ⚠️ 이 값은 **형제펀드를 구분하지 못한다.** 구현은 index.loader 에 있다
       (질문 쪽 `match_known_funds` 와 같은 규칙을 써야 하기 때문 — 규칙이
       갈리면 "질문이 지목한 펀드" 와 "답변이 말한 펀드" 의 판정이 어긋난다).
       구분이 필요한 자리에서는 아래 `_named_funds()` 를 써라.
    """
    return fund_core(name)


def _norm(v: str) -> str:
    """'1,500만 원' 과 '1,500만원' 을 같은 것으로 본다."""
    return re.sub(r"\s+", "", v)


# 소수 표기 정규화용 — "0.310%" 의 앞자리/소수부/꼬리를 가른다.
_DEC = re.compile(r"^(\d[\d,]*)\.(\d+)(.*)$")


def _canon(v: str) -> str:
    """값 토큰을 **한 표기로** 맞춘다 (2026-08-31, 9차 이슈 6-a).

        0.310%  ->  0.31%      2.00%  ->  2%      1,500만 원 -> 1,500만원

    ■ 왜 필요한가
      P-E1 이 `unsupported_values: ["0.310%"]` 로 플래그됐는데 카드 실제값은
      `0.31%` 였다. **값은 맞고 자릿수 표기만 달랐다.** 문자열 교집합이라
      소수점 뒤 0 하나에 근거 있는 값이 환각으로 잡힌다.

      반대 방향으로도 샌다 — 같은 값을 다른 표기로 쓴 청크가 근거로 안 붙는다.

    ■ 왜 값에만 거나
      답변 본문 전체에 걸면 펀드명·문장까지 건드리게 된다. 대조하는 것은
      토큰이므로 토큰만 맞추면 충분하다.
    """
    v = re.sub(r"\s+", "", v)
    m = _DEC.match(v)
    if m:
        frac = m.group(2).rstrip("0")
        v = m.group(1) + (f".{frac}" if frac else "") + m.group(3)
    return v


def _md_table_values(text: str) -> set:
    """마크다운 표의 **숫자만 든 칸**을 백분율 토큰으로 (2026-09-02 신설).

    ■ 왜 필요한가 — 지표가 답변의 표기 습관에 따라 켜졌다 꺼졌다 했다
      `_UNIT_PCT` 는 청크 쪽을 열었다. 그런데 **답변 쪽은 그대로 `%` 를
      요구하고 있었다.** 답변이 표로 쓰면 셀에 `%` 를 안 붙인다.

        9차 답변  "| 종류A-e | -23.33% | 36.13% |"  → 20개 토큰 → 20개 플래그
        10차 답변 "| 종류A-e | -23.33  | 36.13  |"  → **0개 토큰** → 0개 플래그

      10차 P-E5 의 `unsupported_values: 0` 은 값이 검증됐다는 뜻이 아니라
      **볼 것이 없었다**는 뜻이다. 같은 표를 옮긴 같은 답변인데 지표가 뒤집힌다.

    ■ 왜 표 칸으로 한정하나
      본문 아무 데서나 맨숫자를 값으로 읽으면 조항 번호·페이지·배수가 걸린다.
      표 칸은 "이 자리는 값이다" 를 답변 스스로 밝힌 자리다. 소수점을 요구해
      금액·좌수·연도 칸(정수)은 들어오지 않는다 — 청크 쪽 `_BARE_DEC` 와 같은
      기준이라 양쪽이 어긋나지 않는다.
    """
    out = set()
    for row in md_table_rows(text):
        for cell in row:
            v = md_cell_number(cell)
            if v:
                out.add(_canon(v + "%"))
    return out


def value_sequence(text: str) -> List[str]:
    """텍스트에 나온 값들을 **나온 순서 그대로** 돌려준다 (표 행 대조용).

    ■ 왜 집합이 아니라 순서인가 (2026-09-02 신설, 10차 이슈)
      `unsupported_values` 는 집합이라 "이 값이 어딘가에 있는가" 만 본다.
      그래서 **한 표의 한 행을 통째로 옮겼는가** 를 못 본다.

        10차 P-E5 종류C-e 행:  33.33 · 19.33 · -30.91 · 36.00 · -23.42
        p47 원문 종류C-e 행:   -23.42 · 36.00 · -30.91 · 19.33 · (33.86)
                                                                 └ 뒤 청크로 잘렸다

      33.33 은 **바로 윗행(종류C4)의 값**이다. 값 자체는 같은 청크에 있으므로
      집합 대조는 통과한다. 순서를 보면 마지막 칸에서 끊긴다.

    투자설명서 표는 셀마다 줄이 바뀌므로 텍스트 순서가 곧 표의 행 순서다.
    부호는 버린다 — 답변이 `-23.62` 를 `23.62%(손실)` 로 쓰기도 한다.
    """
    text = text or ""
    out: List[str] = []
    if _UNIT_PCT.search(text):
        # 단위를 선언한 표 — 맨숫자도 값이다.
        pat = re.compile(r"(?<![\d.,%])-?(\d{1,3}(?:\.\d{1,2})?)\s*%?(?![\d.])")
        for m in pat.finditer(text):
            if "." in m.group(1):              # 소수만 — 정수는 좌수·금액이다
                out.append(_canon(m.group(1) + "%"))
    else:
        for m in re.finditer(r"(?<![\d.,])-?(\d{1,3}\.\d{1,2})\s*%", text):
            out.append(_canon(m.group(1) + "%"))
    return out


def _tokens(text: str):
    """(높은 특이도, 낮은 특이도, 클래스코드, 기간라벨) 네 집합.

    클래스코드와 기간라벨을 따로 돌려주는 이유는 위 주석 참고 —
    **둘 다 이름표라 부착 문턱에 세면 안 된다.**
    """
    text = text or ""
    hi = {_canon(m.group(0)) for p in _TOK_HIGH for m in p.finditer(text)}
    # 등급은 어느 표기로 쓰였든 "N등급" 하나로 모은다.
    hi |= {(m.group(1) or m.group(2)) + "등급"
           for m in _TOK_GRADE.finditer(text)}
    # 표가 "단위:%" 로 선언했으면 맨숫자도 백분율로 읽는다 — 위 주석 참고.
    if _UNIT_PCT.search(text):
        hi |= {_canon(m.group(1) + "%") for m in _BARE_DEC.finditer(text)}
    # 마크다운 표의 숫자 칸도 같은 기준으로 읽는다 (답변 쪽 대응물).
    hi |= _md_table_values(text)
    lo = {_canon(m.group(0)) for p in _TOK_LOW for m in p.finditer(text)}
    cls = {_norm(m.group(0)) for p in _TOK_CLASS for m in p.finditer(text)}
    per = {_norm(m.group(0)) for m in _TOK_PERIOD.finditer(text)}
    return hi, lo - hi, cls - hi - lo, per - hi - lo - cls


def _named_funds(ordered: List[dict], ans_n: str) -> set:
    """답변이 **실제로 말한** 펀드 정식명칭 집합 (2026-08-31, 9차 이슈 3).

    ■ 예전 판정이 형제펀드에서 무력했다
          named = {fund for fund in ... if _fund_core(fund) in ans_n}

      핵심이름만 보므로 "한국투자골드플랜연금(국공채/채권/주식)" 세 펀드가 전부
      같은 조건을 만족한다. "답변이 특정 펀드를 말했으면 그 펀드만 인용" 이라는
      정밀도 가드가 이 상품군에서는 **아예 작동하지 않았다.**
      코퍼스 실측(2026-09-02 재확인, `scratchpad/audit_counts.py funds` 로 재현):
      92개 중 11개·5개 군이 이 구조다.

          교보악사파워인덱스   1호(주식) · 2호
          미래에셋퇴직플랜     1호(주식) · 1호(채권)
          베어링고배당         투자회사(주식) · 자투자신탁(주식)   ← 변별자가 같다
          삼성클래식연금       제1호[주식] · 제1호[채권]
          한국투자골드플랜연금 1호(국공채) · 1호(채권) · 1호(주식)

      ⚠️ 괄호만 떼는 규칙으로 세면 4개·2군이 나온다. `fund_core` 는 `_FUND_TAIL`
         (`증권|투자신탁|자투자|[`) 의 **첫 일치 앞**을 자르므로 "증권" 에서 먼저
         끊긴다. 그래서 `…증권전환형자투자신탁1호(채권)` 과
         `…증권전환형투자신탁1호(국공채)` 처럼 상품형태 표기가 다른 형제도 같은
         핵심이름으로 묶인다 — 한국투자·삼성 두 군이 여기서 갈린다.

    ■ 형제군에서만 조건을 하나 더 건다
      후보 중 핵심이름이 겹치는 것이 **둘 이상일 때만**, 변별자(괄호 안 국공채·
      채권·주식, 또는 호수)까지 답변에 있어야 그 펀드를 말한 것으로 본다.
      형제가 없는 86개 펀드는 판정이 예전과 완전히 같다 — 회귀가 없다.

    ■ 아무도 특정 못 하면 빈 집합
      답변이 상품군만 부르고 변별자를 안 썼으면 어느 형제인지 확정할 수 없다.
      그때는 빈 집합을 돌려주고, 호출부는 "펀드 특정 없음" 경로로 간다
      (= 예전과 같은 보수적 동작. 특정 못 한 것을 특정했다고 우기지 않는다).
    """
    groups: dict = {}
    for ch in ordered:
        f = _fund_of(ch)
        if f:
            groups.setdefault(_fund_core(f), set()).add(f)

    named = set()
    for core, members in groups.items():
        if not core or core not in ans_n:
            continue
        # ⚠️ 핵심이름이 **다른 펀드 이름의 접두어**인 경우를 막는다 (2026-08-31).
        #    코퍼스 실측 3쌍 — '미래에셋퇴직플랜' ⊂ '미래에셋퇴직플랜단기' 등.
        #    짧은 이름의 등장이 전부 긴 이름 안이면 답변은 그 펀드를 말한 적이 없다.
        #    (둘 다 언급했으면 짧은 쪽 등장 횟수가 더 많으므로 그대로 통과한다)
        longer = [o for o in groups if o != core and core in o and o in ans_n]
        if longer and ans_n.count(core) <= sum(ans_n.count(o) for o in longer):
            continue
        if len(members) == 1:
            named |= members
            continue
        # ⚠️ 변별자가 형제끼리 겹치는 군이 있다 — 베어링고배당은 둘 다 "(주식)".
        #    그때는 상품형태까지 봐야 갈리고, 그래도 안 갈리면 아무도 안 센다
        #    (`index.loader.fund_discriminators` 주석 참고).
        ms = sorted(members)
        for m in ms:
            keys = fund_discriminators(m, ms)
            if keys and any(_norm(k) in ans_n for k in keys):
                named.add(m)
    return named


def attach_citations_by_value(answer: str, ordered: List[dict],
                              cites: List[dict], question: str = "",
                              extra_text: str = ""):
    """답변의 값이 실제로 들어 있는 청크를 인용으로 붙인다.

    `question` — 클래스코드를 근거로 셀지 가르는 데만 쓴다(`_class_is_topic`).
    안 주면 세지 않는다(보수적 기본값).

    `extra_text` — **검색이 아닌 근거**의 본문 (2026-09-02 신설).
    여기서 나온 값은 '근거 있는 값'으로 세지만 **인용은 붙지 않는다.**

      계산기 결과가 이 경우다. 10차 D-B3 은 계산 도구가 낸 중간값을 그대로
      설명했는데 12개 전부 `unsupported_values` 로 잡혔다.

          ② 근속연수공제 = 70,000,000원   ③ 환산급여 = 12,000,000원
          ⑤ 과세표준 = 1,600,000원        ⑦ 퇴직소득세 = 240,000원

      셋 다 도구가 계산한 값이라 코퍼스 청크에 있을 리가 없다. 지표가 **가장
      근거가 확실한 답변**을 최악으로 채점하고 있었다 (10차 32개 중 10개가 이
      한 문항이고, 그중 10개가 재작성이 만든 값으로 기록됐다).
      인용을 붙이지 않는 이유: 계산 결과는 출처 목록에 넣을 문서가 없다.

    반환: (보강된 인용 목록, 어느 청크에도 없는 값 목록)
    """
    hi, lo, cls, per = _tokens(answer)
    if not (hi or lo or cls):
        # 값이 하나도 없는 **서술형 답변**이다. 예전에는 여기서 그냥 끝나서
        # 인용이 붙을 길이 아예 없었다 — 어휘 대조로 한 번 더 본다.
        return (cites or _attach_by_wording(
            answer, ordered, _named_funds(ordered, _norm(answer or "")))), []

    # 질문이 클래스를 묻고 있으면 클래스코드도 특이값으로 센다 — 주석 참고.
    if _class_is_topic(question):
        hi, cls = hi | cls, set()

    already = {c.get("chunk_id") for c in cites}
    nxt = max([c.get("n") or 0 for c in cites] or [0]) + 1

    # 답변이 어느 펀드를 말하는지 — 같은 수치를 가진 **다른 펀드**가 붙는 것을 막는다.
    #
    # "0.28%" 같은 보수율은 여러 펀드에 존재한다. 값만 보면 답변과 무관한 펀드가
    # 근거로 붙을 수 있다. 답변에 펀드명이 나와 있으면 그 펀드 청크를 우선한다.
    # (상품 트랙에 인용 0건이 몰려 있어서 — 31건 중 23건 — 여기가 정밀도의 관건이다)
    # ⚠️ 형제펀드는 핵심이름이 같아 변별자까지 봐야 한다 — `_named_funds` 주석 참고.
    ans_n = _norm(answer or "")
    named = _named_funds(ordered, ans_n)

    hits = []            # (펀드명 일치, HIGH 수, 클래스 수, LOW 수, 청크, 일치값)
    # 검색 밖 근거(계산기)에서 나온 값은 처음부터 '있는 값'으로 둔다.
    ehi, elo, ecls, _eper = _tokens(extra_text) if extra_text else \
        (set(), set(), set(), set())
    covered = (hi | lo | cls) & (ehi | elo | ecls)
    for ch in ordered:
        # ⚠️ 부분문자열이 아니라 **양쪽에서 뽑은 토큰끼리** 맞춘다 (2026-08-26).
        #    부분문자열 대조는 답변의 "0.28%" 가 청크의 "10.28%" 안에서 걸리고,
        #    "C-P2" 가 "C-P2e" 안에서 걸린다 — 둘 다 다른 펀드를 출처로 붙이는
        #    경로다. 같은 정규식으로 청크에서도 값을 뽑아 집합으로 맞추면
        #    경계가 자동으로 지켜진다.
        # ⚠️ 표 연속 조각은 단위 선언을 잃는다 — 앞 조각의 선언 줄을 이어 붙인다.
        bhi, blo, bcls, bper = _tokens(chunk_text_for_values(ch))
        body_vals = bhi | blo | bcls | bper
        mh = hi & body_vals
        ml = lo & body_vals
        mc = cls & body_vals
        mp = per & body_vals
        # ⚠️ `covered` 에는 클래스코드도 넣는다. 이건 "답변이 말한 값이 근거
        #    어딘가에 있는가" 라는 별개 질문이고, 클래스코드가 근거에 실재하는데
        #    '근거 없는 값' 으로 잡히면 그게 오탐이다. 부착 문턱만 달라진다.
        covered |= mh | ml | mc
        if ch.get("chunk_id") in already:
            continue
        fund_ok = bool(named) and _fund_of(ch) in named
        # HIGH 한 개면 충분, LOW 만이면 두 개 이상이어야 붙인다.
        # ⚠️ 클래스코드·기간라벨은 문턱에 세지 않는다 — 이름표가 겹친 것은
        #    근거가 아니다. 정렬에만 쓴다(`_TOK_PERIOD` 주석의 P-E5 실측 참고).
        if mh or len(ml) >= 2:
            hits.append((1 if fund_ok else 0, len(mh), len(mc), len(ml) + len(mp),
                         ch, sorted(mh | ml) + sorted(mc | mp)))

    # 답변이 펀드를 특정했는데 그 펀드 청크가 하나라도 걸렸다면, 다른 펀드는 버린다
    if any(h[0] for h in hits):
        hits = [h for h in hits if h[0]]

    # 클래스코드는 **동점을 가르는 데만** 쓴다 (HIGH 다음, LOW 앞).
    hits.sort(key=lambda x: (-x[0], -x[1], -x[2], -x[3]))
    for _, _, _, _, ch, matched in hits[:_AUTO_MAX]:
        c = to_citation(ch)
        c["n"] = nxt
        c["auto"] = True                       # 코드가 붙인 것임을 표시
        # 값을 앞에 둔다 — quote 가 이름표(`C-e`·`3년`)로만 채워지면 무엇을
        # 근거로 붙였는지 사람이 읽을 수 없다 (10차 P-E5 의 quote 3건이 그랬다).
        c["quote"] = "값 일치: " + ", ".join(matched[:4])
        cites.append(c)
        nxt += 1

    # 값으로 아무것도 못 붙였으면 **어휘**로 한 번 더 본다 — 아래 주석 참고.
    if not cites:
        cites = _attach_by_wording(answer, ordered, named)

    # 어느 청크에도 없는 값 = 근거 없이 말한 수치.
    # 기간라벨(`per`)은 값이 아니므로 넣지 않는다 — 위 `_TOK_PERIOD` 주석 참고.
    unsupported = sorted((hi | lo | cls) - covered)
    return cites, unsupported


# 어휘 대조로 붙일 최대 건수. 값 대조와 달리 "어느 대목을 옮겼나" 라 하나면 족하다.
_WORD_MAX = 1
# 중앙값 대비 이 배수를 넘어야 붙인다. 후보 안에서 **혼자 튀어야** 근거다.
_WORD_MARGIN = 2.5
# 2등과도 이 배수만큼 벌어져야 한다. 1·2등이 붙어 있으면 어느 쪽인지 못 고른다 —
# 그때는 **아무것도 안 붙이는 것**이 맞다. 틀린 출처를 다는 것보다 낫다.
#   실측 P-C4: 1위 2.50x · 2위 2.50x 로 동점인데 정답은 4위였다 → 안 붙인다
#         P-G0b: 4.07x 대 2.93x (1.39배 차) → 붙인다. 정답 청크가 1위였다
_WORD_GAP = 1.15


def _attach_by_wording(answer: str, ordered: List[dict], named: set) -> List[dict]:
    """숫자가 없는 **서술형 답변**의 출처를 어휘로 되찾는다 (2026-09-02 신설).

    ■ 왜 필요한가 — 값 대조는 숫자가 있어야 돈다
      10차 검증 실행의 인용 0건 15건 중 6건이 "답변은 정상인데 인용만 없는" 경우였고,
      전부 숫자가 없는 사실조회였다.

          P-G0b "넣은 돈은 어디에 보관되나요"  → 답변: ㈜우리은행이 신탁업자로 보관
          P-F2  "주로 뭐에 투자해요"            → 답변: 국내 주식 60% 이상
          P-C4  "C1 클래스는 왜 가입이 안 되나" → 답변: 선행 클래스 보유 필요

      셋 다 근거 청크가 검색에 들어와 있었고 답변도 맞았다. 붙일 방법이 없었을 뿐이다.
      근거표시가 평가 지표이므로 이건 그대로 점수 손실이다.

    ■ 왜 어휘(BM25)인가 — 의미 유사도가 아니라 **전사 여부**를 묻는 것이다
      "이 답변이 이 청크의 말을 옮겼는가" 는 어휘 문제다. 의미가 비슷한 것으로
      치면 **다른 펀드의 같은 조항**이 1등으로 올라온다(투자설명서는 문서 92종이
      거의 같은 문구를 쓴다). 어휘 쪽은 고유명사·희소어에 IDF 가 실려 그 펀드
      문서를 고른다.

      실측 — 답변을 질의로 넣고 그 펀드 문서 88청크 안에서 순위를 봤다.
          P-G0b  정답(관련회사·신탁업자) 1위·2위
          P-F2   정답(투자전략 60%)      2위 (1위는 같은 문구의 별첨1)
          P-C4   정답                     4위
      검색 전체가 아니라 **이미 검색·채점을 통과한 후보 안에서만** 고르므로
      실제 조건은 이보다 좁다.

    ■ 붙이는 조건을 좁게 잡는다
      · 값 대조가 **하나도** 못 붙였을 때만 돈다 (값이 있으면 값이 우선이다)
      · 후보 중 **혼자 튈 때만** — 중앙값의 2.5배 미만이면 안 붙인다
      · 최대 1건. 답변이 어디서 왔는지 가리키는 것이지 목록을 채우는 게 아니다
      · 답변이 펀드를 특정했으면 그 펀드 청크만 본다 (값 대조와 같은 가드)
    """
    if not (config.ATTACH_CITES_BY_WORDING and ordered):
        return []
    pool = [c for c in ordered if not named or _fund_of(c) in named or not _fund_of(c)]
    if len(pool) < 3:
        return []
    try:
        from index.store import BM25
        from utils import tokenize_ko
        bm = BM25([tokenize_ko(chunk_text_for_values(c)) for c in pool])
        raw = bm.scores(tokenize_ko(answer or ""), list(range(len(pool))))
    except Exception as e:                       # 인용 부착이 답변을 막으면 안 된다
        log.warning("어휘 인용 대조 건너뜀: %s", e)
        return []
    scores = [raw.get(i, 0.0) for i in range(len(pool))]
    if not any(scores):
        return []
    ranked = sorted(range(len(pool)), key=lambda i: -scores[i])
    mid = sorted(scores)[len(scores) // 2] or 1e-9
    out: List[dict] = []
    second = scores[ranked[1]] if len(ranked) > 1 else 0.0
    for i in ranked[:_WORD_MAX]:
        if scores[i] < mid * _WORD_MARGIN:
            break
        if second and scores[i] < second * _WORD_GAP:
            break                    # 1·2등이 붙어 있다 — 고를 수 없으면 안 붙인다
        c = to_citation(pool[i])
        c["n"] = len(out) + 1
        c["auto"] = True
        c["quote"] = f"서술 근거 (어휘 일치 {scores[i] / mid:.1f}배)"
        out.append(c)
    return out


def _restored_cite_ok(ch: dict, answer: str, hi: set, lo: set,
                      named: set) -> bool:
    """본문 `[n]` 으로 복원하려는 청크가 답변과 실제로 맞물리는지 대조한다.

    ■ 왜 필요한가 (2026-08-26)
      구조화 출력 파싱이 실패하면(clova_client 의 salvage) `citations` 가 **무조건
      빈 배열**이 된다. 그러면 아래 복원 경로가 도는데, 이 경로는 모델이 본문에 쓴
      번호를 **검증 없이 검색 랭킹 순서에 그대로 매핑**했다. 모델이 매긴 번호와
      우리 랭킹이 어긋나면 답변과 무관한 청크가 출처로 붙는다.
      실측 P-A4: 답변 표에 없는 펀드(신영마라톤)가 출처에 붙었다.

    ■ 무엇을 대조하나 — 펀드카드에만 건다
      펀드카드는 "이 답변이 이 펀드를 말하고 있는가" 를 펀드명·수치로 확인할 수 있다.
      규정문서(정책 청크)는 답변이 수치 없이 서술로만 답하는 경우가 많아 같은 대조를
      걸면 정상 인용까지 날아간다. 그래서 펀드카드만 검증한다 —
      오매칭이 실제로 관측된 것도 펀드카드 쪽이다.
    """
    fund = _fund_of(ch)
    if not fund:
        return True                     # 규정문서는 그대로 통과
    if fund in named:
        return True                     # 답변이 이 펀드를 직접 언급했다
                                        # (형제펀드면 변별자까지 맞아야 한다)
    bhi, blo, _bcls, _bper = _tokens(chunk_text_for_values(ch))
    body_vals = bhi | blo               # 클래스코드는 근거로 안 센다
    if hi & body_vals:
        return True                     # 답변의 특이값이 이 카드에 실제로 있다
    return len(lo & body_vals) >= 2


def generate_answer(state: RAGState) -> dict:
    """서브질문 하나에 대한 답을 만든다."""
    question = state["question"]
    original = state.get("original_question") or question
    docs: List[dict] = state.get("documents", []) or []
    qid = state.get("qid", "")

    # 답변 프롬프트는 **근거의 성격**이 고른다. 고정 페르소나는 없다.
    # 서브질문이 추천이거나 검색 결과가 투자설명서 위주면 상품 안내 프롬프트를 쓴다.
    prospectus = sum(1 for d in docs if d.get("tier") == "prospectus")
    is_rec = bool(state.get("is_recommendation")) or (
        bool(docs) and prospectus > len(docs) / 2)
    mode = "recommend" if is_rec else "general"

    if not docs:
        return {
            "answer": "제공된 자료로는 확인되지 않습니다.",
            "citations": [],
            "evidence_ok": False,
            "trace": step("generate_answer", qid=qid, summary="근거 0건 → 미확인 응답"),
        }

    context, ordered = _build_context(docs)
    system = (prompts.answer_recommend_system if is_rec
              else prompts.answer_system)

    # 안전검사가 넘긴 주의 신호를 답변 프롬프트에 얹는다 (2026-08-18).
    #
    # 안전검사는 위법조력·유해·프롬프트공격 의심을 **차단하지 않고** 통과시킨다
    # (safety_checker.py 의 HARM_RECOVERABLE / HARM_PASS_THROUGH 참고).
    # 그 대신 여기로 신호를 넘겨서 답변 단계가 원칙을 지키며 처리하게 한다.
    # 룰이 거르고 LLM 이 다루는 분업이다 — 차단은 복구가 안 되지만 답변은 조정된다.
    caution = (state.get("safety_caution") or "").strip()
    caution_block = (
        f"\n\n[주의 신호] {prompts.caution_guidance(caution)}"
    ) if caution else ""

    # 근거에 형제펀드가 섞여 있으면 그 사실을 명시한다 (없으면 빈 문자열).
    sibling_block = _sibling_notice(ordered)

    user = (
        f"[사용자 원 질문]\n{original}\n\n"
        f"[지금 답해야 할 질문]\n{question}{caution_block}{sibling_block}\n\n"
        f"[근거]\n{context}"
    )

    # ── 구제 경로를 탔는지 표시해 둔다 (2026-08-29) ───────────────────
    #
    # ■ 무엇이 조용히 망가지고 있었나
    #   GeneratedAnswer 는 `citations` 에 default_factory=list, `evidence_sufficient`
    #   에 default=True 가 걸려 있다. 그래서 clova_client 가 구조화 파싱에 실패해
    #   원문을 main_answer 에 담아 구제하면(_salvage_text), 나머지 두 필드는
    #   **기본값으로 채워진 채 Pydantic 검증을 통과한다.**
    #
    #       citations          = []     ← 인용을 못 뽑은 게 아니라 '없다'고 기록됨
    #       evidence_sufficient = True  ← 파싱이 실패했는데 '근거 충분'이라고 기록됨
    #
    #   예외도 경고도 안 난다. 파싱 실패가 "인용 0건인 정상 답변" 으로 둔갑한다.
    #   특히 표를 요구하는 문항에서 모델이 JSON 대신 마크다운 표를 내면
    #   중괄호가 없어 복구 호출조차 건너뛴다(clova_client 의 'json_없음' 경로).
    #
    # ■ 왜 스키마에서 기본값을 빼지 않았나
    #   `citations` 를 필수로 만들면 _free_text_field 가 자유서술형 판정을 안 하게
    #   되고, 구제 경로 자체가 막혀 **파싱 실패한 서브질문이 통째로 죽는다.**
    #   그 실패는 이미 겪었다(1차 실행 6건 유실) — 그래서 구제는 그대로 두고
    #   **구제였다는 사실만 신호로 남긴다.**
    n_salvage_before = len(llm.salvages)

    with Timer() as t:
        res = llm.structured(system, user, GeneratedAnswer,
                             model=config.MODEL_ANSWER, max_tokens=2000,
                             stage=f"answer:{mode}")

    salvaged = len(llm.salvages) > n_salvage_before

    # 인용 번호 → 실제 청크
    cites: List[dict] = []
    for ref in res.citations:
        i = ref.n - 1
        if 0 <= i < len(ordered):
            c = to_citation(ordered[i])
            c["n"] = ref.n
            c["quote"] = truncate(ref.quote, 120)
            cites.append(c)

    # ── 본문에만 [n] 을 쓴 경우 거기서 인용을 되찾는다 (2026-08-20) ────
    #
    #   ⚠️ 이건 삭제한 '상위 3건 폴백'과 **다르다.**
    #        상위 3건 폴백 : 답변과 무관한 검색 결과를 갖다 붙였다 → 틀린 출처
    #        이 복원        : 모델이 **본문에 직접 쓴 번호**를 그대로 해석한다
    #                        → [2] 라고 썼으면 근거 목록의 2번을 붙인다
    #
    #   원래 문제 진단서의 (a) 경로가 이것이다. 모델이 본문에는 [1][2] 를 써 놓고
    #   citations 배열은 비워 두는 일이 잦은데, 코드가 배열만 보고 있어서
    #   "인용 0건" 으로 처리됐다. 특히 구조화 파싱이 실패해 원문을 그대로 담은
    #   경우(구제 경로)에는 배열이 반드시 비므로, 이 복원이 없으면
    #   근거표시 점수를 통째로 잃는다.
    #   ⚠️ 다만 **번호를 그대로 믿지는 않는다** (2026-08-26 보강)
    #        모델이 매긴 번호와 우리 검색 랭킹이 어긋나면 답변과 무관한 청크가
    #        붙는다(P-A4: 답변 표에 없는 펀드가 출처로). 그래서 복원 대상이
    #        펀드카드인 경우 답변이 그 펀드를 실제로 말했는지 대조한다.
    if not cites and res.main_answer:
        hi, lo, _cls, _per = _tokens(res.main_answer)
        # 클래스코드 취급을 값 대조 경로와 똑같이 맞춘다 — 질문이 클래스를
        # 묻는 문항에서만 근거로 센다(`_class_is_topic`).
        if _class_is_topic(f"{original} {question}"):
            hi |= _cls
        named = _named_funds(ordered, _norm(res.main_answer))
        dropped = 0
        for n in sorted({int(x) for x in CITE_IN_TEXT.findall(res.main_answer)}):
            i = n - 1
            if not (0 <= i < len(ordered)):
                continue
            if not _restored_cite_ok(ordered[i], res.main_answer, hi, lo, named):
                dropped += 1
                continue
            c = to_citation(ordered[i])
            c["n"] = n
            c["quote"] = ""            # 모델이 인용문을 안 줬다
            c["restored"] = True       # 본문 번호에서 복원한 것임을 표시
            cites.append(c)
        if cites or dropped:
            log.info("본문 인용 번호에서 근거 %d건 복원 (대조 불일치로 %d건 버림)",
                     len(cites), dropped)

    # ── 상위 3건 폴백을 제거했다 (2026-08-18) ─────────────────────────
    #
    # 예전 코드:
    #     if not cites and res.evidence_sufficient:
    #         cites = [dict(to_citation(c), n=i+1) for i, c in enumerate(ordered[:3])]
    #
    # 의도는 "출처 없는 답변을 내보내면 안 된다" 였는데, 결과는 **틀린 출처를
    # 만들어내는** 것이었다. 답변 내용과 무관하게 검색 결과 상위 3건을 붙였기 때문이다.
    #
    # 실측 P-A1: 답변은 DB/DC/IRP 가 무엇인지에 대한 일반 설명(문서에서 온 것이
    #            아니라 모델이 아는 상식)인데, 출처로 신한초단기채 펀드카드 하나가
    #            붙었다. 답변 내용과 완전히 무관하다.
    #
    # 대회 평가가 근거표시·인용정확도를 보므로 **출처가 없는 것보다 틀린 출처가 더
    # 나쁘다.** 인용이 없으면 없는 대로 두고 evidence_ok=False 로 내려보내
    # 검증 단계가 판단하게 한다.
    #
    # 진짜 문제(모델이 인용 표기를 빠뜨림)는 폴백으로 덮을 게 아니라
    # 프롬프트(EVIDENCE_CONVENTIONS)와 합성 후 검증(combiner)에서 다룬다.
    # ── 값 대조로 인용을 보강한다 (LLM 호출 0회) ─────────────────────
    # 질문은 클래스코드를 근거로 셀지 가르는 데만 쓴다(`_class_is_topic`).
    # 원 질문과 서브질문을 **둘 다** 본다 — 분해기가 "어느 클래스" 를 서브질문에서
    # 처음 꺼내는 경우가 있고, 원 질문에만 코드가 적힌 경우도 있다.
    cites, unsupported = attach_citations_by_value(
        res.main_answer, ordered, cites, question=f"{original} {question}")
    if unsupported:
        log.info("근거에서 못 찾은 값 %d개: %s", len(unsupported), unsupported[:5])

    return {
        "answer": res.main_answer,
        "citations": cites,
        # 답변이 말했는데 어느 근거에도 없는 값. 비어 있지 않으면 환각 후보다.
        "unsupported_values": unsupported,
        # evidence_ok 는 "근거도 충분하고 인용도 달렸나" 라는 **합성 판정**이다.
        # 그런데 인용 0건의 원인을 좁히려면 두 축을 나눠 봐야 한다.
        #   sufficient=False → 모델이 "답할 근거가 없다"고 판단   → 답변 프롬프트 문제
        #   sufficient=True + 인용 0 → 근거는 있는데 표기만 안 함 → 인용 규칙 문제
        # 고칠 곳이 정반대라, 원값을 따로 올려보낸다 (2026-08-20).
        # 구제 경로를 탔으면 evidence_sufficient 는 모델의 판단이 아니라 스키마
        # 기본값(True)이다. 판단한 적이 없으므로 '충분' 으로 세면 안 된다.
        "evidence_sufficient": bool(res.evidence_sufficient) and not salvaged,
        "evidence_ok": bool(res.evidence_sufficient and cites) and not salvaged,
        "llm_calls": 1,
        "trace": step("generate_answer", qid=qid,
                      summary=f"[{mode}] {len(res.main_answer)}자, 인용 {len(cites)}건"
                          + (" ⚠구제경로(파싱실패)" if salvaged else "")
                          + (", 형제펀드 주의 부착" if sibling_block else "")
                              + (f"(자동 {sum(1 for c in cites if c.get('auto'))})"
                                 if any(c.get("auto") for c in cites) else "")
                              + (f"(본문복원 {sum(1 for c in cites if c.get('restored'))})"
                                 if any(c.get("restored") for c in cites) else "")
                              + f", 근거충분={res.evidence_sufficient}"
                              + (f", 미근거값 {len(unsupported)}개" if unsupported else "")
                              + (f", 주의={caution}" if caution else ""),
                      ms=t.ms),
    }
