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
from index import format_chunk_for_llm, to_citation
from llm import llm
from prompt import prompts
from state import RAGState
from utils import Timer, step, truncate

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
    # ⚠️ 앞뒤 경계를 **둘 다** 요구한다 (2026-08-26).
    #    예전 패턴 `C-[A-Za-z]{1,3}\d?` 는 뒷경계가 없어서 "C-P2e" 를 "C-P2" 로
    #    잘랐고, 실제 클래스가 C-P2 인 **다른 펀드**(신영마라톤)와 문자열이 우연히
    #    겹쳐 그 펀드가 P-A4 답변의 출처로 붙었다.
    #    코퍼스 실측 클래스코드: C-P · C-Pe · C-e · C-P2 · C-P2e · C-P1e · C-Pe2 ·
    #    C-P2I … 글자와 숫자가 섞이므로 통째로 잡아야 잘림이 안 생긴다.
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
_TOK_LOW = [
    re.compile(r"\d+\s*개월"),                         # 3개월
    re.compile(r"\d+\s*년"),                           # 20년
    re.compile(r"(?<!\.)\b\d{1,3}\s*%"),               # 30%
]

_AUTO_MAX = 3          # 자동 부착 상한 — 목록이 길어지면 오히려 근거가 흐려진다


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

    답변은 펀드를 축약해 부른다("미래에셋장기성장포커스 사면"). 정식 명칭 전체를
    대조하면 절대 안 맞으므로, 상품 종류를 나타내는 꼬리를 떼고 핵심 이름만 본다.
    """
    core = re.split(r"증권|투자신탁|자투자|[\[(]", name or "", 1)[0]
    return _norm(core)


def _norm(v: str) -> str:
    """'1,500만 원' 과 '1,500만원' 을 같은 것으로 본다."""
    return re.sub(r"\s+", "", v)


def _tokens(text: str):
    """(높은 특이도 집합, 낮은 특이도 집합)"""
    hi = {_norm(m.group(0)) for p in _TOK_HIGH for m in p.finditer(text or "")}
    # 등급은 어느 표기로 쓰였든 "N등급" 하나로 모은다.
    hi |= {(m.group(1) or m.group(2)) + "등급"
           for m in _TOK_GRADE.finditer(text or "")}
    lo = {_norm(m.group(0)) for p in _TOK_LOW for m in p.finditer(text or "")}
    return hi, lo - hi


def attach_citations_by_value(answer: str, ordered: List[dict],
                              cites: List[dict]):
    """답변의 값이 실제로 들어 있는 청크를 인용으로 붙인다.

    반환: (보강된 인용 목록, 어느 청크에도 없는 값 목록)
    """
    hi, lo = _tokens(answer)
    if not (hi or lo):
        return cites, []

    already = {c.get("chunk_id") for c in cites}
    nxt = max([c.get("n") or 0 for c in cites] or [0]) + 1

    # 답변이 어느 펀드를 말하는지 — 같은 수치를 가진 **다른 펀드**가 붙는 것을 막는다.
    #
    # "0.28%" 같은 보수율은 여러 펀드에 존재한다. 값만 보면 답변과 무관한 펀드가
    # 근거로 붙을 수 있다. 답변에 펀드명이 나와 있으면 그 펀드 청크를 우선한다.
    # (상품 트랙에 인용 0건이 몰려 있어서 — 31건 중 23건 — 여기가 정밀도의 관건이다)
    ans_n = _norm(answer or "")
    named = {_fund_of(ch) for ch in ordered
             if _fund_of(ch) and _fund_core(_fund_of(ch)) in ans_n}

    hits = []            # (펀드명 일치, HIGH 수, LOW 수, 청크, 일치값)
    covered = set()
    for ch in ordered:
        # ⚠️ 부분문자열이 아니라 **양쪽에서 뽑은 토큰끼리** 맞춘다 (2026-08-26).
        #    부분문자열 대조는 답변의 "0.28%" 가 청크의 "10.28%" 안에서 걸리고,
        #    "C-P2" 가 "C-P2e" 안에서 걸린다 — 둘 다 다른 펀드를 출처로 붙이는
        #    경로다. 같은 정규식으로 청크에서도 값을 뽑아 집합으로 맞추면
        #    경계가 자동으로 지켜진다.
        bhi, blo = _tokens(ch.get("text") or "")
        body_vals = bhi | blo
        mh = hi & body_vals
        ml = lo & body_vals
        covered |= mh | ml
        if ch.get("chunk_id") in already:
            continue
        fund_ok = bool(named) and _fund_of(ch) in named
        # HIGH 한 개면 충분, LOW 만이면 두 개 이상이어야 붙인다
        if mh or len(ml) >= 2:
            hits.append((1 if fund_ok else 0, len(mh), len(ml), ch,
                         sorted(mh | ml)))

    # 답변이 펀드를 특정했는데 그 펀드 청크가 하나라도 걸렸다면, 다른 펀드는 버린다
    if any(h[0] for h in hits):
        hits = [h for h in hits if h[0]]

    hits.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    for _, _, _, ch, matched in hits[:_AUTO_MAX]:
        c = to_citation(ch)
        c["n"] = nxt
        c["auto"] = True                       # 코드가 붙인 것임을 표시
        c["quote"] = "값 일치: " + ", ".join(matched[:4])
        cites.append(c)
        nxt += 1

    # 어느 청크에도 없는 값 = 근거 없이 말한 수치
    unsupported = sorted((hi | lo) - covered)
    return cites, unsupported


def _restored_cite_ok(ch: dict, answer: str, hi: set, lo: set) -> bool:
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
    if _fund_core(fund) in _norm(answer or ""):
        return True                     # 답변이 이 펀드를 직접 언급했다
    bhi, blo = _tokens(ch.get("text") or "")
    body_vals = bhi | blo
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

    user = (
        f"[사용자 원 질문]\n{original}\n\n"
        f"[지금 답해야 할 질문]\n{question}{caution_block}\n\n"
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
        hi, lo = _tokens(res.main_answer)
        dropped = 0
        for n in sorted({int(x) for x in CITE_IN_TEXT.findall(res.main_answer)}):
            i = n - 1
            if not (0 <= i < len(ordered)):
                continue
            if not _restored_cite_ok(ordered[i], res.main_answer, hi, lo):
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
    cites, unsupported = attach_citations_by_value(res.main_answer, ordered, cites)
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
                              + (f"(자동 {sum(1 for c in cites if c.get('auto'))})"
                                 if any(c.get("auto") for c in cites) else "")
                              + (f"(본문복원 {sum(1 for c in cites if c.get('restored'))})"
                                 if any(c.get("restored") for c in cites) else "")
                              + f", 근거충분={res.evidence_sufficient}"
                              + (f", 미근거값 {len(unsupported)}개" if unsupported else "")
                              + (f", 주의={caution}" if caution else ""),
                      ms=t.ms),
    }
