"""최종 합성 노드 — 서브답변·계산·전략을 하나의 답변으로 묶고 출처를 붙인다.

■ 선행모델 대응
  FinSight 의 `combine_answer_v3` + `append_citations` 자리다.

■ 인용 번호 재매핑 (여기가 까다롭다)
  서브질문마다 독립적으로 답을 만들었으므로, 각 답변 안의 [1] [2] 는
  **그 서브질문의 근거 목록 기준**이다. 세 개의 서브답변을 이어붙이면
  [1] 이 세 번 나오고 각각 다른 문서를 가리킨다.

  해결: 합치기 전에 전역 번호로 다시 매긴다.
    서브답변1의 [1][2] → [1][2]
    서브답변2의 [1][2] → [3][4]
    ...
  본문 문자열을 정규식으로 치환하고, 최종 출처 목록을 그 순서로 만든다.
  (선행모델은 이 문제를 '최종 답변에 다시 LLM 을 태워 인용을 재삽입'하는 방식으로
   풀었는데, 그러면 인용이 원래 문장과 어긋나는 일이 생긴다. 문자열 치환이 안전하다.)
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import config
from llm import llm
from prompt import prompts
from state import OverallState
from utils import Timer, step

CITE_RE = re.compile(r"\[(\d{1,2})\]")


def _renumber(qa_pairs: List[dict]) -> Tuple[List[dict], List[dict]]:
    """서브답변별 지역 인용번호를 전역 번호로 다시 매긴다."""
    global_cites: List[dict] = []
    key_to_global: Dict[str, int] = {}
    rebuilt: List[dict] = []

    for pair in qa_pairs:
        local = {c.get("n"): c for c in (pair.get("citations") or []) if c.get("n")}
        mapping: Dict[int, int] = {}

        for n, c in sorted(local.items()):
            key = c.get("chunk_id") or c.get("url") or c.get("snippet", "")[:60]
            if key not in key_to_global:
                global_cites.append(dict(c))
                key_to_global[key] = len(global_cites)
                global_cites[-1]["n"] = key_to_global[key]
            mapping[n] = key_to_global[key]

        text = pair.get("answer") or ""
        # 한 번에 치환해야 한다. 순차 치환하면 [1]→[3] 후 [3]→[5] 처럼 연쇄된다.
        text = CITE_RE.sub(
            lambda m: f"[{mapping.get(int(m.group(1)), int(m.group(1)))}]", text)

        rebuilt.append({**pair, "answer": text})

    return rebuilt, global_cites


def _format_sources(cites: List[dict]) -> str:
    """출처 목록 마크다운."""
    if not cites:
        return ""
    lines = ["", "---", "### 출처"]
    for c in cites:
        if c.get("source") == "웹검색":
            lines.append(f"[{c['n']}] (웹) {c.get('doc_id') or ''} — {c.get('url', '')}")
            continue
        bits = [f"소스: {c.get('source')}"]
        if c.get("fund_name"):
            bits.append(c["fund_name"])
        if c.get("section"):
            bits.append(f"「{c['section']}」")
        loc = f"{c.get('doc_id')}"
        if c.get("page"):
            loc += f" p.{c['page']}"
        lines.append(f"[{c['n']}] {' / '.join(bits)} — {loc}")
    return "\n".join(lines)


def combine_answer(state: OverallState) -> dict:
    """모든 결과를 하나의 최종 답변으로."""
    question = state["question"]
    qa_pairs: List[dict] = state.get("qa_pairs", []) or []
    calc = state.get("calc_result") or {}
    strategy = state.get("strategy_note")

    if not qa_pairs and not calc and not strategy:
        return {
            "final_answer": "제공된 자료로는 질문에 답할 근거를 찾지 못했습니다.",
            "trace": step("combine_answer", summary="결과 없음"),
        }

    rebuilt, global_cites = _renumber(qa_pairs)

    # ── LLM 으로 최종 답변을 구성한다 (선행모델의 combine_answer_v3 와 동일) ──
    #
    # 서브답변이 1건일 때도 LLM 을 태운다. 서브답변은 "그 서브질문에 대한 답"이지
    # "사용자 질문에 대한 답"이 아니기 때문이다. 예를 들어 사용자가
    # "IRP 중도인출 사유가 뭔가요?" 라고 물었는데 서브답변이 표 행을 나열만 한
    # 상태라면, 합성 단계에서 결론 → 조건 → 주의사항 순으로 재구성해야 답이 된다.
    # LLM 호출 1회가 늘지만 답변 형태의 일관성이 훨씬 중요하다.
    blocks = [f"[서브질문 {i + 1}] {p['question']}\n{p['answer']}"
              for i, p in enumerate(rebuilt)]
    if calc.get("explanation"):
        blocks.append(f"[계산 결과 설명]\n{calc['explanation']}")
    if calc.get("message"):                    # 입력 부족 안내
        blocks.append(f"[계산 안내]\n{calc['message']}")
    if strategy:
        blocks.append(f"[전략 제안]\n{strategy}")

    # 어떤 인용 번호가 유효한지 알려 준다 — 모델이 없는 번호를 만들어내지 않도록.
    valid_ns = ", ".join(f"[{c['n']}]" for c in global_cites) or "(없음)"

    with Timer() as t:
        body = llm.chat(
            prompts.combine_system,
            f"[사용자 질문]\n{question}\n\n"
            f"[사용 가능한 근거 번호] {valid_ns}\n"
            f"위 번호만 쓸 수 있다. 새 번호를 만들지 마라.\n\n"
            + "\n\n".join(blocks),
            model=config.MODEL_ANSWER, max_tokens=2500, stage="combine")
    used_llm = 1
    ms = t.ms

    # 합성 결과가 비었으면(모델 실패) 서브답변을 그대로 이어붙여 폴백한다.
    # 답변을 못 내는 것보다 정돈이 덜 된 답변이 낫다.
    if not (body or "").strip():
        body = "\n\n".join(p["answer"] for p in rebuilt if p.get("answer"))
        used_llm = 1

    # 투자설명서 근거가 섞였으면 고지 문구를 보강한다.
    # ★ corpus 가 아니라 tier 로 본다 — doc33/53/57/58 은 원본 source 가 "상품"이지만
    #   실제로는 규정문서라, corpus 로 판단하면 엉뚱하게 투자권유 고지가 붙는다.
    has_product = any(c.get("tier") == "prospectus" for c in global_cites)
    has_doc = any(c.get("tier") == "doc" for c in global_cites)
    tail = ""
    if has_product and config.PRODUCT_DISCLAIMER not in body:
        tail += f"\n\n{config.PRODUCT_DISCLAIMER}"
    if has_product and has_doc:
        tail += f"\n{config.SOURCE_CONFLICT_NOTE}"

    final = body + tail + _format_sources(global_cites)

    return {
        "final_answer": final,
        "citations": global_cites,
        "llm_calls": used_llm,
        "trace": step("combine_answer",
                      summary=f"LLM 합성 — 서브답변 {len(rebuilt)}건 + 출처 "
                              f"{len(global_cites)}건 → {len(final)}자",
                      ms=ms),
    }


def out_of_scope_answer(state: OverallState) -> dict:
    """도메인 밖 질문에 대한 고정 응답 (LLM 호출 없음)."""
    return {
        "final_answer": prompts.out_of_scope_message,
        "citations": [],
        "trace": step("out_of_scope", summary="범위 밖 안내 반환"),
    }
