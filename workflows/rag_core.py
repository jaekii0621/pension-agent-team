"""RAG 코어 — 서브질문 **하나**를 푸는 자기교정 루프.

■ 선행모델 대응
  FinSight 의 `workflows/rag_e2e.py` 자리다. 구조를 그대로 가져왔다.

      검색 → 채점 → [분기] → 생성 → 검증 → [분기] → 끝
                      ↓ 부족               ↓ 실패
                   재작성 → 재검색       재생성 / 재검색 / 웹검색

■ 이게 왜 'Agentic RAG' 인가
  일반 RAG:  검색 → 생성 → 끝.  검색이 실패해도 그냥 답을 만든다(= 환각).
  이 파이프라인은 **검문소 2개**(문서 채점, 답변 검증)를 두고,
  실패하면 스스로 질문을 고쳐 되돌아간다. 그리고 각 검문소마다
  재시도 한도가 있어서, 넘으면 웹 검색으로 탈출한다.
  "실패를 감지하고 다른 방법을 시도한다" — 이것이 에이전트다.

■ config 로 그래프 모양이 바뀐다 (선행모델과 같은 기법)
  아래 `if config.WORKFLOW_SETTINGS[...]` 는 **그래프를 조립하는 코드**지
  실행 코드가 아니다. import 시점에 한 번 돌아 그래프 모양을 결정한다.
  → 코드 수정 없이 구성요소를 켜/꺼가며 성능을 비교할 수 있다(ablation).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

import config
import edges
import nodes
import state

# fmt: off
graph = StateGraph(state.RAGState)

# ── 노드 등록 ─────────────────────────────────────────────────────────
graph.add_node("retrieve", nodes.retrieve_documents)
graph.add_node("generate", nodes.generate_answer)
graph.add_node("rewrite", nodes.rewrite_question)

if config.WORKFLOW_SETTINGS["grade_documents"]:
    graph.add_node("grade", nodes.grade_documents)

if config.WORKFLOW_SETTINGS["check_hallucination"] or config.WORKFLOW_SETTINGS["grade_answer"]:
    graph.add_node("verify", nodes.verify_answer)

if config.WORKFLOW_SETTINGS["web_search_fallback"]:
    graph.add_node("web_search", nodes.search_web)
    graph.add_node("web_answer", nodes.generate_web_answer)
    graph.add_edge("web_search", "web_answer")
    graph.add_edge("web_answer", END)

# ── 흐름 연결 ─────────────────────────────────────────────────────────
#
# 선행모델(FinSight)은 rag_e2e 의 첫 노드가 extract_metadata 였다 — 분해 후
# 서브질문마다 필터를 다시 뽑는 구조다. 우리도 예전에는 같은 자리에 정제 노드
# (refine_filters)를 뒀는데, 2026-08 개편에서 **제거했다.**
#
#   · 필터가 scope 하나로 줄어서 정제할 축이 없다.
#   · scope 는 분해 시점에 서브질문별로 이미 결정된다(is_recommendation).
#   · 선행모델이 서브질문마다 필터를 다시 뽑을 수 있었던 건 분해 규칙에서
#     self-contained 를 강제해 회사명·연도가 문장에 남았기 때문이다. 우리도
#     같은 규칙을 분해 프롬프트에 넣었으므로, 조건이 문장에 살아 있다.
#
# → 서브질문 1개당 LLM 호출이 1회 줄었다.
graph.add_edge(START, "retrieve")

_web_target = "web_search" if config.WORKFLOW_SETTINGS["web_search_fallback"] else END

if config.WORKFLOW_SETTINGS["grade_documents"]:
    graph.add_edge("retrieve", "grade")

    if config.WORKFLOW_SETTINGS["assess_graded_documents"]:
        graph.add_conditional_edges(
            "grade",
            edges.assess_documents,
            {
                "enough": "generate",
                "rewrite": "rewrite",
                "web": _web_target,
                "give_up": "generate",     # 근거 없이도 '확인 불가' 응답은 내야 한다
            },
        )
    else:
        graph.add_edge("grade", "generate")
else:
    graph.add_edge("retrieve", "generate")

graph.add_edge("rewrite", "retrieve")

# ── 검증 루프 ─────────────────────────────────────────────────────────
if config.WORKFLOW_SETTINGS["check_hallucination"] or config.WORKFLOW_SETTINGS["grade_answer"]:
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        edges.assess_verdict,
        {
            "ok": END,
            "regenerate": "generate",   # 환각 → 같은 근거로 다시 쓴다
            "rewrite": "rewrite",       # 품질 미달 → 근거부터 다시 모은다
            "give_up": END,
        },
    )
else:
    graph.add_edge("generate", END)
# fmt: on

# recursion_limit: 재작성↔재검색 루프가 예상보다 길어질 때를 대비한 안전망.
# (한도 계산: (검색+채점+생성+검증) 4노드 × 최대 재시도 3회 + 여유)
rag_core = graph.compile()
