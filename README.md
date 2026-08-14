# 연금 에이전트 (Pension Agentic RAG)

퇴직연금·개인연금 문서(제도·세제·절차·상품)에 대한 멀티에이전트 RAG 시스템.
선행모델 **FinSight (Inter-IIT Tech Meet 13.0)** 의 구조를 연금 도메인과 CLOVA Studio 환경에 맞게 재설계했다.

> 현재 단계는 **서버 배포가 아니라 성능 측정**이다. 그래서 서버/이중화/시맨틱캐시/자동완성은 없고,
> 대신 `eval/` 이 사실상 주인공이다.

---

## 1. 30초 요약

```
질문 → 안전성 검사 → 질의 분석(소스·주제 필터) → 라우터(경로·전문가 배정)
     → 질문 분리 → [서브질문 병렬] 검색 → 채점 → 답변 → 검증 → [실패 시 재작성/웹]
     → 취합 → 충분? (아니면 잔여 재분해, 최대 2라운드)
     → 계산기 / 전략 → 최종 합성 + 인용 → 답변
```

에이전트 종류

| 에이전트 | 파일 | 역할 |
|---|---|---|
| 질문 분리 | `nodes/question_decomposer.py` | 복합 질문을 검색 가능한 단위로 쪼갬 + 잔여 재분해(RRR) |
| 필터 정제 | `nodes/filter_refiner.py` | 서브질문별로 검색 필터를 다듬음 (전역값 상속 + 지역 덮어쓰기) |
| 라우터 | `nodes/router.py` | 처리 경로(rag/calc/recommend) + 도메인 전문가 배정 |
| 제도 / 세제 / 일반절차 / 중도인출절차 / 상품 | `nodes/answer_generator.py` | 소스별 전문 프롬프트로 답변 생성 |
| 검증 | `nodes/verifier.py` | 환각 검사 + 품질 채점 (HHEM 미사용, LLM 1회로 통합) |
| 계산기 | `nodes/calculator.py` + `pension_tools.py` | 연금 특화 계산 6종 (**계산은 코드가 한다**) |
| 전략 | `nodes/strategy.py` | 검증된 사실만으로 실행 계획 수립 |

---

## 2. 설치와 실행

```bash
pip install -r requirements.txt
copy .env.example .env        # 값 채우기 (키는 나중에 넣어도 됨)
```

```bash
python run_cli.py --check                      # 데이터·설정 점검 (API 키 불필요)
```

```bash
python run_cli.py "IRP 중도인출 사유가 뭔가요?"
```

```bash
python -m eval.runner --retrieval-only         # 검색 성능만 (LLM 호출 0회)
```

### API 키가 아직 없을 때 — 드라이런

`CLOVA_API_KEY` 가 비어 있으면 자동으로 **드라이런 모드**로 돈다 (`llm/offline.py`).
LLM 이 할 판단을 규칙으로 대신해서 다음을 무료로 검증한다.

- 노드 연결·분기가 의도대로 도는가
- `Send` 병렬 fan-out 과 리듀서 병합이 맞는가
- 검색 필터·소스 가중치·인용 번호 재매핑이 맞는가
- 재시도 한도가 무한루프를 막는가

검증할 수 **없는** 것: 답변 품질. 그건 실제 키로만 측정한다.

### 키를 받은 뒤

```bash
python -m index.build_index --probe            # 임베딩 엔드포인트 확인
```

```bash
python -m index.build_index                    # 494청크 임베딩 → cache/
```

---

## 3. 폴더 구조

```
pension_agent/
├── config.py            ★ 스위치판. WORKFLOW_SETTINGS 를 바꾸면 그래프 모양이 바뀐다
├── state.py             ★ 상자 설계도 (State + 리듀서)
├── prompt.py               프롬프트 전부
├── utils.py                로깅·트레이스 (단순화됨)
│
├── llm/                    CLOVA Studio 호출 통로
│   ├── clova_client.py     Chat Completions + 구조화 출력
│   ├── embeddings.py       clir-sts-dolphin + 디스크 캐시
│   └── offline.py          드라이런
│
├── index/                  검색 계층 (인메모리)
│   ├── loader.py           JSONL → 통일 스키마 + 태그 파생
│   ├── store.py            하이브리드 검색 + 필터 + 소스 신뢰도
│   └── build_index.py      임베딩 사전 계산
│
├── nodes/                  에이전트(작업대) 14개
├── edges/                  분기 함수 6개
├── workflows/
│   ├── rag_core.py         서브질문 1개용 자기교정 루프
│   └── main_graph.py       최상위 그래프
│
├── eval/                ★ 성능 측정 (현 단계의 목적)
│   ├── metrics.py          검색·정확도·비용 지표
│   ├── goldset.jsonl       실제 chunk_id 기반 정답셋 12건
│   └── runner.py           실행기
│
├── data/tax_tables.json    세율표 (코드가 읽음, LLM 아님)
└── docs/PIPELINE.md     ★ 질문이 지나가는 전 과정 상세 설명
```

---

## 4. 데이터

| 파일 | source | 건수 | 신뢰도 |
|---|---|---:|---|
| `chunks_genre_a.jsonl` | 제도 | 90 | 규정 문서 (높음) |
| `chunks_genre_b.jsonl` | 세제 | 151 | 규정 문서 (높음) |
| `chunks_genre_c.jsonl` | **중도인출절차** | 20 | 규정 문서 (높음) |
| `chunks_genre_d.jsonl` | **일반절차** | 141 | 규정 문서 (높음) |
| `fund_cards.jsonl` | 상품 | 92 | 투자설명서 (낮음) |

> 원본 JSONL 은 genre_c/genre_d 를 모두 `source="절차"` 로 표기한다.
> 두 파일의 성격이 완전히 달라(중도인출 사유·서류 vs 부담금 납입·지급) **파일 단위로 소스를 분리**했다.

**총 494 청크.** 소스 신뢰도는 `config.SOURCE_TRUST` 로 검색 점수에 반영된다 (상품 ×0.85).

로딩 시 파생되는 태그 (`index/loader.py`)

- `계좌유형` : IRP 117, 연금저축 83, DC 48, DB 43, 중소기업퇴직연금 4
- `금융주제` : 21종 (부담금납입 192, 적립금운용 187, 중도인출 85 …)
- `금융목적` : 주택구입 7, 요양 4, 파산개인회생 3, 전세보증금 3, 재난피해 3
- `distractor` : 37건 (제도 파일의 '첨부-관련법령') → 검색 점수 ×0.35

---

## 5. 측정된 결과 (BM25 단독, 임베딩 미적용 상태)

`python -m eval.runner --retrieval-only` — 골드셋 12건(검색 지표 대상)

| 조건 | Recall | Hit | MRR | Precision |
|---|---:|---:|---:|---:|
| 필터 없음 | 0.600 | 0.750 | 0.541 | 0.174 |
| **소스 필터 적용** | **0.717** | **0.917** | **0.653** | **0.224** |

→ **소스 필터 하나로 Recall +11.7%p, Hit +16.7%p.**
"필터로 검색 범위를 좁히는 것이 RAG 정확도의 절반"이라는 설계 전제가 우리 데이터에서 실측으로 확인됐다.

⚠️ `--retrieval-only` 는 질문을 쪼개지 않고 통째로 검색하므로 **복합 질문을 과소평가**한다.
G013("세금 얼마고 서류는 뭐?")은 단일 검색에선 recall 0.00이지만, 파이프라인에 태우면
세제/중도인출절차 2개로 분리되어 양쪽에서 근거를 찾는다.
→ 단일 질문 검색의 **하한선**으로 읽어야 한다.

clir-sts-dolphin 임베딩을 붙이면 여기서 더 오른다 (특히 어휘가 다른 구어체 질문 — HyDE 와 함께).

---

## 6. 성능 실험 설계

`config.WORKFLOW_SETTINGS` 를 바꿔가며 `--tag` 를 다르게 주고 돌린 뒤 `eval/results/` 를 비교한다.

| 실험 | 바꿀 설정 | 보는 것 |
|---|---|---|
| baseline | grade_documents=F, check_hallucination=F, grade_answer=F | 순수 검색+생성 기준선 |
| +문서채점 | grade_documents=T | 관련성 채점의 기여 |
| +검증 | check_hallucination=T, grade_answer=T | 환각·품질 검증의 기여 |
| +HyDE | rewrite_with_hyde=T | HyDE 의 기여 (재작성 경로에서만 발동) |
| +서브질문필터 | per_subquestion_filters=T | 서브질문별 필터 정제의 기여 |
| +RRR | repeat_residual=T, MAX_ROUNDS=2 | 잔여 재분해의 기여 |
| 임베딩 | build_index 실행 전/후 | 하이브리드 검색의 기여 |
| top-k | PENSION_TOP_K=3/5/7 | 폭 vs 지연 |

---

## 7. 선행모델(FinSight) 대비 주요 변경

| 항목 | FinSight | 우리 | 이유 |
|---|---|---|---|
| 입력 | PDF → OpenParse 파싱 | **JSONL (파싱 완료)** | 스키마가 이미 확정 |
| 필터 | 회사명 + 연도 | **소스(5) + 계좌유형 + 금융주제 + 금융목적** | 연금 도메인 |
| 검색 인프라 | Pathway 서버 + 이중화 | **인메모리 인덱스** | 494청크 고정, 배포 단계 아님 |
| 페르소나 | LLM 이 매번 동적 생성 | **고정 5 전문가(=소스)** | 재현 가능한 측정 |
| 환각 검사 | HHEM 전용 모델 | **LLM 검사** (요구사항) | 모델 하나 덜 관리 |
| 계산 | LLM 이 코드 생성 | **검증된 파이썬 함수** | 세법 공식은 고정, 검증 가능해야 함 |
| 웹 검색 | Tavily→Google→Bing | **네이버 검색** (요구사항) | 한국어 연금 정보 |
| HyDE | 재작성 결과와 `xxxxxxxxxx` 로 이어붙여 전달 | **`hyde_doc` State 필드로 분리** | 읽기 쉽고, 근거에 섞일 위험 없음 |
| 로깅 | 별도 서버 + 웹소켓 그래프 | **평평한 trace 리스트** (요구사항) | 단순화 |
| RRR 반복 | 노드 3벌 복제 | **순환 엣지** | 노드 1/3, 라운드 수 설정 가능 |
| 제외 | — | 시맨틱캐시·자동완성·이중화 | 요구사항 |

---

## 8. 다음 할 일

1. **API 키 입력** → `python -m index.build_index --probe` 로 임베딩 경로 확정
2. `python -m index.build_index` 로 벡터 생성 → 하이브리드 검색 활성화
3. 골드셋 12건 → **40~60건으로 확장** (현재는 각 소스·유형 최소 커버)
4. `python -m eval.runner --tag full` 로 전체 지표 측정
5. `docs/PIPELINE.md` 의 실험 순서대로 ablation 진행
