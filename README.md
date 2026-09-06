# 연금 에이전트 (Pension Agentic RAG)

퇴직연금·개인연금 문서(제도·세제·절차·상품)에 대한 멀티에이전트 RAG 시스템.
선행모델 **FinSight (Inter-IIT Tech Meet 13.0)** 의 구조를 연금 도메인과 CLOVA Studio 환경에 맞게 재설계했다.

> **현재 단계는 성능 측정을 마치고 대회 제출용 API 서버로 배포된 상태다.**
> NCP(Naver Cloud Platform) 인스턴스에서 FastAPI 서버(`server.py`)가 상시 구동 중이며,
> 예선 평가기간(2026.09.07~09.20) 동안 이 서버가 평가 요청을 그대로 받는다.

---

## 1. 30초 요약

```
질문 → 안전성 검사 → 질의 분석 → 슬롯 해소(사유 확정, 실패 시 되묻고 종료)
     → 라우터(경로·전문가 배정) → (계산 필요시 계산기 먼저)
     → 질문 분리 → [서브질문 병렬 fan-out] → rag_core 서브그래프 × N
     → 취합 → 충분? (아니면 잔여 재분해, 최대 MAX_ROUNDS)
     → 전략(필요 시) → 최종 합성 + 인용 재매핑 → 답변
```

서브질문 하나가 지나가는 `rag_core` (자기교정 루프):

```
검색 → [채점] → [부족하면 재작성 → 재검색 / 재시도 소진 시 그냥 생성]
     → 생성 → [검증] → [환각이면 재생성 / 품질미달이면 재작성 / 통과·포기 시 종료]
```

> 웹 검색 폴백은 2026-08-22에 제거됨 — 근거를 chunk_id로 채점하는 구조에서 검증되지 않은 외부 문서를 근거로 쓸 수 없기 때문. 검색·재시도를 다 소진하면 근거 없이 "확인되지 않습니다"로 답변.

에이전트 종류 (2026-09-05, `nodes/__init__.py` / `main_graph.py` / `rag_core.py` 실제 코드 대조 확정)

| 에이전트 | 파일 | 역할 |
|---|---|---|
| 안전성 검사 | `nodes/safety_checker.py` | 범위·유해 입력 차단 |
| 질의 분석 | `nodes/query_analyzer.py` | 질문유형·펀드명·상품관여도 추출 |
| 슬롯 해소 / 되묻기 | `nodes/slot_filler.py` | 답이 갈리는 값(사유 등) 확정 시도, 실패 시 역질문 후 정상 종료 |
| 라우터 | `nodes/router.py` | 처리 경로(rag/calc/out_of_scope) + 도메인 전문가 배정 |
| 질문 분리 | `nodes/question_decomposer.py` | 복합 질문 분해 + 잔여 재분해(RRR) + 충분성 판정 |
| 문서 검색 | `nodes/document_retriever.py` | 서브질문 1건에 대한 검색 |
| 문서 채점 | `nodes/document_grader.py` | 검색된 문서의 관련성 채점 |
| 질문 재작성 | `nodes/question_rewriter.py` | 검색 부족·품질 미달 시 서브질문 재작성 |
| 답변 생성 (제도/세제/절차/상품) | `nodes/answer_generator.py` | 소스별 전문 프롬프트로 답변 생성 |
| 검증 | `nodes/verifier.py` | 환각 검사 + 품질 채점 (HHEM 미사용, LLM 1회로 통합) |
| 계산기 | `nodes/calculator.py` + `pension_tools.py` | 연금 특화 계산 (**계산은 코드가 한다**) |
| 전략 | `nodes/strategy.py` | 검증된 사실만으로 실행 계획 수립 |
| 합성 / 범위 밖 안내 | `nodes/combiner.py` | 최종 답변 합성 + 인용 전역 재매핑 (`combine_answer`), 범위 밖 질의 안내 (`out_of_scope_answer`). 사후검증도 겸함 — 유효하지 않은 인용번호 제거, 위법조력·유해 범주 오출력 거절문구 제거, 코퍼스에 없는 자리표시자 상품명 탐지, 자산유형/안정성 서술이 카드와 어긋나는지 대조, 표가 여러 출처에서 섞여 합성됐는지 대조. 전부 LLM 재호출 없이 문자열/규칙 기반 |
| 되묻기 이어받기 | `workflows/followup.py` | 직전 턴 되묻기에 대한 답변을 원 질문에 병합 |

> ℹ️ 이전 버전 README(8/22)에 있던 "필터 정제(filter_refiner.py)" 에이전트는 실재한 적 없는 이름이 아니라, **2026-08 개편 때 의도적으로 제거된 노드**입니다 (`rag_core.py` 주석: 필터가 scope 하나로 줄면서 서브질문별로 다시 정제할 축이 없어져 제거, 그만큼 서브질문당 LLM 호출 1회 절감).

---

## 2. 설치와 실행

### 2-1. 로컬/코랩 (개발·성능 측정용)

```bash
pip install -r requirements.txt
copy .env.example .env        # 값 채우기
```

```bash
python run_cli.py --check                      # 설정·데이터 점검 (API 키 불필요)
python run_cli.py "IRP 중도인출 사유가 뭔가요?"  # 질문 1건
python run_cli.py                               # 대화형 (빈 줄/exit로 종료)
python run_cli.py "..." --mermaid               # 실행 경로를 mermaid로 출력
python run_cli.py "..." --json                  # 최종 state 출력
python run_cli.py "..." --profile '{"total_salary":60000000}'
python run_cli.py --queryset                    # 자체 평가 질의세트 78문항 배치 실행 (eval/run_queryset.py)
```

### 2-2. 서버 (평가용 API — 이 프로젝트의 제출 대상)

**환경 준비**

```bash
git clone <repo_url>
cd pension-agent-team
pip3 install --break-system-packages -r requirements.txt
```

`requirements.txt`에 `--extra-index-url https://download.pytorch.org/whl/cpu`가 명시되어 있어 torch는 CPU 전용 빌드로 설치된다.

⚠️ 설치 중 `rich` 패키지가 apt(시스템)로 이미 깔려 있어 pip uninstall 충돌이 날 수 있다. 그 경우:

```bash
pip3 install sentence-transformers --break-system-packages --ignore-installed rich
```

로 재시도한다. (시스템 `rich`는 그대로 두고 pip 환경에만 별도 설치)

**환경변수 설정**

```bash
cp .env.example .env
nano .env   # CLOVA_API_KEY 등 값 채우기
```

`.env`에 `HF_HUB_OFFLINE=1`을 추가하면 매 요청마다 HuggingFace로 보내는 원격 HEAD 요청이 없어져 모델 로드 시간이 10초대에서 3초대로 줄어든다.

**서버 기동**

```bash
nohup python3 server.py > server.log 2>&1 &
```

- 포트: 8000 (0.0.0.0 바인딩, Public 망 통신 가능해야 함 — 대회 요구사항)
- 기동 확인: `curl -s http://localhost:8000/health` → `{"status":"ok","embeddings":{...}}`이면 임베딩 모델까지 정상 로드된 것

**서버 종료 후 재기동 시 주의**

`server.py`나 파이프라인 코드를 pull한 뒤에는 반드시 기존 프로세스를 kill하고 다시 띄워야 한다 (파일만 바꾼다고 실행 중인 프로세스에 반영되지 않음).

```bash
ps -ef | grep server.py     # PID 확인
kill <PID>
nohup python3 server.py > server.log 2>&1 &
```

### API 키가 아직 없을 때 (로컬/코랩 개발 시)

키 없이 되는 것:

```bash
python run_cli.py --check
```

코퍼스 로딩, 벡터 아티팩트 상태, 검색 단독 동작까지 확인된다.
파이프라인 실행(`run_cli.py "질문"`, `eval.run_queryset`, 서버 `/answer`)에는 키가 필요하다.

### 키를 받은 뒤

```bash
python -m index.build_index --probe            # 임베딩 엔드포인트 확인
python -m index.build_index                    # 청크 임베딩 → cache/
```

---

## 3. API 서버 명세

평가용 API 엔드포인트 및 요청/응답 스키마 전체는 별도 문서 **`API_SPEC.md`** 참고.

- Endpoint: `GET http://211.188.56.99:8000/answer`
- Request: `question_id`, `question` (둘 다 string, 쿼리 파라미터)
- Response: `question_id`, `question`, `retrieved_context`, `think_trace`, `answer` (전부 string)
- 타임아웃: 300초
- 부가: `/health` (헬스체크용, 대회 스펙 아님)

---

## 4. 폴더 구조

```
pension_agent/
├── config.py            ★ 스위치판. WORKFLOW_SETTINGS 를 바꾸면 그래프 모양이 바뀐다
├── state.py             ★ 상자 설계도 (State + 리듀서)
├── prompt.py               프롬프트 전부
├── utils.py                로깅·트레이스 (단순화됨)
├── server.py             ★ 평가용 API 서버 (FastAPI + uvicorn, /answer, /health)
│
├── llm/                    CLOVA Studio 호출 통로
│   ├── clova_client.py     Chat Completions + 구조화 출력
│   └── embeddings.py       multilingual-e5-large (질의 인코딩 + 빌드 캐시)
│
├── index/                  검색 계층 (인메모리)
│   ├── loader.py           JSONL → 통일 스키마 + 태그 파생
│   ├── store.py            [서빙] 하이브리드 검색 + 계층 쿼터 + 가중
│   ├── artifact.py         벡터 아티팩트 저장/로드 + 지문 검증
│   └── build_index.py      [빌드] 코퍼스 → 벡터 아티팩트
│
├── nodes/                  에이전트(작업대) 14개
│   ├── safety_checker.py, query_analyzer.py, slot_filler.py, router.py
│   ├── question_decomposer.py, document_retriever.py, document_grader.py
│   ├── question_rewriter.py, answer_generator.py, verifier.py
│   └── calculator.py, pension_tools.py, strategy.py, combiner.py
├── edges/                  분기 함수 6개
│   ├── routing.py            safety_gate, clarify_gate, route_after_router
│   ├── decomposition.py      send_sub_questions(Send 기반 fan-out), should_repeat(RRR 라운드 제어)
│   ├── retrieval_check.py    assess_documents (검색·채점 결과 → enough/rewrite/give_up)
│   └── verification_check.py assess_verdict (검증 결과 → ok/regenerate/rewrite/give_up)
├── workflows/
│   ├── rag_core.py         서브질문 1개용 자기교정 루프
│   ├── main_graph.py       최상위 그래프
│   └── followup.py         되묻기 이어받기
│
├── eval/                   성능 측정
│   ├── metrics.py            검색·정확도·비용 지표
│   ├── gold/                 골드셋 — policy.jsonl(28건=정책 dev19+holdout9), product.jsonl(25건) → 총 53건(공식 평가셋)
│   ├── gold.py               채점 스크립트
│   ├── runner.py             실행기
│   └── run_queryset.py       질의세트 배치 실행
│
├── data/                    코퍼스 원본 (청크 파일 전체 — 5절 데이터 표 참고)
│   └── tax_tables.json      세율표 (코드가 읽음, LLM 아님)
├── API_SPEC.md           ★ 평가용 API 서버 명세 (요청/응답 스키마)
└── docs/PIPELINE.md       ★ 질문이 지나가는 전 과정 상세 설명
```

---

## 5. 데이터

코퍼스는 정책 트랙(제도·세제·절차)과 상품 트랙(투자설명서) 둘로 나뉘며, 검색 자원 배분과 출처 신뢰도 가중에 이 구분이 쓰인다. 아래 수치는 기술제안서 3-1절 기준(2026-09-06 최종 확정)이며, 서버 `data/` 폴더 실제 파일과 라인 수 대조 완료.

| 트랙 | 파일 | chunk_type | 건수 |
|---|---|---|---:|
| 정책 (1,173) | `data/chunks_genre_a.jsonl` | qa (제도 Q&A) | 208 |
| | `data/chunks_genre_b.jsonl` | tax (세제 해설·세율표) | 160 |
| | `data/chunks_genre_c.jsonl` | withdrawal (중도인출 사유별 요건) | 38 |
| | `data/chunks_genre_d.jsonl` + `data/chunks_ocr_batch.jsonl` | procedure (업무 절차·매뉴얼) | 767 (244+523) |
| 상품 (7,904) | `data/fund_cards.jsonl` | fund_card (펀드 단위 정본 카드) | 92 |
| | `data/class_cards.jsonl` | class_card (클래스 단위 정본 카드) | 91 |
| | `data/section_chunks.jsonl` | prospectus_section (투자설명서 섹션 원문) | 7,721 |

**총 9,077 청크** (원본 문서 150건 — 상품 92, 정책 58).

로딩 시 파생되는 태그 (`index/loader.py`)

- `계좌유형` : IRP, 연금저축, DC, DB, 중소기업퇴직연금
- `금융주제` : 21종 (부담금납입, 적립금운용, 중도인출 등)
- `금융목적` : 주택구입, 요양, 파산개인회생, 전세보증금, 재난피해
- `distractor` : 제도 파일의 '첨부-관련법령' → 검색 점수 하향 반영 (가중치 0.35)

---

## 6. 측정된 결과

파이프라인 11차 실행 결과 (2026-09-02, `colab_full_20260902_141622.json` / `FIX_11차_20260902.md`)

| 지표 | 값 |
|---|---:|
| 총 인용(citations) | 164건 |
| LLM 호출 | 1,688회 |
| 토큰 사용량 | 2,548,982 |
| 게이트 거절(gate refusals) | 4 → 3 |
| 근거 불충분 값(unsupported_values) | 32 → 11 |

> 이전 버전에 있던 BM25 단독 검색 성능표(494청크, 하드 필터 기준)는 삭제함 — `config.py` 확인 결과 2026-08-16 재개편으로 하드 필터 자체가 폐지되어(소프트 필터+계층 쿼터로 대체) 더 이상 유효한 실험 조건이 아님. 최신 차수별 지표는 기술제안서 7-2절 참고.

---

## 7. 성능 실험 설계

`config.WORKFLOW_SETTINGS`를 바꿔가며 `--tag`를 다르게 주고 돌린 뒤 `eval/results/`를 비교한다. (2026-09-06, `config.py` 실제 키 대조 완료)

| 실험 | 바꿀 설정 | 보는 것 |
|---|---|---|
| baseline | grade_documents=F, check_hallucination=F, grade_answer=F | 순수 검색+생성 기준선 |
| +문서채점 | grade_documents=T | 관련성 채점의 기여 |
| +검증 | check_hallucination=T, grade_answer=T | 환각·품질 검증의 기여 |
| +HyDE | rewrite_with_hyde=T | HyDE의 기여 (재작성 경로에서만 발동) |
| +소프트필터 | metadata_filtering=T | 사유·펀드명 소프트필터 + 단계적 완화의 기여 |
| +RRR | repeat_residual=T, `PENSION_MAX_ROUNDS`(기본 2) | 잔여 재분해의 기여 |
| 임베딩 | `python -m index.build_index` 실행 전/후 | 하이브리드 검색(BM25+벡터)의 기여 |
| top-k | `PENSION_TOP_K`(기본 8) / `PENSION_TOP_K_RECOMMEND`(기본 24) | 폭 vs 지연 |

---

## 8. 선행모델(FinSight) 대비 주요 변경

| 항목 | FinSight | 우리 | 이유 |
|---|---|---|---|
| 입력 | PDF → OpenParse 파싱 | **JSONL (파싱 완료)** | 스키마가 이미 확정 |
| 필터 | 회사명 + 연도 | **소스(5) + 계좌유형 + 금융주제 + 금융목적** | 연금 도메인 |
| 검색 인프라 | Pathway 서버 + 이중화 | **인메모리 인덱스** | 청크 고정, 서버는 API 요청/응답만 담당 |
| 페르소나 | LLM 이 매번 동적 생성 | **고정 5 전문가(=소스)** | 재현 가능한 측정 |
| 환각 검사 | HHEM 전용 모델 | **LLM 검사** (요구사항) | 모델 하나 덜 관리 |
| 계산 | LLM 이 코드 생성 | **검증된 파이썬 함수** | 세법 공식은 고정, 검증 가능해야 함 |
| 웹 검색 폴백 | Tavily→Google→Bing | **제거함** (2026-08-22) | chunk_id 기반 채점 구조에서 검증 안 된 외부 문서를 근거로 못 씀 |
| HyDE | 재작성 결과와 이어붙여 전달 | **`hyde_doc` State 필드로 분리** | 읽기 쉽고, 근거에 섞일 위험 없음 |
| 로깅 | 별도 서버 + 웹소켓 그래프 | **평평한 trace 리스트** (요구사항) | 단순화 |
| RRR 반복 | 노드 3벌 복제 | **순환 엣지** | 노드 1/3, 라운드 수 설정 가능 |
| 배포 | 서버 없음(로컬/코랩) | **NCP FastAPI 서버 상시 구동** | 대회 제출 요구사항 (09.07~09.20 API 활성화 유지) |
| 제외 | — | 시맨틱캐시·자동완성·이중화 | 요구사항 |

---

## 9. 제출 관련 참고

- 제출 채널: 주최 측 Github Organization Private Repository
- 제출 항목: ①소스코드(이 repo 전체 + 본 README) ②기술제안서(별도 문서) ③평가용 API 서버 정보(`API_SPEC.md`)
- 마감: 2026.09.06 — 이후 커밋/push/서버 변경 시 실격
- 평가기간 중 서버 유지 필수 구간: 2026.09.07 ~ 09.20
