# 연금 RAG 에이전트 — Colab 판

로컬 `pension_agent` 와 **파이프라인 동작이 완전히 같습니다.**
검색·필터·프롬프트·모델 배정·평가 로직은 한 줄도 다르지 않고,
**설정값을 어디서 읽어 오는가**와 **경로를 어떻게 찾는가**만 코랩에 맞췄습니다.

---

## 빠른 시작

1. 이 폴더를 zip 으로 압축해 코랩 파일 탭에 업로드
   (또는 구글 드라이브에 올리고 마운트)
2. `PensionAgent_Colab.ipynb` 를 열고 위에서부터 실행
3. 런타임 → 런타임 유형 변경 → **T4 GPU** (임베딩 속도)
4. 왼쪽 🔑 → `CLOVA_API_KEY` 등록 → **노트북 액세스** 켜기

키가 없으면 질문·질의세트 실행은 되지 않습니다. 키 없이 가능한 것은 코퍼스 로딩, **벡터 아티팩트 빌드**, 검색 단독 점검입니다.
(규칙 기반 드라이런은 제거했습니다 — LLM 없이 만든 답변이 기록 파일에 실제 실행과 똑같은 모양으로 남아 구분이 안 되기 때문입니다.)

---

## 로컬판과 무엇이 다른가

바뀐 파일은 **`config.py` 하나**이고, `colab_setup.py` 와 노트북이 추가됐습니다.

| | 로컬 | 코랩 |
|---|---|---|
| 설정값 | `.env` 파일 | `.env` → **Colab Secrets(🔑)** → 환경변수 |
| 데이터 경로 | `PENSION_DATA_DIR` 한 곳 | **후보 경로 자동 탐색** (업로드/드라이브/클론 어디든) |
| 임베딩 캐시 | `cache/` | **구글 드라이브** (세션이 끊겨도 유지) |
| 디바이스 | CPU 고정 | **GPU 자동 감지** + 배치 32 → 64 |
| 세율표 경로 | `ROOT/data` 고정 | `DATA_DIR` 를 따라감 |

### ① 설정값 — Colab Secrets

노트북 셀에 API 키를 그대로 적으면 노트북을 공유하는 순간 키가 샙니다.
Secrets 는 노트북 파일에 저장되지 않으므로 그쪽을 씁니다.

```
왼쪽 사이드바 🔑 → 이름 CLOVA_API_KEY → 값 입력 → '노트북 액세스' 켜기
```

읽는 순서: `.env` → Colab Secrets → 기존 환경변수.
이미 환경변수에 있으면 덮어쓰지 않으므로, 노트북에서 `os.environ[...]` 로 준 값이
가장 우선합니다.

인식하는 이름:
`CLOVA_API_KEY` `CLOVA_API_BASE` `CLOVA_MODEL_PLAN` `CLOVA_MODEL_GRADE`
`CLOVA_MODEL_ANSWER` `CLOVA_MODEL_VERIFY`
`PENSION_DATA_DIR` `PENSION_EMBED_MODEL` `PENSION_EMBED_DIM`

### ② 데이터 경로 — 후보 탐색

업로드했는지 / 드라이브에 뒀는지에 따라 위치가 달라지므로, JSONL 이 실제로 있는
폴더를 순서대로 찾습니다.

```
PENSION_DATA_DIR (지정했다면) → 그 아래 data/
ROOT/data → ROOT
/content/pension_agent_colab/data
/content/pension_agent/data
/content/data
/content/drive/MyDrive/pension_agent_colab/data
/content/drive/MyDrive/pension_agent/data
/content/drive/MyDrive/pension_data
```

### ③ 캐시 — 드라이브에 둔다

코랩 로컬 디스크(`/content`)는 **세션이 끊기면 사라집니다.**
3,853건 임베딩을 매번 다시 계산하지 않으려면 캐시가 드라이브에 있어야 합니다.
`colab_setup.setup()` 이 마운트하고 `PENSION_CACHE_DIR` 를 잡아 줍니다.

```
/content/drive/MyDrive/pension_cache/
    emb_multilingual-e5-large.jsonl        문장 해시 → 벡터
    vectors_multilingual-e5-large.npy      (3853, 1024)
    vectors_multilingual-e5-large.ids.txt  chunk_id 순서
    eval_results/                          평가 결과
```

드라이브를 쓰지 않으려면 `colab_setup.setup(use_drive=False)`.

### ④ 디바이스 — GPU 자동 감지

`torch.cuda.is_available()` 로 판별해 `EMBED_DEVICE` 를 정하고, GPU면 배치를
32 → 64 로 올립니다.

| | 3,853건 임베딩 |
|---|---|
| T4 GPU | 1~2분 |
| CPU | 수십 분 |

한 번만 돌리면 캐시에 남습니다.

---

## 노트북 구성

| 셀 | 내용 |
|---|---|
| 0 | 프로젝트 올리기 (zip 자동 해제 / 드라이브 / 클론) |
| 1 | `colab_setup.setup()` — 의존성·드라이브·키·데이터 한 번에 |
| 2 | 코퍼스 점검 + scope 필터 동작 확인 (**LLM 0회**) |
| 3 | 임베딩 인덱스 생성 (`build_index`) |
| 4 | 질문하기 — 사실조회 / 추천 / 혼합 / 계산 4가지 |
| 5 | 평가 — 검색만 → 파이프라인 → 전체 (3단계) |
| 6 | Ablation — 가중치 비교 예시 |
| 7 | 결과 내려받기 |

---

## 메모리 가드

`e5-large` 는 fp32 가중치만 2.2GB 입니다. 여유 메모리가 부족한 상태에서 로드하면
파이썬 예외가 아니라 **네이티브 세그폴트(exit 139)** 로 프로세스가 통째로 죽습니다.
`try/except` 로 못 잡습니다.

그래서 **로드 전에 여유 메모리를 재고**, 모자라면 아예 시도하지 않고 BM25 단독으로
떨어집니다. 코랩 기본 런타임은 RAM 약 12GB 라 문제없이 올라갑니다.
(이 가드는 저사양 로컬 환경을 위한 것입니다 — 개발 PC 총 3.9GB 에서 실제로 죽었습니다)

메모리가 빠듯하면 모델만 내리면 됩니다.

```python
os.environ["PENSION_EMBED_MODEL"] = "intfloat/multilingual-e5-base"
os.environ["PENSION_EMBED_DIM"] = "768"
```

| 모델 | 적재 | 차원 |
|---|---:|---:|
| `multilingual-e5-large` | ~2.4GB | 1024 |
| `multilingual-e5-base` | ~1.2GB | 768 |
| `multilingual-e5-small` | ~0.55GB | 384 |

차원을 바꾸면 `python -m index.build_index` 로 인덱스를 다시 만들어야 합니다.

---

## 알아 둘 것

**골드셋 13건 중 8건이 못 씁니다.** 정답 청크(`doc20`/`doc48`/`doc49`/`doc55`)가
현재 코퍼스에 없습니다 — genre_b/c/d 파일이 데이터에서 빠졌기 때문입니다.
해당 8건(G004~G010, G013)은 recall 이 구조적으로 0 이 나옵니다.
**개편 전후를 비교하려면 새 데이터 기준 골드셋이 먼저입니다.**
노트북 5장 첫 셀이 어떤 케이스가 유효한지 자동으로 확인해 줍니다.

**파이프라인 설명은 `docs/PIPELINE.md`** 에 있습니다 — 필터 설계 근거, 선행모델
비교, 실패 사례, 모델 티어링, 토큰 계측까지 전부 거기 있습니다.

---

## 로컬 원본과 동기화

로직을 고칠 때는 **로컬 원본을 고치고 이 폴더로 복사**하는 쪽이 안전합니다.
`config.py` 만 코랩 전용 변경이 들어 있으므로 그것만 빼고 덮어쓰면 됩니다.

```bash
# 로컬 원본 → 코랩 폴더 (config.py 제외)
for d in llm index nodes edges workflows; do
  cp pension_agent/$d/*.py pension_agent_colab/$d/
done
cp pension_agent/{prompt,state,utils,run_cli}.py pension_agent_colab/
cp pension_agent/eval/*.py pension_agent_colab/eval/
cp pension_agent/docs/*.md pension_agent_colab/docs/
```
