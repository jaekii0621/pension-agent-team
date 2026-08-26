# 벡터 캐시 폴더 구조

`pension_cache/` (코랩에서는 `/content/drive/MyDrive/pension_cache/`, 로컬에서는
`pension_agent/cache/`) 에 코드가 만드는 파일은 **세 종류뿐**이다.

경로는 `config.CACHE_DIR` 이 정하고, `.env` 의 `PENSION_CACHE_DIR` 로 통째로 옮길 수
있다. 코랩에서 구글 드라이브로 보내는 것이 이 원리다.

```
pension_cache/
├── index_multilingual-e5-large/     ← ① 벡터 아티팩트 (본체)
│   ├── vectors.npy
│   ├── ids.txt
│   └── meta.json
├── emb_multilingual-e5-large.jsonl  ← ② 빌드용 중간 캐시
└── eval_results/                    ← ③ 실행 기록
    ├── <tag>_<시각>.json
    ├── <tag>_<시각>.compact.jsonl
    └── <tag>_<시각>.report.md
```

폴더 이름의 `multilingual-e5-large` 는 임베딩 모델 슬러그(`config.EMBED_SLUG`)다.
모델을 바꾸면 폴더가 따로 생기므로, 서로 다른 모델의 벡터가 섞이지 않는다.

---

## ① `index_multilingual-e5-large/` — 벡터 아티팩트

**보관 필수.** 서빙 단계가 읽는 유일한 벡터 소스다. 다른 곳에 복사하거나 백업할
것이 있다면 이 폴더 하나면 된다.

| 파일 | 크기 | 내용 |
|---|---|---|
| `vectors.npy` | 약 15.7MB | (7939, 1024) float16 행렬. L2 정규화 완료 |
| `ids.txt` | 약 200KB | 7,939줄. **N번째 줄 = N번째 벡터의 chunk_id** |
| `meta.json` | 1KB 미만 | 지문·모델명·차원·빌드 시각 |

`ids.txt` 를 따로 두는 이유는 코퍼스 로딩 순서가 바뀌어도 `chunk_id` 로 다시 짝지어
붙이기 위해서다. 행 순서에만 의존하면 파일 구성이 조금만 바뀌어도 전부 어긋난다.

`meta.json` 의 `fingerprint` 가 "이 벡터가 지금 코퍼스와 맞는가" 를 판정한다.
`chunk_id` 와 임베딩 대상 텍스트를 함께 해시하므로 청크 증감뿐 아니라 **본문 수정**도
잡아낸다. 맞지 않으면 조용히 넘어가지 않고 예외로 실패한다.

`float16` 으로 저장하는 이유: e5 벡터는 정규화돼 있어 값이 [-1, 1] 이고 코사인 순위를
매기는 데 이 정밀도면 충분하다. `float32` 대비 용량이 절반(31MB → 15.7MB)이다.
로드할 때 `float32` 로 되돌리므로 검색 속도는 그대로다.

---

## ② `emb_multilingual-e5-large.jsonl` — 빌드용 중간 캐시

**지워도 검색은 멀쩡하다.** 다음 빌드가 느려질 뿐이다.

텍스트 SHA-1 → 벡터를 한 줄씩 쌓아둔 파일이다. 청크 몇 개만 수정했을 때 **바뀐 것만
다시 인코딩**하려고 존재한다.

```jsonl
{"h": "3f2a1b...", "v": [0.021, -0.113, ...]}
```

| 항목 | 내용 |
|---|---|
| 용량 | JSON 텍스트라 `.npy` 의 4~5배 (50~80MB) |
| 읽는 쪽 | **빌드 단계만.** 서빙 단계는 이 파일을 열지 않는다 |
| 지우면 | 다음 빌드가 전부 처음부터 다시 계산한다 |

이 파일이 예전에 메모리를 잡아먹던 원인이었다. 프로세스가 뜰 때마다 전체를 파이썬
리스트로 올렸는데, 정작 검색은 `.npy` 행렬만 본다. 지금은 빌드 단계에서만 올린다.

---

## ③ `eval_results/` — 실행 기록

`colab_setup` 이 `PENSION_RESULTS_DIR` 를 여기로 잡아둔다. 세션이 끊겨도 남는다.

| 파일 | 용도 |
|---|---|
| `<tag>_<시각>.json` | 전체 원본 (답변 전문 포함) |
| `<tag>_<시각>.compact.jsonl` | **분석에 쓰는 파일** (78문항 기준 약 190KB) |
| `<tag>_<시각>.report.md` | 사람이 훑어보는 용도 |

세 파일은 `eval.run_queryset`(질의 세트 78문항)이 만든다.
`eval.runner`(골드셋 채점)로 돌리면 `.json` 하나만 생긴다.
