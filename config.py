"""전역 설정 — 이 파일 하나가 시스템 전체의 스위치판이다.

선행모델(FinSight)의 config.py와 같은 역할을 한다.
  · WORKFLOW_SETTINGS 의 True/False 를 바꾸면 **그래프 구조 자체가 바뀐다**
    (workflows/*.py 가 이 값을 보고 노드를 붙이거나 뺀다).
    → 코드를 고치지 않고 구성요소를 하나씩 켜/꺼가며 성능을 비교(ablation)할 수 있다.
  · 상수(top-k, 재시도 한도 등)도 전부 여기 모은다. 노드 코드에 숫자를 박지 않는다.

지금 단계는 **서버 배포가 아니라 성능 측정**이므로, 서버/포트/이중화 설정은 없다.

──────────────────────────────────────────────────────────────────────────
■ 필터 설계 (2026-08 개편) — 선행모델의 결론을 그대로 따른다

선행모델이 실제로 하드 필터로 쓴 값은 딱 둘이다.

    METADATA_FILTER_INIT = ["company_name", "year"]

topics(금융 35종)는 LLM 으로 뽑기까지 해놓고 **검색 필터에서 제외**했고,
청크 쪽 topic 은 아예 "Other" 로 고정돼 있었다. 이유는 명확하다 —
필터는 **질문 쪽 값과 청크 쪽 값의 완전일치 매칭**이라 양쪽 모두 100% 신뢰할 수
있어야 성립하는데, 주제 태그는 어느 쪽도 그렇지 못하기 때문이다.

우리 데이터로 실측한 결과도 같았다.
    · 금융주제 태그가 검색을 망친다: "세액공제" 필터 하나로 펀드카드 92장 전멸
    · 계좌유형은 88.7% 미태깅이라 필터로서 사실상 무효
    · 금융목적은 원본에 '사유' 필드가 없어 항상 None (죽은 필터)

그리고 연금 도메인에는 회사명·연도 같은 **문서 단위 축이 애초에 없다.**

■ 2026-08-16 재개편 — scope 하드필터를 없앴다

한동안 필터를 `scope: "doc" | "all"` 하나로 줄여서 썼다. 추천이면 투자설명서를 열고
아니면 규정문서만 보는 이분법이다. 이것도 결국 같은 병을 앓았다 —
**LLM 판정 1회의 실패 비용이 코퍼스 절반**이라는 병이다.

지금 하드 필터는 **하나도 없다.** 대신 두 가지로 대체했다.

    product_affinity : 0.0~1.0.  top-k 안에서 규정문서와 투자설명서가 **각각 몇 자리**를
                       가져갈지 정한다. 틀려도 배분이 기울 뿐 후보가 사라지지 않는다.
    사유             : 장르C(중도인출) 전용. 소프트 필터 — 결과가 모자라면 자동으로 풀린다.
                       확정하지 못하면 검색 대신 **되묻는다**(역질문).

    fund_name : 질문에 특정 펀드명이 명시됐을 때만 거는 유일한 배제형 조건.
                1문서=1펀드라 100% 확정값이고, 펀드명이 없는 청크는 배제하지 않는다.

나머지(주제·계좌유형·목적·자산유형·위험등급)는 **점수 가중이나 프롬프트로 내리고
하드 필터에서는 전부 뺐다.**
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ==========================================================================
# 0. 실행 환경 판별 — 로컬 / Google Colab
# ==========================================================================
# ★ 이 폴더는 **로컬 원본과 파이프라인 동작이 같아야 한다.**
#   코랩 대응은 "설정값을 어디서 읽어 오는가"와 "경로를 어떻게 찾는가"만 다르다.
#   검색·필터·프롬프트·모델 배정은 원본과 완전히 동일하다.
def _in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.isdir("/content")


IS_COLAB = _in_colab()


# --------------------------------------------------------------------------
# 설정값 로딩 — .env → Colab Secrets → 환경변수
# --------------------------------------------------------------------------
def _load_dotenv() -> None:
    """`.env` 가 있으면 읽는다. 코랩에는 보통 없으므로 조용히 넘어간다."""
    for f in (ROOT / ".env", Path("/content/.env")):
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return


def _load_colab_secrets() -> None:
    """코랩 좌측 🔑 '보안 비밀'에 넣어 둔 값을 환경변수로 올린다.

    노트북 셀에 API 키를 그대로 적으면 노트북을 공유하는 순간 키가 새어 나간다.
    Secrets 는 노트북 파일에 저장되지 않으므로 그쪽을 쓴다.

        Colab 왼쪽 사이드바 → 🔑 → 이름 `CLOVA_API_KEY`, 값 입력 → '노트북 액세스' 켜기

    미등록 이름은 조용히 건너뛴다(전부 선택 사항이다). 이미 환경변수에 있으면
    덮어쓰지 않는다 — 노트북에서 `os.environ[...]` 로 준 값이 우선이다.
    """
    if not IS_COLAB:
        return
    try:
        from google.colab import userdata          # type: ignore
    except ImportError:
        return

    for k in (
        "CLOVA_API_KEY", "CLOVA_API_BASE",
        "CLOVA_MODEL_PLAN", "CLOVA_MODEL_GRADE",
        "CLOVA_MODEL_ANSWER", "CLOVA_MODEL_VERIFY",
        "PENSION_DATA_DIR", "PENSION_EMBED_MODEL", "PENSION_EMBED_DIM",
    ):
        if os.getenv(k):
            continue
        try:
            v = userdata.get(k)
        except Exception:                           # 미등록·권한 거부 등
            continue
        if v:
            os.environ[k] = str(v)


_load_dotenv()
_load_colab_secrets()


def env(k: str, d: str = "") -> str:
    v = os.getenv(k)
    return v if v not in (None, "") else d


def env_int(k: str, d: int) -> int:
    try:
        return int(env(k, str(d)))
    except ValueError:
        return d


def env_float(k: str, d: float) -> float:
    try:
        return float(env(k, str(d)))
    except ValueError:
        return d


def env_bool(k: str, d: bool) -> bool:
    v = env(k, "")
    return d if not v else v.strip().lower() in {"1", "true", "yes", "y", "on"}


# ==========================================================================
# 1. 데이터 경로 + 코퍼스 구성
# ==========================================================================
def _has_jsonl(p: Path) -> bool:
    try:
        return any(p.glob("*.jsonl"))
    except OSError:
        return False


def _resolve_data_dir() -> Path:
    """JSONL 이 실제로 있는 디렉토리를 찾는다.

    로컬에서는 `.env` 의 PENSION_DATA_DIR 가 프로젝트 루트를 가리키는 실수가 잦고
    (파일은 그 아래 data/ 에 있다), 코랩에서는 데이터가 어디에 놓일지 —
    저장소 클론이냐 / 드라이브 마운트냐 / 직접 업로드냐 — 에 따라 달라진다.
    그래서 **후보 경로를 순서대로 훑어 실제로 JSONL 이 있는 곳**을 쓴다.

    설정 오타나 업로드 위치 차이 하나로 코퍼스가 통째로 비는 걸 막는다.
    """
    explicit = env("PENSION_DATA_DIR", "")
    candidates: list[Path] = []
    if explicit:
        candidates += [Path(explicit), Path(explicit) / "data"]

    candidates += [ROOT / "data", ROOT]

    if IS_COLAB:
        candidates += [
            Path("/content/pension_agent_colab/data"),
            Path("/content/pension_agent/data"),
            Path("/content/data"),
            Path("/content/drive/MyDrive/pension_agent_colab/data"),
            Path("/content/drive/MyDrive/pension_agent/data"),
            Path("/content/drive/MyDrive/pension_data"),
        ]

    for c in candidates:
        if _has_jsonl(c):
            return c

    # 못 찾으면 첫 후보를 돌려준다 — loader 가 무엇이 없는지 알려 준다.
    return candidates[0]


DATA_DIR = _resolve_data_dir()

# 파일 → (corpus, tier, chunk_type) 매핑. loader.py 가 이 표를 보고 읽는다.
#
# ★ tier 가 핵심이다. 원본 JSONL 의 `source` 필드는 doc 문서 일부(doc37 의 14건 등)가
#   "상품" 으로 찍혀 있어서, 그대로 쓰면 **고신뢰 doc 문서가 투자설명서와 같은 취급**을
#   받는다(신뢰도 가중에서 0.85 로 감점). 그래서 "장르(corpus)" 와 "출처 신뢰도(tier)" 를
#   별도 축으로 분리했다.
#
#   corpus 가 빈 문자열이면 레코드의 source 필드를 그대로 쓴다.
#   (genre_d 는 절차 240 + 세제 4, doc37 은 절차 22 + 상품 14 + 세제 1 로 섞여 있어
#    파일 단위로 못 정한다)
#
# ⚠️ 2026-08-16 코퍼스 개편 반영
#   · 독립 OCR 파일 5종(doc22/33/53/57/58)은 **genre_d 로 병합**되었다. 같은 chunk_id 를
#     가진 파일이 남아 있어도 다시 읽지 않는다 — 읽으면 전 청크가 2벌이 되어 BM25 의
#     문서빈도(df)가 왜곡되고 벡터 캐시 매칭이 깨진다.
#   · genre_b/c/d 는 "예전 코퍼스"가 아니라 **현행 정식 파일**이다.
#     (예전 주석이 이것을 legacy 로 적어 둔 탓에 세제·중도인출 코퍼스가 통째로 빠진 채
#      파이프라인이 조용히 돌던 이력이 있다. 아래 REQUIRED_CORPUS_FILES 로 재발을 막는다.)
#
# ⚠️ 2026-08-18 OCR 파일 통합 반영
#   개별 OCR 파일 2종(chunks_doc21_ocr / chunks_doc37_ocr)이
#   **chunks_ocr_batch.jsonl 한 개로 병합**되었다(523건, 18개 문서:
#   doc1·2·3·4·5·7·8·9·21·24·27·28·30·31·32·37·54·56).
#
#   병합 직후 이 표에는 없어진 두 파일만 남아 있고 새 파일이 등록되지 않았다.
#   두 파일은 REQUIRED 목록에도 없어서 "선택 파일 없음(건너뜀)" 한 줄만 남기고
#   지나갔고, 결과적으로 **523건 전체가 검색 코퍼스에서 빠진 채** 돌았다.
#   같은 종류의 조용한 실패가 세 번째다(genre_b/c → OCR 5종 → 이번 건).
#   그래서 새 파일은 아래 REQUIRED_CORPUS_FILES 에 **반드시** 넣는다.
#
#   chunk_type 을 "procedure" 하나로 두는 이유:
#     523건의 source 는 제도 201 / 절차 185 / 상품 74 / 세제 63 으로 섞여 있는데,
#     chunk_type 은 파일 단위 상수라 하나만 고를 수 있다. 다만 doc 계층에서
#     chunk_type 은 **집계·리포트 라벨로만 쓰인다** — 후보 필터(_passes)·점수
#     가중(_weights)·표 가산(is_table_chunk 는 origin/section 을 본다)·LLM 프롬프트
#     어디에도 관여하지 않는다. 레코드별 의미는 corpus="" 로 두어 각 레코드의
#     source 를 그대로 쓰므로 프롬프트의 `분류=` 표기는 정확하게 유지된다.
CORPUS_FILES: list[tuple[str, str, str, str]] = [
    # 파일명                      corpus          tier          chunk_type
    ("chunks_genre_a.jsonl",     "제도",         "doc",        "qa"),
    ("chunks_genre_b.jsonl",     "세제",         "doc",        "tax"),
    ("chunks_genre_c.jsonl",     "중도인출절차", "doc",        "withdrawal"),
    ("chunks_genre_d.jsonl",     "",             "doc",        "procedure"),
    ("chunks_ocr_batch.jsonl",   "",             "doc",        "procedure"),
    ("fund_cards.jsonl",         "상품",         "prospectus", "fund_card"),
    ("class_cards.jsonl",        "상품",         "prospectus", "class_card"),
    ("section_chunks.jsonl",     "상품",         "prospectus", "prospectus_section"),
]

# 이 파일들이 없으면 **로딩을 실패시킨다.**
#
# 예전에는 없는 파일을 전부 조용히 건너뛰었다. 그래서 세제(genre_b)·중도인출(genre_c)
# 코퍼스가 통째로 빠진 상태에서도 에러 없이 돌았고, 그 질문들에 "근거 없는 그럴듯한 답"
# 이 나갔다. 파일 누락은 경고가 아니라 **실패**여야 한다.
REQUIRED_CORPUS_FILES: tuple[str, ...] = (
    "chunks_genre_a.jsonl",
    "chunks_genre_b.jsonl",
    "chunks_genre_c.jsonl",
    "chunks_genre_d.jsonl",
    "chunks_ocr_batch.jsonl",
    "fund_cards.jsonl",
    "class_cards.jsonl",
    "section_chunks.jsonl",
)

# ── 캐시·결과 경로 ────────────────────────────────────────────────────
# ⚠️ 코랩 런타임의 로컬 디스크(/content)는 **세션이 끊기면 사라진다.**
#    임베딩 캐시(3,853건, 수십 MB)와 평가 결과를 매번 다시 만들지 않으려면
#    구글 드라이브를 마운트하고 PENSION_CACHE_DIR 를 드라이브 경로로 주면 된다.
#
#        from google.colab import drive; drive.mount('/content/drive')
#        os.environ['PENSION_CACHE_DIR'] = '/content/drive/MyDrive/pension_cache'
#
#    노트북 1번 셀(setup)이 이 작업을 대신 해 준다.
CACHE_DIR = Path(env("PENSION_CACHE_DIR", str(ROOT / "cache")))
TABLES_DIR = DATA_DIR               # 세율표 등 결정 테이블 (데이터와 같은 폴더)
RESULTS_DIR = Path(env("PENSION_RESULTS_DIR", str(ROOT / "eval" / "results")))
# ── 골드셋 ────────────────────────────────────────────────────────────
#
# ⚠️ 단일 파일에서 **폴더**로 바꿨다 (2026-08-20, 골드셋_형식_확정 §11).
#
#   예전: GOLDSET_PATH = eval/goldset.jsonl   ← 아무도 이 이름으로 안 만든다
#   지금: GOLD_DIR     = eval/gold/
#           ├─ product.jsonl   상품 트랙 (민서)
#           └─ policy.jsonl    제도 트랙 (재희)
#
#   트랙마다 코퍼스 구조와 함정이 달라 담당자가 자기 트랙을 쓴다.
#   파일이 없으면 **조용히 빈 리스트로 돌지 않고 실패**한다(eval/gold.py) —
#   파일 누락이 조용히 넘어가 사고가 난 이력이 세 번 있다.
GOLD_DIR = Path(env("PENSION_GOLD_DIR", str(ROOT / "eval" / "gold")))

# 하위 호환 — 옛 이름으로 부르는 코드가 남아 있을 수 있다.
GOLDSET_PATH = Path(env("PENSION_GOLDSET", str(GOLD_DIR / "policy.jsonl")))


# ==========================================================================
# 2. CLOVA Studio (LLM) — 단계별 모델 분리
# ==========================================================================
CLOVA_API_KEY = env("CLOVA_API_KEY", "")
CLOVA_BASE = env("CLOVA_API_BASE", "https://clovastudio.stream.ntruss.com")

# ※ 드라이런(규칙 기반 대역 LLM)은 제거했다.
#   LLM 없이 돌린 결과는 답변 품질을 전혀 대변하지 못하는데, 기록 파일에는 실제
#   실행과 똑같은 모양으로 남아서 나중에 섞이면 구분이 안 된다. 배선 확인은
#   `python -m index.build_index --stats` 와 검색 단독 점검(`run_cli.py --check`)으로
#   충분하고, 그쪽은 LLM 키가 없어도 돌아간다.

# ── 단계별 모델 배정 ──────────────────────────────────────────────────
# CLOVA Studio Chat Completions v3 가 제공하는 모델은 두 종류다.
#   · HCX-005        고성능(비전 포함). 생성·검증처럼 품질이 곧 결과인 단계에 쓴다.
#   · HCX-DASH-002   경량. 빠르고 싸다. 분류·판정처럼 **출력이 짧고 정답이 좁은** 단계에 쓴다.
#
# 우리 파이프라인의 LLM 호출은 성격이 두 갈래로 뚜렷이 갈린다.
#   (A) 판정/분류형 : 안전성, 추천여부, 분해, 문서채점, 재작성, HyDE
#       → 출력이 몇 단어~몇 줄. 큰 모델을 써도 정확도 차이가 작고 호출 수가 많다.
#   (B) 생성/검증형 : 답변 생성, 환각·품질 검증, 전략, 최종 합성
#       → 출력이 길고 사실 정확도가 곧 시스템 품질이다.
# (A)를 DASH 로 내리면 질문 1건당 호출 10~15회 중 8~10회가 경량 모델로 간다.
MODEL_PLAN = env("CLOVA_MODEL_PLAN", "HCX-DASH-002")      # 분해·추천판단·라우팅·재작성
MODEL_GRADE = env("CLOVA_MODEL_GRADE", "HCX-DASH-002")    # 문서 관련성 채점
MODEL_ANSWER = env("CLOVA_MODEL_ANSWER", "HCX-005")       # 답변 생성·전략·합성
MODEL_VERIFY = env("CLOVA_MODEL_VERIFY", "HCX-005")       # 환각·품질 검증

# ── (A) 중에서 다시 끌어올린 세 자리 (2026-08-29) ─────────────────────
#
# 위 (A)/(B) 이분법은 "출력이 짧으면 DASH" 였는데, 짧은 출력이라고 다 같지 않다.
# **틀렸을 때 문항 하나가 통째로 망가지는** 자리가 (A) 안에 섞여 있었다.
# 호출 수가 적어 지연·레이트리밋 부담은 거의 없으므로 셋만 HCX-005 로 올린다.
#
#   MODEL_DECOMPOSE  질문 분해. 질문당 1~2회.
#       분해가 어긋나면 그 아래 검색·채점·생성이 전부 엉뚱한 것을 푼다.
#       뒤 단계가 아무리 좋아도 복구되지 않는 유일한 자리다.
#
#   MODEL_SAFETY     안전성 범주 판정. 질문당 1회.
#       범주가 곧 출력 문구를 가른다(prompt.CAUTION_GUIDANCE / blocked_messages).
#       '프롬프트공격' 을 놓치면 평가지표 6번 대응 문장이 아예 안 나간다.
#
#   MODEL_REPAIR     구조화 출력 복구 호출. 파싱이 실패했을 때만 돈다.
#       ⚠️ 여기가 실패하면 `citations` 배열이 통째로 빈다 — 즉 근거표시 0건.
#       예전엔 MODEL_PLAN 고정이라, HCX-005 가 실패한 JSON 을 DASH 에게
#       고쳐 달라고 보내는 모양이었다.
#       ※ 단, 응답에 중괄호가 하나도 없으면 복구 호출 자체를 건너뛴다
#         (clova_client 의 'json_없음' 경로). 그 경우는 이 상수와 무관하다.
# ⚠️ HCX-005 → HCX-DASH-002 로 되돌린다 (2026-08-31, 9차 검토 이슈 7).
#    9차에 셋을 한꺼번에 올렸는데 분해 쪽 부작용이 컸다. raw 대조 결과:
#      · 서브질문 92 → 125개(+36%). 같은 질문을 표현만 바꿔 3~4개로 쪼갠 사례
#        (P-G0b 1→3, D-H2 1→4) — 쪼갤수록 검색·채점·검증이 배로 돈다.
#      · 되묻기 오발동 재발(P-G3). slot_filler 8/20 주석에 대표 사례로 적혀 있던
#        문항이고, 8차엔 정상이었는데 9차에 슬롯 판정이 통째로 뒤집혔다.
#      · 총 토큰 251만 → 323만(+28%), LLM 호출 2,088 → 2,410(+14%).
#    얻은 것은 평균 인용 2.04 → 2.30 뿐이라 비용 대비 남는 게 없다.
#    safety·repair 는 호출 수가 적고 부작용이 관측되지 않아 HCX-005 로 남긴다.
MODEL_DECOMPOSE = env("CLOVA_MODEL_DECOMPOSE", "HCX-DASH-002")  # 질문 분해
MODEL_SAFETY = env("CLOVA_MODEL_SAFETY", "HCX-005")        # 안전성 범주 판정
MODEL_REPAIR = env("CLOVA_MODEL_REPAIR", "HCX-005")        # 구조화 출력 복구

# 문서 채점(MODEL_GRADE)은 DASH 로 남겼다.
#   질문당 최대 8회를 GRADE_WORKERS 만큼 **동시에** 쏘는 유일한 자리라,
#   여기를 올리면 6차에서 P-B1 을 죽인 429 버스트가 그대로 재현된다.
#   올려서 보려면 동시성을 함께 내려라:
#       CLOVA_MODEL_GRADE=HCX-005 PENSION_GRADE_WORKERS=2

LLM_TIMEOUT = env_float("CLOVA_TIMEOUT", 60.0)
# ⚠️ 4 → 6 (2026-08-28). P-B1 이 6·7·8차 **세 번 연속** 429 로 죽었다.
#    traceback 상 문서 채점이 아니라 답변 생성 단계에서 나는데, 순차 호출 하나가
#    재시도 4회(3→6→12→24초, 약 45초)를 다 쓰고 실패했다. 순간 버스트가 아니라
#    누적 요청률 문제라 워커 축소(8→4)로는 안 막혔다.
#    6회 + 아래 백오프면 최대 대기가 약 95초 — 300초 예산 안에 들어온다.
LLM_MAX_RETRY = env_int("CLOVA_MAX_RETRY", 6)     # HTTP 레벨 재시도
LLM_TEMPERATURE = env_float("CLOVA_TEMPERATURE", 0.0)

# ── 레이트리밋(429) 백오프 ────────────────────────────────────────────
# 429 는 "내가 너무 빨리 보냈다" 는 신호라, 짧게 기다렸다 다시 보내면 같은 이유로
# 또 걸린다. 지수로 늘린다 (기본 3초 → 6 → 12 → 24, 상한 30초).
# 서버가 Retry-After 를 주면 그 값을 우선한다.
# 3.0 → 5.0 (2026-08-28). 3초 시작은 레이트리밋 창을 못 넘겼다.
LLM_BACKOFF_BASE = env_float("CLOVA_BACKOFF_BASE", 5.0)
LLM_BACKOFF_MAX = env_float("CLOVA_BACKOFF_MAX", 30.0)

# 질의 세트처럼 많은 문항을 연속 실행할 때 문항 사이에 두는 간격(초).
# 0 이면 쉬지 않는다. 레이트리밋이 잦으면 0.5~2 로 올린다.
# 0.0 → 0.5 (2026-08-28). 배치에서 문항을 쉬지 않고 붙여 보내면 누적 요청률이
# 올라가 뒤쪽 문항이 429 를 맞는다. 78문항 × 0.5초 = 39초로 비용이 작다.
# ⚠️ 이건 **배치 전용**이다. 서버는 요청을 하나씩 받으므로 효과가 없다.
REQUEST_GAP = env_float("PENSION_REQUEST_GAP", 0.5)

# ── 문서 채점 동시 요청 수 (2026-08-27 신설) ──────────────────────────
#
# ⚠️ REQUEST_GAP 은 **문항 사이** 간격이라 배치 실행에만 듣는다. 서버는 요청을
#    하나씩 받으므로 아무 효과가 없다. 6차 실행에서 P-B1 이 429 로 죽은 원인은
#    문항 사이 간격이 아니라 **질의 하나 안에서 나는 버스트**였다 —
#    document_grader 가 문서 k개를 8워커로 동시에 채점한다.
#
#    평가 시스템이 팀당 순차 1건이어도 이 버스트는 그대로 난다. 서버에서는
#    이 값을 낮추는 것이 429 를 막는 유일한 손잡이다.
GRADE_WORKERS = env_int("PENSION_GRADE_WORKERS", 4)

# ── 채점 전멸 시 살릴 검색 상위 건수 (2026-08-31 신설, 9차 이슈 5) ─────
#
# head 8건이 **전부** "no" 로 떨어지면 kept 가 빈 리스트가 되고, 그 뒤로
# 답변 생성이 "근거 0건 → 미확인 응답" 으로 빠져 게이트 거절까지 간다.
# 9차에 7건(78문항의 9%)이 이 경로였고, 8차와 겹치는 건 1건뿐이다 —
# 매 실행마다 **다른 문항**이 걸린다는 뜻이라 특정 질문의 문제가 아니다.
#
# 같은 파일 `_grade_one` 은 개별 채점이 실패하면 이미 통과시킨다("채점 실패는
# 문서를 버리는 근거가 못 된다"). 전멸도 같은 성격이다 — 8건이 전부 no 라는 건
# 채점기가 과하게 엄격했다는 신호일 수 있다.
#
# ⚠️ 삭제한 '상위 3건 폴백'과 다르다. 그건 답변과 무관한 청크를 **인용**으로
#    갖다 붙였고, 이건 검색 결과를 답변 생성에 넘기는 것뿐이다. 인용은 여전히
#    값 대조나 모델 판단을 거친다.
# 0 이면 폴백을 끈다.
GRADE_WIPEOUT_KEEP = env_int("PENSION_GRADE_WIPEOUT_KEEP", 2)

# ── 형제펀드(같은 상품군) 처리 (2026-08-31 신설, 9차 이슈 3) ───────────
#
# 코퍼스 92개 펀드 중 11개가 "○○증권전환형투자신탁1호(국공채/채권/주식)" 처럼
# **괄호 안 한 단어로만** 갈리는 형제 관계다(5개 군). 그중 4개 군은 등급이나
# 자산유형이 서로 다르다 — 뭉개지면 위험도가 반대로 나간다.
#
#   실측 P-B1: 채권 펀드 수익률을 국공채 펀드 값(5.91%)으로 답했다. 실제는 5.74%.
#              채권 펀드 문서는 애초에 검색조차 안 됐다 — 펀드명 필터가
#              형제 중 하나만 잡아 나머지를 배제했기 때문이다.
#
# ① 질문이 상품군을 지목하면 형제를 **전부** 필터에 넣는다 (query_analyzer)
# ② 지목된 펀드의 카드가 검색 결과에 없으면 채워 넣는다 (document_retriever)
# ③ 답변 프롬프트에 형제 목록을 명시해 수치를 섞지 못하게 한다 (answer_generator)
MAX_SIBLING_FUNDS = env_int("PENSION_MAX_SIBLING_FUNDS", 4)
ENSURE_NAMED_FUND_CARDS = env_bool("PENSION_ENSURE_FUND_CARDS", True)

# 합성(combine_answer)의 출력 상한. 답변 잘림(P-A4 류) 대응을 서버에서 조절할 수
# 있게 뺀다 (2026-08-29, 재희님 요청). 기본값 2500 그대로라 동작은 안 바뀐다.
COMBINE_MAX_TOKENS = env_int("PENSION_COMBINE_MAX_TOKENS", 2500)

# ── LLM 동시 요청 상한 (2026-08-29 신설, 재희님 요청 3번) ──────────────
#
# ■ 무엇이 문제인가
#   서브질문이 `Send` 로 병렬 fan-out 되는데(rag_worker × 최대 4) 그 fan-out 에는
#   동시 실행 상한이 없다. `GRADE_WORKERS` 는 **서브질문 하나 안의** 채점 병렬도만
#   막는다. 그래서 순간 최대 4 × 4 = 16 요청이 동시에 나갈 수 있다.
#
# ■ 왜 fan-out 이 아니라 여기인가
#   그래프 쪽에 세마포어를 걸면 fan-out 만 막힌다. 채점·검증·합성처럼 다른 곳에서
#   나가는 요청은 그대로다. **HTTP 호출 지점 하나**를 막으면 어디서 나가든 전부
#   걸린다 — 손댈 곳이 한 군데고 빠뜨릴 경로가 없다.
#
#   ⚠️ 재시도 대기(백오프)는 이 상한 **밖에서** 기다린다. 안 그러면 30초 자는
#      스레드가 남의 자리를 붙들고 있게 된다.
#
# ■ 기본값을 왜 0(꺼짐)으로 두나
#   9차 실행에서 429 가 해결됐다(재시도 6회 + 백오프 5초 + 문항 간격 0.5초).
#   지금 상한을 켜면 코랩 배치가 느려지기만 한다 — 서브질문 4개가 자리를
#   두고 다투면 채점 단계가 직렬화된다.
#   재희님도 "급한 건 아니고 429 계속 나면" 이라고 하셨으므로, 손잡이만
#   만들어 두고 기본은 끈다. 서버에서 429 가 나면 PENSION_LLM_CONCURRENCY=4
#   (또는 2) 로 켜면 된다.
#
# 0 이하면 상한을 끄고 예전처럼 무제한으로 보낸다.
LLM_MAX_CONCURRENCY = env_int("PENSION_LLM_CONCURRENCY", 0)


# ── 토큰 계측 (CLOVA Tokenizer API) ───────────────────────────────────
# POST /v1/api-tools/chat-tokenize/{modelName}
#   요청: {"messages":[{"role":..., "content":...}]}
#   응답: {"result":{"messages":[{..., "count": <int>}]}}
#
# Chat Completions 응답의 usage(promptTokens/completionTokens)만으로도 사후 집계는
# 되지만, 토크나이저 API 는 **호출 전에** 입력 토큰을 알 수 있다. 프롬프트를 고쳤을 때
# 비용이 얼마나 늘지 미리 재고, 컨텍스트 한도를 넘기지 않는지 사전에 확인하는 용도다.
TOKENIZE_PATH = env("CLOVA_TOKENIZE_PATH", "/v1/api-tools/chat-tokenize")
COUNT_TOKENS = env_bool("PENSION_COUNT_TOKENS", True)   # usage 기반 집계(무료)
PRECOUNT_TOKENS = env_bool("PENSION_PRECOUNT_TOKENS", False)  # 토크나이저 API 사전 호출

# ── 임베딩: multilingual-e5-large ─────────────────────────────────────
# CLOVA 의 clir-sts-dolphin 에서 교체했다.
#   · 오픈 가중치라 API 호출·과금·레이트리밋이 없다 → 전 코퍼스 재임베딩이 자유롭다
#   · 한국어 포함 100개 언어로 학습된 검색 특화 모델이고, MTEB 한국어 검색에서 강하다
#   · 1024차원으로 기존 캐시 구조(.npy)를 그대로 쓴다
#
# ⚠️ E5 계열은 **접두사가 필수**다. 질의에는 "query: ", 문서에는 "passage: " 를 붙인다.
#    이걸 빼면 성능이 눈에 띄게 떨어진다(같은 공간에 안 놓인다).
EMBED_MODEL = env("PENSION_EMBED_MODEL", "intfloat/multilingual-e5-large")
EMBED_DIM = env_int("PENSION_EMBED_DIM", 1024)
EMBED_QUERY_PREFIX = env("PENSION_EMBED_QUERY_PREFIX", "query: ")
EMBED_PASSAGE_PREFIX = env("PENSION_EMBED_PASSAGE_PREFIX", "passage: ")
EMBED_MAX_SEQ = env_int("PENSION_EMBED_MAX_SEQ", 512)  # E5 학습 길이

# ── 디바이스 / 배치 ───────────────────────────────────────────────────
# 코랩은 GPU(T4 등)를 붙일 수 있다. GPU 가 있으면 배치를 키워 3,853건 임베딩
# 시간을 크게 줄인다 (CPU 수십 분 → GPU 1~2분).
#   런타임 → 런타임 유형 변경 → 하드웨어 가속기: T4 GPU
def _auto_device() -> str:
    """"" 를 주면 sentence-transformers 가 알아서 고르지만, 로그에 무엇을 쓰는지
    남기려고 여기서 먼저 판별한다."""
    forced = env("PENSION_EMBED_DEVICE", "")
    if forced:
        return forced
    try:
        import torch                                 # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


EMBED_DEVICE = _auto_device()
EMBED_BATCH_SIZE = env_int("PENSION_EMBED_BATCH", 64 if EMBED_DEVICE == "cuda" else 32)

# ⚠️ 메모리 가드 — 끄지 말 것을 권한다.
# e5-large 는 fp32 가중치만 2.2GB 다. 여유 메모리가 부족한 상태에서 로드하면
# 파이썬 예외가 아니라 **세그폴트(exit 139)** 로 프로세스가 통째로 죽는다.
# try/except 로 못 잡으므로, 로드 전에 여유 메모리를 재고 모자라면 시도조차 하지 않는다.
# (그 경우 BM25 단독으로 계속 동작한다)
#
# 코랩 기본 런타임은 RAM 약 12GB 라 e5-large 가 문제없이 올라간다.
# 이 가드는 로컬 저사양 환경(개발 PC: 총 3.9GB)을 위한 것이다.
EMBED_MEMORY_GUARD = env_bool("PENSION_EMBED_MEMORY_GUARD", True)

# 캐시 파일명에 쓸 안전한 슬러그 ("intfloat/multilingual-e5-large" → "multilingual-e5-large")
EMBED_SLUG = EMBED_MODEL.split("/")[-1].replace(":", "-")


# ── 벡터 아티팩트: 빌드 단계와 서빙 단계의 경계 ───────────────────────
#
#   [빌드]  python -m index.build_index   → cache/index_<슬러그>/ 에 벡터 저장
#   [서빙]  get_index()                   → 그 폴더를 읽어 붙인다 (재임베딩 없음)
#
# 아티팩트가 없을 때 어떻게 할지를 정한다.
#   True  : 실패시킨다. 의미 검색이 꺼진 줄 모르고 돌리는 사고를 막는다(권장).
#   False : 경고만 하고 BM25 단독으로 진행한다.
#           임베딩을 못 쓰는 환경(메모리 부족 등)에서 일부러 쓰는 모드다.
#
# ※ 아티팩트가 **낡은** 경우는 이 설정과 무관하게 항상 실패한다.
#   없는 것은 선택일 수 있지만, 낡은 것은 언제나 버그이기 때문이다.
REQUIRE_VECTORS = env_bool("PENSION_REQUIRE_VECTORS", True)


# ==========================================================================
# 3. 네이버 웹 검색 (문서에서 못 찾았을 때의 최후 수단)
# ==========================================================================
NUM_WEB_RESULTS = env_int("PENSION_WEB_K", 5)


# ==========================================================================
# 4. 검색 파라미터
# ==========================================================================
NUM_DOCS_TO_RETRIEVE = env_int("PENSION_TOP_K", 8)

# ★ 상품 비중이 큰 질문일 때는 넓게 본다.
#   이유: 사실조회는 정답 청크가 1~2개로 좁지만, 추천·비교는 후보군을 폭넓게 본 뒤
#   답변 단계에서 조건으로 걸러야 답이 된다.
NUM_DOCS_RECOMMEND = env_int("PENSION_TOP_K_RECOMMEND", 24)

# ──────────────────────────────────────────────────────────────────────
# ■ 계층 분할 검색 (stratified retrieval) — 옛 scope 하드필터의 대체물
#
# 예전 구조: 서브질문마다 LLM 이 "추천인가?" 를 판단해
#     추천    → scope="all" (전체)
#     아니다  → scope="doc" (규정문서만, 투자설명서 7,339건을 **삭제**)
#
# 이 이분법의 문제는 **판단 1회의 실패 비용이 코퍼스 전체**라는 점이다.
#   · "미래에셋장기성장포커스 총보수 얼마?" 는 추천이 아니지만 답은 투자설명서에만 있다.
#     추천=false 로 찍히는 순간 정답 후보가 0건이 되고, 완화가 돌기 전까지 헛돈다.
#   · "세액공제 한도까지 채우고 그 돈으로 살 채권형 펀드도" 처럼 두 코퍼스가 **동시에**
#     필요한 질문은 어느 쪽으로 찍어도 절반이 사라진다.
#   · LLM 판정이라 실행마다 흔들리는데, 흔들림이 곧 정답 유실이다(측정 불가능해진다).
#
# 새 구조: **어느 코퍼스도 지우지 않는다.** 대신 top-k 안의 자리를 나눠 준다.
#
#     product_affinity  0.0 ~ 1.0   "이 질문이 개별 상품에 관한 정도"
#         → 투자설명서 몫 = round(top_k × affinity)
#         → 규정문서 몫   = 나머지
#         → 양쪽 모두 후보가 있으면 최소 MIN_STRATUM_QUOTA 자리는 보장한다
#
# 판단이 틀려도 **자리 배분이 기울 뿐 정답이 사라지지 않는다.** 이것이 하드필터와의
# 결정적 차이다. affinity 는 서브질문마다 LLM 이 3단계 라벨로 준다(아래 매핑).
PRODUCT_INTENT_LEVELS: dict[str, float] = {
    "제도": 0.15,   # 제도·세제·절차·운용규제를 묻는다
    "혼합": 0.50,   # 제도와 상품이 함께 필요하다
    "상품": 0.85,   # 개별 펀드의 이름·수치·클래스를 묻는다
}
DEFAULT_PRODUCT_INTENT = env_float("PENSION_DEFAULT_INTENT", 0.35)

# 양쪽 계층에 최소한으로 보장하는 자리 수. 0 으로 두면 옛 하드필터와 같아진다.
MIN_STRATUM_QUOTA = env_int("PENSION_MIN_STRATUM_QUOTA", 2)

# 한 문서(doc_id)가 top-k 를 독식하지 못하게 하는 상한.
#
# 섹션 원문이 7,156건으로 전체의 90%다. 상한이 없으면 "채권형 펀드 추천" 같은 질의의
# top-24 가 특정 펀드 한두 개의 원문 조각으로 도배되어, 정작 비교할 펀드가 후보에
# 들어오지 못한다. 근거 완전성(다른 펀드 근거 혼입)에도 직접 영향을 준다.
MAX_CHUNKS_PER_DOC = env_int("PENSION_MAX_PER_DOC", 3)
# 단, 질문이 특정 펀드 1개를 지목했다면 그 문서만 깊게 파야 하므로 상한을 푼다.
MAX_CHUNKS_PER_DOC_FOCUSED = env_int("PENSION_MAX_PER_DOC_FOCUSED", 12)

# 하이브리드 검색 가중치 (BM25 = 어휘 일치, VEC = 의미 유사도)
#
# 0.45/0.55 → 0.35/0.65 로 조정했다. multilingual-e5-large 는 검색 목적으로 학습된
# 모델이라 clir-sts-dolphin 보다 의미 매칭이 강하다. 다만 BM25 를 0 으로 두지는
# 않는다 — "환산급여공제", "C-P2e" 같은 도메인 고유어·코드는 여전히 어휘 일치가 이긴다.
BM25_WEIGHT = env_float("PENSION_BM25_W", 0.35)
VEC_WEIGHT = env_float("PENSION_VEC_W", 0.65)

# ── 출처 신뢰도 가중치 ────────────────────────────────────────────────
# 확정 사항: doc 문서(제도/세제/절차/운용규제) > 투자설명서.
# **corpus(장르)가 아니라 tier(출처)로 판단한다.** doc33/53/57/58 은 원본 source 가
# "상품" 이지만 실제로는 고신뢰 doc 문서다.
TIER_TRUST: dict[str, float] = {
    "doc": 1.00,
    "prospectus": 0.85,
}

# distractor=True (제도 파일의 '첨부-관련법령' 37건) 는 실제 답이 거의 안 나오는
# 들러리 청크다. 지우지 않고 점수만 깎는다 — 법령 원문이 진짜 답인 질문도 있으므로.
DISTRACTOR_PENALTY = env_float("PENSION_DISTRACTOR_PENALTY", 0.35)

# 표 청크는 정량/계산 질문에서 가치가 높고 서술형 질문에선 낮다.
#
# 판정 축이 둘이다 — 둘 중 하나만 맞아도 표로 본다.
#   · origin  : genre_b/d 가 원본 형태를 남겨 준다("table_row"/"calc_box_text")
#   · section : 그 밖의 코퍼스는 섹션명이 "표1"/"표2" 로 들어온다
# 예전 주석은 "현재 데이터에 origin 이 없다"고 적혀 있었는데, 2026-08-16 코퍼스부터
# genre_b(79건)·genre_d(125건)에 origin=table_row 가 실제로 존재한다. section 만 보면
# 세제 해설·업무매뉴얼의 표 행이 통째로 가산에서 빠진다(세율표·한도표가 정답인 구간).
TABLE_BOOST_ON_CALC = env_float("PENSION_TABLE_BOOST", 1.25)
TABLE_SECTION_PREFIXES = ("표",)
TABLE_ORIGINS = ("table_row", "calc_box_text")

# 카드(펀드/클래스) 가산 — 상품 질문일수록 세게 준다.
#
# 섹션 원문 7,156 : 카드 183 = 39:1 이라, 가중이 없으면 카드가 구조적으로 밀린다.
# 그런데 "이 펀드 총보수/위험등급/클래스" 류의 답은 **카드가 정본**이다.
# 실제 배수 = 1 + CARD_BOOST × product_affinity (상품 0.85 → ×1.21, 제도 0.15 → ×1.04)
#
# ⚠️ 0.5 → 0.25 로 내렸다 (2026-08-18). 근거: 78문항 실행에서 검색 결과 1,920건 중
#    카드가 46.1% 였다(코퍼스 비중 2.3%). 섹션 원문은 8.6%(코퍼스 90.1%).
#    환매수수료·기준가격·관련회사처럼 **카드에 없고 섹션에만 있는** 항목이
#    구조적으로 도달 불가가 됐다(P-F4·P-G0a·P-G0b).
#
#    ■ ablation 실측 (코퍼스 7,939건, top_k=8, affinity=0.85, BM25 단독)
#      섹션이 정답인 질문 4개의 섹션 확보:  0.50 → 16칸 / 0.25 → 23칸
#      카드가 정본인 질문 4개의 카드 확보:  0.50 → 11칸 / 0.25 → 10칸
#      섹션을 7칸 얻고 카드를 1칸 내주는 교환이다. 명확히 남는 장사다.
#
#    ⚠️ 처음에는 "이 값만 내려서는 안 고쳐지고 섹션 쿼터가 있어야 한다" 고 봤는데,
#       측정 결과는 반대였다 — **개선은 전부 이 값이 만들었고** 쿼터는 이득이
#       없었다(MIN_SECTION_QUOTA 주석의 표 참고). 나머지 병목이었던
#       product_affinity 오판은 query_analyzer 의 펀드명 목록 대조로 따로 고쳤다.
CARD_BOOST = env_float("PENSION_CARD_BOOST", 0.25)

# 투자설명서 몫 안에서 **섹션 원문**에 보장하는 최소 자리 수 (2026-08-18 신설).
#
# MIN_STRATUM_QUOTA 는 doc(규정문서) vs prospectus(투자설명서) **계층 사이**에만
# 걸린다. 투자설명서 안에서 카드 183건과 섹션 7,156건이 자리를 다투는 것은
# 아무도 조정하지 않았다. 이 값은 "투자설명서 자리 중 최소 N칸은 섹션에 준다"는 뜻.
# 카드 자리를 0 으로 만들지 않도록 실제 적용값은 min(MIN_SECTION_QUOTA, 몫//2) 이다.
#
# ══════════════════════════════════════════════════════════════════════
# ⚠️ 기본값 0 (비활성)이다 — **실측에서 이득이 없고 비용만 있었다.**
# ══════════════════════════════════════════════════════════════════════
#
# 처음에는 2로 넣었다. "CARD_BOOST 를 내리는 것만으로는 카드가 자리를 다 가져간다"
# 고 판단했기 때문이다. 그런데 ablation 을 돌려 보니 그 판단이 틀렸다.
#
#   조건: 실제 코퍼스 7,939건, top_k=8, product_affinity=0.85
#         섹션이 정답인 질문 4개(P-F4·P-G0a·P-G0b·P-E4)
#         카드가 정본인 질문 4개(총보수·위험등급·클래스자격·수익률)
#
#     설정                        섹션질문의 섹션 확보   카드질문의 카드 확보
#     boost .50 / 쿼터 0                  16                   11   ← 수정 전
#     boost .25 / 쿼터 0                  23                   10
#     boost .25 / 쿼터 1                  23                    9
#     boost .25 / 쿼터 2                  23                    8
#
#   읽는 법:
#     · 섹션 개선(16→23)은 **전부 CARD_BOOST 인하가 만들었다.**
#     · 쿼터는 섹션을 하나도 더 늘리지 못했다(23→23→23). 자연 순위가 이미
#       쿼터보다 많은 섹션을 뽑고 있어서 **바닥값이 걸리지 않는다.**
#     · 반대로 카드가 정본인 질문에서는 쿼터가 걸려서 카드를 밀어냈다
#       (C3 "C-P 클래스 가입자격": 6장 → 5장 → 4장).
#
#   즉 지금 데이터에서 쿼터는 **이득 0 / 비용 있음**이다. 켜 둘 이유가 없다.
#
# ■ 그런데 왜 코드는 남겨 두나
#
#   위 측정은 **벡터 아티팩트가 없어 BM25 단독**으로 돌린 것이다
#   (index/artifact 미생성 → 의미검색 가중 0.65 가 꺼진 상태).
#   원래 문제를 만든 실측(카드 46.1% / 섹션 8.6%)은 **벡터가 켜진** 실행이었다.
#   카드는 짧고 의미가 응축돼 있어 코사인에서 훨씬 유리하므로,
#   벡터를 켜면 카드가 다시 섹션을 밀어낼 가능성이 있다. 그때는 바닥값이 필요하다.
#
#   그래서 기제는 남기고 기본값만 끈다. 벡터를 다시 만든 뒤
#
#       python -m index.build_index
#       python scratchpad/ablation.py        # 위 표를 다시 뽑는다
#
#   에서 "쿼터 0" 의 섹션 확보가 다시 떨어지면 2로 올린다.
#   (환경변수 PENSION_MIN_SECTION_QUOTA 로 코드 수정 없이 바꿀 수 있다)
MIN_SECTION_QUOTA = env_int("PENSION_MIN_SECTION_QUOTA", 0)

# ⚠️ confidence 를 점수에 곱하지 않는다 (2026-08-16 계약 반영)
#
# 계약(INTERFACES.md §2-1)에서 confidence 는 **상수**로 확정됐다.
#     1.0  = 텍스트 추출 / 0.95 = OCR·좌표 재조립
# 즉 이 필드는 품질 점수가 아니라 **파싱 방식 표시**다. 여기에 곱셈을 걸면
# "OCR 로 읽은 문서는 내용과 무관하게 5% 감점" 이 되는데, 하필 OCR 84건은
# 운용규제·전환 FAQ 처럼 **정답이 그 문서에만 있는** 구간이다.
# (예전 데이터의 confidence 는 0.5~0.89 의 실측 품질값이었고, 그때는 곱셈이 맞았다.
#  같은 필드의 의미가 바뀌었으므로 로직도 바뀌어야 한다.)
#
# 품질 가중이 필요하면 fill_ratio 를 쓴다 — 다만 이 필드는 **투자설명서 카드에만**
# 있으므로 전역 곱셈에 쓰면 안 된다(안내문서가 일방적으로 불리해진다).
USE_CONFIDENCE_WEIGHT = env_bool("PENSION_USE_CONFIDENCE", False)


# ==========================================================================
# 5. 재시도 한도 — 에이전트가 무한루프에 빠지지 않게 하는 안전장치
# ==========================================================================
MAX_FILTER_RETRIES = env_int("PENSION_MAX_FILTER_RETRY", 1)   # 소프트필터 완화 재검색
MAX_GRADING_RETRIES = env_int("PENSION_MAX_GRADE_RETRY", 2)   # 문서 부족 → 질문 재작성
MAX_VERIFY_RETRIES = env_int("PENSION_MAX_VERIFY_RETRY", 1)   # 검증 실패 → 답변 재생성

# 질문 분해 (RRR: 잔여 해결 반복)
MAX_SUB_QUESTIONS = env_int("PENSION_MAX_SUBQ", 4)   # 한 라운드당 최대 서브질문
MAX_ROUNDS = env_int("PENSION_MAX_ROUNDS", 2)       # 최대 반복 라운드

# 문서 관련성 통과 최소 개수
DOCS_RELEVANCE_THRESHOLD = env_int("PENSION_DOC_THRESHOLD", 1)

# 범위 밖 판정을 뒤집는 코퍼스 신호 임계값 (raw BM25 top-1, 규정문서 대상).
#
# 안전검사는 1번 노드라 여기서 막히면 복구 경로가 없다. 오탐(정상 질문 차단)은
# 전체 실패고, 미탐(엉뚱한 질문 통과)은 LLM 몇 회 낭비로 끝난다. 비대칭이 크므로
# **의심스러우면 통과**시킨다. 이 값을 올리면 더 많이 막고, 내리면 더 많이 통과시킨다.
#
# 실측 분포: 도메인 안 29~49 / 도메인 밖 1.7~12.4  → 20 이면 깨끗하게 갈린다.
SCOPE_MIN_BM25 = env_float("PENSION_SCOPE_MIN_BM25", 20.0)

# 유해 판정(위법조력·유해)을 뒤집는 코퍼스 신호 임계값 (2026-08-18 신설).
#
# 범위 판정에만 있던 보정을 유해 판정에도 넣는다. 다만 **임계값을 더 높게** 둔다 —
# "도메인 안이다" 보다 "차단할 사안이 아니다" 가 더 무거운 주장이기 때문이다.
#
# 실측(safety_checker.py 상단 표): D-B4 류 규약 변경 질문 29.3,
# "돈 미리 빼 쓸 수 있나요" 31.1, IRP 중도인출 48.6 → 25 면 전부 통과한다.
# 도메인 밖 최고값은 12.4 라 여유가 있다.
#
# ⚠️ 개인정보 범주에는 이 보정이 적용되지 않는다(HARM_ALWAYS_BLOCK).
HARM_MIN_BM25 = env_float("PENSION_HARM_MIN_BM25", 25.0)

# 상품 질문 판정 기준 — BM25(투자설명서) / BM25(규정문서) 비율.
#
# 1.0 은 "두 코퍼스가 동점" 이라는 정의상 중립점이다. 평가 문항을 보고 정한 값이
# 아니므로 특정 질문에 맞춰진 게 없다. 올리면 더 보수적(제도 쪽), 내리면 더 공격적.
#
# 실측(78문항): 제도 트랙 37건 중 상품 오판 0건 / 상품 트랙 41건 중 26건 포착.
# 한쪽으로만 정확한 신호라 **자리 배분을 올리는 용도로만** 쓴다.
PRODUCT_SIGNAL_RATIO = env_float("PENSION_PRODUCT_RATIO", 1.0)

# ── LLM 호출 예산 ──────────────────────────────────────────────────────
#
# ⚠️ 예전에는 이 값이 **한 곳에서만** 검사됐다: edges/decomposition.py 의
#    should_repeat(), 즉 "RRR 라운드를 한 번 더 돌까?" 를 정하는 순간뿐이다.
#    서브질문 fan-out 안에서 도는 재검색·재작성·재생성 루프는 전혀 안 봤다.
#    그래서 예산 40 인데 실측 최대 95 회가 나왔다(중앙 29).
#
# 지금은 3중으로 건다.
#   1) 호출 레벨   llm.calls_for_question() 이 질문당 호출 수를 실시간으로 센다
#   2) 루프 레벨   재시도 분기(edges/*)가 예산을 넘겼으면 재시도를 포기한다
#   3) 하드 상한   LLM_CALL_HARD_CAP 을 넘으면 LLMError 로 즉시 끊는다
#
# 1·2 는 **부드러운** 제동이다 — 답변은 나오되 품질 향상 시도를 그만둔다.
# 3 은 폭주 방지용 최후 수단이다.
#
# ⚠️ 하드 상한을 60 → 120 으로 올렸다 (2026-08-19). 60 은 너무 빡빡했다.
#
#   실측(차단 11건 재실행): D-B4 가 60회를 넘겨 **문항 전체가 실패**했다.
#   상한에 걸리면 그때까지 쓴 호출이 통째로 버려지므로, 잘못 잡은 상한은
#   비용을 아끼는 게 아니라 비용을 버린다.
#
#   왜 40(소프트)과 60(하드) 사이 20회로 부족했나:
#     · 소프트 게이트는 **재시도 분기**에서만 검사한다. 그 사이의 검색·채점·
#       생성·검증 호출은 그대로 나간다.
#     · 서브질문이 병렬로 흩어지므로 여러 워커가 동시에 통과 판정을 받는다.
#     · 구조화 파싱이 실패하면 복구 호출이 붙어 **호출이 2배**가 된다.
#       (이 마지막 항목은 llm/clova_client.py 의 _repair_json 으로 상당 부분 없앴다)
#
#   40 은 "이 이상은 품질 향상 시도를 그만둔다"는 신호이고,
#   120 은 "이건 정상 실행이 아니다"는 차단선이다. 성격이 다르므로 간격을 둔다.
#   (실측 최대 95회 → 120 이면 정상 범위는 다 통과하고 폭주만 걸린다)
LLM_CALL_BUDGET = env_int("PENSION_LLM_BUDGET", 40)
LLM_CALL_HARD_CAP = env_int("PENSION_LLM_HARD_CAP", 120)


# ==========================================================================
# 6. 도메인 어휘 — 이제 '필터'가 아니라 '분류 라벨'이다
# ==========================================================================
# 상품 관여도 라벨. 서브질문마다 LLM 이 이 셋 중 하나를 고른다.
# **필터가 아니라 자리 배분값**이다 (PRODUCT_INTENT_LEVELS 참조).
PRODUCT_INTENTS = list(PRODUCT_INTENT_LEVELS)          # ["제도", "혼합", "상품"]

# 출처 계층. TIER_TRUST 의 키와 같다.
TIERS = ["doc", "prospectus"]

# 장르 라벨 — 필터가 아니라 **출처 표기와 리포트 집계**에만 쓴다.
CORPORA = ["제도", "세제", "절차", "중도인출절차", "운용규제", "상품"]

# 청크 형태 — 답변 프롬프트에서 "이 근거가 무엇인지" 알려주는 데 쓴다.
CHUNK_TYPES = ["qa", "tax", "procedure", "withdrawal",
               "fund_card", "class_card", "prospectus_section"]

# ──────────────────────────────────────────────────────────────────────
# ■ 슬롯(되물어야 답이 갈리는 값) — 지금은 '사유' 하나뿐이다
#
# 장르C(중도인출, doc46~50)는 **도입부 문장이 5개 문서 모두 동일**해서, 사유를 모르면
# BM25 도 벡터도 다섯을 구분하지 못한다. 텍스트가 실제로 같기 때문에 검색 기법으로는
# 풀 수 없는 문제다 — 질문 쪽에서 사유를 확정해야 한다.
#
# 값은 데이터(chunks_genre_c.jsonl 의 `사유` 필드)에 실재하는 5종뿐이다.
# LLM 이 이 목록 밖의 값을 뱉으면 버린다(없는 값으로 필터를 걸면 후보가 0건이 된다).
REASON_VALUES = ["요양", "파산개인회생", "전세보증금", "주택구입", "재난피해"]

# 사유를 확정하지 못했을 때 **되묻는다**(역질문). 끄면 사유 없이 그냥 검색한다.
CLARIFY_MISSING_SLOTS = env_bool("PENSION_CLARIFY", True)

# ── 되묻기 이어받기 (2026-08-27 신설) ──────────────────────────────────
#
# 되묻기는 그래프를 끝낸다. 사용자가 "IRP요" 라고 답해도 그 입력은 새 질문으로
# 처음부터 다시 돌아서, 원래 무엇을 물었는지가 사라진다.
# true 면 같은 session_id 의 다음 입력이 그 답으로 보일 때 **원 질문에 붙여**
# 다시 태운다 (workflows/followup.py).
#
# ⚠️ 대회 평가는 1요청 1응답이라 여기서는 작동하지 않는다. CLI·노트북·실제
#    서비스처럼 사람이 다음 턴을 칠 수 있는 곳에서만 의미가 있다.
#    평가 배치(eval.run_queryset)는 문항마다 session_id 가 달라 영향이 없다.
CLARIFY_FOLLOWUP = env_bool("PENSION_CLARIFY_FOLLOWUP", True)

# 되묻고 나서 답을 기다리는 시간(초). 지나면 새 질문으로 본다.
CLARIFY_FOLLOWUP_TTL = env_int("PENSION_CLARIFY_FOLLOWUP_TTL", 900)

# ── 되묻기 두 번째 축: 계좌유형 (2026-08-20 신설) ─────────────────────
#
# ■ 왜 필요한가 — 코퍼스가 계좌유형별로 **다른 규정**을 명시한다
#
#     doc48#p1#c001  "전월세사유 중도인출의 경우 DC 는 하나의 사업장에서 재직 중
#                     1회에 한하여 가능, **개인형IRP 가입자는 횟수제한 없음**"
#     doc46#p1#c001  "의료비총액이 … 12.5% 초과하는 경우 신청가능
#                     **(개인형IRP 는 … 무관)**"
#
#   즉 계좌유형을 모르면 어느 쪽으로 답해도 절반은 틀린다.
#   실측에서 이 축의 문항(D-A1·D-A2)이 되묻지 않고 한쪽으로 단정했다.
#
# ■ '사유' 축과 성격이 다르다
#     사유     — **필터 축**. genre_c 5개 문서의 도입부가 같아 검색이 못 가른다.
#                그래서 확정되면 소프트 필터로 넘긴다.
#     계좌유형 — **답변 축**. 같은 청크 안에 두 규정이 함께 적혀 있어 검색은
#                잘 되지만, 답변이 조건을 나누거나 되물어야 한다.
#                청크에 계좌유형 메타데이터가 없으므로 필터로 쓰지 않는다.
#
#   따라서 계좌유형은 **되묻기 판정에만** 쓰고 filters 에 넣지 않는다.
ACCOUNT_VALUES = ["DB", "DC", "IRP", "연금저축"]

# 질문에서 계좌유형을 결정론적으로 찾을 때 쓰는 표기 변형.
# 목록 대조라 LLM 판단이 아니고, 사용자가 이미 밝힌 계좌유형을 다시 묻지 않게 한다.
ACCOUNT_ALIASES = {
    "DB": ("DB", "확정급여", "디비형"),
    "DC": ("DC", "확정기여", "디씨형"),
    "IRP": ("IRP", "아이알피", "개인형퇴직연금", "개인형 퇴직연금"),
    "연금저축": ("연금저축", "연금 저축"),
}


# ==========================================================================
# 7. 질문유형 — 검색 폭과 라우팅을 좌우한다
# ==========================================================================
QUERY_TYPES = ["사실조회", "계산", "추천", "절차", "혼합"]

# HyDE(가상 문서 임베딩) 로 만든 가상 답변의 최대 길이.
HYDE_MAX_TOKENS = env_int("PENSION_HYDE_MAX_TOKENS", 400)


# ==========================================================================
# 8. 워크플로 스위치 — 여기를 바꾸면 그래프 모양이 바뀐다
# ==========================================================================
WORKFLOW_SETTINGS: dict[str, bool] = {
    "check_safety": True,          # 범위 이탈/유해 질문 차단
    "resolve_slots": True,         # 사유 등 되물어야 할 값 추출 + 역질문
    "decompose_question": True,    # 질문 분리 에이전트
    "metadata_filtering": True,    # 소프트필터(사유·펀드명) 사용 + 단계적 완화
    "grade_documents": True,       # 검색 결과 관련성 채점
    "assess_graded_documents": True,  # 문서 부족 시 재작성 분기
    "rewrite_with_hyde": True,     # 질문 재작성 시 HyDE(가상 답변)까지 생성해 함께 검색
    "check_hallucination": True,   # 검증 에이전트 - 환각
    "grade_answer": True,          # 검증 에이전트 - 품질
    "use_calculator": True,        # 연금 계산기 에이전트
    "use_strategy": True,          # 전략 에이전트
    "repeat_residual": True,       # RRR: 못 답한 부분 재분해
    "trace": True,                 # 실행 경로 기록
}


# ==========================================================================
# 9. 응답 정책
# ==========================================================================
# 문서 근거가 없으면 지어내지 말고 모른다고 답한다.
REFUSE_WHEN_NO_EVIDENCE = True

# ── 인용 0건이어도 **검증을 통과한** 서브답변은 살린다 (2026-08-28 신설) ──
#
# 위 게이트는 인용이 0건이면 답변을 통째로 버리고 고정 문구로 바꾼다.
# 그런데 8차에서 6문항이 여기 걸렸고, 그중 P-G0a 는 서브답변이
# **verified=True, score=0.9** 였다. 검색 1라운드에 검증까지 통과한 답을
# "근거를 찾지 못했습니다" 로 덮어썼다.
#
# 골드 기대가 "거절 시 실패" 인 문항이 그 안에 여럿이다(D-E3, D-E5 …).
#
# ⚠️ 이건 "틀린 출처를 붙이지 않는다" 는 원칙과 충돌하지 않는다.
#    출처를 지어내는 게 아니라, 출처 없이 답하되 **그 사실을 밝힌다.**
#    아래 고지 문구가 답변 끝에 붙는다.
KEEP_VERIFIED_WITHOUT_CITATION = env_bool("PENSION_KEEP_VERIFIED", True)

NO_CITATION_NOTE = (
    "※ 위 내용은 제공된 자료에서 직접 인용할 문장을 찾지 못했습니다. "
    "일반적인 제도 설명이므로 실제 적용 전에 가입하신 금융기관이나 "
    "관할 기관에 확인해 주세요.")

# ── 서브답변이 **전부** 검증에 떨어졌을 때 (2026-08-31 신설, 9차 이슈 1) ──
#
# 위 KEEP_VERIFIED_WITHOUT_CITATION 은 "인용 0건 + 검증 통과" 를 살리는 구제다.
# 그 반대 경우 — **인용은 붙었는데 서브답변이 전부 검증 실패** — 는 지금까지
# 아무 데도 안 걸렸다. 인용이 1건이라도 있으면 근거 게이트를 통과하고,
# combine 은 verified 값과 무관하게 모든 서브답변을 합성 프롬프트에 넣는다.
#
#   실측 P-A2: 서브질문 1개, verified=false·score=0.6, 인용 1건.
#              코퍼스에 없는 가공 펀드("A채권형펀드")를 최종 추천했다.
#   실측 P-B1: evidence_sufficient=False 인데 품질 0.8 로 통과.
#
# "빼는" 것은 위험하다 — 서브답변이 1개뿐이면 합성에 넣을 게 없어져 빈 답변이
# 나간다. 그래서 **살리되 고지한다.** 게이트 구조는 rescue 와 대칭이다.
KEEP_UNVERIFIED_WITH_NOTE = env_bool("PENSION_KEEP_UNVERIFIED", True)

UNVERIFIED_NOTE = (
    "※ 위 내용은 자동 검증 단계를 통과하지 못했습니다. "
    "수치·요건은 가입하신 금융기관이나 관할 기관에서 반드시 다시 확인해 주세요.")

# ── 코퍼스에 없는 자리표시자 상품명 고지 (2026-08-31 신설, 9차 이슈 1) ──
#
# P-A2 가 "A채권형펀드", "B혼합형펀드" 라는 **없는 상품**을 만들어 최종 추천했다.
# 이름의 변별부가 한 글자이거나 비어 있으면(A/B/○○/상품1) 자리표시자다 —
# 실제 상품명은 운용사명이 붙어 최소 두 음절 이상이다.
# 판정은 문자열 대조라 LLM 호출이 없고, 코퍼스 펀드명과 겹치면 걸리지 않는다.
FLAG_PLACEHOLDER_FUNDS = env_bool("PENSION_FLAG_PLACEHOLDER_FUNDS", True)

UNKNOWN_FUND_NOTE = (
    "※ 답변에 등장한 다음 상품명은 제공된 자료에서 확인되지 않았습니다: {names}. "
    "실제 판매 중인 상품이 아닐 수 있으니 상품 선택의 근거로 삼지 마세요.")

# ── 자산유형 서술 대조 (2026-08-31 신설, 9차 이슈 6-b) ─────────────────
#
# `unsupported_values` 는 문자열 대조라 **숫자로 안 쓴 오류**를 원리상 못 잡는다.
#   P-A2: 2등급 주식형 펀드 둘을 "과거 성과에서 안정성을 보여준다" 는 예시로 들고,
#         3등급 주식형을 "위험 낮고 채권 위주" 묶음에 넣었다. 등급을 숫자로 안 써서
#         지표에 하나도 안 잡혔다.
#
# 오탐을 줄이려고 조건을 좁게 잡는다 — 한 문장 안에서
#   ① 카드가 있는 펀드명이 나오고
#   ② 카드의 자산유형과 **반대되는** 유형 단어만 있고
#   ③ 맞는 유형 단어는 없을 때
# 만 잡는다. "신한코리아롱숏은 주식형이라 채권형보다 위험합니다" 같은 정상 문장은
# 두 단어가 같이 나오므로 걸리지 않는다.
#
# 답변을 고치지 않고 **카드 원값을 각주로 덧붙인다.** 판정이 틀렸어도 사실을 한 줄
# 더 적는 것뿐이라 손해가 없고, 맞았으면 위험도 역전을 바로잡는다.
CHECK_ASSET_CLAIM = env_bool("PENSION_CHECK_ASSET_CLAIM", True)

ASSET_CLAIM_NOTE = (
    "※ 자료 기준 상품 정보: {facts}. "
    "본문 서술과 다르면 이 값이 기준입니다.")

# 상품 추천 시 반드시 붙일 고지
PRODUCT_DISCLAIMER = (
    "※ 본 내용은 제공된 투자설명서 요약에 근거한 정보 제공이며 투자 권유가 아닙니다. "
    "투자 결정 전 반드시 최신 투자설명서와 간이투자설명서를 확인하세요."
)

# 소스 충돌 시 안내 문구 (doc > 투자설명서)
SOURCE_CONFLICT_NOTE = (
    "※ 제도·세제·절차 문서와 상품 투자설명서의 내용이 다를 경우, "
    "제도·세제·절차 문서를 우선했습니다."
)
