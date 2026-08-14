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
그래서 필터를 하나로 줄였다.

    scope : "doc" | "all"      ← 서브질문마다 LLM 이 판단
            추천 질문이 아니면 doc 문서(제도·세제·절차·운용규제)만 본다
            추천 질문이면 투자설명서(상품)까지 포함한다

여기에 선행모델의 company_name 과 성격이 같은 값 하나만 선택적으로 더한다.

    fund_name : 질문에 특정 펀드명이 명시됐을 때만. 1문서=1펀드라 100% 확정값이다.

나머지(주제·계좌유형·목적·자산유형·위험등급)는 **점수 가중이나 프롬프트로 내리고
하드 필터에서는 전부 뺐다.**
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# .env 로딩 (의존성 없이 직접 읽는다)
# --------------------------------------------------------------------------
def _load_dotenv() -> None:
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


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
def _resolve_data_dir() -> Path:
    """JSONL 이 실제로 있는 디렉토리를 찾는다.

    `.env` 의 PENSION_DATA_DIR 가 프로젝트 루트를 가리키는 실수가 잦아서
    (파일은 그 아래 data/ 에 있다), 지정한 경로에 JSONL 이 없고 하위 data/ 에
    있으면 그쪽을 쓴다. 설정 오타 하나로 코퍼스가 통째로 비는 걸 막는다.
    """
    p = Path(env("PENSION_DATA_DIR", str(ROOT / "data")))
    if not any(p.glob("*.jsonl")) and any((p / "data").glob("*.jsonl")):
        return p / "data"
    return p


DATA_DIR = _resolve_data_dir()

# 파일 → (corpus, tier, chunk_type) 매핑. loader.py 가 이 표를 보고 읽는다.
#
# ★ tier 가 핵심이다. 원본 JSONL 의 `source` 필드는 doc 문서 일부(doc33/53/57/58)가
#   "상품" 으로 찍혀 있어서, 그대로 쓰면 **고신뢰 doc 문서가 투자설명서와 같은 취급**을
#   받는다(신뢰도 가중에서 0.85 로 감점). 그래서 "장르(corpus)" 와 "출처 신뢰도(tier)" 를
#   별도 축으로 분리했다.
#
#   corpus 가 빈 문자열이면 레코드의 source 필드를 그대로 쓴다.
#   (doc22 는 한 파일 안에 절차 20건 + 세제 4건이 섞여 있어 파일 단위로 못 정한다)
CORPUS_FILES: list[tuple[str, str, str, str]] = [
    # 파일명                      corpus      tier          chunk_type
    ("chunks_genre_a.jsonl",     "제도",     "doc",        "qa"),
    ("chunks_doc22_ocr.jsonl",   "",         "doc",        "procedure"),
    ("chunks_doc33_ocr.jsonl",   "절차",     "doc",        "procedure"),
    ("chunks_doc53_ocr.jsonl",   "운용규제", "doc",        "procedure"),
    ("chunks_doc57_ocr.jsonl",   "절차",     "doc",        "procedure"),
    ("chunks_doc58_ocr.jsonl",   "운용규제", "doc",        "procedure"),
    ("fund_cards.jsonl",         "상품",     "prospectus", "fund_card"),
    ("class_cards.jsonl",        "상품",     "prospectus", "class_card"),
    ("section_chunks.jsonl",     "상품",     "prospectus", "prospectus_section"),
    # 아래는 예전 코퍼스. 파일이 있으면 자동으로 함께 로드된다(없으면 조용히 건너뜀).
    ("chunks_genre_b.jsonl",     "세제",     "doc",        "qa"),
    ("chunks_genre_c.jsonl",     "중도인출절차", "doc",     "procedure"),
    ("chunks_genre_d.jsonl",     "일반절차", "doc",        "procedure"),
]

CACHE_DIR = ROOT / "cache"          # 임베딩 캐시(.npy)
TABLES_DIR = ROOT / "data"          # 세율표 등 결정 테이블
RESULTS_DIR = ROOT / "eval" / "results"
GOLDSET_PATH = ROOT / "eval" / "goldset.jsonl"


# ==========================================================================
# 2. CLOVA Studio (LLM) — 단계별 모델 분리
# ==========================================================================
CLOVA_API_KEY = env("CLOVA_API_KEY", "")
CLOVA_BASE = env("CLOVA_API_BASE", "https://clovastudio.stream.ntruss.com")

# 드라이런: LLM 을 부르지 않고 규칙 기반 응답으로 파이프라인 배선만 검증한다.
# 키가 없으면 자동으로 켜진다 (llm/__init__.py).
DRY_RUN = env_bool("PENSION_DRY_RUN", False)

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

LLM_TIMEOUT = env_float("CLOVA_TIMEOUT", 60.0)
LLM_MAX_RETRY = env_int("CLOVA_MAX_RETRY", 3)     # HTTP 레벨 재시도
LLM_TEMPERATURE = env_float("CLOVA_TEMPERATURE", 0.0)

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
EMBED_BATCH_SIZE = env_int("PENSION_EMBED_BATCH", 32)
EMBED_DEVICE = env("PENSION_EMBED_DEVICE", "")        # "" = 자동(cuda 있으면 cuda)
EMBED_MAX_SEQ = env_int("PENSION_EMBED_MAX_SEQ", 512)  # E5 학습 길이

# ⚠️ 메모리 가드 — 끄지 말 것을 권한다.
# e5-large 는 fp32 가중치만 2.2GB 다. 여유 메모리가 부족한 상태에서 로드하면
# 파이썬 예외가 아니라 **세그폴트(exit 139)** 로 프로세스가 통째로 죽는다.
# try/except 로 못 잡으므로, 로드 전에 여유 메모리를 재고 모자라면 시도조차 하지 않는다.
# (그 경우 BM25 단독으로 계속 동작한다)
EMBED_MEMORY_GUARD = env_bool("PENSION_EMBED_MEMORY_GUARD", True)

# 캐시 파일명에 쓸 안전한 슬러그 ("intfloat/multilingual-e5-large" → "multilingual-e5-large")
EMBED_SLUG = EMBED_MODEL.split("/")[-1].replace(":", "-")


# ==========================================================================
# 3. 네이버 웹 검색 (문서에서 못 찾았을 때의 최후 수단)
# ==========================================================================
NAVER_CLIENT_ID = env("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = env("NAVER_CLIENT_SECRET", "")
NAVER_SEARCH_URL = "https://openapi.naver.com/v1/search/webkr.json"
NUM_WEB_RESULTS = env_int("PENSION_WEB_K", 5)


# ==========================================================================
# 4. 검색 파라미터
# ==========================================================================
NUM_DOCS_TO_RETRIEVE = env_int("PENSION_TOP_K", 5)

# ★ 추천 질문일 때는 넓게 본다.
#   이유: 사실조회는 정답 청크가 1~2개로 좁지만, 추천은 후보군을 폭넓게 본 뒤
#   답변 단계에서 조건으로 걸러야 답이 된다.
NUM_DOCS_RECOMMEND = env_int("PENSION_TOP_K_RECOMMEND", 24)

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
# ⚠️ 예전에는 `origin == "table_row"` 로 판정했는데 현재 데이터에 origin 필드가 없어
#    한 번도 발동하지 않았다. 실제 표는 section 이 "표1"/"표2" 로 들어온다.
TABLE_BOOST_ON_CALC = env_float("PENSION_TABLE_BOOST", 1.25)
TABLE_SECTION_PREFIXES = ("표",)

# 파싱 신뢰도(confidence)를 점수에 반영한다.
# class_cards 의 좌표 파싱(parse_method="coord") 23건은 confidence 0.5~0.55 로,
# 표가 어긋났을 가능성이 있다. 0 으로 만들지는 않되 순위에서 뒤로 민다.
USE_CONFIDENCE_WEIGHT = env_bool("PENSION_USE_CONFIDENCE", True)


# ==========================================================================
# 5. 재시도 한도 — 에이전트가 무한루프에 빠지지 않게 하는 안전장치
# ==========================================================================
MAX_FILTER_RETRIES = env_int("PENSION_MAX_FILTER_RETRY", 1)   # scope 완화 재검색
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

# 전체 LLM 호출 예산 (성능 측정 시 비용 폭주 방지)
LLM_CALL_BUDGET = env_int("PENSION_LLM_BUDGET", 40)


# ==========================================================================
# 6. 도메인 어휘 — 이제 '필터'가 아니라 '분류 라벨'이다
# ==========================================================================
# 검색 범위. 서브질문마다 LLM 이 이 둘 중 하나를 고른다.
#   "doc" : 제도·세제·절차·운용규제 문서만  (사실조회·절차·계산 질문)
#   "all" : doc + 투자설명서(상품)          (추천 질문)
SCOPES = ["doc", "all"]
DEFAULT_SCOPE = "doc"

# 출처 계층. TIER_TRUST 의 키와 같다.
TIERS = ["doc", "prospectus"]

# 장르 라벨 — 필터가 아니라 **출처 표기와 리포트 집계**에만 쓴다.
CORPORA = ["제도", "세제", "절차", "일반절차", "중도인출절차", "운용규제", "상품"]

# 청크 형태 — 답변 프롬프트에서 "이 근거가 무엇인지" 알려주는 데 쓴다.
CHUNK_TYPES = ["qa", "procedure", "fund_card", "class_card", "prospectus_section"]


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
    "decompose_question": True,    # 질문 분리 에이전트
    "metadata_filtering": True,    # scope 필터로 검색 범위 축소
    "grade_documents": True,       # 검색 결과 관련성 채점
    "assess_graded_documents": True,  # 문서 부족 시 재작성/웹검색 분기
    "rewrite_with_hyde": True,     # 질문 재작성 시 HyDE(가상 답변)까지 생성해 함께 검색
    "check_hallucination": True,   # 검증 에이전트 - 환각
    "grade_answer": True,          # 검증 에이전트 - 품질
    "use_calculator": True,        # 연금 계산기 에이전트
    "use_strategy": True,          # 전략 에이전트
    "web_search_fallback": True,   # 네이버 웹검색 폴백
    "repeat_residual": True,       # RRR: 못 답한 부분 재분해
    "trace": True,                 # 실행 경로 기록
}


# ==========================================================================
# 9. 응답 정책
# ==========================================================================
# 문서 근거가 없으면 지어내지 말고 모른다고 답한다.
REFUSE_WHEN_NO_EVIDENCE = True

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
