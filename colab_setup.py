"""코랩 실행 준비 — 노트북 첫 셀에서 부르는 헬퍼.

    import colab_setup
    colab_setup.setup()

이 모듈이 하는 일은 **환경 준비뿐**이다. 파이프라인 로직은 로컬 원본과 완전히 같다.

  1. 의존성 설치 (이미 있으면 건너뜀)
  2. 구글 드라이브 마운트 → 임베딩 캐시를 드라이브에 둔다 (선택)
     · 코랩 로컬 디스크는 세션이 끊기면 사라진다. 3,853건 임베딩을 매번 다시
       계산하지 않으려면 캐시를 드라이브에 둬야 한다.
  3. API 키 확인 (Colab Secrets → 환경변수)
  4. 데이터 위치 확인
  5. 환경 리포트 출력

⚠️ 이 모듈은 **config 를 import 하기 전에** 호출해야 한다.
   config 는 import 시점에 환경변수를 읽어 경로·모델을 확정하기 때문이다.
   (이미 import 했다면 `colab_setup.setup(reload=True)` 로 다시 읽힌다)
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

# 노트북에서 그대로 쓰는 이름들
__all__ = ["setup", "in_colab", "install_deps", "mount_drive", "report"]


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.isdir("/content")


# ==========================================================================
# 1. 의존성
# ==========================================================================
_PKGS = [
    ("langgraph", "langgraph>=0.2.0"),
    ("pydantic", "pydantic>=2.0"),
    ("numpy", "numpy>=1.24"),
    ("sentence_transformers", "sentence-transformers>=3.0"),
    ("psutil", "psutil>=5.9"),
]


def install_deps(quiet: bool = True) -> None:
    """없는 것만 설치한다. 코랩은 numpy·torch 가 이미 깔려 있다."""
    missing = []
    for mod, spec in _PKGS:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(spec)

    if not missing:
        print("의존성 확인 — 이미 전부 설치됨")
        return

    print(f"설치 중: {', '.join(missing)}")
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd += missing
    subprocess.run(cmd, check=True)
    print("설치 완료")


# ==========================================================================
# 2. 드라이브 (임베딩 캐시 영속화)
# ==========================================================================
def mount_drive(cache_subdir: str = "pension_cache") -> Path | None:
    """구글 드라이브를 마운트하고 캐시 경로를 환경변수에 설정한다.

    코랩 런타임이 끊기면 /content 는 사라진다. 임베딩 캐시(3,853건)와 벡터
    행렬(.npy, 약 15MB)을 드라이브에 두면 다음 세션에서 재계산이 필요 없다.
    """
    if not in_colab():
        print("코랩이 아니므로 드라이브 마운트를 건너뜁니다.")
        return None
    try:
        from google.colab import drive              # type: ignore
    except ImportError:
        print("google.colab 을 불러올 수 없어 드라이브 마운트를 건너뜁니다.")
        return None

    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")

    cache = Path("/content/drive/MyDrive") / cache_subdir
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["PENSION_CACHE_DIR"] = str(cache)
    os.environ.setdefault("PENSION_RESULTS_DIR", str(cache / "eval_results"))
    print(f"임베딩 캐시 → {cache}  (세션이 끊겨도 유지됩니다)")
    return cache


# ==========================================================================
# 3. API 키
# ==========================================================================
def _load_secret(name: str) -> str:
    """Colab Secrets → 환경변수 순으로 찾는다."""
    v = os.getenv(name)
    if v:
        return v
    if in_colab():
        try:
            from google.colab import userdata       # type: ignore
            v = userdata.get(name)
            if v:
                os.environ[name] = str(v)
                return str(v)
        except Exception:
            pass
    return ""


def check_keys() -> dict:
    """키 상태를 확인한다.

    키가 없으면 **파이프라인 실행은 안 된다.** 예전에는 규칙 기반 드라이런으로
    조용히 전환됐는데, 그 결과는 답변 품질을 대변하지 못하면서 기록 파일에는
    실제 실행과 똑같은 모양으로 남아 나중에 구분이 안 됐다. 그래서 없앴다.

    키 없이도 되는 것: 코퍼스 로딩, 벡터 아티팩트 빌드, 검색 단독 점검.
    """
    clova = _load_secret("CLOVA_API_KEY")

    if not clova:
        print(
            "CLOVA_API_KEY 없음 — 파이프라인 실행(질문·질의세트)은 불가합니다.\n"
            "  키 없이 가능한 것: 코퍼스 로딩 / 벡터 아티팩트 빌드 / 검색 단독 점검\n"
            "  키를 넣으려면: 왼쪽 사이드바 🔑 → 이름 CLOVA_API_KEY → 값 입력 →\n"
            "                 '노트북 액세스' 켜기 → 이 셀 다시 실행"
        )
    return {"clova": bool(clova)}


# ==========================================================================
# 4. 데이터 위치
# ==========================================================================
_REQUIRED = ["chunks_genre_a.jsonl", "fund_cards.jsonl",
             "class_cards.jsonl", "section_chunks.jsonl"]


def find_data() -> Path | None:
    """JSONL 이 실제로 있는 폴더를 찾아 PENSION_DATA_DIR 로 고정한다."""
    here = Path(__file__).resolve().parent
    cands = [
        here / "data",
        Path("/content/pension_agent_colab/data"),
        Path("/content/pension_agent/data"),
        Path("/content/data"),
        Path("/content/drive/MyDrive/pension_agent_colab/data"),
        Path("/content/drive/MyDrive/pension_agent/data"),
        Path("/content/drive/MyDrive/pension_data"),
    ]
    for c in cands:
        try:
            if any(c.glob("*.jsonl")):
                os.environ["PENSION_DATA_DIR"] = str(c)
                have = sum(1 for f in _REQUIRED if (c / f).exists())
                print(f"데이터 폴더 → {c}  (주요 파일 {have}/{len(_REQUIRED)}개 확인)")
                return c
        except OSError:
            continue

    print(
        "데이터 폴더를 찾지 못했습니다.\n"
        "  다음 중 하나를 하세요:\n"
        "   ① 이 폴더를 통째로 업로드했다면 data/ 하위에 JSONL 이 있는지 확인\n"
        "   ② 드라이브에 뒀다면: colab_setup.mount_drive() 후\n"
        "      os.environ['PENSION_DATA_DIR'] = '/content/drive/MyDrive/…/data'\n"
        "   ③ 직접 업로드: 왼쪽 파일 탭 → data 폴더 생성 → JSONL 끌어다 놓기"
    )
    return None


# ==========================================================================
# 5. 리포트
# ==========================================================================
def report() -> None:
    """확정된 설정을 출력한다. config 를 import 한 뒤에 부를 것."""
    import config

    print("\n" + "=" * 66)
    print("실행 환경")
    print("=" * 66)
    print(f"  코랩            : {config.IS_COLAB}")
    print(f"  데이터          : {config.DATA_DIR}")
    print(f"  캐시            : {config.CACHE_DIR}")
    print(f"  API 키          : {'있음' if config.CLOVA_API_KEY else '없음 (파이프라인 실행 불가)'}")
    print(f"  모델(판정/채점) : {config.MODEL_PLAN} / {config.MODEL_GRADE}")
    print(f"  모델(생성/검증) : {config.MODEL_ANSWER} / {config.MODEL_VERIFY}")
    print(f"  임베딩          : {config.EMBED_MODEL}")
    print(f"  디바이스        : {config.EMBED_DEVICE} (배치 {config.EMBED_BATCH_SIZE})")
    print(f"  하이브리드      : BM25 {config.BM25_WEIGHT} : 벡터 {config.VEC_WEIGHT}")

    try:
        import torch                                 # type: ignore
        if torch.cuda.is_available():
            print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        else:
            print("  GPU             : 없음 (런타임 유형 변경 → T4 GPU 권장)")
    except Exception:
        pass

    # 벡터 아티팩트 — 빌드 단계 산출물. 드라이브에 두면 세션이 끊겨도 유지된다.
    try:
        from index import artifact
        info = artifact.describe()
        if info["exists"]:
            print(f"  벡터 아티팩트   : {info['n_vectors']}건 × {info['embed_dim']}차원, "
                  f"{info['size_mb']}MB (빌드 {info['built_at']})")
            print(f"                    {info['path']}")
        else:
            print("  벡터 아티팩트   : 없음 → `!python -m index.build_index` 필요")
            print(f"                    {info['path']}")
    except Exception as e:
        print(f"  벡터 아티팩트   : 확인 실패 ({e})")
    print("=" * 66)


# ==========================================================================
# 통합 진입점
# ==========================================================================
def setup(*, use_drive: bool = True, install: bool = True,
          reload: bool = False) -> None:
    """노트북 1번 셀에서 이것 하나만 부르면 된다."""
    print("=" * 66)
    print("연금 에이전트 — 코랩 준비")
    print("=" * 66)

    if install:
        install_deps()
    if use_drive:
        mount_drive()
    check_keys()
    find_data()

    # config 가 이미 import 돼 있으면 새 환경변수를 못 봤으므로 다시 읽힌다.
    if reload or "config" in sys.modules:
        for m in list(sys.modules):
            if m == "config" or m.split(".")[0] in {
                "llm", "index", "nodes", "edges", "workflows", "prompt", "state"
            }:
                sys.modules.pop(m, None)
        print("모듈 캐시를 비웠습니다 (새 설정으로 다시 import 됩니다).")

    report()
