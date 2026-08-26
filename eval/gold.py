"""골드셋 로더 — `eval/gold/` 두 파일을 합쳐 읽는다.

■ 왜 파일이 둘인가 (골드셋_형식_확정_0820 §11)

    eval/gold/
    ├─ product.jsonl   상품 트랙 (민서)  16건 = dev 11 / holdout 5
    └─ policy.jsonl    제도 트랙 (재희)  24건 = dev 19 / holdout 5

  트랙마다 코퍼스 구조와 함정이 다르다(같은 클래스 코드가 문서마다 다른 뜻,
  `(코드규칙)` 추론 표기 등). 담당자가 자기 트랙을 쓰는 편이 안전해서
  파일을 나눴고, `id` 접두어(`P-` / `D-`)로 트랙이 구분된다.

■ 파일이 없으면 **조용히 넘어가지 않는다**

  예전 `GOLDSET_PATH` 는 `eval/goldset.jsonl` 파일 하나를 가리켰다.
  그런데 아무도 그 이름으로 만들지 않으므로, 그대로 두면 **빈 리스트로 조용히
  돌면서 "채점 대상 0건"** 이 된다. 코퍼스 파일 누락으로 같은 사고가 세 번
  있었기 때문에(config.REQUIRED_CORPUS_FILES 주석 참고) 여기서는 처음부터
  **없으면 실패**시킨다.

■ `//` 주석을 허용하지 않는다

  `queryset.jsonl` 이 jsonc 라 표준 `json.loads` 가 첫 줄부터 막히는 문제가
  있었다. 골드셋은 순수 jsonl 이고, 설명은 `note` 필드에 넣는다.
  그래도 실수로 들어온 주석 줄은 읽을 때 걸러 주되 **경고를 남긴다.**
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

log = logging.getLogger(__name__)

TRACK_FILES = {"product": "product.jsonl", "policy": "policy.jsonl"}

# 형식 확정본이 못박은 enum — 오타가 들어오면 집계가 조용히 깨진다
VALID_KINDS = {"plain", "correction", "clarify", "calc",
               "absent", "distinct", "conflict", "refuse_ok"}
VALID_BLOCKS = {"none", "개인정보", "위법조력", "유해", "범위밖"}
VALID_CAUTIONS = {"none", "위법조력", "유해", "프롬프트공격"}
VALID_TESTS = {"정확성", "근거완전성", "요구사항충족", "근거기반",
               "추론논리성", "안전성신뢰성", "정보한계대응", "근거문서표시"}
VALID_SPLITS = {"dev", "holdout"}


def gold_dir() -> Path:
    return Path(config.GOLD_DIR)


def _read(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for i, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("//"):
                log.warning("%s:%d 주석 줄을 건너뜁니다 — 골드셋은 순수 jsonl 입니다",
                            path.name, i)
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}:{i} JSON 파싱 실패 — {e}") from e
    return rows


def validate(items: List[Dict[str, Any]]) -> List[str]:
    """형식 위반을 모아 돌려준다. **빈 리스트면 정상.**

    채점 도중에 터지는 것보다 시작 전에 한 번에 보여 주는 편이 낫다.
    """
    problems: List[str] = []
    seen: Dict[str, str] = {}

    for it in items:
        gid = it.get("id") or "(id 없음)"
        if not it.get("id"):
            problems.append(f"{gid}: id 가 없다")
        elif gid in seen:
            problems.append(f"{gid}: id 중복")
        else:
            seen[gid] = gid

        if not it.get("question"):
            problems.append(f"{gid}: question 이 없다")

        kind = it.get("kind")
        if kind not in VALID_KINDS:
            problems.append(f"{gid}: kind={kind!r} — {sorted(VALID_KINDS)} 중 하나여야 한다")

        split = it.get("split")
        if split not in VALID_SPLITS:
            problems.append(f"{gid}: split={split!r} — dev|holdout")

        blk = it.get("expect_block", "none")
        if blk not in VALID_BLOCKS:
            problems.append(f"{gid}: expect_block={blk!r}")
        cau = it.get("expect_caution", "none")
        if cau not in VALID_CAUTIONS:
            problems.append(f"{gid}: expect_caution={cau!r}")

        for t in it.get("tests") or []:
            if t not in VALID_TESTS:
                problems.append(f"{gid}: tests 에 {t!r} — enum 밖 값 "
                                f"(중점·괄호·띄어쓰기 없이 쓴다)")

        # kind 별 필수 필드
        if kind == "calc" and not (it.get("calc") or {}).get("value") is not None:
            problems.append(f"{gid}: kind=calc 인데 calc.value 가 없다")
        if kind == "clarify" and not it.get("clarify_slots"):
            problems.append(f"{gid}: kind=clarify 인데 clarify_slots 가 없다")
        if kind == "conflict" and not it.get("conflict_values"):
            problems.append(f"{gid}: kind=conflict 인데 conflict_values 가 없다")
        if kind == "refuse_ok" and it.get("expect_block", "none") == "none":
            problems.append(f"{gid}: kind=refuse_ok 인데 expect_block 이 none")

        # clarify_slots 는 **동의어 묶음**(배열의 배열)이어야 한다 (§4-3)
        for g in it.get("clarify_slots") or []:
            if isinstance(g, str):
                problems.append(f"{gid}: clarify_slots 원소가 문자열이다 — "
                                f"동의어 묶음(배열)이어야 매칭이 산다")
                break

        # evidence 는 딕셔너리이거나 null (§1)
        ev = it.get("evidence", "MISSING")
        if ev != "MISSING" and ev is not None and not isinstance(ev, dict):
            problems.append(f"{gid}: evidence 가 딕셔너리가 아니다 ({type(ev).__name__})")

        # absent 는 evidence 를 비우지 않는다 (§10)
        if kind == "absent" and not isinstance(ev, dict):
            problems.append(f"{gid}: kind=absent 는 evidence 가 있어야 한다 "
                            f"— '정답이 있어야 했을 자리'를 가리킨다")

    return problems


def load_goldset(split: str = "", track: str = "",
                 gold_dir_override: Optional[str] = None,
                 strict: bool = True) -> List[Dict[str, Any]]:
    """골드셋을 읽는다.

    split : "" | "dev" | "holdout"
    track : "" | "product" | "policy"
    strict: 형식 위반이 있으면 예외를 던진다 (기본 True)
    """
    d = Path(gold_dir_override) if gold_dir_override else gold_dir()
    if not d.is_dir():
        raise FileNotFoundError(
            f"골드셋 폴더가 없습니다: {d}\n"
            f"  {sorted(TRACK_FILES.values())} 를 이 폴더에 두세요.\n"
            "  (PENSION_GOLD_DIR 로 경로를 바꿀 수 있습니다)")

    wanted = ({track: TRACK_FILES[track]} if track in TRACK_FILES else TRACK_FILES)
    items: List[Dict[str, Any]] = []
    missing: List[str] = []

    for tname, fname in wanted.items():
        p = d / fname
        if not p.exists():
            missing.append(fname)
            continue
        rows = _read(p)
        for r in rows:
            r.setdefault("track", tname)
        items += rows
        log.info("골드셋 로드 %-14s %3d건", fname, len(rows))

    if missing:
        # 한 트랙만 요청했는데 그게 없으면 실패. 전체 요청이면 둘 다 없을 때만 실패.
        if track or len(missing) == len(wanted):
            raise FileNotFoundError(
                f"골드셋 파일이 없습니다: {', '.join(missing)}\n"
                f"  경로: {d}\n"
                "  파일 누락을 조용히 넘기면 '채점 대상 0건'으로 도는 사고가 납니다.")
        log.warning("골드셋 일부 없음(건너뜀): %s", ", ".join(missing))

    if split:
        items = [x for x in items if (x.get("split") or "dev") == split]

    problems = validate(items)
    if problems:
        msg = "골드셋 형식 위반 %d건:\n  " % len(problems) + "\n  ".join(problems[:20])
        if strict:
            raise ValueError(msg)
        log.warning(msg)

    return items


def stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """구성 요약 — 로드 직후 눈으로 확인하는 용도."""
    from collections import Counter
    return {
        "total": len(items),
        "by_split": dict(Counter(i.get("split") for i in items)),
        "by_track": dict(Counter(i.get("track") for i in items)),
        "by_kind": dict(Counter(i.get("kind") for i in items)),
        "with_evidence": sum(1 for i in items if isinstance(i.get("evidence"), dict)),
        "expect_block": dict(Counter(i.get("expect_block", "none") for i in items)),
        "expect_caution": dict(Counter(i.get("expect_caution", "none") for i in items)),
    }
