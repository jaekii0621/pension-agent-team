"""JSONL 코퍼스 로딩 + 스키마 정규화.

■ 선행모델과의 가장 큰 구조적 차이
  선행모델은 PDF 를 OpenParse 로 파싱하고, 비전 LLM 으로 표를 읽고, 첫 10페이지를
  LLM 에 보내 회사명·연도를 뽑는 **인덱싱 파이프라인**을 직접 돌렸다.
  우리는 그 단계가 이미 끝나 있다(파싱 완료 + 스키마 확정). 그래서 이 파일은
  "PDF → 청크"가 아니라 **"JSONL → 통일된 청크"** 를 담당한다.

■ 규칙 기반 태그 파생을 걷어냈다 (2026-08 개편)
  예전에는 본문 정규식으로 `계좌유형`·`금융주제`·`금융목적` 을 붙이고 그걸 검색
  필터로 썼다. 실측 결과 세 가지 모두 필터로 쓸 수 없었다.

    · 금융주제 : 상품종류 84.4% / 적립금운용 43.5% 로 변별력이 없고,
                 반대로 펀드카드에는 5종만 붙어서 그 밖의 주제가 필터로 들어오면
                 92장이 통째로 배제된다 (실측: "세액공제" 필터 → fund_card 0건)
    · 계좌유형 : 88.7% 가 미태깅이라 필터로서 무효
    · 금융목적 : 원본에 '사유' 필드가 없어 항상 None (죽은 필터)

  선행모델도 같은 결론에 도달해 topics 를 하드 필터에서 뺐다
  (METADATA_FILTER_INIT = ["company_name", "year"]).
  → 우리도 태그를 만들지 않는다. 만들지 않으면 잘못 걸 일도 없다.

■ 대신 **문서 단위로 확정되는 값**만 청크에 박는다
  선행모델이 문서 앞 10페이지에서 뽑은 company_name/year 를 그 문서의 모든 청크에
  복사한 것과 같다. 우리 쪽 대응물:

    tier        doc | prospectus       ← 출처 신뢰도. 파일에서 확정
    corpus      제도/세제/절차/운용규제/상품 ← 장르. 파일(또는 레코드 source)에서 확정
    chunk_type  qa/procedure/fund_card/class_card/prospectus_section
    fund_*      투자설명서 1권 = 펀드 1개이므로 그 문서의 전 청크에 참

  전부 **누락이 구조적으로 불가능한 값**이라 하드 필터로 써도 정답이 사라지지 않는다.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

log = logging.getLogger(__name__)


# ==========================================================================
# 정규화
# ==========================================================================
def _normalize(raw: dict, corpus: str, tier: str, chunk_type: str) -> dict:
    """파일마다 다른 필드를 하나의 청크 스키마로 통일한다.

    `corpus` 가 빈 문자열이면 레코드의 source 필드를 그대로 쓴다.
    (doc22 는 한 파일 안에 절차 20건 + 세제 4건이 섞여 있어 파일 단위로 못 정한다)
    """
    text = raw.get("text", "") or ""
    section = raw.get("section")

    chunk: Dict[str, Any] = {
        # --- 공통 (전 파일 보유) ---
        "chunk_id": raw.get("chunk_id", ""),
        "doc_id": raw.get("doc_id", ""),
        "corpus": corpus or raw.get("source") or "기타",
        "tier": tier,
        "chunk_type": chunk_type,
        "page": raw.get("page"),
        "section": section,
        "text": text,
        # 임베딩·BM25 가 볼 텍스트. 파서가 만들어 둔 embed_text 가 있으면 그것을 쓴다
        # (펀드명·섹션명이 이미 앞에 붙어 있는 형태).
        "embed_text": raw.get("embed_text") or "",
        "parse_method": raw.get("parse_method", "text"),
        # ⚠️ 상수다(1.0=텍스트, 0.95=OCR·좌표). 품질 점수가 아니므로 랭킹에 곱하지 않는다.
        #    config.USE_CONFIDENCE_WEIGHT 주석 참고.
        "confidence": float(raw.get("confidence", 1.0) or 1.0),
        "distractor": bool(raw.get("distractor", False)),

        # --- 원본 형태 (genre_b/d) — 표 가산 판정에 쓴다 ---
        #     "paragraph" | "table_row" | "calc_box_text" | "ocr_block"
        "origin": raw.get("origin"),

        # --- 페이지 근사 표시 (genre_b/d, 52건) ---
        #     docx 에 페이지 개념이 없어 PDF 대조로 사후 추정한 청크.
        #     인용할 때 "p.8" 을 확정처럼 쓰면 안 된다는 뜻이다.
        "page_unverified": bool(raw.get("page_unverified", False)),

        # --- 장르C(중도인출) 전용 — 소프트 필터 축 ---
        #     5개 문서의 도입부가 동일해서, 이 값 없이는 검색으로 구분이 안 된다.
        "사유": raw.get("사유"),
        "하위유형": raw.get("하위유형"),

        # --- 품질 지표 (투자설명서 카드에만 존재) ---
        #     confidence 와 별개. 전역 곱셈에 쓰면 안 된다(안내문서엔 없는 필드).
        "fill_ratio": raw.get("fill_ratio"),

        # --- 하위 호환: 기존 코드/리포트가 chunk["source"] 를 읽는다 ---
        "source": corpus or raw.get("source") or "기타",
    }

    # --- 상품(투자설명서) 전용 필드 ---
    # 1문서 = 1펀드이므로 이 값들은 그 문서의 모든 청크에 대해 참이다.
    # = 선행모델의 company_name 과 같은 성격 → 하드 필터로 안전하다.
    if raw.get("fund_name") or raw.get("fund_code"):
        chunk["fund"] = {
            "code": raw.get("fund_code") or [],
            "name": raw.get("fund_name"),
            "risk_grade": raw.get("위험등급"),
            "asset_type": raw.get("자산유형"),
            "연금저축_가능": raw.get("연금저축_가능"),
            "퇴직연금_가능": raw.get("퇴직연금_가능"),
        }
    else:
        chunk["fund"] = {}

    return chunk


# ==========================================================================
# 공개 API
# ==========================================================================
@lru_cache(maxsize=1)
def load_corpus() -> List[dict]:
    """설정된 모든 JSONL 을 읽어 정규화된 청크 리스트로 돌려준다 (프로세스당 1회 캐싱).

    ⚠️ **필수 파일이 없으면 죽는다** (config.REQUIRED_CORPUS_FILES).

    예전에는 없는 파일을 전부 조용히 건너뛰었다. "코퍼스 구성이 바뀌는 단계라 파일
    하나 빠졌다고 멈추면 실험을 못 돈다"는 이유였는데, 실제로 벌어진 일은 그 반대였다.
    세제(genre_b)·중도인출(genre_c) 코퍼스가 통째로 빠진 상태에서 에러 없이 돌았고,
    그 질문들에 **근거 없는 그럴듯한 답**이 나갔다. 조용한 실패가 시끄러운 실패보다
    훨씬 비싸다 — 에러는 고치지만 침묵은 못 고친다.

    선택 파일(REQUIRED 에 없는 것)은 지금도 조용히 건너뛴다.
    """
    chunks: List[dict] = []
    missing: List[str] = []
    seen_ids: Dict[str, str] = {}                       # chunk_id → 최초 등장 파일

    for fname, corpus, tier, chunk_type in config.CORPUS_FILES:
        path = Path(config.DATA_DIR) / fname
        if not path.exists():
            missing.append(fname)
            continue
        n = dup = 0
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("JSON 파싱 실패 (%s): %s", fname, line[:80])
                    continue

                # 같은 chunk_id 가 두 파일에 있으면 **뒤엣것을 버린다.**
                # 독립 OCR 파일 5종이 genre_d 로 병합된 뒤에도 옛 파일이 폴더에 남아
                # 있으면 전 청크가 2벌이 되는데, 그러면 BM25 의 문서빈도(df)가 왜곡되고
                # 중복 근거가 인용 목록에 두 번 찍힌다.
                cid = raw.get("chunk_id", "")
                if cid and cid in seen_ids:
                    dup += 1
                    continue
                if cid:
                    seen_ids[cid] = fname

                chunks.append(_normalize(raw, corpus, tier, chunk_type))
                n += 1
        log.info("로드 %-26s %5d건  (tier=%s, type=%s)%s",
                 fname, n, tier, chunk_type, f"  ⚠️중복 {dup}건 무시" if dup else "")

    # ── 필수 파일 누락은 실패다 ──────────────────────────────────────
    required_missing = [f for f in missing if f in config.REQUIRED_CORPUS_FILES]
    if required_missing:
        raise FileNotFoundError(
            "필수 코퍼스 파일이 없습니다 — 이대로 돌리면 해당 주제 질문에 "
            "'근거 없는 그럴듯한 답'이 나갑니다.\n"
            f"  데이터 경로: {config.DATA_DIR}\n"
            f"  누락: {', '.join(required_missing)}\n"
            "  최신 jsonl 을 data/ 에 넣은 뒤 `python -m index.build_index` 로 "
            "벡터를 재생성하세요."
        )
    if missing:
        log.info("선택 파일 없음(건너뜀): %s", ", ".join(missing))

    if not chunks:
        raise FileNotFoundError(
            f"코퍼스가 비었습니다. config.DATA_DIR 를 확인하세요.\n"
            f"  현재: {config.DATA_DIR}"
        )

    log.info("총 %d 청크 로드", len(chunks))
    return chunks


def searchable_text(c: dict) -> str:
    """BM25·임베딩이 함께 보는 텍스트.

    파서가 만들어 둔 `embed_text` 가 있으면 그것을 쓴다 — 펀드명·섹션명이 이미
    앞에 붙어 있어 문맥이 살아 있다. 없으면 섹션+본문+펀드명으로 조립한다.

    BM25 와 벡터가 **같은 텍스트**를 보므로 두 점수를 합칠 때 의미가 어긋나지 않는다.
    """
    if c.get("embed_text"):
        return c["embed_text"]

    parts = [c.get("section") or "", c.get("text") or ""]
    f = c.get("fund") or {}
    if f.get("name"):
        parts.append(f["name"])
        parts.append(f"{f.get('asset_type') or ''} 위험등급{f.get('risk_grade') or ''}")
    return "\n".join(p for p in parts if p)


@lru_cache(maxsize=1)
def fund_names() -> List[str]:
    """코퍼스에 실제로 존재하는 펀드명 목록 (중복 제거, 정렬).

    선행모델이 `db.get_companies()` 로 회사 목록을 뽑아 **프롬프트에 주입**한 것과
    같은 용도다. 질문 분석 LLM 에게 이 목록을 주면
      · 없는 펀드명을 지어내지 못하고
      · "미래에셋 배당" 같은 부분 표현을 정식 명칭으로 정규화할 수 있다.
    목록에 없는 값으로 필터를 걸면 후보가 0건이 되어 파이프라인이 헛돈다.
    """
    seen: Dict[str, None] = {}
    for c in load_corpus():
        n = (c.get("fund") or {}).get("name")
        if n:
            seen.setdefault(n, None)
    return sorted(seen)


# ==========================================================================
# 펀드 이름 다루기 — 핵심이름 / 변별자 / 형제 (2026-08-31 신설)
#
# ■ 왜 한 곳에 모으나
#   같은 판단을 answer_generator(`_fund_core`)와 여기(`match_known_fund`)가
#   **서로 다른 규칙으로** 하고 있었다. 앞쪽은 "증권|투자신탁|자투자|[(" 앞을
#   자르고, 뒤쪽은 괄호 앞만 자른다. 규칙이 갈리면 "답변이 이 펀드를 말했나" 와
#   "질문이 이 펀드를 지목했나" 의 판정이 어긋난다.
#
# ■ 이 코퍼스의 이름 구조
#       한국투자골드플랜연금 증권전환형투자신탁1호 (국공채)
#       └ 핵심이름 ────────┘ └ 상품형태 ───────┘ └ 변별자 ┘
#   핵심이름만 보면 형제 세 펀드가 **하나로 뭉개진다.** 변별자는 상품형태 뒤에
#   붙으므로 "증권 앞부분만 자르기" 로는 절대 잡히지 않는다.
#   실측: 92개 카드 중 11개(5개 군)가 이렇게 뭉개진다.
# ==========================================================================

# 상품형태 꼬리 — 여기부터 뒤는 이름의 변별부가 아니다.
_FUND_TAIL = re.compile(r"증권|투자신탁|자투자|[\[(]")

# 변별자 — 이름 끝의 괄호/대괄호 안 내용. "(국공채)" · "[주식]" · "(채권)"
_FUND_VARIANT = re.compile(r"[\[(]\s*([^\[\]()]{1,20})\s*[\])]\s*$")

# 상품형태 — 변별자가 겹치는 형제를 가르는 마지막 축 (`fund_discriminators`).
_FUND_FORM = re.compile(r"자투자신탁|투자신탁|투자회사")

# 호수 — "제1호" · "1호" · "2호". 괄호가 없는 형제(1호/2호)를 가른다.
_FUND_SERIES = re.compile(r"제?\s*(\d{1,2})\s*호")


def fund_core(name: str) -> str:
    """정식 명칭에서 **핵심이름**만 남긴다 (공백 제거).

    '미래에셋장기성장포커스증권자투자신탁1호(주식)' -> '미래에셋장기성장포커스'

    답변도 질문도 펀드를 축약해 부르므로("미래에셋장기성장포커스 사면"),
    정식 명칭 전체로 대조하면 절대 안 맞는다. 그래서 핵심이름으로 본다.
    ⚠️ 핵심이름은 형제펀드를 구분하지 **못한다.** 구분이 필요하면
       `fund_variant()` 를 같이 봐라.
    """
    return re.sub(r"\s+", "", _FUND_TAIL.split(name or "", 1)[0])


def fund_variant(name: str) -> str:
    """형제펀드를 가르는 **변별자**. 없으면 빈 문자열.

    '…투자신탁1호(국공채)' -> '국공채'   '…투자신탁 제1호[주식]' -> '주식'
    괄호가 없으면 호수로 대신한다 — '…투자신탁2호' -> '2호'
    """
    m = _FUND_VARIANT.search((name or "").strip())
    if m:
        return re.sub(r"\s+", "", m.group(1))
    s = _FUND_SERIES.search(name or "")
    return f"{s.group(1)}호" if s else ""


def fund_discriminators(name: str, group: List[str]) -> List[str]:
    """형제군 안에서 `name` 을 **혼자만** 가리키는 문자열들 (공백 제거).

    ■ 왜 `fund_variant()` 하나로 부족한가 (2026-09-02, 코퍼스 실측)
      괄호 변별자가 형제끼리 **같은** 군이 있다.

          베어링 고배당 증권 투자회사(주식)      -> variant '주식'
          베어링 고배당 증권자투자신탁(주식)      -> variant '주식'

      이 군에서 변별자만 보면 답변이 "베어링 고배당 … 주식형" 이라고만 써도
      **두 펀드가 모두 지목된 것으로 판정된다.** 지목 판정이 정밀도 가드인데
      여기서는 가드가 열려 버린다.

      그래서 변별자가 겹치면 상품형태("투자회사"·"자투자신탁"·"투자신탁")를
      뒤에 더 본다. 그래도 안 갈리면 **빈 목록**을 돌려준다 — 특정할 수 없다는
      뜻이고, 호출부는 "펀드 특정 없음" 경로로 가야 한다.

    ■ 왜 정식 명칭 전체를 쓰지 않나
      답변도 질문도 펀드를 축약해 부른다("베어링 고배당 주식형"). 정식 명칭
      전체로 대조하면 어느 형제도 안 걸려서 가드가 반대 방향으로 무력해진다.
    """
    others = [o for o in group if o != name]
    if not others:
        v = fund_variant(name)
        return [re.sub(r"\s+", "", v)] if v else []

    mine = fund_variant(name)
    if mine and all(fund_variant(o) != mine for o in others):
        return [re.sub(r"\s+", "", mine)]

    # 변별자가 겹친다 — 상품형태를 덧붙여 다시 시도한다.
    def _form(n: str) -> str:
        m = _FUND_FORM.search(n or "")
        return m.group(0) if m else ""

    mine_form = _form(name)
    if mine_form and all(_form(o) != mine_form for o in others):
        joined = mine_form + mine if mine else mine_form
        return [re.sub(r"\s+", "", joined)]
    return []


def _fund_groups() -> Dict[str, List[str]]:
    """핵심이름 -> 그 이름을 공유하는 정식 명칭 목록."""
    groups: Dict[str, List[str]] = {}
    for n in fund_names():
        groups.setdefault(fund_core(n), []).append(n)
    return groups


def sibling_funds(name: str) -> List[str]:
    """같은 상품군의 형제 펀드 전부 (자기 자신 포함, 정렬).

    형제가 없으면 `[name]` 하나만 돌려준다 — 호출부가 분기를 안 해도 된다.
    """
    if not name:
        return []
    return sorted(_fund_groups().get(fund_core(name), [name]))


# 펀드명 지목으로 인정할 최소 접두어 길이.
# 너무 짧으면 운용사명("미래에셋")만으로 걸려 변별력이 없고, 너무 길면 사용자가
# 줄여 말한 경우를 놓친다. 8자면 "미래에셋장기성장" 수준이라 상품 특정이 된다.
MIN_FUND_PREFIX = 8


def match_known_fund(question: str) -> str:
    """질문이 **우리가 보유한 펀드**를 지목하고 있는가. 지목했으면 정식 명칭.

    ■ 왜 여기(index 계층)에 있나 — 두 노드가 같은 판단을 해야 하기 때문 (2026-08-18)

      원래 이 함수는 `nodes/safety_checker.py` 안에만 있었다. 그런데 **똑같은 판단이
      query_analyzer 에도 필요하다.** query_analyzer 는 펀드명 추출을 LLM 에게만
      맡기고 있었는데, 실측에서 LLM(HCX-DASH-002)이 이걸 놓쳤다.

        P-F4  "미래에셋장기성장포커스 중간에 깨면 환매수수료 물어야 하나요?"
        P-G0a "미래에셋장기성장포커스 운용사가 미래에셋증권이랑 같은 계열인가요?"
        P-G0b "미래에셋장기성장포커스에 넣은 돈은 어디에 보관되나요?"

      세 문항 모두 펀드명이 질문에 그대로 들어 있는데 fund_name 이 빈 배열로 나왔고,
      그 결과 product_affinity 가 기본값 0.35 에 머물렀다. 관여도 0.35 면
      투자설명서가 top-k 8칸 중 3칸만 받는다 — 답이 있는 섹션 원문에 도달할 수 없다.

      **목록 대조는 LLM 판단이 아니라 문자열 비교라 오탐이 없다.** 92개 중 하나가
      질문에 들어 있으면 그것만으로 확정이다. LLM 이 놓쳐도 이쪽이 잡는다.

    ■ 왜 접두어로 보나
      정식 명칭은 "미래에셋장기성장포커스증권자투자신탁1호(주식)" 처럼 길어서
      사용자가 통째로 적는 일이 없다. 보통 앞부분만 말한다. 그래서 **이름의 앞부분이
      질문에 들어 있는지** 를 본다 — 긴 접두어부터 확인해 가장 구체적인 매칭을 고른다.
    """
    hits = match_known_funds(question)
    return hits[0] if hits else ""


def match_known_funds(question: str) -> List[str]:
    """지목된 펀드를 **전부** 돌려준다 — 형제펀드를 하나로 줄이지 않는다.

    ■ 왜 목록인가 (2026-08-31, 9차 이슈 3)

      `match_known_fund` 는 접두어가 가장 긴 것 **하나**만 돌려줬다. 그런데
      형제펀드는 접두어가 정확히 같은 길이에서 동점이다.

          한국투자골드플랜연금증권전환형투자신탁1호(국공채)
          한국투자골드플랜연금증권전환형투자신탁1호(채권)
          한국투자골드플랜연금증권전환형투자신탁1호(주식)
                              ↑ 세 이름의 공통 접두어가 여기까지다

      `if k > len(best)` 라 **먼저 만난 하나만** 남는다. 그 값이 그대로
      `filters["fund_name"]` 이 되고, store 의 `_passes` 가
      `any(n in fn or fn in n ...)` 로 대조하므로 **나머지 형제는 검색에서
      통째로 배제된다.**

      실측 P-B1: 질문이 "국공채·채권·주식, 뭐가 달라요?" 인데 채권 펀드 문서
      (R2_KR5113420013)가 검색 후보에 아예 없었다. 답변은 채권 펀드의 수익률을
      국공채 값(5.91%)으로 채웠다 — 실제 값은 5.74% 다.
      **모델이 지어낸 게 아니라 자료를 못 받은 것이다.**

    ■ 동점만 묶는다
      접두어 길이가 같은 것만 형제로 본다. 더 긴 접두어로 특정된 펀드가 있으면
      그것 하나만 남으므로, "이 펀드 총보수 얼마?" 류 단일 지목은 그대로다.
    """
    try:
        q = (question or "").replace(" ", "")
        if not q:
            return []
        best = 0
        matched: List[str] = []
        for name in fund_names():
            head = re.split(r"[\[\(]", name)[0].replace(" ", "")
            for k in range(len(head), MIN_FUND_PREFIX - 1, -1):
                if head[:k] in q:
                    if k > best:
                        best, matched = k, [name]
                    elif k == best:
                        matched.append(name)
                    break
        return sorted(matched)[:config.MAX_SIBLING_FUNDS]
    except Exception:                    # 인덱스 문제로 판단을 막지 않는다
        return []


def corpus_stats() -> dict:
    """코퍼스 구성 요약. 리포트/점검용."""
    from collections import Counter

    chunks = load_corpus()
    return {
        "total": len(chunks),
        "by_tier": dict(Counter(c["tier"] for c in chunks)),
        "by_corpus": dict(Counter(c["corpus"] for c in chunks)),
        "by_chunk_type": dict(Counter(c["chunk_type"] for c in chunks)),
        "distractor": sum(1 for c in chunks if c["distractor"]),
        "table_chunks": sum(1 for c in chunks if is_table_chunk(c)),
        "by_사유": dict(Counter(c["사유"] for c in chunks if c.get("사유"))),
        "page_unverified": sum(1 for c in chunks if c.get("page_unverified")),
        "funds": len(fund_names()),
        "ocr_chunks": sum(1 for c in chunks if c.get("parse_method") == "ocr"),
    }


def is_table_chunk(c: dict) -> bool:
    """이 청크가 표(또는 계산 박스)인가.

    판정 축이 둘이다 — genre_b/d 는 `origin` 을 남겨 주고, 그 밖의 코퍼스는
    섹션명이 "표1"/"표2" 로 들어온다. 둘 중 하나만 맞아도 표로 본다.
    (section 만 보면 세제·업무매뉴얼의 표 행 204건이 통째로 빠진다)
    """
    if (c.get("origin") or "") in config.TABLE_ORIGINS:
        return True
    return str(c.get("section") or "").startswith(config.TABLE_SECTION_PREFIXES)


# 표가 청크 경계에서 잘렸을 때 뒤 조각의 머리에 붙는 표시.
# 실측: section_chunks 7,721건 중 3,134건(40.6%)이 이걸 달고 있다.
_TABLE_CONT = "(표 계속)"
# 청크 id 는 "<doc>#p<페이지>#c<번호>" 형태다.
_CHUNK_ID = re.compile(r"^(.*#p\d+#c)(\d+)$")


@lru_cache(maxsize=1)
def chunk_by_id() -> Dict[str, dict]:
    """chunk_id -> 청크. 이웃 청크를 id 로 찾아올 때 쓴다."""
    return {c["chunk_id"]: c for c in load_corpus() if c.get("chunk_id")}


def is_table_continuation(c: dict) -> bool:
    """이 청크가 **앞 청크에서 잘린 표의 뒷부분**인가.

    뒷조각은 열 머리가 없다 — 「최근1년 (표 계속)」 처럼 첫 열 이름 한 개만
    남는다. 그래서 단독으로 주면 어느 표의 어느 열인지 알 수 없고, 반대로
    앞조각만 주면 마지막 행들의 값이 통째로 빠진다.
    """
    return _TABLE_CONT in (c.get("text") or "")[:200]


def table_neighbor_ids(chunk_id: str) -> List[str]:
    """잘린 표를 복원하려면 같이 봐야 하는 이웃 청크 id 목록 (없으면 빈 목록).

    ■ 양방향이다
      · 이 청크가 뒷조각이면  → 앞 조각(열 머리가 있는 쪽)
      · 이 청크 다음이 뒷조각이면 → 그 뒷조각(남은 행·칸이 있는 쪽)

    ■ 왜 id 산술로 찾나
      청크 스키마에 이웃 포인터가 없다. id 가 `#c010` → `#c011` 로 **연속**이고
      같은 문서·같은 페이지 안에서만 이어지므로, 번호 ±1 로 찾은 뒤 문서·섹션이
      같은지 확인하면 충분하다. 페이지를 넘어가는 표는 잇지 않는다 — 페이지가
      바뀌면 표 머리가 다시 인쇄되므로 앞조각이 이미 완결이다.
    """
    m = _CHUNK_ID.match(chunk_id or "")
    if not m:
        return []
    head, num = m.group(1), m.group(2)
    by_id = chunk_by_id()
    me = by_id.get(chunk_id)
    if not me:
        return []
    width = len(num)
    out: List[str] = []
    for delta in (-1, 1):
        nid = f"{head}{int(num) + delta:0{width}d}"
        nb = by_id.get(nid)
        if not nb or nb.get("doc_id") != me.get("doc_id"):
            continue
        if nb.get("section") != me.get("section"):
            continue
        # 뒤쪽 조각이 「(표 계속)」인 경우에만 잇는다. 앞으로 갈 때는 내가 뒷조각일 때.
        later = nb if delta > 0 else me
        if is_table_continuation(later):
            out.append(nid)
    return out


# 단위 선언 줄 — 표 머리에 한 번만 인쇄되므로 뒤 조각은 이걸 잃는다.
_UNIT_LINE = re.compile(r"^.*단위\s*[:：]?\s*\(?\s*%.*$", re.M)


def chunk_text_for_values(c: dict) -> str:
    """값 대조가 볼 본문. 표 연속 청크는 앞 조각의 **단위 선언 줄**을 이어 붙인다.

    ■ 왜 필요한가 (2026-09-02)
      "이 표의 숫자는 %다" 는 선언이 표 머리에 **한 번만** 인쇄된다. 표가 청크
      경계에서 잘리면 뒤 조각에는 선언이 없다.

          p47#c010  나. 연도별 수익률 추이(단위:%)   … 종류C-e -23.42 36.00 …
          p47#c011  최근1년 (표 계속)  33.86  종류F …  투자신탁 -22.43 37.62 …
                    └ 선언이 없어서 이 조각의 값은 **하나도 추출되지 않는다**

      10차 P-E5 답변의 투자신탁·비교지수 행(35.68 · 20.87 · 37.62 …)은 전부
      c011 의 값인데, 그 조각을 근거로 줘도 값 대조가 못 읽는다. 잘린 표의
      뒷조각이 section_chunks 의 40.6% 라 이 구멍은 넓다.

    ■ 선언 줄만 가져온다
      앞 조각을 통째로 붙이면 **앞 조각의 값이 이 조각에 있는 것으로 잡힌다.**
      선언 줄에는 값이 없으므로 그 줄만 붙이면 부작용이 없다.
    """
    text = c.get("text") or ""
    if not is_table_continuation(c) or _UNIT_LINE.search(text):
        return text
    by_id = chunk_by_id()
    cid = c.get("chunk_id") or ""
    m = _CHUNK_ID.match(cid)
    if not m:
        return text
    head, num, width = m.group(1), int(m.group(2)), len(m.group(2))
    # 연속 조각이 여러 개일 수 있다 — 선언이 나올 때까지 앞으로 몇 칸만 본다.
    for back in range(1, 4):
        prev = by_id.get(f"{head}{num - back:0{width}d}")
        if not prev or prev.get("doc_id") != c.get("doc_id"):
            break
        found = _UNIT_LINE.search(prev.get("text") or "")
        if found:
            return found.group(0) + "\n" + text
        if not is_table_continuation(prev):
            break
    return text


def format_chunk_for_llm(chunk: dict, idx: Optional[int] = None) -> str:
    """청크를 LLM 프롬프트에 넣을 형태로 직렬화한다.

    출처 표기를 **본문 위에** 붙이는 이유: 모델이 "어느 출처에서 온 문장인지"를
    보고 신뢰도를 판단하게 하려는 것. (doc 문서 > 투자설명서)
    """
    tier_label = "규정문서" if chunk["tier"] == "doc" else "투자설명서"
    head_bits = [f"출처={tier_label}", f"분류={chunk['corpus']}"]

    # 근거의 '성격'을 알려 준다 — 카드(정리된 표)와 원문(섹션)은 쓰임이 다르다.
    #   카드   : 펀드 단위 정리. 총보수·위험등급 같은 축의 **정본**
    #   원문   : 환매수수료·기준가격처럼 카드에 없는 항목은 여기에만 있다
    #
    # ⚠️ 예전 라벨이 "가공요약" · "가공표" 였다 (2026-08-25 수정).
    #    모델이 이걸 **인용하면 안 되는 2차 가공물**로 읽은 정황이 있다.
    #    실측 5차: 인용률이 fund_card 1.6% · class_card 6.2% 인데
    #    doc/tax 는 12.2% 였다. 검색은 상품 청크를 가장 많이 물어오는데
    #    (fund_card 308건) 인용은 5건뿐이다.
    #    라벨에 **인용 가능**을 명시해 그 오해를 없앤다.
    kind = {
        "fund_card": "펀드카드(총보수·위험등급 정본 — 인용 가능)",
        "class_card": "클래스카드(클래스별 보수·가입자격 정본 — 인용 가능)",
        "prospectus_section": "투자설명서 원문 — 인용 가능",
    }.get(chunk.get("chunk_type") or "")
    if kind:
        head_bits.append(f"형태={kind}")

    if chunk.get("section"):
        head_bits.append(f"섹션={chunk['section']}")
    if chunk.get("사유"):
        head_bits.append(f"사유={chunk['사유']}")
    if chunk.get("하위유형"):
        head_bits.append(f"하위유형={chunk['하위유형']}")

    f = chunk.get("fund") or {}
    if f.get("name"):
        head_bits.append(f"펀드={f['name']}")
        if f.get("risk_grade"):
            head_bits.append(f"위험등급={f['risk_grade']}")
        if f.get("asset_type"):
            head_bits.append(f"자산유형={f['asset_type']}")

    head_bits.append(f"문서={chunk['doc_id']}")
    if chunk.get("page"):
        # 페이지가 추정값이면 그렇게 적는다. 확정처럼 인용하면 안 되기 때문이다.
        head_bits.append(f"p.{chunk['page']}"
                         + ("(추정)" if chunk.get("page_unverified") else ""))

    tag = f"[{idx}] " if idx is not None else ""
    return f"{tag}({', '.join(head_bits)})\n{chunk['text']}"


def to_citation(chunk: dict) -> dict:
    """청크 → 인용(citation) 레코드."""
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "source": chunk["corpus"],
        "tier": chunk["tier"],
        "chunk_type": chunk["chunk_type"],
        "page": chunk.get("page"),
        "page_unverified": bool(chunk.get("page_unverified")),
        "section": chunk.get("section"),
        "사유": chunk.get("사유"),
        "snippet": (chunk["text"][:160] + "…") if len(chunk["text"]) > 160 else chunk["text"],
        "fund_name": (chunk.get("fund") or {}).get("name"),
    }
