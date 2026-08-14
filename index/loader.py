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
        "confidence": float(raw.get("confidence", 1.0) or 1.0),
        "distractor": bool(raw.get("distractor", False)),

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

    ⚠️ 없는 파일은 **건너뛴다**(예전에는 FileNotFoundError 로 죽었다).
    코퍼스 구성이 계속 바뀌는 단계라, 파일 하나가 빠졌다고 파이프라인 전체가
    멈추면 실험을 못 돈다. 대신 무엇이 빠졌는지는 로그로 반드시 남긴다.
    """
    chunks: List[dict] = []
    missing: List[str] = []

    for fname, corpus, tier, chunk_type in config.CORPUS_FILES:
        path = Path(config.DATA_DIR) / fname
        if not path.exists():
            missing.append(fname)
            continue
        n = 0
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
                chunks.append(_normalize(raw, corpus, tier, chunk_type))
                n += 1
        log.info("로드 %-26s %5d건  (tier=%s, type=%s)", fname, n, tier, chunk_type)

    if missing:
        log.info("코퍼스에 없는 파일(건너뜀): %s", ", ".join(missing))

    if not chunks:
        raise FileNotFoundError(
            f"코퍼스가 비었습니다. config.DATA_DIR 를 확인하세요.\n"
            f"  현재: {config.DATA_DIR}\n"
            f"  찾은 파일 없음: {', '.join(missing)}"
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
        "table_chunks": sum(
            1 for c in chunks
            if str(c.get("section") or "").startswith(config.TABLE_SECTION_PREFIXES)
        ),
        "funds": len(fund_names()),
        "low_confidence": sum(1 for c in chunks if c["confidence"] < 0.8),
    }


def format_chunk_for_llm(chunk: dict, idx: Optional[int] = None) -> str:
    """청크를 LLM 프롬프트에 넣을 형태로 직렬화한다.

    출처 표기를 **본문 위에** 붙이는 이유: 모델이 "어느 출처에서 온 문장인지"를
    보고 신뢰도를 판단하게 하려는 것. (doc 문서 > 투자설명서)
    """
    tier_label = "규정문서" if chunk["tier"] == "doc" else "투자설명서"
    head_bits = [f"출처={tier_label}", f"분류={chunk['corpus']}"]
    if chunk.get("section"):
        head_bits.append(f"섹션={chunk['section']}")

    f = chunk.get("fund") or {}
    if f.get("name"):
        head_bits.append(f"펀드={f['name']}")
        if f.get("risk_grade"):
            head_bits.append(f"위험등급={f['risk_grade']}")
        if f.get("asset_type"):
            head_bits.append(f"자산유형={f['asset_type']}")

    head_bits.append(f"문서={chunk['doc_id']}")
    if chunk.get("page"):
        head_bits.append(f"p.{chunk['page']}")

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
        "section": chunk.get("section"),
        "snippet": (chunk["text"][:160] + "…") if len(chunk["text"]) > 160 else chunk["text"],
        "fund_name": (chunk.get("fund") or {}).get("name"),
    }
