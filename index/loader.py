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
    try:
        q = (question or "").replace(" ", "")
        if not q:
            return ""
        best = ""
        matched = ""
        for name in fund_names():
            head = re.split(r"[\[\(]", name)[0].replace(" ", "")
            for k in range(len(head), MIN_FUND_PREFIX - 1, -1):
                if head[:k] in q:
                    if k > len(best):
                        best, matched = head[:k], name
                    break
        return matched if best else ""
    except Exception:                    # 인덱스 문제로 판단을 막지 않는다
        return ""


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
