"""
1패스 — 전 페이지 텍스트 덤프 (1회, 캐시)

fitz(PyMuPDF)가 있으면 쓰고, 없으면 pdfplumber로 폴백한다.
fitz가 10~20배 빠르지만 결과는 거의 같다.

⚠️ 라이선스: fitz는 AGPL. 소스를 제출하는 대회에선 문제없지만,
   비공개 서비스에 쓸 땐 pdfplumber(MIT)로 바꿔야 한다.
   USE_FITZ=0 으로 강제 폴백 가능.

산출: data/dump/{doc_id}.json
  {"doc_id":..., "pages":[{"page":1,"text":"..."}, ...]}
"""
import os, sys, json, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import DUMP_DIR, TABLE_DIR, load_json, save_json

warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

USE_FITZ = os.environ.get("USE_FITZ", "1") == "1"
try:
    import fitz
    HAS_FITZ = USE_FITZ
except ImportError:
    HAS_FITZ = False


def dump_fitz(path):
    doc = fitz.open(path)
    pages = [{"page": i + 1, "text": p.get_text()} for i, p in enumerate(doc)]
    doc.close()
    return pages


def dump_plumber(path):
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return [{"page": i + 1, "text": p.extract_text() or ""}
                for i, p in enumerate(pdf.pages)]


def main():
    docs = load_json(os.path.join(TABLE_DIR, "docs.json"))
    os.makedirs(DUMP_DIR, exist_ok=True)
    engine = "fitz" if HAS_FITZ else "pdfplumber"
    print(f"엔진: {engine} / 문서 {len(docs)}개")

    done = skipped = 0
    for i, d in enumerate(docs, 1):
        out = os.path.join(DUMP_DIR, d["doc_id"] + ".json")
        if os.path.exists(out):          # ★ 캐시 — 재실행 시 건너뜀
            skipped += 1
            continue
        try:
            pages = dump_fitz(d["path"]) if HAS_FITZ else dump_plumber(d["path"])
        except Exception as e:
            print(f"  [!] {d['doc_id']}: {e}")
            continue
        save_json({"doc_id": d["doc_id"], "fund_codes": d["fund_codes"],
                   "engine": engine, "pages": pages}, out)
        done += 1
        if i % 10 == 0:
            print(f"  {i}/{len(docs)}")
    print(f"완료: 신규 {done} / 캐시 {skipped}")


def check():
    """덤프 건전성 검사 — 한글 깨짐·빈 페이지 탐지."""
    import glob, re
    files = sorted(glob.glob(os.path.join(DUMP_DIR, "*.json")))
    print(f"\n[검사] 덤프 {len(files)}개")
    bad = []
    for f in files:
        d = load_json(f)
        pages = d["pages"]
        total = sum(len(p["text"]) for p in pages)
        han = sum(len(re.findall(r"[가-힣]", p["text"])) for p in pages)
        empty = sum(1 for p in pages if len(p["text"].strip()) < 10)
        ratio = han / total * 100 if total else 0
        if ratio < 20 or empty > len(pages) * 0.3:
            bad.append((d["doc_id"], len(pages), total, round(ratio, 1), empty))
    if bad:
        print("  ⚠️ 이상 문서:")
        for b in bad:
            print(f"    {b[0]}  {b[1]}p  {b[2]}자  한글{b[3]}%  빈페이지{b[4]}")
    else:
        print("  ✅ 전부 정상 (한글 비율 20% 이상)")


if __name__ == "__main__":
    main()
    check()
