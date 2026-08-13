"""
0패스 — MD5 중복 제거.
투자설명서는 전 클래스를 한 문서에 담기 때문에, 클래스별 표준코드마다
같은 파일이 배포된다. 100개 중 7그룹 8파일이 바이트 단위로 동일하다.

산출: data/tables/docs.json
  [{doc_id, path, fund_codes: [...]}, ...]   ← fund_code가 1:N
"""
import glob, hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import RAW_DIR, TABLE_DIR, save_json

def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*", "*.pdf")))
    if not files:
        print(f"[!] {RAW_DIR} 아래에 PDF가 없습니다. RAW_DIR 환경변수를 확인하세요.")
        return
    by_hash = {}
    for f in files:
        code = os.path.basename(os.path.dirname(f))   # KR5127420034
        h = md5(f)
        by_hash.setdefault(h, {"path": f, "fund_codes": []})["fund_codes"].append(code)

    docs = []
    for h, v in by_hash.items():
        codes = sorted(v["fund_codes"])
        docs.append({
            "doc_id": os.path.basename(v["path"]).replace(".pdf", ""),
            "path": v["path"],
            "fund_codes": codes,
            "md5": h,
        })
    docs.sort(key=lambda d: d["doc_id"])
    save_json(docs, os.path.join(TABLE_DIR, "docs.json"))

    dup = [d for d in docs if len(d["fund_codes"]) > 1]
    print(f"파일 {len(files)}개 → 고유 문서 {len(docs)}개")
    print(f"중복 그룹 {len(dup)}개:")
    for d in dup:
        print("  ", d["fund_codes"])

if __name__ == "__main__":
    main()
