"""
data/chunks/ 폴더 안의 모든 jsonl 파일을 열어서
INTERFACES.md 스키마 기준 fund_code, table_ref 필드가 없으면 null로 추가함.
이미 필드가 있으면 건드리지 않음 (재실행해도 안전).
"""
import json
import glob
import os

CHUNKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "chunks")
CHUNKS_DIR = os.path.normpath(CHUNKS_DIR)

jsonl_files = glob.glob(os.path.join(CHUNKS_DIR, "*.jsonl"))

print(f"대상 파일 {len(jsonl_files)}개 발견:")
for f in jsonl_files:
    print(f"  - {os.path.basename(f)}")

total_added = 0
total_already_ok = 0

for filepath in jsonl_files:
    lines = []
    changed = False
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if "fund_code" not in c:
                c["fund_code"] = None
                changed = True
                total_added += 1
            else:
                total_already_ok += 1
            if "table_ref" not in c:
                c["table_ref"] = None
                changed = True
            lines.append(c)

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            for c in lines:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"  ✅ {os.path.basename(filepath)} 수정됨 ({len(lines)}개 청크)")
    else:
        print(f"  ⏭️  {os.path.basename(filepath)} 이미 정상 (건너뜀)")

print(f"\n완료! fund_code/table_ref 새로 추가된 청크: {total_added}개")
