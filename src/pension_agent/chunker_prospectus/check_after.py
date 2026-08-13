# -*- coding: utf-8 -*-
"""
재실행 후 검증 — 패치가 실제로 먹었는지 숫자로 확인한다.
BASE 는 data/ 의 부모 디렉터리(run_all.py 가 있는 곳).
"""
import os, re, json, glob, collections

BASE = ""          # 예: "/content/drive/MyDrive/prospectus"


def find_base():
    if BASE:
        return BASE
    for root in ["/content", ".", "/content/drive/MyDrive"]:
        if not os.path.isdir(root):
            continue
        hits = glob.glob(os.path.join(root, "**", "data", "tables", "six_axis.json"),
                         recursive=True)
        if hits:
            return os.path.dirname(os.path.dirname(os.path.dirname(sorted(hits, key=len)[0])))
    raise SystemExit("[!] six_axis.json 을 못 찾았습니다. BASE 를 직접 지정하세요.")


B = find_base()
T = os.path.join(B, "data", "tables")
C = os.path.join(B, "data", "chunks")
six = json.load(open(os.path.join(T, "six_axis.json"), encoding="utf-8"))
fc = [json.loads(l) for l in open(os.path.join(C, "fund_cards.jsonl"), encoding="utf-8")]
cc = [json.loads(l) for l in open(os.path.join(C, "class_cards.jsonl"), encoding="utf-8")]
n = len(six)

rows = []


def chk(name, got, want, ok):
    rows.append(("✅" if ok else "❌", name, got, want))


# ── A-1 / A-2 ──────────────────────────────────────────
polluted = sum(1 for r in six for x in (r.get("보수") or [])
               if str(x.get("class", "")).startswith("수수료"))
chk("보수 행 class 오염(수수료…)", polluted, "0", polluted == 0)

bad_card = sum(1 for c in fc
               for line in c["text"].split("\n")
               if line.startswith("클래스별 총보수") and "수수료" in line)
chk("펀드카드에 설명문구 노출", bad_card, "0", bad_card == 0)

fee_docs = sum(1 for r in six if r.get("보수"))
chk("[회귀] 보수 있는 문서", f"{fee_docs}/{n}", "70 유지", fee_docs >= 70)

unk_docs = {r["doc_id"] for r in six
            if any(x.get("클래스명미상") for x in (r.get("보수") or []))}
flag_docs = {r["doc_id"] for r in six if r.get("보수_신뢰불가")}
chk("플래그 정합(미상행 문서 ⊆ 신뢰불가)", f"{len(unk_docs)}/{len(flag_docs)}",
    "두 수가 같음", unk_docs <= flag_docs)

# ── A-4 ────────────────────────────────────────────────
manual = sum(1 for r in six if r.get("수익률_수동확인필요"))
noperf = sum(1 for r in six if not r.get("펀드수익률"))
chk("수익률_수동확인필요", manual, f"{noperf} 내외(>0)", manual > 0)

# ── A-6 ────────────────────────────────────────────────
na_to = sum(1 for r in six if r.get("매매회전율_해당없음"))
undet = sum(1 for r in six
            if r.get("매매회전율") is None and not r.get("매매회전율_해당없음"))
chk("회전율 해당없음(문서 명시)", na_to, "20에서 감소", na_to < 20)
chk("회전율 미확인(라벨 미검출)", undet, "0보다 큼", undet > 0)

# ── confidence 가 신호를 갖는지 ─────────────────────────
c10 = sum(1 for c in fc if c["confidence"] >= 1.0)
chk("펀드카드 confidence 1.0", f"{c10}/{len(fc)}", "90에서 감소", c10 < 90)

# ── 실제 매핑이 얼마나 늘었나 (정보용, 판정 아님) ────────
tot = mapped = 0
for c in cc:
    for line in c["text"].split("\n")[3:]:
        m = re.match(r"^(\S+)\s\s+(\S+)", line)
        if not m:
            continue
        tot += 1
        mapped += (m.group(2) != "미확인")
INFO = f"클래스카드 총보수 매핑 {mapped}/{tot}행  (패치 전 192/473)"

# ── 회귀 확인 (줄면 안 되는 것들) ───────────────────────
for k, want in [("fund_name", 92), ("위험등급", 92), ("상품분류", 92),
                ("시장잔고_백만원", 90), ("펀드수익률", 87)]:
    got = sum(1 for r in six if r.get(k) is not None)
    chk(f"[회귀] {k}", f"{got}/{n}", f"{want} 이상", got >= want)

chk("[회귀] 클래스 카드 수", len(cc), "64 이상", len(cc) >= 64)

cls_docs = sum(1 for r in six if r.get("클래스"))
cls_rows = sum(len(r.get("클래스") or []) for r in six)
chk("[회귀] 클래스 있는 문서", f"{cls_docs}/{n}", "64 이상", cls_docs >= 64)
chk("[회귀] 클래스 총 개수", cls_rows, "473 이상", cls_rows >= 473)

bad_code = sorted({c["class"] for r in six for c in (r.get("클래스") or [])
                   if c["class"].endswith(("-", "_"))})
chk("끝이 구분자인 클래스 코드", bad_code or 0, "0", not bad_code)
chk("[회귀] 펀드 카드 수", len(fc), "92", len(fc) == n)

w1 = max(len(r[1]) for r in rows)
w2 = max(len(str(r[2])) for r in rows)
print(f"{'':2} {'항목':<{w1}}  {'현재':>{w2}}   기대")
print("-" * (w1 + w2 + 22))
for m, name, got, want in rows:
    print(f"{m} {name:<{w1}}  {str(got):>{w2}}   {want}")

nfail = sum(1 for r in rows if r[0] == "❌")
print(f"\n{'통과' if not nfail else '실패 ' + str(nfail) + '건'}")

# 참고용 분포
print("\n── 참고 ──")
print(INFO)
print("보수 출처:", collections.Counter(x.get("출처")
                                     for r in six for x in (r.get("보수") or [])))
print("클래스 출처:", collections.Counter(c.get("출처")
                                      for r in six for c in (r.get("클래스") or [])))
print("클래스카드 confidence:",
      collections.Counter(c["confidence"] for c in cc).most_common(5))