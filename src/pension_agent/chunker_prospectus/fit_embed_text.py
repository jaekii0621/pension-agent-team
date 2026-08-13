# -*- coding: utf-8 -*-
"""
fit_embed_text.py — embed_text를 토큰 기준으로 마무리 절단한다.

왜 필요한가
  patch3의 EMBED_MAX는 **글자** 예산이다. XLM-R 계열 토크나이저는 숫자·구분자를
  잘게 쪼개서, 표 블록은 같은 글자 수라도 토큰이 1.5~2배 나온다.
  → 평균 비율로 잡은 글자 예산은 표에서 초과한다(실측 238/2492).
  여기서 실제 토크나이저로 재서 초과분만 잘라낸다. 산문 청크는 건드리지 않는다.

  ⚠️ text(인용용 원문)는 절대 건드리지 않는다. embed_text만 수정한다.

실행 위치: run_all.py --chunk 뒤 (청커를 다시 돌리면 이 스크립트도 다시 돌려야 한다)
"""
import os, json, glob, shutil

MODEL = "intfloat/multilingual-e5-large"
PREFIX = "passage: "     # e5 계열은 문서 쪽에 이 접두어를 붙여 임베딩한다
LIMIT = 508              # 512 - 여유 4 (특수토큰 포함해서 셈)
CHUNK_DIR = ""           # 비우면 자동 탐색


def find_chunk_dir():
    if CHUNK_DIR:
        return CHUNK_DIR
    for root in ["/content", ".", "/content/drive/MyDrive"]:
        if not os.path.isdir(root):
            continue
        hits = glob.glob(os.path.join(root, "**", "data", "chunks",
                                      "section_chunks.jsonl"), recursive=True)
        if hits:
            return os.path.dirname(sorted(hits, key=len)[0])
    raise SystemExit("[!] data/chunks 를 못 찾았습니다. CHUNK_DIR 을 지정하세요.")


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
D = find_chunk_dir()
print(f"CHUNK_DIR = {D}\n")


def ntok(s):
    return len(tok.encode(PREFIX + s))


def fit(s):
    """LIMIT 안에 들어오도록 뒤를 자른다. 이미 들어오면 그대로."""
    if ntok(s) <= LIMIT:
        return s, False

    # ① 오프셋으로 한 번에 자를 위치를 찾는다 (fast tokenizer)
    try:
        enc = tok(PREFIX + s, return_offsets_mapping=True, add_special_tokens=True)
        offs = [o for o in enc["offset_mapping"] if o != (0, 0)]
        if len(offs) > LIMIT - 4:
            end = offs[LIMIT - 4][0] - len(PREFIX)
            if end > 0:
                s = s[:end]
    except Exception:
        pass

    # ② 검산 — 오프셋이 없거나 어긋나면 줄여가며 맞춘다
    while s and ntok(s) > LIMIT:
        s = s[:int(len(s) * 0.92)]

    # ③ 단어·문장 경계로 다듬는다
    p = max(s.rfind(". "), s.rfind(" "), s.rfind("\n"))
    if p > len(s) * 0.8:
        s = s[:p]
    return s.rstrip() + "…", True


total = changed = 0
for name in ["fund_cards", "class_cards", "section_chunks"]:
    path = os.path.join(D, name + ".jsonl")
    if not os.path.exists(path):
        print(f"  {name}: 파일 없음 — 건너뜀")
        continue
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if not os.path.exists(path + ".bak"):
        shutil.copy2(path, path + ".bak")

    n_cut = 0
    for r in rows:
        if not r.get("embed_text"):
            continue
        new, cut = fit(r["embed_text"])
        if cut:
            r["embed_text"] = new
            n_cut += 1
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    over = sum(1 for r in rows if r.get("embed_text") and ntok(r["embed_text"]) > 512)
    lens = sorted(ntok(r["embed_text"]) for r in rows if r.get("embed_text"))
    print(f"  {name:<16} n={len(rows):>4}  추가절단 {n_cut:>4}  "
          f"512초과 {over}  토큰 중앙 {lens[len(lens)//2]} / max {lens[-1]}")
    total += len(rows)
    changed += n_cut

print(f"\n총 {total}개 중 {changed}개 절단. text(원문)는 변경하지 않았습니다.")
print("되돌리려면 data/chunks/*.jsonl.bak 을 복사하세요.")
