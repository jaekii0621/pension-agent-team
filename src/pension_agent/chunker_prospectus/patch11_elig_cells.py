# -*- coding: utf-8 -*-
"""
patch11 — 가입자격 표가 칸 단위로 세로 분해된 문서 복구 (PDF 불필요)

  E-1  pass5 에 eligibility_cells() 추가
  E-2  find_eligibility / scan_eligibility 실패 시 3차 폴백으로 연결
       (부분만 잡힌 문서는 빠진 클래스만 채운다 — 기존 값은 안 건드린다)

배경: 클래스 802행 중 511행이 가입자격 미확인, 그중 138행은 11항 원문에 문구가 있다.
      상위 12개 문서에 96행(70%)이 몰려 있고, 형식은 두 가지다.

        A형   '종류 C-P' / 조건문장 …
        B형   '종류C-P' / '수수료미징구-…' / 조건문장 / '있음'|'없음'

멱등, .bak13 백업.  실행 후: python run_all.py --from 5 --skip 6 7
"""
import os, glob, shutil

PARSE_DIR = ""
SEARCH_ROOTS = ["/content", ".", "/content/drive/MyDrive"]
_LOG = []

FUNC = '\n\n# ==============================================================\n# ★ 3차 폴백 — 가입자격 표가 **칸 단위로 세로 분해**된 문서\n#\n#   find_eligibility / scan_eligibility 가 둘 다 실패하는 문서가 많다(511행 미확인).\n#   원인은 보수 표와 같다: PDF 표의 한 칸이 여러 줄로 흘러나온다.\n#\n#     A형   \'종류 C-P\' / 조건문장 …\n#     B형   \'종류C-P\' / \'수수료미징구-…\' / 조건문장 / \'있음\'|\'없음\'\n#\n#   \'종류X\' 줄을 기준으로 다음 \'종류Y\'까지를 모으고,\n#   판매유형·부과여부 칸만 걸러낸다.\n# ==============================================================\nCLS = r"[A-Za-z][A-Za-z0-9\\-]{0,7}"\nHEAD = re.compile(rf"^종\\s*류\\s*({CLS})\\s*$")          # \'종류 C-P\' 단독 줄\n# ⚠️ \'가. 집합투자기구\'처럼 조건 본문이 가/나/다로 시작하는 문서가 있다(C-F 사례).\n#   섹션 헤딩만 STOP으로 본다.\nSTOP = re.compile(r"^(부과기준|주\\s*\\d+\\s*\\)|\\(\\s*\\d+\\s*\\)\\s*[가-힣]|\\d+\\s*[.．]\\s*[가-힣]{2,})")\nDROP = re.compile(r"^(있음|없음|해당\\s*없음|-|—|구\\s*분|종\\s*류|판매수수료|부과여부|"\n                  r"수수료(선취|미징구|후취)[^\\s]*)$")\nPAGE = re.compile(r"^-?\\s*\\d{1,3}\\s*-?$")\n# 실제 가입자격 문장에 나타나는 말들 — 이게 있으면 판매유형 꼬리가 아니다\n_COND = re.compile(r"제한|없음|투자자|가입|전용|자격|기관|법|계좌|이상|한하여|대상|"\n                   r"온라인|누구나|개인|법인|기금")\n\n\ndef eligibility_cells(lines):\n    """{class: 조건문장}. \'종류X\' 줄 다음부터 다음 \'종류Y\' 전까지를 모은다."""\n    out, cur, buf = {}, None, []\n    fee_tail = False        # 직전 줄이 판매유형 칸이면 그 꼬리를 흘려보낸다\n\n    def flush():\n        if cur and buf:\n            s = " ".join(buf).strip()\n            s = re.sub(r"\\s+", " ", s)\n            if len(s) >= 3:\n                out.setdefault(cur, s[:200])\n\n    for raw in lines:\n        L = raw.strip()\n        if not L or PAGE.match(L):\n            continue\n        m = HEAD.match(L.replace(" ", "").replace("종류", "종류 ", 1)) or HEAD.match(L)\n        if m:\n            flush()\n            cur, buf = m.group(1), []\n            continue\n        if STOP.match(L):\n            flush()\n            cur, buf = None, []\n            continue\n        if cur is None:\n            continue\n        # ★ 판매유형 칸이 두 줄로 쪼개진 경우의 꼬리("…무권유저" + "비용")\n        #   ⚠️ 짧다고 무조건 버리면 \'제한없음\', \'온라인 투자자\' 같은 실제 조건까지 사라진다.\n        #   조건 키워드가 하나도 없는 짧은 조각만 꼬리로 본다.\n        if fee_tail and not buf and len(L) <= 12 and not _COND.search(L):\n            fee_tail = False\n            continue\n        fee_tail = False\n        if DROP.match(L.replace(" ", "")):\n            if re.match(r"^수수료(선취|미징구|후취)", L.replace(" ", "")):\n                fee_tail = True\n            # 판매유형/부과여부 칸은 조건이 아니다. 다만 조건을 이미 모았으면 거기서 끊는다\n            if buf:\n                flush()\n                cur, buf = None, []\n            continue\n        buf.append(L)\n    flush()\n    return out\n'
OLD_HOOK = '        # ★ 2차 폴백 — 하위 헤딩이 아예 없는 문서는 줄 단위로 직접 스캔\n        if not elig:\n            elig = scan_eligibility(t)\n'
NEW_HOOK = '        # ★ 2차 폴백 — 하위 헤딩이 아예 없는 문서는 줄 단위로 직접 스캔\n        if not elig:\n            elig = scan_eligibility(t)\n        # ★ 3차 폴백 — 표가 칸 단위로 세로 분해된 문서\n        #   ⚠️ 기존 값은 건드리지 않는다. 빠진 클래스만 채운다.\n        _cells = eligibility_cells(t.split("\\n"))\n        for _k, _v in _cells.items():\n            elig.setdefault(_k, _v)\n'


def find_parse_dir():
    if PARSE_DIR:
        return PARSE_DIR
    for root in SEARCH_ROOTS:
        hits = [h for h in glob.glob(os.path.join(root, "**", "pass5_eligibility.py"),
                                     recursive=True) if "__pycache__" not in h]
        if hits:
            return os.path.dirname(sorted(hits, key=len)[0])
    raise SystemExit("[!] pass5_eligibility.py 를 못 찾았습니다.")


def patch(tag, old, new, marker=None):
    global SRC
    if marker and marker in SRC:
        _LOG.append(("SKIP", tag, "이미 적용됨")); return
    n = SRC.count(old)
    if n != 1:
        _LOG.append(("FAIL", tag, f"앵커 {n}회 (1회여야 함)")); return
    SRC = SRC.replace(old, new, 1)
    _LOG.append(("OK", tag, ""))


DIR = find_parse_dir()
PATH = os.path.join(DIR, "pass5_eligibility.py")
SRC = open(PATH, encoding="utf-8").read()
print(f"PARSE_DIR = {DIR}\n")

patch("E-1 함수 추가", "def classify(text):", FUNC + "\ndef classify(text):",
      marker="def eligibility_cells(")
patch("E-2 폴백 연결", OLD_HOOK, NEW_HOOK, marker="eligibility_cells(t.split")

if any(s == "OK" for s, _, _ in _LOG):
    if not os.path.exists(PATH + ".bak13"):
        shutil.copy2(PATH, PATH + ".bak13")
    open(PATH, "w", encoding="utf-8").write(SRC)

w = max(len(t) for _, t, _ in _LOG)
for st, tag, msg in _LOG:
    print(f"{ {'OK': '✅', 'SKIP': '•', 'FAIL': '❌'}[st] } {tag:<{w}}  {msg}")
print(f"\n적용 {sum(1 for s,_,_ in _LOG if s=='OK')} / 건너뜀 "
      f"{sum(1 for s,_,_ in _LOG if s=='SKIP')} / 실패 {sum(1 for s,_,_ in _LOG if s=='FAIL')}")

import py_compile
py_compile.compile(PATH, doraise=True)
print("✅ 문법 정상")

import sys, importlib
sys.path.insert(0, os.path.dirname(DIR))
m = importlib.import_module("parse.pass5_eligibility")
importlib.reload(m)
A = ["구분", "가입자격", "종류 A", "선취판매수수료가 징구되는 수익증권에 가입하고자 하는 투자자",
     "종류 C-F", "가. 집합투자기구", "나. 법시행령 제10 조제2 항에서 정하는 기관투자자",
     "종류 C-P", "소득세법 제20조의3에 따른 연금저축계좌를 통하여 가입", "- 33 -",
     "종류 C-P2", "근로자퇴직급여보장법에 의한 퇴직연금제도 가입자", "부과기준 매입시"]
B = ["종류", "구분", "판매수수료", "부과여부",
     "종류A", "수수료선취-오프라인", "제한없음", "있음",
     "종류A-e", "수수료선취-온라인", "온라인 투자자", "있음",
     "종류AG", "수수료선취-오프라인-무권유저", "비용",
     "투자자문업자로부터 투자자문을 받고 그 결과에 따라", "있음",
     "종류C-P", "수수료미징구-오프라인-개인연금", "소득세법에 따른 연금저축계좌 가입자", "없음"]
print("\n── 단위 확인 ──")
for name, ls, want in [("A형", A, ["A", "C-F", "C-P", "C-P2"]),
                       ("B형", B, ["A", "A-e", "AG", "C-P"])]:
    got = list(m.eligibility_cells(ls))
    print(f"  {'✅' if got == want else '❌'} {name}  {got}  (기대 {want})")

print("\n다음: python run_all.py --from 5 --skip 6 7")
