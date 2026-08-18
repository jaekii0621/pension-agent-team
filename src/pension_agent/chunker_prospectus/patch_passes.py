# -*- coding: utf-8 -*-
"""
A그룹 + B그룹 패치를 pass 파일에 직접 적용한다.

· 같은 셀을 두 번 돌려도 안전하다(이미 적용된 패치는 SKIP).
· 최초 1회만 원본을 <파일>.bak 으로 남긴다.
· 앵커 문자열이 정확히 1회 매칭되지 않으면 그 패치만 FAIL로 표시하고 나머지는 계속한다.

사용:
    PARSE_DIR 을 pass3_extract.py 가 있는 디렉터리로 지정 (보통 <프로젝트>/src/parse)
    비워두면 현재 경로 아래에서 자동 탐색한다.
"""
import os, glob, shutil, sys

PARSE_DIR = ""          # 예: "/content/drive/MyDrive/prospectus/src/parse"
SEARCH_ROOTS = ["/content", ".", "/content/drive/MyDrive"]


# ── 유틸 ────────────────────────────────────────────────
def find_parse_dir():
    if PARSE_DIR:
        return PARSE_DIR
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        hits = glob.glob(os.path.join(root, "**", "pass3_extract.py"), recursive=True)
        hits = [h for h in hits if "__pycache__" not in h]
        if hits:
            return os.path.dirname(sorted(hits, key=len)[0])
    raise SystemExit("[!] pass3_extract.py 를 못 찾았습니다. PARSE_DIR 을 직접 지정하세요.")


_LOG = []


def patch(fname, tag, old, new, done_marker=None):
    """old → new 치환. done_marker 가 이미 파일에 있으면 건너뛴다."""
    path = os.path.join(DIR, fname)
    if not os.path.exists(path):
        _LOG.append(("FAIL", tag, f"{fname} 없음"))
        return
    src = open(path, encoding="utf-8").read()

    if done_marker and done_marker in src:
        _LOG.append(("SKIP", tag, "이미 적용됨"))
        return
    n = src.count(old)
    if n == 0:
        _LOG.append(("SKIP", tag, "앵커 없음 — 이미 적용된 것으로 보임"))
        return
    if n > 1:
        _LOG.append(("FAIL", tag, f"앵커 {n}회 매칭 (1회여야 함)"))
        return

    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
    _LOG.append(("OK", tag, fname))


DIR = find_parse_dir()
print(f"PARSE_DIR = {DIR}\n")

# ══════════════════════════════════════════════════════
# A-1  pass3 extract_fees(table) — 클래스명 검증 + #N 부여
# ══════════════════════════════════════════════════════
patch(
    "pass3_extract.py", "A-1 보수 클래스명 검증",
    '''                    if val and re.match(r"^[\\d.]+$", norm(val)):
                        cm = re.match(r"^(?:종류)?([A-Za-z][A-Za-z0-9\\-]{0,7})", cname)
                        rows.append({"class": cm.group(1) if cm else cname[:8],
                                     "설명": cname[:40],
                                     "총보수": float(norm(val)),
                                     "출처": "table"})
    if rows:
        return rows''',
    '''                    if val and re.match(r"^[\\d.]+$", norm(val)):
                        # ★ 첫 칸이 설명 문구인 문서가 많다("수수료미징구-오프라인-…").
                        #   코드로 볼 수 없으면 class를 비우고 아래에서 순서 인덱스를 준다.
                        #   값 자체는 정확하므로 버리지 않는다.
                        c = normalize_class(cname)
                        rows.append({"class": c,
                                     "설명": cname[:40],
                                     "총보수": float(norm(val)),
                                     "출처": "table",
                                     "클래스명미상": c is None})
    for i, r in enumerate(rows):
        if r["class"] is None:
            r["class"] = f"#{i+1}"
    if rows:
        return rows''',
    done_marker='"클래스명미상": c is None')

# ══════════════════════════════════════════════════════
# A-2  pass3 main — 보수_신뢰불가 플래그
# ══════════════════════════════════════════════════════
patch(
    "pass3_extract.py", "A-2 보수_신뢰불가 플래그",
    '''        rec["보수"] = fees
        if err:''',
    '''        rec["보수"] = fees
        # ★ 클래스 매핑이 없는 보수 행이 하나라도 있으면 문서 단위로 표시한다.
        #   pass8 _fee_line / pass9 _fee_map 의 방어가 이 플래그로 작동한다.
        if any(r.get("클래스명미상") for r in fees):
            rec["보수_신뢰불가"] = True
        if err:''',
    done_marker='if any(r.get("클래스명미상") for r in fees):')

# ══════════════════════════════════════════════════════
# A-3  pass7 — coord-noclass 분기 도달 가능하게
# ══════════════════════════════════════════════════════
patch(
    "pass7_fee_recover.py", "A-3 #N 폴백 활성화",
    '''                        if len(vals) == len(known):        # 정확히 일치 → 확정
                            best = (vals, "coord-order")
                            break''',
    '''                        if known and len(vals) == len(known):   # 정확히 일치 → 확정
                            best = (vals, "coord-order")
                            break
                        # ★ 클래스 목록 자체가 없는 문서 — 순서만 보존해 살린다.
                        #   (known이 비면 위 조건은 절대 성립하지 않아 예전엔 죽은 코드였다)
                        if not known and best is None:
                            best = (vals, "coord-noclass")''',
    done_marker='if not known and best is None:')

# ══════════════════════════════════════════════════════
# A-4  pass6 — 수익률_수동확인필요 를 rec 에 실제로 붙인다
# ══════════════════════════════════════════════════════
patch(
    "pass6_broken.py", "A-4 수동확인필요 플래그 병합",
    '''    save_json(six, os.path.join(TABLE_DIR, "six_axis.json"))
    n = len(six)
    ok = sum(1 for r in six if r.get("펀드수익률"))''',
    '''    # ★ 복구까지 실패한 문서에 플래그를 세운다.
    #   parse_perf 안에서 세우면 main의 병합 조건에 걸려 절대 rec에 붙지 않는다.
    #   pop은 재실행 시 낡은 플래그가 남는 것을 막는다.
    for rec in six:
        if not (rec.get("펀드수익률") or rec.get("클래스수익률")):
            rec["수익률_수동확인필요"] = True
        else:
            rec.pop("수익률_수동확인필요", None)

    save_json(six, os.path.join(TABLE_DIR, "six_axis.json"))
    n = len(six)
    ok = sum(1 for r in six if r.get("펀드수익률"))''',
    done_marker='rec.pop("수익률_수동확인필요", None)')

# ══════════════════════════════════════════════════════
# A-5  pass6 — 중복 DATE_ANY 정의 제거
# ══════════════════════════════════════════════════════
patch(
    "pass6_broken.py", "A-5a DATE_ANY 중복 제거",
    '''# 펀드명 라벨 뒤 어디에나 오는 설정일 제거용
DATE_ANY = re.compile(r"\\d{4}[-./]\\d{1,2}[-./]\\d{1,2}")
# 펀드명 라벨 뒤 어디에나 오는 설정일 제거용
DATE_ANY = re.compile(r"\\d{4}[-./]\\d{1,2}[-./]\\d{1,2}")''',
    '''# 펀드명 라벨 뒤 어디에나 오는 설정일 제거용
DATE_ANY = re.compile(r"\\d{4}[-./]\\d{1,2}[-./]\\d{1,2}")''')

# ══════════════════════════════════════════════════════
# A-5b pass6 — 복붙된 두 번째 폴백 블록 삭제
#      (사본에는 _looks_like_period 필터가 빠져 있다)
# ══════════════════════════════════════════════════════
patch(
    "pass6_broken.py", "A-5b 중복 폴백 블록 삭제",
    '''                # ★ 펀드 행 라벨이 **펀드명 자체**인 문서가 있다:
                #   "Tomorrow장기우량증권투자신탁K-1호(채권)2006-09-054.856.87…"
                #   고정 패턴으로 못 잡으므로 위치로 잡는다 — 비교지수 바로 위 행에서
                #   같은 개수의 숫자가 나오면 그게 펀드 행이다(4패스와 동일한 논리).
                # ⚠️ 비교지수가 페이지 첫 줄이면 펀드 행은 이전 페이지에 있다 → 추측하지 않는다.
                if out["펀드수익률"] is None:
                    for k in range(i - 1, max(-1, i - 4), -1):
                        prev = lines[k].replace(" ", "")
                        if prev.startswith(("비교지수", "수익률변동성", "참조지수")):
                            continue
                        pv = split_nums(DATE_ANY.sub("", prev))
                        if pv and len(pv) == len(v):
                            out["펀드수익률"] = pv[:5]
                            break

    # ── 라벨 없는 폴백''',
    '''
    # ── 라벨 없는 폴백''')

# ══════════════════════════════════════════════════════
# A-6  pass4 find_turnover — 라벨 미검출 vs 값 없음 분리
# ══════════════════════════════════════════════════════
patch(
    "pass4_perf.py", "A-6a 회전율 라벨 감지",
    '''    for p in pages:
        if not (p3 <= p["page"] <= p3 + 14):
            continue
        lines = p["text"].split("\\n")
        for i, line in enumerate(lines):
            L = line.strip()
            s = L.replace(" ", "")
            if not s.startswith("매매회전율") or len(s) > 60:
                continue''',
    '''    saw_label = False          # ★ 라벨을 봤는데 값이 없다 vs 라벨 자체가 없다
    for p in pages:
        if not (p3 <= p["page"] <= p3 + 14):
            continue
        lines = p["text"].split("\\n")
        for i, line in enumerate(lines):
            L = line.strip()
            s = L.replace(" ", "")
            if not s.startswith("매매회전율") or len(s) > 60:
                continue
            saw_label = True''',
    done_marker="saw_label = False")

patch(
    "pass4_perf.py", "A-6b 회전율 반환값 분기",
    '''            if v:                      # ★ 0도 유효값이다 (실제로 '0 0 0'인 펀드가 있음)
                return v[0], v
    return None, None''',
    '''            if v:                      # ★ 0도 유효값이다 (실제로 '0 0 0'인 펀드가 있음)
                return v[0], v
    # 라벨은 있는데 값이 '-'/'_' → 문서가 명시한 '해당없음'
    # 라벨 자체가 없음 → 파싱 미검출이지 '해당없음'이 아니다
    return ("N/A", None) if saw_label else (None, None)''')

patch(
    "pass4_perf.py", "A-6c 미검출을 해당없음으로 단정하지 않기",
    '''            rec["매매회전율_해당없음"] = to is None''',
    '''            # ★ 여기 오는 to is None 은 '라벨 자체 미검출' = 미확인이다.
            #   해당없음은 위 "N/A" 분기에서만 세운다.
            rec["매매회전율_해당없음"] = False''',
    done_marker='rec["매매회전율_해당없음"] = False')

# ══════════════════════════════════════════════════════
# B-1  pass8 — 회전율 미확인도 줄로 명시
# ══════════════════════════════════════════════════════
patch(
    "pass8_chunk_fund.py", "B-1 회전율 미확인 명시",
    '''    elif isinstance(to, (int, float)) and to > 0:
        lines.append(f"매매회전율: {to:g}%")''',
    '''    elif isinstance(to, (int, float)) and to > 0:
        lines.append(f"매매회전율: {to:g}%")
    elif to is None:
        lines.append(f"매매회전율: {NA}")      # 빈 값은 생략하지 않고 명시''',
    done_marker='lines.append(f"매매회전율: {NA}")')


# ── 결과 ────────────────────────────────────────────────
w = max(len(t) for _, t, _ in _LOG)
for st, tag, msg in _LOG:
    mark = {"OK": "✅", "SKIP": "•", "FAIL": "❌"}[st]
    print(f"{mark} {tag:<{w}}  {msg}")

nf = sum(1 for s, _, _ in _LOG if s == "FAIL")
print(f"\n적용 {sum(1 for s,_,_ in _LOG if s=='OK')} / 건너뜀 "
      f"{sum(1 for s,_,_ in _LOG if s=='SKIP')} / 실패 {nf}")

# 문법 검사 — 깨진 채로 파이프라인을 돌리지 않는다
import py_compile
bad = []
for f in ["pass3_extract.py", "pass4_perf.py", "pass6_broken.py",
          "pass7_fee_recover.py", "pass8_chunk_fund.py"]:
    p = os.path.join(DIR, f)
    try:
        py_compile.compile(p, doraise=True)
    except Exception as e:
        bad.append((f, str(e)[:120]))
if bad:
    print("\n❌ 문법 오류 — 되돌리려면 .bak 을 복사하세요")
    for f, e in bad:
        print(f"   {f}: {e}")
else:
    print("✅ 전 파일 문법 정상")
if nf:
    print("\n⚠️ FAIL 이 있으면 해당 패치는 적용되지 않았습니다. 파일이 이미 수정된 상태일 수 있습니다.")
