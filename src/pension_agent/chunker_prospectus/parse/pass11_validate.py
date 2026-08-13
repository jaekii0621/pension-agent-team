"""
11패스 — 범위 검증 (six_axis.json)

⚠️ 원칙: **걸린 값을 고치지 않는다. 비우고 사유를 남긴다.**
   어떤 값이 맞는지 모르는 상태에서 추측해 넣으면 그게 가장 위험하다.
   None + 사유 플래그로 두면 에이전트가 "확인하지 못했습니다"라고
   정직하게 답할 수 있다(평가지표: 정보한계대응).

이 그물이 잡으려는 것 — 지금까지 실제로 발생했던 조용한 오답들:
  · 시장잔고 4,781,815,218,648 백만원  (원 단위 미환산 — '[단위:원]' 대괄호 표기 미인식)
  · 수익률 [24.05, 23.05, 22.05]        (결산기 연월 헤더를 수익률로 오인)
  · VaR 50.0 × 6건                       ('재산의 50% 이상' 비중 문구 오인)
  · 총보수 0.811                          (소수 자릿수 혼용으로 잘못 분리)
셋 다 정상 범위 밖이거나 형태가 이상해서, 값만 보고도 판정할 수 있다.

앞으로 새 운용사 문서를 넣었을 때 **모르는 형식에서 생긴 오류가 조용히
통과하는 것**을 막는 것이 이 패스의 목적이다.

산출: six_axis.json 갱신 + data/tables/validation_report.json
"""
import os, re, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import TABLE_DIR, load_json, save_json

# ── 규칙 ─────────────────────────────────────────────
# (필드, 판정함수, 사유)  — True면 '이상'
RULES = []


def rule(field, reason):
    def deco(fn):
        RULES.append((field, fn, reason))
        return fn
    return deco


@rule("시장잔고_백만원", "AUM 10조원 초과 — 원/천원 단위 미환산 의심")
def _aum(v, rec):
    return isinstance(v, (int, float)) and v > 1e7


@rule("시장잔고_백만원", "AUM이 0 이하")
def _aum0(v, rec):
    return isinstance(v, (int, float)) and v <= 0


@rule("위험등급", "위험등급이 1~6 범위 밖")
def _grade(v, rec):
    return v is not None and not (isinstance(v, int) and 1 <= v <= 6)


@rule("VaR_97.5", "VaR가 비중 표현으로 보이는 정수값(50/60/80/100)")
def _var_int(v, rec):
    return isinstance(v, (int, float)) and float(v) in (50.0, 60.0, 80.0, 100.0)


@rule("VaR_97.5", "VaR가 0~100% 범위 밖")
def _var_rng(v, rec):
    return isinstance(v, (int, float)) and not (0 < v <= 100)


@rule("매매회전율", "매매회전율이 음수 또는 5000% 초과")
def _turnover(v, rec):
    return isinstance(v, (int, float)) and not (0 <= v <= 5000)


@rule("펀드수익률", "수익률이 연도 표기로 보임(15~30 구간 양수만)")
def _perf_period(v, rec):
    if not isinstance(v, list) or len(v) < 3:
        return False
    if len(set(v)) == 1:                                   # 전부 같은 값 = 날짜 반복
        return True
    h = [x for x in v if isinstance(x, (int, float)) and 15 <= x <= 30]
    if len(h) >= 3 and all(round(h[i] - h[i + 1], 1) in (1.0, 2.0)
                           for i in range(len(h) - 1)):     # 1~2씩 감소 = 연월
        return True
    return len(h) == len(v) and all(x > 0 for x in v)


@rule("펀드수익률", "수익률이 -100% ~ +200% 범위 밖")
def _perf_rng(v, rec):
    return isinstance(v, list) and any(
        isinstance(x, (int, float)) and not (-100 < x < 200) for x in v)


@rule("비교지수수익률", "비교지수 수익률이 -100% ~ +200% 범위 밖")
def _bm_rng(v, rec):
    return isinstance(v, list) and any(
        isinstance(x, (int, float)) and not (-100 < x < 200) for x in v)


@rule("수익률변동성", "변동성이 0~200% 범위 밖")
def _vol(v, rec):
    return isinstance(v, (int, float)) and not (0 <= v <= 200)


# ── 보수는 행 단위라 별도 처리 ──
FEE_MIN, FEE_MAX = 0.0, 5.0
CLASS_OK = re.compile(r"^[A-Za-z](?:[A-Za-z0-9\-]{0,5}[A-Za-z0-9])?$")


def check_fees(rec):
    """총보수 행별 검증. 이상 행만 제거하고, 남는 게 없으면 필드를 비운다."""
    rows = rec.get("보수")
    if not rows:
        return []
    issues, keep = [], []
    for r in rows:
        v = r.get("총보수")
        cls = r.get("class") or ""
        if not isinstance(v, (int, float)) or not (FEE_MIN < v <= FEE_MAX):
            issues.append(f"총보수 {v!r}({cls}) 0~5% 범위 밖")
            continue
        # ⚠️ 보수 행의 클래스명은 검사하지 않는다.
        #   13항 표에서 설명 문구가 클래스명 자리에 들어오는 문서가 있지만
        #   **총보수 값 자체는 정확**하다. 여기서 버리면 보수가 69→49로 떨어진다.
        #   클래스 매핑 신뢰도는 '보수_신뢰불가' 플래그로 따로 표시한다.
        keep.append(r)
    if issues:
        rec["보수"] = keep or None
    return issues


def check_classes(rec):
    """클래스 코드 형식 검증."""
    cls = rec.get("클래스")
    if not cls:
        return []
    issues, keep = [], []
    for c in cls:
        name = c.get("class") or ""
        if name.startswith("#"):            # 순서 인덱스는 허용(클래스명 미상)
            keep.append(c)
            continue
        if not CLASS_OK.match(name):
            issues.append(f"클래스 코드 형식 이상: {name!r}")
            continue
        keep.append(c)
    if issues:
        rec["클래스"] = keep or None
    return issues


def main():
    path = os.path.join(TABLE_DIR, "six_axis.json")
    six = load_json(path)

    report, cleared = [], 0
    for rec in six:
        rec.pop("검증실패", None)          # ★ 재실행 시 사유 중복 누적 방지
        issues = []
        for field, fn, reason in RULES:
            v = rec.get(field)
            if v is None:
                continue
            try:
                bad = fn(v, rec)
            except Exception:
                bad = True
            if bad:
                issues.append({"필드": field, "값": v, "사유": reason})
                rec[field] = None
                rec.setdefault("검증실패", []).append(f"{field}: {reason}")
                cleared += 1
        for msg in check_fees(rec):
            issues.append({"필드": "보수", "값": None, "사유": msg})
            rec.setdefault("검증실패", []).append(f"보수: {msg}")
        for msg in check_classes(rec):
            issues.append({"필드": "클래스", "값": None, "사유": msg})
            rec.setdefault("검증실패", []).append(f"클래스: {msg}")
        if issues:
            report.append({"doc_id": rec["doc_id"],
                           "fund_name": rec.get("fund_name"), "issues": issues})

    save_json(six, path)
    rpath = os.path.join(TABLE_DIR, "validation_report.json")
    save_json(report, rpath)

    # ── 리포트 ★ 제안서 §데이터 구조화 근거표가 된다 ──
    from collections import Counter
    n = len(six)
    reasons = Counter(i["사유"] for r in report for i in r["issues"])
    print(f"범위 검증 — 문서 {n}개 / 이상 {len(report)}건 / 값 {cleared}개 격리")
    print(f"  리포트: {rpath}\n")
    if reasons:
        print(f"{'사유':<46}{'건수':>5}")
        print("-" * 52)
        for k, v in reasons.most_common():
            print(f"{k:<46}{v:>5}")
    else:
        print("  ✅ 전 항목 정상 범위")

    print(f"\n{'필드':<18}{'유효':>8}{'검증실패':>9}")
    print("-" * 36)
    for k in ["fund_name", "위험등급", "상품분류", "시장잔고_백만원",
              "펀드수익률", "보수", "클래스", "매매회전율", "VaR_97.5"]:
        # ★ 클래스·보수는 []로 남는다. is not None 이면 빈 표가 '유효'로 잡혀
        #   보수 92/92, 클래스 91/92 처럼 실제(74·54)와 딴판인 숫자가 찍힌다.
        ok = sum(1 for r in six if r.get(k) not in (None, [], {}, ""))
        ng = sum(1 for r in six
                 if any(f.startswith(k + ":") for f in (r.get("검증실패") or [])))
        print(f"{k:<18}{ok:>5}/{n}{ng:>9}")
    print("\n※ 격리된 값은 고치지 않고 None + '검증실패' 사유로 남긴다.")
    print("   틀린 값을 인용하는 것보다 '확인하지 못했다'가 낫다.")


if __name__ == "__main__":
    main()
