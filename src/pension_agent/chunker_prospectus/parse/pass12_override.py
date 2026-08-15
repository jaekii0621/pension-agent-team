"""
12패스 — 수작업 보정 적용 (manual_override.json → six_axis.json)

⚠️ 왜 별도 패스인가
   수작업으로 확인한 값을 six_axis.json에 **직접 쓰면 안 된다.**
   3패스를 다시 돌리는 순간 통째로 사라진다.
   별도 파일에 보관하고 파이프라인 마지막에 덮어씌워야 재실행에도 살아남는다.

⚠️ 실행 위치: 11패스(범위 검증) **뒤**, 청커(8~10패스) **앞**
   검증이 격리한 값을 수작업으로 되살리는 경우가 있으므로 순서가 중요하다.

원칙
  · 수작업 값은 파싱 값을 **덮어쓴다** (사람이 원문을 직접 본 결과가 우선)
  · 어느 필드가 수작업인지 `수작업보정` 리스트로 남긴다
    → 청커가 parse_method="manual"로 표시하고,
      나중에 "수작업분에서 오답이 나는가"를 분석할 수 있다
  · 검증이 격리한 필드를 수작업으로 채우면 해당 '검증실패' 사유를 지운다

manual_override.json 형식:
  {
    "R2_KR5118420006": {"위험등급": 5, "위험라벨": "낮은 위험"},
    "R2_KR5153450009": {"연금저축_가능": false, "퇴직연금_가능": true}
  }

산출: six_axis.json 갱신
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse.common import TABLE_DIR, load_json, save_json

OV_PATH = os.path.join(TABLE_DIR, "manual_override.json")
# ★ 수작업 값의 근거(페이지·원문 한 줄). refine_evidence.py 가 만든다.
#   확인필요=True 인 항목은 붙이지 않는다 — 틀린 근거를 다는 것보다 없는 게 낫다.
EV_PATH = os.path.join(TABLE_DIR, "override_evidence.json")


def main():
    path = os.path.join(TABLE_DIR, "six_axis.json")
    six = load_json(path)

    if not os.path.exists(OV_PATH):
        print(f"수작업 보정 파일 없음 ({OV_PATH}) — 건너뜀")
        return
    ov = load_json(OV_PATH)
    ev = load_json(EV_PATH) if os.path.exists(EV_PATH) else {}
    if not ev:
        print("  (근거 파일 없음 — refine_evidence.py 를 먼저 돌리면 근거가 붙습니다)")

    by_id = {r["doc_id"]: r for r in six}
    applied, unknown, fields = 0, [], {}

    for did, patch in ov.items():
        rec = by_id.get(did)
        if rec is None:
            unknown.append(did)
            continue
        for k, v in patch.items():
            rec[k] = v
            fields[k] = fields.get(k, 0) + 1
        # ★ 보수를 수작업으로 다시 넣었으면 파싱 단계의 신뢰불가 플래그는 무효다.
        #   안 지우면 이 플래그를 보는 코드가 정상 값을 계속 버린다.
        if "보수" in patch:
            rec.pop("보수_신뢰불가", None)
        rec["수작업보정"] = sorted(set((rec.get("수작업보정") or []) + list(patch)))
        # ★ 근거 부착 — 페이지가 있고 확인필요가 아닌 것만.
        #   답변에서 "연금저축 가입 가능"을 인용할 때 쓸 span이 여기서 나온다.
        src_ev = {k: v for k, v in (ev.get(did) or {}).items()
                  if isinstance(v, dict) and v.get("page") and not v.get("확인필요")}
        if src_ev:
            rec["수작업근거"] = {k: {"page": v["page"], "quote": v.get("quote")}
                             for k, v in src_ev.items()}
        # ★ 검증이 격리했던 필드를 수작업으로 채웠으면 그 사유는 해소된 것
        fails = [f for f in (rec.get("검증실패") or [])
                 if f.split(":")[0] not in patch]
        if fails:
            rec["검증실패"] = fails
        else:
            rec.pop("검증실패", None)
        applied += 1

    save_json(six, path)

    n = len(six)
    print(f"수작업 보정 {applied}/{len(ov)}개 문서 적용")
    if unknown:
        print(f"  ⚠️ six_axis에 없는 doc_id {len(unknown)}건: {unknown[:5]}")
    print(f"\n{'필드':<18}{'보정':>5}{'전체유효':>9}")
    print("-" * 33)
    for k, c in sorted(fields.items(), key=lambda t: -t[1]):
        ok = sum(1 for r in six if r.get(k) is not None)
        print(f"{k:<18}{c:>5}{ok:>6}/{n}")
    ev_docs = sum(1 for r in six if r.get("수작업근거"))
    ev_flds = sum(len(r.get("수작업근거") or {}) for r in six)
    tgt = sum(len([k for k in p if not k.startswith("_")]) for p in ov.values())
    print(f"\n근거 부착 {ev_flds}/{tgt}건 (문서 {ev_docs}개)")
    miss = [r["doc_id"] for r in six
            if r.get("수작업보정") and not r.get("수작업근거")]
    if miss:
        print(f"  근거 없는 문서 {len(miss)}건: {miss[:5]}")
    print("\n※ 수작업 값은 파싱 값을 덮어쓴다. 어느 필드인지는 '수작업보정'에 기록된다.")


if __name__ == "__main__":
    main()
