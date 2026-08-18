#!/usr/bin/env python3
"""patch24 — pass10 의 다음 섹션 경계 탐색 범위를 넓힌다.

■ 무엇이 문제였나
  `next_boundary()` 는 다음 섹션 헤딩을 찾으면 그 오프셋에서 정확히 자른다.
  그런데 스캔 범위가 `span()` 이 준 `b` 까지라, 다음 섹션 헤딩이 `b` 너머에
  있으면 아예 보지 못한다. 그 결과 **다음 섹션 시작 페이지의 헤딩 앞부분**
  (= 이 섹션의 정당한 꼬리)이 어느 섹션에도 속하지 못하고 사라진다.

  실측 (R2_KR510902511M):
    투자위험이 section_map 에 없어 resolve_sec 이 p19 로 잡음
      → span(smax=5) 이 다음 섹션(11항매입환매 p21) 앞인 p20 까지로 자름
      → next_boundary 가 p19~p20 만 훑어 헤딩 미검출 → (20, None)
      → p21 의 97.5% VaR 52.95% + 위험등급 기준표 714자가 통째로 유실

  같은 원인으로 VaR 48건 중 20건이 섹션 청크에 없다. 위험등급 기준표·
  판매경로 설명(랩/무권유저비용/온라인슈퍼)·평가기준도 같이 날아갔다.

■ 어떻게 고치나
  경계 탐색만 `b + LOOKAHEAD` 까지 넓힌다. **실제 헤딩을 찾은 경우에만**
  확장분을 쓰고, 못 찾으면 기존 `b` 로 되돌린다. 헤딩을 찾았다는 것은 다음
  섹션 위치가 확정됐다는 뜻이므로 폭주가 구조적으로 불가능하다.
  (`smax` 는 헤딩 미검출 시 폭주를 막는 안전장치인데, 그 상한이 탐색 시야까지
   막고 있었던 것이 원인이다.)

■ 안전한 이유
  · 배제 섹션 보호 유지 — next_boundary 가 _ALL(KEEP + BOUNDARY_ONLY)을
    모두 보므로 14항 과세·12항 기준가격이 먼저 나오면 거기서 멈춘다.
    check_bleed 0건이 깨지지 않는다.
  · chunk_id 충돌 없음 — build() 의 seq 가 문서 단위 페이지별 카운터라
    한 페이지에 두 섹션이 붙어도 #c010, #c011 로 자동 채번된다.

■ 사용법
    python patch24_section_tail.py            # 적용
    python patch24_section_tail.py --revert   # 직전 .bak 으로 되돌리기
    python patch24_section_tail.py --dry      # 미리보기만

  적용 후:
    python run_all.py --chunk
    python fit_embed_text.py
    python check_after.py
"""

import argparse
import os
import re
import shutil
import sys

TARGET = "src/parse/pass10_chunk_section.py"

OLD = '''    end_pg, end_off = next_boundary(by_page, a, start_off, b, key, toc)
    if end_off is not None:
        STATS["tail_trimmed"] = STATS.get("tail_trimmed", 0) + 1
'''

NEW = '''    # ★ patch24 — 경계 탐색만 span 밖으로 넓힌다.
    #   span 의 smax 는 헤딩 미검출 시 폭주를 막는 안전장치인데, 그 상한이
    #   next_boundary 의 시야까지 막아 다음 섹션 시작 페이지의 '헤딩 앞부분'
    #   (= 이 섹션의 정당한 꼬리)이 통째로 버려졌다. VaR 20건이 이 경우다.
    #   실제 헤딩을 찾은 경우에만 확장분을 쓰므로 폭주 위험은 없다.
    LOOKAHEAD = 3
    end_pg, end_off = next_boundary(
        by_page, a, start_off, min(npages, b + LOOKAHEAD), key, toc)
    if end_off is None:
        end_pg, end_off = b, None          # 헤딩 미검출 → 기존 동작(smax 상한)
    elif end_pg > b:
        STATS["tail_recovered"] = STATS.get("tail_recovered", 0) + 1
    else:
        STATS["tail_trimmed"] = STATS.get("tail_trimmed", 0) + 1
'''


def find_target(explicit=None):
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (os.getcwd(), here, "/content/pension"):
        p = os.path.join(root, TARGET)
        if os.path.exists(p):
            return p
        p = os.path.join(root, os.path.basename(TARGET))
        if os.path.exists(p):
            return p
    return None


def next_bak(path):
    if not os.path.exists(path + ".bak"):
        return path + ".bak"
    n = 2
    while os.path.exists(f"{path}.bak{n}"):
        n += 1
    return f"{path}.bak{n}"


def latest_bak(path):
    cands = []
    d = os.path.dirname(path) or "."
    base = os.path.basename(path)
    for f in os.listdir(d):
        m = re.fullmatch(re.escape(base) + r"\.bak(\d*)", f)
        if m:
            cands.append((int(m.group(1) or 1), os.path.join(d, f)))
    return max(cands)[1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="pass10_chunk_section.py 경로 (자동 탐색 실패 시)")
    ap.add_argument("--dry", action="store_true", help="적용하지 않고 미리보기")
    ap.add_argument("--revert", action="store_true", help="가장 최근 .bak 으로 복원")
    args = ap.parse_args()

    path = find_target(args.path)
    if not path:
        sys.exit(f"[x] 대상 파일을 못 찾았습니다. --path 로 지정하세요 ({TARGET})")
    print(f"[대상] {path}")

    if args.revert:
        bak = latest_bak(path)
        if not bak:
            sys.exit("[x] 복원할 .bak 이 없습니다")
        shutil.copy2(bak, path)
        print(f"[✓] 복원 완료 — {os.path.basename(bak)} → {os.path.basename(path)}")
        return

    src = open(path, encoding="utf-8").read()

    if "patch24" in src:
        print("[!] 이미 적용된 파일입니다. 아무것도 하지 않습니다.")
        return

    n = src.count(OLD)
    if n == 0:
        print("[x] 대상 블록을 찾지 못했습니다. 파일이 이미 수정됐을 수 있습니다.")
        print("    찾던 블록:")
        for l in OLD.rstrip("\n").split("\n"):
            print("      " + l)
        sys.exit(1)
    if n > 1:
        sys.exit(f"[x] 대상 블록이 {n}곳에서 발견됐습니다. 수동 확인이 필요합니다.")

    new_src = src.replace(OLD, NEW)

    # 문법 검사 — 깨진 파일을 쓰지 않는다
    try:
        compile(new_src, path, "exec")
    except SyntaxError as e:
        sys.exit(f"[x] 패치 결과가 문법 오류입니다: {e}")

    if args.dry:
        print("[dry] 아래로 교체됩니다:\n")
        print(NEW)
        return

    bak = next_bak(path)
    shutil.copy2(path, bak)
    open(path, "w", encoding="utf-8").write(new_src)

    print(f"[✓] 백업  {os.path.basename(bak)}")
    print(f"[✓] 패치 적용 완료")
    print()
    print("다음 순서로 실행하세요:")
    print("  python run_all.py --chunk")
    print("  python fit_embed_text.py       # 새로 붙은 꼬리가 512토큰을 넘을 수 있음")
    print("  python check_after.py")
    print()
    print("STATS 출력에 tail_recovered 가 새로 찍힙니다. 그게 되살린 건수입니다.")


if __name__ == "__main__":
    main()
