# 골드셋

`골드셋_형식_확정_0820.md`(민서) 기준. 이 폴더가 `config.GOLD_DIR` 이다.

```
eval/gold/
├─ product.jsonl   상품 트랙 (민서)  16건 = dev 11 / holdout 5
├─ policy.jsonl    제도 트랙 (재희)  24건 = dev 19 / holdout 5
└─ README.md       이 파일
```

- 한 줄에 한 레코드. **순수 jsonl** — `//` 주석을 넣지 않는다(설명은 `note` 필드).
- `id` 접두어로 트랙 구분 (`P-` 상품 / `D-` 제도).
- 파일이 **둘 다 없으면 즉시 실패**한다. 조용히 "채점 대상 0건"으로 도는 사고를 막는다.

## 돌리는 법

```bash
# 파이프라인을 돌리면서 채점
python -m eval.runner --gold --split dev

# 이미 있는 실행 기록으로 재채점 (LLM 호출 0회)
python -m eval.runner --gold --from-compact eval/results/queryset_XXXX.compact.jsonl

# 트랙 한쪽만
python -m eval.runner --gold --track policy --split dev
```

## 형식 검증

로드할 때 자동으로 돈다. 위반이 있으면 **채점 전에** 한 번에 보여주고 멈춘다.

| 검사 | 내용 |
|---|---|
| `id` | 있는가 / 중복 아닌가 |
| `kind` | 8종 enum 안인가 |
| `split` | `dev` \| `holdout` |
| `expect_block` | 5종 enum 안인가 |
| `expect_caution` | 4종 enum 안인가 |
| `tests` | 8종 enum 안인가 (중점·괄호·띄어쓰기 금지) |
| `evidence` | 딕셔너리이거나 `null` |
| `clarify_slots` | **동의어 묶음**(배열의 배열)인가 — 문자열이면 매칭이 통째로 실패 |
| kind별 필수 | `calc.value` / `clarify_slots` / `conflict_values` / `refuse_ok`의 `expect_block` |
| `absent` | `evidence`를 비우지 않았는가 (§10) |

수동 확인:

```python
from eval import gold
items = gold.load_goldset(strict=False)
print(gold.stats(items))
print(gold.validate(items))     # 빈 리스트면 정상
```

## 채점 결과 읽는 법

```jsonc
{
  "id": "D-C1",
  "answer_score": 2,        // 자동 — must/must_not, kind별 규칙
  "evidence_score": 2,      // 자동 — tier → chunk_type → chunk_id/doc_id
  "requirement_score": null, // 사람이 채운다
  "attitude_score": null,    // 사람이 채운다
  "gate_pass": true,         // 차단·주의 게이트 (총점 밖)
  "auto_score": 4, "auto_max": 4,
  "needs_human": false,      // calc 보류 / must_not 위반 시 true
  "failed": false            // 게이트 실패 또는 답변 0점
}
```

요약에는 차단 네 지표가 **총점과 별도로** 나온다.

```jsonc
"block": {
  "오차단율": 0.0,   // expect_block=none 인데 막힌 비율
  "미차단율": 0.0,   // expect_block≠none 인데 안 막힌 비율
  "복구율": 0.57     // 통과했지만 caution이 붙은 비율 = 룰이 구제한 것
}
```

**복구율이 중요하다.** 판정 자체가 정확해진 것과, LLM은 오판했는데 코퍼스 신호가
구제한 것은 다르다. 후자는 표현이 바뀌면 다시 막힌다.

## 채점기 수정 이력

| 날짜 | 내용 |
|---|---|
| 2026-08-20 | `gold_ids_for`가 `evidence`를 리스트로 순회하던 것 수정 — 확정 형식은 **딕셔너리 하나**라 항상 `[]`를 반환했다 |
| 2026-08-20 | `_doc_of` 구분자 `-` → `#` (우리 규약은 `doc52#p4#c003`). doc 수준 폴백이 같이 죽어 있었다 |
| 2026-08-20 | `_final_amount`가 `steps`를 함께 보도록 보완 — §1은 steps에 전제 금액을 넣으라 하고 §6-2는 표지 없으면 마지막 금액을 쓰라 해서, 답변이 전제를 되풀이하며 끝나면 정답이 오답이 됐다 |

## 검증

```bash
python scratchpad/verify_gold.py
```

형식 확정본의 필수 검증 케이스를 그대로 돌린다 — §4-4 conflict 6종,
§6-3 calc 9종, §7 정규식 패턴, 근거 축 6단계, 차단 게이트 6종.
