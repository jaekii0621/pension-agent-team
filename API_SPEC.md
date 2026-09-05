# 평가용 API 서버 명세서

제10회 2026 미래에셋증권 AI Festival — 주제2 "연금 Agent"
제출 항목 ③ 평가용 API 서버 정보

---

## Endpoint

```
GET http://211.188.56.99:8000/answer
```

- 서버 환경: NCP (Naver Cloud Platform), Ubuntu 24.04, s2-g3a (2vCPU/8GB)
- 운영 기간: 2026.09.07 ~ 2026.09.20 (평가기간 중 반드시 API 활성화 유지)
- 요청 방식: 순차 요청만 지원 (동시 요청 다건 시 429 발생 가능 — 대회 규정상 평가는 순차 요청으로 진행됨을 확인함)
- 타임아웃: 300초

## Request

| 파라미터 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `question_id` | string | Y | 평가 질문 ID (예: `Q-001`) |
| `question` | string | Y | 평가 질문 원문 |

쿼리 파라미터로 전달 (GET).

### 요청 예시 (cURL)

```bash
curl -G "http://211.188.56.99:8000/answer" \
  --data-urlencode "question_id=Q-001" \
  --data-urlencode "question=DC와 DB, 퇴직금이 정해지는 방식이랑 운용 주체가 어떻게 다른가요?"
```

### 요청 예시 (Python)

```python
import requests

resp = requests.get(
    "http://211.188.56.99:8000/answer",
    params={"question_id": "Q-001", "question": "DC와 DB, 퇴직금이 정해지는 방식이랑 운용 주체가 어떻게 다른가요?"},
    timeout=300,
)
result = resp.json()
```

## Response

전 필드 string, JSON 형식.

| 필드 | 타입 | 설명 |
|---|---|---|
| `question_id` | string | 요청받은 질문 ID 그대로 반환 |
| `question` | string | 요청받은 질문 원문 그대로 반환 |
| `retrieved_context` | string | 답변 생성에 사용된 근거 문서 발췌 (실제 원문 스니펫 기반, LLM 합성 답변 아님) |
| `think_trace` | string | 사고·추론·도구 사용 과정 로그 |
| `answer` | string | 최종 생성 답변 (출처 표기 포함) |

### 응답 예시

```json
{
  "question_id": "FINAL-TEST-1",
  "question": "한국투자골드플랜연금 국공채·채권·주식, 뭐가 달라요? 안정적인 걸 원해요.",
  "retrieved_context": "[1] 한국투자골드플랜연금증권전환형자투자신탁1호(채권) / 펀드요약 (p.5)\n...",
  "think_trace": "통과 — 질문이 연금 상품에 대한 내용을 다루고 있으며...",
  "answer": "**한국투자골드플랜연금 국공채, 채권, 주식의 차이점과...**\n..."
}
```

(전체 원문은 2026-09-05 최종 서버 테스트 응답 참고)

## 오류 응답

| 상황 | HTTP 상태 | 응답 |
|---|---|---|
| 처리 시간 300초 초과 | 504 | 5개 필드 모두 포함, `answer`/`retrieved_context`/`think_trace`는 빈 문자열 |
| 서버 내부 오류 | 500 | 5개 필드 모두 포함, `answer`에 오류 메시지 |

## 부가 엔드포인트 (대회 규정 외, 헬스체크용)

```
GET /health
```

서버·임베딩 모델 정상 로드 여부 확인용. 대회 평가 대상 아님.

```json
{"status": "ok", "embeddings": {"ok": true, "model": "intfloat/multilingual-e5-large", "dim": 1024}}
```
