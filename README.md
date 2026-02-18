# lol-report-mvp

Riot API + LLM 기반 `LoL 자동 해설 리포트` MVP.
목표는 완성도가 아니라 **한 번이라도 end-to-end 성공**입니다.

## 프로젝트 구조
- `api`: FastAPI 백엔드
- `web`: Next.js 프론트엔드

## Step 0 (완료)
- `api`, `web` 폴더 확인
- git 초기화
- 공통 파일 추가: `AGENTS.md`, `README.md`, `.env.example`, `.gitignore`

## Step 1 (완료) Web 더미 라우트
- `/`
- `/pay`
- `/start`
- `/generating/[report_id]` (status polling + 실패 UX)
- `/report/[report_id]` (마크다운 렌더 + 복사 버튼 + 실패 UX)

## Step 2 (완료) API 더미 Contract
- `POST /create-report`
- `GET /report-status?report_id=...`
- `GET /report?report_id=...`

백엔드는 SQLite(`report_jobs`)에 상태/진행률/섹션을 저장하고 background thread로 작업을 처리합니다.

## Step 3 (완료) Riot API 연결
- region 매핑: `KR -> (kr1, asia)`, `NA -> (na1, americas)`, `EUW -> (euw1, europe)`
- `riot_id -> puuid` 조회
- 최근 20게임 id 조회
- match detail 동시성 제한(세마포어 4)
- 429 처리: `Retry-After` 우선, 없으면 지수 backoff
- 요약 JSON 생성 (games_analyzed, win_rate, role/champ/kda/cs/streak/early proxy 등)

## Step 4 (완료) LLM 연결
- 입력: summary_json + tone + language
- 출력: 5섹션 markdown JSON
- 시스템/유저 프롬프트 분리
- OpenAI 키 없으면 fallback 더미 섹션 사용

## Step 5 (완료) 안정화 최소 기능
- 동일 입력 캐시 키 재사용 (`riot_id/region/tone/language`)
- 실패 시 `failed + error` 저장
- 프론트 재시도 버튼 제공
- 서버 로그 추가

## API 실행
```bash
cd api
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Web 실행
```bash
cd web
npm install
npm run dev
```

## 환경 변수
루트 `.env.example` 참고:
- `RIOT_API_KEY`
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional, default: `gpt-5-mini`)
- `OPENAI_MAX_OUTPUT_TOKENS` (optional, default: `2000`)
- `DATABASE_URL`
- `NEXT_PUBLIC_API_BASE_URL`

OpenAI 호출은 Responses API(`client.responses.create`)를 사용하며, GPT-5 mini 호환을 위해
`temperature`/`top_p`/`logprobs`는 전달하지 않습니다.

## curl end-to-end 테스트
1. create
```bash
curl -X POST "http://localhost:8000/create-report" ^
  -H "Content-Type: application/json" ^
  -d "{\"riot_id\":\"Hide on bush#KR1\",\"region\":\"KR\",\"tone\":\"funny\",\"language\":\"ko\"}"
```

2. status polling 예시
```bash
curl "http://localhost:8000/report-status?report_id=<REPORT_ID>"
```

3. report get
```bash
curl "http://localhost:8000/report?report_id=<REPORT_ID>"
```

## 확인 방법
1. API와 Web을 각각 실행
2. `http://localhost:3000` 접속
3. `/pay` -> 결제 완료(테스트) -> `/start` 폼 제출
4. `/generating/[report_id]`에서 진행률 확인
5. 완료 시 `/report/[report_id]` 렌더 확인
6. 실패 상태에서는 실패 메시지/다시 시도 버튼 확인

## Step 3.1 스냅 UX 수동 체크리스트
- 트랙패드 관성 스크롤 시 강제 점프/잠금 없이 자연스럽게 스냅되는지 확인
- 마우스 휠 스크롤 시 섹션 단위로 자연스럽게 스냅되는지 확인
- 모바일 터치 스크롤에서 스냅이 유지되고 스크롤 트래핑이 없는지 확인
- 우측 도트 버튼이 Tab으로 포커스되고 Enter/Space로 섹션 이동되는지 확인
- `← 메인으로 돌아가기` 동작 후 다른 페이지 스크롤이 정상인지 확인
