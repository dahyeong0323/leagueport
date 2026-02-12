# LoL Auto Commentary Report MVP Rules

## Mission
- 2주 내 `1회`라도 end-to-end 성공이 최우선이다.
- 완성도보다 동작하는 한 바퀴를 우선한다.

## Definition of Done
1. 테스트 결제 이후 Riot ID 입력
2. 최근 게임 데이터 기반 요약 JSON 생성
3. LLM 5섹션 마크다운 리포트 생성
4. 프론트에서 생성 상태 polling 후 결과 렌더
5. 실패 시 에러 화면 및 재시도 UX 제공

## Hard Constraints
- Riot 공식 API만 사용 (크롤링 금지)
- 지표는 10~15개만 유지
- 로그인/구독/친구비교/랭킹/PDF/다국어 확장/디자인 과몰입 금지
- 원본 match JSON을 LLM에 직접 전달 금지

## API Contract (Fixed)
- `POST /create-report`
- `GET /report-status?report_id=...`
- `GET /report?report_id=...`

## Runtime Commands
- API: `uvicorn app.main:app --reload --port 8000` (workdir: `api`)
- WEB: `npm run dev` (workdir: `web`)

## Environment Keys
- `RIOT_API_KEY`
- `OPENAI_API_KEY`
- `DATABASE_URL`
- `NEXT_PUBLIC_API_BASE_URL`
- `API_BASE_URL`

