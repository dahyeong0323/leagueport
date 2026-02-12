import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Base, SessionLocal, engine, get_db
from app.repository import (
    clone_done_into_new_job,
    create_job,
    find_done_cache,
    get_job,
    make_cache_key,
    store_report,
    update_status,
)
from app.schemas import (
    CreateReportRequest,
    CreateReportResponse,
    ReportMeta,
    ReportResponse,
    ReportSection,
    ReportStatusResponse,
)
from app.services.llm_service import LlmGenerationError, generate_sections
from app.services.riot_service import RiotApiError, fetch_riot_summary


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lol-report-api")

app = FastAPI(title="LoL Report API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("database initialized")


def _dummy_summary() -> dict:
    return {
        "games_analyzed": 20,
        "win_rate": 0.55,
        "main_role": "MIDDLE",
        "role_consistency": 0.8,
        "champion_pool_size": 6,
        "most_played_champ": "Ahri",
        "avg_kda": "7.0/4.2/8.1",
        "deaths_per_game": 4.2,
        "avg_cs_per_min": 6.8,
        "avg_game_duration_min": 30.5,
        "streak_state": {"type": "W", "length": 2},
        "early_impact_proxy": 0.43,
        "vision_proxy": 0.9,
        "objective_proxy": 1.7,
    }


def _dummy_sections() -> list[dict[str, str]]:
    return [
        {"title": "한 줄 요약", "content_markdown": "- 더미 리포트: 캐리 욕심과 안정성 사이에서 줄타기 중"},
        {"title": "최근 흐름", "content_markdown": "- 더미 리포트: 최근 20게임 기준 기복은 있지만 상승 여지 충분"},
        {"title": "플레이 스타일", "content_markdown": "- 더미 리포트: 라인전 주도는 좋고, 한타 진입 타이밍 보정 필요"},
        {"title": "챔프/포지션 성향", "content_markdown": "- 더미 리포트: 미드 고정 성향, 시그니처 챔프 집중형"},
        {
            "title": "한 줄 처방 + 다음 3판 미션",
            "content_markdown": "- 처방: 과감함 유지하되 데스만 절제\n- 다음 3판 미션\n- 첫 10분 데스 1회 이하\n- 시야 점수 팀 내 4위 이내\n- 같은 역할/챔프군으로 2판 연속",
        },
    ]


def _run_report_pipeline(report_id: str) -> None:
    db = SessionLocal()
    try:
        job = get_job(db, report_id)
        if not job:
            return
        logger.info("pipeline start report_id=%s", report_id)
        update_status(db, report_id, status="processing", progress=15)

        try:
            update_status(db, report_id, status="processing", progress=35)
            if settings.riot_api_key:
                summary = asyncio.run(fetch_riot_summary(job.riot_id, job.region, settings.riot_api_key))
            else:
                summary = _dummy_summary()

            update_status(db, report_id, status="processing", progress=70)
            if settings.openai_api_key or not settings.riot_api_key:
                sections = generate_sections(summary, tone=job.tone, language=job.language, openai_api_key=settings.openai_api_key)
            else:
                raise LlmGenerationError("OPENAI_API_KEY가 설정되지 않았습니다.")

            update_status(db, report_id, status="processing", progress=90)
            store_report(
                db,
                report_id=report_id,
                sections=sections if sections else _dummy_sections(),
                summary_json=summary,
                games_analyzed=int(summary.get("games_analyzed", 0)),
            )
            logger.info("pipeline done report_id=%s", report_id)
        except RiotApiError as exc:
            logger.error("riot error report_id=%s error=%s", report_id, exc)
            update_status(db, report_id, status="failed", progress=100, error=str(exc))
        except LlmGenerationError as exc:
            logger.error("llm error report_id=%s error=%s", report_id, exc)
            update_status(db, report_id, status="failed", progress=100, error=f"{exc} 재시도 해주세요.")
        except Exception as exc:
            logger.exception("unexpected error report_id=%s", report_id)
            update_status(db, report_id, status="failed", progress=100, error=f"알 수 없는 오류: {exc}")
    finally:
        db.close()


@app.post("/create-report", response_model=CreateReportResponse)
def create_report(payload: CreateReportRequest, db: Session = Depends(get_db)):
    report_id = str(uuid.uuid4())
    cache_key = make_cache_key(payload.riot_id, payload.region, payload.tone, payload.language)

    cached = find_done_cache(db, cache_key)
    if cached:
        clone_done_into_new_job(db, cached, new_report_id=report_id, cache_key=cache_key)
        return CreateReportResponse(report_id=report_id, status="queued")

    create_job(
        db=db,
        report_id=report_id,
        riot_id=payload.riot_id,
        region=payload.region,
        tone=payload.tone,
        language=payload.language,
        cache_key=cache_key,
    )
    threading.Thread(target=_run_report_pipeline, args=(report_id,), daemon=True).start()
    return CreateReportResponse(report_id=report_id, status="queued")


@app.get("/report-status", response_model=ReportStatusResponse)
def report_status(report_id: str = Query(...), db: Session = Depends(get_db)):
    job = get_job(db, report_id)
    if not job:
        raise HTTPException(status_code=404, detail="report_id를 찾을 수 없습니다.")
    return ReportStatusResponse(status=job.status, progress=job.progress, error=job.error)


@app.get("/report", response_model=ReportResponse)
def report(report_id: str = Query(...), db: Session = Depends(get_db)):
    job = get_job(db, report_id)
    if not job:
        raise HTTPException(status_code=404, detail="report_id를 찾을 수 없습니다.")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="리포트가 아직 완료되지 않았습니다.")
    if not job.sections_json:
        raise HTTPException(status_code=500, detail="리포트 섹션이 비어 있습니다.")

    sections_raw = json.loads(job.sections_json)
    sections = [ReportSection(title=s["title"], content_markdown=s["content_markdown"]) for s in sections_raw]
    meta = ReportMeta(
        riot_id=job.riot_id,
        region=job.region,
        games_analyzed=job.games_analyzed,
        created_at=(job.created_at or datetime.now(timezone.utc)).isoformat(),
        tone=job.tone,
        language=job.language,
    )
    return ReportResponse(status="done", report_id=job.report_id, sections=sections, meta=meta)
