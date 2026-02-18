import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import openai
from sqlalchemy.orm import Session

load_dotenv()

from app.config import settings
from app.database import Base, SessionLocal, engine, get_db
from app.repository import (
    create_job,
    get_job,
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
from app.services.riot_service import (
    RiotIdParseError,
    RiotUpstreamError,
    RiotUserInputError,
    fetch_riot_summary,
    normalize_region,
)


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
    logger.info("openai sdk version=%s", getattr(openai, "__version__", "unknown"))
    logger.info("database initialized")


def _failed_summary(riot_error: str | None) -> dict:
    return {
        "data_source": "failed",
        "riot_error": riot_error,
        "matches_fetched": 0,
        "puuid": None,
        "games_analyzed": 0,
        "win_rate": 0.0,
        "main_role": "UNKNOWN",
        "role_consistency": 0.0,
        "champion_pool_size": 0,
        "most_played_champ": "Unknown",
        "avg_kda": "0.0/0.0/0.0",
        "deaths_per_game": 0.0,
        "avg_cs_per_min": 0.0,
        "avg_game_duration_min": 0.0,
        "streak_state": {"type": "NONE", "length": 0},
        "early_impact_proxy": 0.0,
        "vision_proxy": 0.0,
        "objective_proxy": 0.0,
    }


def _run_report_pipeline(report_id: str) -> None:
    db = SessionLocal()
    try:
        job = get_job(db, report_id)
        if not job:
            return

        logger.info("pipeline start report_id=%s riot_id=%s region=%s", report_id, job.riot_id, job.region)
        update_status(db, report_id, status="processing", progress=15)
        summary: dict

        try:
            update_status(db, report_id, status="processing", progress=35)
            if not settings.riot_api_key:
                logger.error("riot fetch unavailable report_id=%s reason=RIOT_API_KEY missing", report_id)
                update_status(db, report_id, status="failed", progress=100, error="RIOT_API_KEY is missing.")
                return
            else:
                try:
                    summary = asyncio.run(fetch_riot_summary(job.riot_id, job.region, settings.riot_api_key))
                except RiotIdParseError as exc:
                    logger.exception("riot id parse error report_id=%s error=%s", report_id, exc)
                    update_status(db, report_id, status="failed", progress=100, error=str(exc))
                    return
                except RiotUserInputError as exc:
                    logger.exception("riot user input error report_id=%s error=%s", report_id, exc)
                    update_status(db, report_id, status="failed", progress=100, error=str(exc))
                    return
                except RiotUpstreamError as exc:
                    logger.exception("riot upstream error report_id=%s error=%s", report_id, exc)
                    update_status(db, report_id, status="failed", progress=100, error=f"Riot API fetch failed: {exc}")
                    return

            update_status(db, report_id, status="processing", progress=70)
            sections = generate_sections(
                summary,
                tone=job.tone,
                language=job.language,
                openai_api_key=settings.openai_api_key,
            )

            update_status(db, report_id, status="processing", progress=90)
            store_report(
                db,
                report_id=report_id,
                sections=sections,
                summary_json=summary,
                games_analyzed=int(summary.get("games_analyzed", 0)),
            )
            logger.info(
                "pipeline done report_id=%s data_source=%s matches_fetched=%s",
                report_id,
                summary.get("data_source"),
                summary.get("matches_fetched"),
            )
        except LlmGenerationError as exc:
            logger.error("llm error report_id=%s error=%s", report_id, exc)
            update_status(db, report_id, status="failed", progress=100, error=f"{exc}. Please retry.")
        except Exception as exc:
            logger.exception("unexpected error report_id=%s", report_id)
            update_status(db, report_id, status="failed", progress=100, error=f"Unexpected error: {exc}")
    finally:
        db.close()


@app.post("/create-report", response_model=CreateReportResponse)
def create_report(payload: CreateReportRequest, db: Session = Depends(get_db)):
    try:
        canonical_region = normalize_region(payload.region)
    except RiotUserInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    report_id = str(uuid.uuid4())

    create_job(
        db=db,
        report_id=report_id,
        riot_id=payload.riot_id,
        region=canonical_region,
        tone=payload.tone,
        language=payload.language,
    )
    threading.Thread(target=_run_report_pipeline, args=(report_id,), daemon=True).start()
    return CreateReportResponse(report_id=report_id, status="queued")


@app.get("/report-status", response_model=ReportStatusResponse)
def report_status(report_id: str = Query(...), db: Session = Depends(get_db)):
    job = get_job(db, report_id)
    if not job:
        raise HTTPException(status_code=404, detail="report_id not found")
    return ReportStatusResponse(status=job.status, progress=job.progress, error=job.error)


@app.get("/report", response_model=ReportResponse)
def report(report_id: str = Query(...), db: Session = Depends(get_db)):
    job = get_job(db, report_id)
    if not job:
        raise HTTPException(status_code=404, detail="report_id not found")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="report is not completed")
    if not job.sections_json:
        raise HTTPException(status_code=500, detail="report sections are empty")

    summary = json.loads(job.summary_json) if job.summary_json else {}
    sections_raw = json.loads(job.sections_json)
    sections = [ReportSection(title=s["title"], content_markdown=s["content_markdown"]) for s in sections_raw]
    meta = ReportMeta(
        riot_id=job.riot_id,
        region=job.region,
        games_analyzed=job.games_analyzed,
        created_at=(job.created_at or datetime.now(timezone.utc)).isoformat(),
        tone=job.tone,
        language=job.language,
        data_source=summary.get("data_source", "failed"),
        riot_error=summary.get("riot_error"),
        matches_fetched=int(summary.get("matches_fetched", 0)),
        puuid=summary.get("puuid"),
    )
    return ReportResponse(status="done", report_id=job.report_id, sections=sections, meta=meta)
