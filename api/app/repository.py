import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ReportJob


def create_job(
    db: Session,
    report_id: str,
    riot_id: str,
    region: str,
    tone: str,
    language: str,
) -> ReportJob:
    job = ReportJob(
        report_id=report_id,
        riot_id=riot_id,
        region=region,
        tone=tone,
        language=language,
        status="queued",
        progress=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, report_id: str) -> ReportJob | None:
    stmt = select(ReportJob).where(ReportJob.report_id == report_id)
    return db.execute(stmt).scalars().first()


def update_status(db: Session, report_id: str, status: str, progress: int, error: str | None = None) -> None:
    job = get_job(db, report_id)
    if not job:
        return
    job.status = status
    job.progress = progress
    job.error = error
    db.add(job)
    db.commit()


def store_report(db: Session, report_id: str, sections: list[dict[str, Any]], summary_json: dict[str, Any], games_analyzed: int) -> None:
    job = get_job(db, report_id)
    if not job:
        return
    job.status = "done"
    job.progress = 100
    job.error = None
    job.sections_json = json.dumps(sections, ensure_ascii=False)
    job.summary_json = json.dumps(summary_json, ensure_ascii=False)
    job.games_analyzed = games_analyzed
    db.add(job)
    db.commit()
