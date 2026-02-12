import json
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.models import ReportJob


def make_cache_key(riot_id: str, region: str, tone: str, language: str) -> str:
    return f"{riot_id.strip().lower()}::{region.strip().upper()}::{tone}::{language}"


def create_job(
    db: Session,
    report_id: str,
    riot_id: str,
    region: str,
    tone: str,
    language: str,
    cache_key: str,
) -> ReportJob:
    job = ReportJob(
        report_id=report_id,
        riot_id=riot_id,
        region=region,
        tone=tone,
        language=language,
        status="queued",
        progress=0,
        cache_key=cache_key,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def find_done_cache(db: Session, cache_key: str) -> ReportJob | None:
    stmt = select(ReportJob).where(ReportJob.cache_key == cache_key, ReportJob.status == "done").order_by(desc(ReportJob.created_at))
    return db.execute(stmt).scalars().first()


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


def clone_done_into_new_job(db: Session, source_job: ReportJob, new_report_id: str, cache_key: str) -> ReportJob:
    clone = ReportJob(
        report_id=new_report_id,
        riot_id=source_job.riot_id,
        region=source_job.region,
        tone=source_job.tone,
        language=source_job.language,
        status="done",
        progress=100,
        error=None,
        sections_json=source_job.sections_json,
        summary_json=source_job.summary_json,
        games_analyzed=source_job.games_analyzed,
        cache_key=cache_key,
    )
    db.add(clone)
    db.commit()
    db.refresh(clone)
    return clone
