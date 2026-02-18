from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from app.database import Base


class ReportJob(Base):
    __tablename__ = "report_jobs"

    report_id = Column(String(36), primary_key=True, index=True)
    riot_id = Column(String(120), nullable=False, index=True)
    region = Column(String(8), nullable=False)
    tone = Column(String(16), nullable=False, default="funny")
    language = Column(String(8), nullable=False, default="ko")
    status = Column(String(16), nullable=False, default="queued")
    progress = Column(Integer, nullable=False, default=0)
    error = Column(Text, nullable=True)
    sections_json = Column(Text, nullable=True)
    summary_json = Column(Text, nullable=True)
    games_analyzed = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
