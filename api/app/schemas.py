from typing import Literal

from pydantic import BaseModel, Field


class CreateReportRequest(BaseModel):
    riot_id: str = Field(min_length=1, max_length=120)
    region: str = Field(min_length=2, max_length=8)
    tone: Literal["funny", "roast", "sweet"] = "funny"
    language: str = Field(default="ko", min_length=2, max_length=8)


class CreateReportResponse(BaseModel):
    report_id: str
    status: Literal["queued"]


class ReportStatusResponse(BaseModel):
    status: Literal["queued", "processing", "done", "failed"]
    progress: int = Field(ge=0, le=100)
    error: str | None = None


class ReportSection(BaseModel):
    title: str
    content_markdown: str


class ReportMeta(BaseModel):
    riot_id: str
    region: str
    games_analyzed: int
    created_at: str
    tone: str
    language: str
    data_source: Literal["riot", "fallback", "failed"]
    riot_error: str | None = None
    matches_fetched: int = 0
    puuid: str | None = None


class ReportResponse(BaseModel):
    status: Literal["done"]
    report_id: str
    sections: list[ReportSection]
    meta: ReportMeta
    generation_warning: str | None = None
