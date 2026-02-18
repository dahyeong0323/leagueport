import argparse
import os
import sys
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.llm_service import FALLBACK_MARKER, SECTION_TITLES
from app.services.llm_service import _validate_style_requirements


class _SectionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    content_markdown: str


class _SectionsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sections: list[_SectionSchema] = Field(min_length=len(SECTION_TITLES), max_length=len(SECTION_TITLES))


def _fail(message: str) -> int:
    print(f"[FAIL] {message}")
    return 1


def _validate_sections_payload(sections: object, tone: str) -> tuple[int, str]:
    try:
        parsed = _SectionsSchema.model_validate({"sections": sections})
    except ValidationError as exc:
        raise ValueError(f"sections schema validation failed: {exc}") from exc

    if any(FALLBACK_MARKER in s.content_markdown for s in parsed.sections):
        raise ValueError("fallback marker text found in report sections")

    try:
        _validate_style_requirements(
            [{"title": s.title, "content_markdown": s.content_markdown} for s in parsed.sections],
            tone=tone,
        )
    except Exception as exc:
        raise ValueError(f"style validation failed: {exc}") from exc

    return len(parsed.sections), parsed.sections[0].title


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for LoL report API pipeline.")
    parser.add_argument("--api-base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--riot-id", required=True)
    parser.add_argument("--region", default="KR")
    parser.add_argument("--tone", default="funny", choices=["funny", "roast", "sweet"])
    parser.add_argument("--language", default="ko")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    args = parser.parse_args()

    payload = {
        "riot_id": args.riot_id,
        "region": args.region,
        "tone": args.tone,
        "language": args.language,
    }
    base = args.api_base_url.rstrip("/")

    print(f"[INFO] create-report POST {base}/create-report")
    with httpx.Client(timeout=20.0) as client:
        create_res = client.post(f"{base}/create-report", json=payload)
        if create_res.status_code != 200:
            return _fail(f"create-report failed: status={create_res.status_code}, body={create_res.text}")

        create_body = create_res.json()
        report_id = create_body.get("report_id")
        if not report_id:
            return _fail(f"create-report missing report_id: {create_body}")

        print(f"[INFO] report_id={report_id}")
        deadline = time.time() + args.timeout_sec

        while time.time() < deadline:
            status_res = client.get(f"{base}/report-status", params={"report_id": report_id})
            if status_res.status_code != 200:
                return _fail(f"report-status failed: status={status_res.status_code}, body={status_res.text}")

            status_body = status_res.json()
            status = status_body.get("status")
            progress = status_body.get("progress")
            error = status_body.get("error")
            print(f"[INFO] status={status} progress={progress} error={error}")

            if status == "done":
                break
            if status == "failed":
                return _fail(f"pipeline failed: {error}")
            time.sleep(args.poll_interval_sec)
        else:
            return _fail(f"timeout waiting for completion ({args.timeout_sec}s)")

        report_res = client.get(f"{base}/report", params={"report_id": report_id})
        if report_res.status_code != 200:
            return _fail(f"report fetch failed: status={report_res.status_code}, body={report_res.text}")

        report_body = report_res.json()
        sections = report_body.get("sections")
        if report_body.get("status") != "done":
            return _fail(f"unexpected report status payload: {report_body}")
        if not isinstance(sections, list) or not sections:
            return _fail(f"sections missing/empty: {report_body}")

        try:
            count, first_title = _validate_sections_payload(sections, tone=args.tone)
        except ValueError as exc:
            return _fail(str(exc))

        print(f"[PASS] report generated. sections={count} first_title={first_title}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
