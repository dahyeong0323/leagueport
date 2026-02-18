import argparse
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, ValidationError


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app  # noqa: E402
from app.services.llm_service import (  # noqa: E402
    FALLBACK_MARKER,
    SECTION_TITLES,
    _validate_style_requirements,
)


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


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sanitized = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(sanitized)


def _validate_sections_payload(sections: object, tone: str) -> tuple[int, str, _SectionsSchema]:
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

    return len(parsed.sections), parsed.sections[0].title, parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="In-process smoke test for LoL report API.")
    parser.add_argument("--riot-id", required=True)
    parser.add_argument("--region", default="KR")
    parser.add_argument("--tone", default="funny", choices=["funny", "roast", "sweet"])
    parser.add_argument("--language", default="ko")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--excerpt-lines", type=int, default=5)
    args = parser.parse_args()

    payload = {
        "riot_id": args.riot_id,
        "region": args.region,
        "tone": args.tone,
        "language": args.language,
    }

    with TestClient(app) as client:
        create_res = client.post("/create-report", json=payload)
        if create_res.status_code != 200:
            return _fail(f"create-report failed: status={create_res.status_code}, body={create_res.text}")

        create_body = create_res.json()
        report_id = create_body.get("report_id")
        if not report_id:
            return _fail(f"create-report missing report_id: {create_body}")
        print(f"[INFO] report_id={report_id}")

        deadline = time.time() + args.timeout_sec
        while time.time() < deadline:
            status_res = client.get("/report-status", params={"report_id": report_id})
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

        report_res = client.get("/report", params={"report_id": report_id})
        if report_res.status_code != 200:
            return _fail(f"report failed: status={report_res.status_code}, body={report_res.text}")

        report_body = report_res.json()
        if report_body.get("status") != "done":
            return _fail(f"unexpected report payload status: {report_body}")

        sections = report_body.get("sections")
        if not isinstance(sections, list) or not sections:
            return _fail(f"sections missing/empty: {report_body}")

        try:
            count, first_title, parsed = _validate_sections_payload(sections, tone=args.tone)
        except ValueError as exc:
            return _fail(str(exc))

        first_content = parsed.sections[0].content_markdown
        excerpt = "\n".join(first_content.splitlines()[: max(1, args.excerpt_lines)])
        print(f"[PASS] report ready. sections={count} first_title={first_title}")
        print("[EXCERPT]")
        _safe_print(excerpt)
        return 0


if __name__ == "__main__":
    sys.exit(main())
