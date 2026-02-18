from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


KNOWN_TEXT = "테스트 한글 123 ABC"


def _normalize(text: str) -> str:
    return " ".join(text.split())


def generate_pdf(output_path: Path, font_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont("NotoSansKRPDF", str(font_path)))

    c = canvas.Canvas(str(output_path), pagesize=A4)
    c.setTitle("Korean PDF Regression")
    c.setFont("NotoSansKRPDF", 18)
    c.drawString(72, 780, KNOWN_TEXT)
    c.save()


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate a Korean UTF-8 PDF sample.")
    parser.add_argument(
        "--font-path",
        default=str(Path(__file__).resolve().parents[2] / "web" / "public" / "fonts" / "NotoSansKR-Variable.ttf"),
    )
    parser.add_argument("--output-path", default=str(Path(__file__).resolve().parents[1] / "tmp" / "korean-regression.pdf"))
    args = parser.parse_args()

    font_path = Path(args.font_path).resolve()
    output_path = Path(args.output_path).resolve()

    if not font_path.exists():
        print(f"[FAIL] font not found: {font_path}")
        return 1

    generate_pdf(output_path=output_path, font_path=font_path)
    extracted = extract_text(output_path)
    normalized = _normalize(extracted)

    has_exact = KNOWN_TEXT in normalized
    has_question = "?" in normalized

    print(f"[INFO] pdf_path={output_path}")
    print(f"[INFO] extracted_text={normalized}")
    print(f"[INFO] contains_known_text={has_exact}")
    print(f"[INFO] contains_question_mark={has_question}")

    if not has_exact:
        print("[FAIL] exact korean sample text was not extracted from PDF")
        return 1
    if has_question:
        print("[FAIL] question mark replacement detected in extracted PDF text")
        return 1

    print("[PASS] korean PDF text is preserved without mojibake")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
