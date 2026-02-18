import unittest

from app.services.llm_service import (
    SECTION_TITLES,
    LlmStructuredOutputError,
    _validate_parsed_sections,
)


class TestLlmServiceSchema(unittest.TestCase):
    def test_validate_parsed_sections_accepts_exact_title_sequence(self) -> None:
        payload = {
            "sections": [
                {"title": title, "content_markdown": f"{title} content"}
                for title in SECTION_TITLES
            ],
            "meta": {"tone": "funny", "language": "ko"},
        }
        sections = _validate_parsed_sections(payload)
        self.assertEqual(len(sections), len(SECTION_TITLES))
        self.assertEqual([s["title"] for s in sections], SECTION_TITLES)

    def test_validate_parsed_sections_rejects_title_sequence_mismatch(self) -> None:
        wrong_titles = SECTION_TITLES[1:] + [SECTION_TITLES[0]]
        payload = {
            "sections": [
                {"title": title, "content_markdown": f"{title} content"}
                for title in wrong_titles
            ],
            "meta": {"tone": "funny", "language": "ko"},
        }
        with self.assertRaises(LlmStructuredOutputError):
            _validate_parsed_sections(payload)

    def test_validate_parsed_sections_requires_meta(self) -> None:
        payload = {
            "sections": [
                {"title": title, "content_markdown": f"{title} content"}
                for title in SECTION_TITLES
            ]
        }
        with self.assertRaises(LlmStructuredOutputError):
            _validate_parsed_sections(payload)


if __name__ == "__main__":
    unittest.main()
