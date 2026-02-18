import unittest

from app.services.llm_service import (
    LlmStructuredOutputError,
    SECTION_TITLES,
    _validate_style_requirements,
)


def _long_valid_body() -> str:
    base = (
        "마치 롤러코스터를 타는 듯한 흐름이지만, 조급함만 눌러도 경기의 무게중심을 안정적으로 잡을 수 있다. "
        "초반에는 라인 손실을 줄이고, 중반에는 시야 주도권을 먼저 확보하고, 후반에는 진입 타이밍을 팀과 맞추는 순서가 중요하다. "
        "이런 루틴이 쌓이면 한타의 체감 난도가 확실히 내려간다. "
        "첫 제안은 교전 전 2초 체크 습관을 만들어 불필요한 데스를 줄이는 것이다. "
        "두 번째 제안은 오브젝트 45초 전 라인 정리를 먼저 실행해 합류 속도를 높이는 것이다. "
        "세 번째 제안은 챔피언 폭을 무리하게 넓히지 말고, 익숙한 군에서 판단 품질을 높이는 것이다. "
        "이 과정을 지키면 승부의 기복이 줄고, 마지막 순간의 선택이 더 선명해진다."
    )
    return base + " 추가 문장으로 분량을 보강해도 핵심은 같다. 판단의 일관성이 승률을 만든다."


def _make_sections(content: str, *, make_unique: bool) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    for index, title in enumerate(SECTION_TITLES):
        body = content
        if make_unique:
            body = f"섹션 {index + 1} 오프닝 문장이다. {content} 섹션별 구분 포인트 {index + 1}."
        sections.append({"title": title, "content_markdown": body})
    return sections


class TestLlmServiceStyle(unittest.TestCase):
    def test_accepts_long_metaphor_rich_section(self) -> None:
        _validate_style_requirements(_make_sections(_long_valid_body(), make_unique=True), tone="funny")

    def test_rejects_forbidden_template_phrase(self) -> None:
        bad = _long_valid_body() + " 톤: 이렇게 쓰면 안 된다."
        with self.assertRaises(LlmStructuredOutputError):
            _validate_style_requirements(_make_sections(bad, make_unique=True), tone="funny")

    def test_rejects_raw_stat_key_leak(self) -> None:
        bad = _long_valid_body() + " win_rate 라는 키를 노출하면 실패해야 한다."
        with self.assertRaises(LlmStructuredOutputError):
            _validate_style_requirements(_make_sections(bad, make_unique=True), tone="funny")

    def test_rejects_short_section(self) -> None:
        short = "마치 파도처럼 흔들린다. 제안은 있다."
        with self.assertRaises(LlmStructuredOutputError):
            _validate_style_requirements(_make_sections(short, make_unique=True), tone="funny")

    def test_rejects_duplicate_section_bodies(self) -> None:
        with self.assertRaises(LlmStructuredOutputError):
            _validate_style_requirements(_make_sections(_long_valid_body(), make_unique=False), tone="funny")


if __name__ == "__main__":
    unittest.main()
