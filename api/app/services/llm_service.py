from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.services.llm_input_transformer import build_llm_input_summary


DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 6000
MIN_MAX_OUTPUT_TOKENS = 3000
MAX_MAX_OUTPUT_TOKENS = 7000
DEFAULT_TEMPERATURE = 0.9

FALLBACK_MARKER = "LLM structured output failed; fallback narrative used."
RAW_PREVIEW_CHARS = 400
PROMPT_PREVIEW_CHARS = 700
MIN_SECTION_CHARS = 120
TARGET_SECTION_CHARS = 500

logger = logging.getLogger("lol-report-llm")

SECTION_TITLES = [
    "한 줄 요약",
    "최근 흐름 분석",
    "플레이 스타일 정밀 해부",
    "챔피언/포지션 성향 분석",
    "승률 개선 전략 로드맵",
    "다음 3판 구체 미션",
]

FORBIDDEN_PHRASES = (
    "tone:",
    "evidence one",
)

FORBIDDEN_OPENING_PREFIXES = (
    "최근 20판",
)

RAW_STAT_KEYS = (
    "data_source",
    "matches_fetched",
    "games_analyzed",
    "win_rate",
    "avg_kda",
    "deaths_per_game",
    "avg_cs_per_min",
    "avg_game_duration_min",
    "main_role",
    "role_consistency",
    "champion_pool_size",
    "most_played_champ",
    "early_impact_proxy",
    "vision_proxy",
    "objective_proxy",
    "streak_state",
)

LLM_INPUT_KEYS = (
    "sample_size",
    "recent_result",
    "avg_kda",
    "avg_deaths",
    "cs_per_min",
    "main_position",
    "main_champion",
    "champion_pool_size",
    "early_game_presence",
    "avg_game_duration",
)

METAPHOR_CUES = (
    "마치",
    "처럼",
    "롤러코스터",
    "파도",
    "항해",
    "엔진",
    "지도",
    "불꽃",
    "스프링",
    "드라마",
)

SECTION_PURPOSES = (
    "Section 01: Summarize the one-line core diagnosis and game rhythm.",
    "Section 02: Analyze recent flow by time windows and momentum changes.",
    "Section 03: Break down engage timing, risk control, and decision speed.",
    "Section 04: Analyze champion/position patterns and suggest pick strategy.",
    "Section 05: Provide three prioritized actions to improve win rate.",
    "Section 06: Provide a concrete checklist mission for the next 3 games.",
)

SECTION_STYLE_MODES = (
    "전술 분석",
    "차분한 코치",
    "스포츠 중계",
    "도발적/직설",
    "스토리텔링",
    "데이터 브리핑",
)

FALLBACK_OPENERS = (
    "이 리포트의 핵심은 경기 리듬을 먼저 바로잡는 데 있다.",
    "최근 로그를 시간대별로 보면 강점과 약점이 분명하게 갈린다.",
    "플레이 패턴을 해부해보면 진입 각도와 리스크 관리가 승패를 가른다.",
    "챔피언 선택은 좋지만 운영 디테일에서 손실이 누적된다.",
    "승률을 올리려면 화려함보다 반복 가능한 루틴이 먼저다.",
    "다음 3판은 개선 효과를 검증할 수 있는 가장 좋은 구간이다.",
)


class LlmGenerationError(Exception):
    pass


class LlmStructuredOutputError(Exception):
    pass


class _SectionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    content_markdown: str


class _MetaSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tone: str
    language: str


class _SectionsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[_SectionSchema] = Field(min_length=len(SECTION_TITLES), max_length=len(SECTION_TITLES))
    meta: _MetaSchema

    @model_validator(mode="after")
    def _validate_exact_titles(self) -> "_SectionsSchema":
        actual_titles = [s.title for s in self.sections]
        if actual_titles != SECTION_TITLES:
            raise ValueError(f"title sequence mismatch: expected={SECTION_TITLES}, got={actual_titles}")
        return self


def _resolve_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def _resolve_max_output_tokens() -> int:
    raw = os.getenv("OPENAI_MAX_OUTPUT_TOKENS") or os.getenv("OPENAI_MAX_TOKENS")
    try:
        parsed = int(raw) if raw else DEFAULT_MAX_OUTPUT_TOKENS
    except ValueError:
        parsed = DEFAULT_MAX_OUTPUT_TOKENS
    return max(MIN_MAX_OUTPUT_TOKENS, min(MAX_MAX_OUTPUT_TOKENS, parsed))


def _resolve_temperature() -> float:
    raw = os.getenv("OPENAI_TEMPERATURE")
    try:
        parsed = float(raw) if raw else DEFAULT_TEMPERATURE
    except ValueError:
        parsed = DEFAULT_TEMPERATURE
    return max(0.0, min(1.2, parsed))


def _sections_json_schema() -> dict[str, Any]:
    return {
        "name": "lol_report_sections",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "sections": {
                    "type": "array",
                    "minItems": len(SECTION_TITLES),
                    "maxItems": len(SECTION_TITLES),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {"type": "string", "enum": SECTION_TITLES},
                            "content_markdown": {"type": "string"},
                        },
                        "required": ["title", "content_markdown"],
                    },
                },
                "meta": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tone": {"type": "string"},
                        "language": {"type": "string"},
                    },
                    "required": ["tone", "language"],
                },
            },
            "required": ["sections", "meta"],
        },
        "strict": True,
    }


def _response_text_format() -> dict[str, Any]:
    schema = _sections_json_schema()
    return {
        "type": "json_schema",
        "name": schema["name"],
        "schema": schema["schema"],
        "strict": schema["strict"],
    }


def _response_text_config() -> dict[str, Any]:
    return {
        "format": _response_text_format(),
        "verbosity": "high",
    }


def _section_modes(llm_input_summary: dict[str, Any]) -> list[str]:
    seed_parts = [
        str(llm_input_summary.get("main_champion", "")),
        str(llm_input_summary.get("main_position", "")),
        str(llm_input_summary.get("sample_size", "")),
        str(llm_input_summary.get("avg_kda", "")),
        str(llm_input_summary.get("avg_deaths", "")),
        str(llm_input_summary.get("cs_per_min", "")),
    ]
    seed_value = sum(ord(ch) for ch in "|".join(seed_parts))
    start = seed_value % len(SECTION_STYLE_MODES)
    return [SECTION_STYLE_MODES[(start + i) % len(SECTION_STYLE_MODES)] for i in range(len(SECTION_TITLES))]


def _system_prompt(strict_json_retry: bool) -> str:
    retry_line = (
        "The previous output violated rules. Follow every rule strictly on this retry.\n"
        if strict_json_retry
        else ""
    )
    return (
        "You are a Korean esports performance analyst and writer.\n"
        "Write in Korean only.\n"
        "Output must be immersive but concrete, not generic templates.\n"
        "Never expose raw internal field names.\n"
        "Never reuse a fixed opener phrase.\n"
        "Every section must use a different opening sentence pattern.\n"
        "Every section must contain 2-3 paragraphs and exactly 3 actionable bullet points.\n"
        "Include concrete numbers only when they exist in the provided summary JSON.\n"
        "If a requested number is unavailable, explicitly say data is unavailable.\n"
        "Do not hallucinate numbers.\n"
        "Avoid two-decimal percentages.\n"
        "No profanity.\n"
        f"{retry_line}"
        "Output must be one strict JSON object only, and follow the provided schema exactly.\n"
    )


def _user_prompt(
    llm_input_summary: dict[str, Any],
    tone: str,
    language: str,
    section_modes: list[str],
) -> str:
    summary_text = json.dumps(llm_input_summary, ensure_ascii=False, indent=2)
    title_lines = "\n".join(f"{idx + 1}. {title}" for idx, title in enumerate(SECTION_TITLES))
    purpose_lines = "\n".join(
        f"{idx + 1}. {title} - purpose: {purpose} - style_mode: {section_modes[idx]}"
        for idx, (title, purpose) in enumerate(zip(SECTION_TITLES, SECTION_PURPOSES))
    )
    return (
        f"language={language}\n"
        f"tone={tone}\n"
        "Below is the only trustworthy summary JSON.\n"
        f"{summary_text}\n\n"
        "Keep the exact section titles and order below:\n"
        f"{title_lines}\n\n"
        "Section purposes and style mode assignments (no duplicates in one report):\n"
        f"{purpose_lines}\n\n"
        "Structure rules per section:\n"
        "- 2 to 3 paragraphs before bullets.\n"
        "- Exactly 3 actionable bullets using '-' markdown format.\n"
        "- Mention concrete stats numbers from summary JSON when available.\n"
        "- If a number is unavailable, write '데이터 없음' instead of inventing it.\n"
        f"- Recommended depth per section: >= {TARGET_SECTION_CHARS} characters.\n"
        "- Never reuse the same opener phrase across sections.\n"
        "- Do not leak internal key names from JSON.\n"
    )


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _short_preview(value: Any, max_len: int = RAW_PREVIEW_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text[:max_len]


def _log_prompt_excerpt(system_prompt: str, user_prompt: str, tone: str, language: str, strict_json_retry: bool) -> None:
    logger.info(
        "llm prompt tone=%s language=%s strict_json_retry=%s system_preview=%s user_preview=%s",
        tone,
        language,
        strict_json_retry,
        _short_preview(system_prompt, max_len=PROMPT_PREVIEW_CHARS),
        _short_preview(user_prompt, max_len=PROMPT_PREVIEW_CHARS),
    )


def _extract_structured_candidates(response: Any) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    output_parsed = _obj_get(response, "output_parsed")
    if isinstance(output_parsed, dict):
        candidates.append(("output_parsed", output_parsed))

    for item_idx, item in enumerate(_obj_get(response, "output", []) or []):
        for content_idx, content in enumerate(_obj_get(item, "content", []) or []):
            parsed = _obj_get(content, "parsed")
            if isinstance(parsed, dict):
                candidates.append((f"output[{item_idx}].content[{content_idx}].parsed", parsed))
    return candidates


def _extract_text_candidates(response: Any) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    output_text = (_obj_get(response, "output_text", "") or "").strip()
    if output_text:
        candidates.append(("output_text", output_text))

    for item_idx, item in enumerate(_obj_get(response, "output", []) or []):
        for content_idx, content in enumerate(_obj_get(item, "content", []) or []):
            text = (_obj_get(content, "text", "") or "").strip()
            if text:
                candidates.append((f"output[{item_idx}].content[{content_idx}].text", text))
    return candidates


def _validate_parsed_sections(parsed: dict[str, Any]) -> list[dict[str, str]]:
    try:
        validated = _SectionsSchema.model_validate(parsed)
    except ValidationError as exc:
        errors = exc.errors(include_url=False)
        first_error = errors[0] if errors else str(exc)
        raise LlmStructuredOutputError(f"schema validation failed: {first_error}") from exc

    return [{"title": s.title, "content_markdown": s.content_markdown} for s in validated.sections]


def _contains_leaked_key(text: str, key: str) -> bool:
    return re.search(rf"\b{re.escape(key.lower())}\b", text.lower()) is not None


def _normalize_content_for_dup(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _opening_sentence(text: str) -> str:
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not first_line:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)
    return parts[0].strip()[:120]


def _find_duplicate_section_indexes(sections: list[dict[str, str]]) -> list[list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, section in enumerate(sections):
        normalized = _normalize_content_for_dup(section.get("content_markdown", ""))
        grouped.setdefault(normalized, []).append(index)
    return [indexes for indexes in grouped.values() if len(indexes) > 1]


def _paragraph_count(content: str) -> int:
    paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", content) if chunk.strip()]
    return len(paragraphs)


def _actionable_bullet_count(content: str) -> int:
    return len(re.findall(r"(?m)^\s*[-*]\s+\S+", content))


def _validate_style_requirements(
    sections: list[dict[str, str]],
    tone: str,
    enforce_tone_cue: bool = True,
) -> None:
    del tone  # tone still exists in signature for compatibility with existing callers/tests

    opening_sentences: list[str] = []

    for section in sections:
        title = section.get("title", "")
        content = section.get("content_markdown", "").strip()
        if len(content) < MIN_SECTION_CHARS:
            logger.warning(
                "section below recommended length threshold title=%s chars=%s min=%s",
                title,
                len(content),
                MIN_SECTION_CHARS,
            )

        lower_content = content.lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase.lower() in lower_content:
                raise LlmStructuredOutputError(f"forbidden phrase found: {phrase}")

        for key in (*RAW_STAT_KEYS, *LLM_INPUT_KEYS):
            if _contains_leaked_key(content, key):
                raise LlmStructuredOutputError(f"internal key leaked in output: {key}")

        if re.search(r"\b\d+\.\d{2}%", content):
            raise LlmStructuredOutputError("percentage with two decimals is not allowed")

        if enforce_tone_cue and not any(cue in content for cue in METAPHOR_CUES):
            raise LlmStructuredOutputError(f"metaphor cue missing in section: {title}")

        opening = _opening_sentence(content)
        if not opening:
            raise LlmStructuredOutputError(f"opening sentence missing: {title}")
        for prefix in FORBIDDEN_OPENING_PREFIXES:
            if opening.startswith(prefix):
                raise LlmStructuredOutputError(f"forbidden opening prefix found: {prefix}")

        if _paragraph_count(content) < 2:
            logger.warning("section paragraph count too low title=%s", title)

        if _actionable_bullet_count(content) < 3:
            logger.warning("section actionable bullet count too low title=%s", title)

        if not re.search(r"\d", content):
            logger.warning("numeric evidence missing in section title=%s", title)

        opening_sentences.append(opening.lower())

    if len(set(opening_sentences)) != len(opening_sentences):
        logger.warning("duplicate section opener detected")


def _parse_sections_from_response(response: Any, tone: str) -> list[dict[str, str]]:
    for source, parsed in _extract_structured_candidates(response):
        logger.info("llm structured candidate source=%s preview=%s", source, _short_preview(parsed))
        try:
            sections = _validate_parsed_sections(parsed)
            _validate_style_requirements(sections, tone=tone, enforce_tone_cue=False)
            return sections
        except LlmStructuredOutputError as exc:
            logger.error("structured candidate rejected source=%s error=%s", source, exc)

    errors: list[str] = []
    for source, raw_text in _extract_text_candidates(response):
        logger.info("llm text candidate source=%s preview=%s", source, _short_preview(raw_text))
        try:
            parsed = json.loads(raw_text)
            sections = _validate_parsed_sections(parsed)
            _validate_style_requirements(sections, tone=tone, enforce_tone_cue=False)
            return sections
        except Exception as exc:
            errors.append(f"{source}: {exc}")

    if errors:
        raise LlmStructuredOutputError("; ".join(errors))
    raise LlmStructuredOutputError("no parseable content found in response")


def _create_sections_response(
    client: OpenAI,
    model: str,
    llm_input_summary: dict[str, Any],
    tone: str,
    language: str,
    max_output_tokens: int,
    temperature: float,
    strict_json_retry: bool,
) -> Any:
    system_prompt = _system_prompt(strict_json_retry=strict_json_retry)
    user_prompt = _user_prompt(
        llm_input_summary=llm_input_summary,
        tone=tone,
        language=language,
        section_modes=_section_modes(llm_input_summary),
    )
    _log_prompt_excerpt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tone=tone,
        language=language,
        strict_json_retry=strict_json_retry,
    )

    request_payload: dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
        "text": _response_text_config(),
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }

    try:
        return client.responses.create(**request_payload)
    except Exception as exc:
        # Some model families may reject temperature. Retry once without it.
        if "temperature" in str(exc).lower():
            logger.warning("temperature parameter rejected by model; retrying without temperature")
            request_payload.pop("temperature", None)
            return client.responses.create(**request_payload)
        raise


def _single_section_response_text_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "lol_report_single_section",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"content_markdown": {"type": "string"}},
            "required": ["content_markdown"],
        },
        "strict": True,
    }


def _single_section_response_text_config() -> dict[str, Any]:
    return {
        "format": _single_section_response_text_format(),
        "verbosity": "high",
    }


def _single_section_system_prompt() -> str:
    return (
        "You are a Korean esports performance analyst and writer.\n"
        "Write in Korean only.\n"
        "Return one strict JSON object with key content_markdown.\n"
        "Write only body markdown for one section (no title header).\n"
        "Use 2-3 paragraphs, then exactly 3 actionable '-' bullets.\n"
        "Do not leak raw internal JSON key names.\n"
        "Use concrete numbers only when they exist in summary.\n"
        "If data is missing, say data is unavailable.\n"
    )


def _single_section_user_prompt(
    llm_input_summary: dict[str, Any],
    tone: str,
    language: str,
    section_index: int,
    title: str,
    duplicate_content: str,
    other_sections: list[dict[str, str]],
) -> str:
    section_mode = _section_modes(llm_input_summary)[section_index]
    purpose = SECTION_PURPOSES[section_index]
    summary_text = json.dumps(llm_input_summary, ensure_ascii=False, indent=2)
    other_content_lines = "\n\n".join(
        f"- title: {section.get('title', '')}\n  content_preview: {_short_preview(section.get('content_markdown', ''), max_len=300)}"
        for idx, section in enumerate(other_sections)
        if idx != section_index
    )
    return (
        f"language={language}\n"
        f"tone={tone}\n"
        f"target_title={title}\n"
        f"target_style_mode={section_mode}\n"
        f"target_purpose={purpose}\n\n"
        "Summary JSON:\n"
        f"{summary_text}\n\n"
        "This section content was duplicated and must be rewritten uniquely:\n"
        f"{duplicate_content}\n\n"
        "Other section previews (must stay distinct from them):\n"
        f"{other_content_lines}\n\n"
        "Return only JSON object with content_markdown."
    )


def _create_single_section_response(
    client: OpenAI,
    model: str,
    llm_input_summary: dict[str, Any],
    tone: str,
    language: str,
    section_index: int,
    title: str,
    duplicate_content: str,
    other_sections: list[dict[str, str]],
    max_output_tokens: int,
    temperature: float,
) -> Any:
    request_payload: dict[str, Any] = {
        "model": model,
        "instructions": _single_section_system_prompt(),
        "input": _single_section_user_prompt(
            llm_input_summary=llm_input_summary,
            tone=tone,
            language=language,
            section_index=section_index,
            title=title,
            duplicate_content=duplicate_content,
            other_sections=other_sections,
        ),
        "text": _single_section_response_text_config(),
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }
    try:
        return client.responses.create(**request_payload)
    except Exception as exc:
        if "temperature" in str(exc).lower():
            logger.warning("single section temperature rejected; retrying without temperature")
            request_payload.pop("temperature", None)
            return client.responses.create(**request_payload)
        raise


def _parse_single_section_response(response: Any) -> str:
    for source, parsed in _extract_structured_candidates(response):
        content = parsed.get("content_markdown") if isinstance(parsed, dict) else None
        if isinstance(content, str) and content.strip():
            logger.info("single section structured candidate source=%s", source)
            return content.strip()

    errors: list[str] = []
    for source, raw_text in _extract_text_candidates(response):
        try:
            parsed = json.loads(raw_text)
            content = parsed.get("content_markdown") if isinstance(parsed, dict) else None
            if isinstance(content, str) and content.strip():
                logger.info("single section text candidate source=%s", source)
                return content.strip()
            errors.append(f"{source}: missing content_markdown")
        except Exception as exc:
            errors.append(f"{source}: {exc}")

    raise LlmStructuredOutputError("; ".join(errors) if errors else "no parseable single section content found")


def _regenerate_single_duplicate_section(
    client: OpenAI,
    model: str,
    llm_input_summary: dict[str, Any],
    tone: str,
    language: str,
    sections: list[dict[str, str]],
    section_index: int,
    max_output_tokens: int,
    temperature: float,
) -> str | None:
    section = sections[section_index]
    title = section.get("title") or SECTION_TITLES[section_index]
    content = section.get("content_markdown", "")
    try:
        response = _create_single_section_response(
            client=client,
            model=model,
            llm_input_summary=llm_input_summary,
            tone=tone,
            language=language,
            section_index=section_index,
            title=title,
            duplicate_content=content,
            other_sections=sections,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        regenerated_content = _parse_single_section_response(response)
        return regenerated_content
    except Exception as exc:
        logger.warning("duplicate section regeneration failed index=%s title=%s error=%s", section_index, title, exc)
        return None


def _fallback_section(title: str, llm_input_summary: dict[str, Any], tone: str, section_index: int) -> dict[str, str]:
    sample_size = int(llm_input_summary.get("sample_size", 0))
    main_position = llm_input_summary.get("main_position", "UNKNOWN")
    main_champion = llm_input_summary.get("main_champion", "Unknown")
    avg_deaths = float(llm_input_summary.get("avg_deaths", 0.0) or 0.0)
    cs_per_min = float(llm_input_summary.get("cs_per_min", 0.0) or 0.0)
    avg_duration = float(llm_input_summary.get("avg_game_duration", 0.0) or 0.0)

    tone_line = {
        "funny": "분위기는 좋지만 리듬이 종종 끊기는 장면이 보여서, 한 템포 빠른 판단이 필요하다.",
        "roast": "좋은 장면을 만들 실력은 충분한데, 고점과 저점 간격이 커서 손해가 누적되고 있다.",
        "sweet": "기본기는 단단하고 방향도 맞지만, 몇 가지 고정 루틴만 더하면 체감 성장이 빠를 흐름이다.",
    }.get(tone, "핵심 지표는 선명하고, 실행 루틴만 정교해지면 승률 개선 여지가 크다.")

    section_focus = {
        SECTION_TITLES[0]: "이 섹션은 전체 리포트의 핵심 진단을 한 줄로 고정하고 우선순위를 정한다.",
        SECTION_TITLES[1]: "이 섹션은 초중후반 흐름을 시간축으로 분리해서 흔들린 구간을 찾는다.",
        SECTION_TITLES[2]: "이 섹션은 교전 진입, 위험관리, 이탈 타이밍을 플레이 성향 관점에서 해부한다.",
        SECTION_TITLES[3]: "이 섹션은 챔피언 풀과 포지션 선택 패턴을 점검해 선택 전략을 제시한다.",
        SECTION_TITLES[4]: "이 섹션은 다음 경기에서 바로 적용 가능한 승률 개선 행동을 우선순위로 제시한다.",
        SECTION_TITLES[5]: "이 섹션은 다음 3판 미션을 체크리스트로 고정해 실행력을 확보한다.",
    }.get(title, "이 섹션은 현재 경기력에서 먼저 바꿔야 할 행동 규칙을 구체적으로 제시한다.")

    opener_seed = sum(ord(ch) for ch in f"{main_champion}|{main_position}|{sample_size}|{cs_per_min:.2f}")
    opener = FALLBACK_OPENERS[(opener_seed + section_index) % len(FALLBACK_OPENERS)]
    section_mode = _section_modes(llm_input_summary)[section_index]
    paragraph_1 = f"[{section_mode}] {opener} {tone_line} {section_focus}"
    paragraph_2 = (
        f"최근 표본은 {sample_size}판이고, 평균 데스는 {avg_deaths:.1f}, 분당 CS는 {cs_per_min:.1f}, "
        f"평균 경기 시간은 {avg_duration:.1f}분이다. 주 포지션은 {main_position}, 대표 챔피언은 {main_champion}으로 "
        "확인되며, 이 숫자는 라인 운영과 교전 합류 타이밍을 재설계할 근거가 된다."
    )
    bullets = [
        "- 실행 1: 8분 이전 라인 상태와 시야 체크를 고정 루틴으로 만들어 초반 손해를 줄인다.",
        "- 실행 2: 교전 진입 전 2초 점검(아군 위치, 상대 핵심 스킬, 퇴각 경로)을 매번 수행한다.",
        "- 실행 3: 오브젝트 45초 전 라인 정리와 합류 콜을 먼저 맞춰 후반 변수 관리를 안정화한다.",
    ]
    paragraph_3 = "핵심은 새로운 기술보다 반복 가능한 선택 규칙을 만드는 것이다. 이 규칙이 고정되면 승률은 자연스럽게 따라온다."

    content = "\n\n".join([paragraph_1, paragraph_2, "\n".join(bullets), paragraph_3])
    if len(content) < MIN_SECTION_CHARS:
        content += "\n\n데이터 없음 항목은 추측하지 않고, 다음 게임에서 추가 수집 후 재평가한다."
    return {"title": title, "content_markdown": content}


def _dummy_sections(summary_json: dict[str, Any], tone: str, language: str) -> list[dict[str, str]]:
    del language
    llm_input_summary = build_llm_input_summary(summary_json)
    sections = [
        _fallback_section(title, llm_input_summary, tone, section_index=index)
        for index, title in enumerate(SECTION_TITLES)
    ]
    return sections


def _resolve_duplicate_sections_once(
    sections: list[dict[str, str]],
    regenerate_section: Callable[[int, list[dict[str, str]]], str | None] | None = None,
) -> tuple[list[dict[str, str]], str | None]:
    normalized_sections = [dict(section) for section in sections]
    duplicate_groups = _find_duplicate_section_indexes(normalized_sections)
    if not duplicate_groups:
        return normalized_sections, None

    regenerated_indexes: list[int] = []
    for indexes in duplicate_groups:
        for duplicate_index in indexes[1:]:
            if regenerate_section is None:
                continue
            regenerated = regenerate_section(duplicate_index, normalized_sections)
            if regenerated:
                normalized_sections[duplicate_index]["content_markdown"] = regenerated
                regenerated_indexes.append(duplicate_index)

    remaining_groups = _find_duplicate_section_indexes(normalized_sections)
    if not remaining_groups:
        if regenerated_indexes:
            logger.warning("duplicate sections detected and regenerated once indexes=%s", regenerated_indexes)
        return normalized_sections, None

    warning = "duplicate section content remained after one regeneration pass"
    logger.warning("%s groups=%s regenerated_indexes=%s", warning, remaining_groups, regenerated_indexes)
    return normalized_sections, warning


def generate_sections(
    summary_json: dict[str, Any],
    tone: str,
    language: str,
    openai_api_key: str,
) -> tuple[list[dict[str, str]], str | None]:
    llm_input_summary = build_llm_input_summary(summary_json)
    logger.info("llm input summary=%s", _short_preview(llm_input_summary))

    if not openai_api_key:
        sections = _dummy_sections(summary_json, tone, language)
        _validate_style_requirements(sections, tone=tone, enforce_tone_cue=False)
        sections, generation_warning = _resolve_duplicate_sections_once(sections)
        logger.info("section distinctness check passed source=fallback_no_key count=%s", len(sections))
        return sections, generation_warning

    client = OpenAI(api_key=openai_api_key)
    model = _resolve_model()
    tokens = _resolve_max_output_tokens()
    temperature = _resolve_temperature()
    strict_json_retry = False
    last_error = ""

    try:
        for attempt in range(1, 4):
            logger.info(
                "llm request attempt=%s model=%s max_output_tokens=%s temperature=%s strict_json_retry=%s",
                attempt,
                model,
                tokens,
                temperature,
                strict_json_retry,
            )
            response = _create_sections_response(
                client=client,
                model=model,
                llm_input_summary=llm_input_summary,
                tone=tone,
                language=language,
                max_output_tokens=tokens,
                temperature=temperature,
                strict_json_retry=strict_json_retry,
            )

            try:
                sections = _parse_sections_from_response(response, tone=tone)
                sections, generation_warning = _resolve_duplicate_sections_once(
                    sections,
                    regenerate_section=lambda duplicate_index, current_sections: _regenerate_single_duplicate_section(
                        client=client,
                        model=model,
                        llm_input_summary=llm_input_summary,
                        tone=tone,
                        language=language,
                        sections=current_sections,
                        section_index=duplicate_index,
                        max_output_tokens=tokens,
                        temperature=temperature,
                    ),
                )
                logger.info("section distinctness check passed source=llm count=%s", len(sections))
                return sections, generation_warning
            except LlmStructuredOutputError as exc:
                last_error = str(exc)
                logger.error("llm output validation failed attempt=%s error=%s", attempt, exc)
                strict_json_retry = True
                continue

        logger.error("llm fallback after retries; last_error=%s", last_error)
        sections = _dummy_sections(summary_json, tone, language)
        _validate_style_requirements(sections, tone=tone, enforce_tone_cue=False)
        sections, generation_warning = _resolve_duplicate_sections_once(sections)
        logger.info("section distinctness check passed source=fallback_retry count=%s", len(sections))
        return sections, generation_warning
    except Exception as exc:
        logger.exception("llm report generation failed")
        raise LlmGenerationError(f"LLM report generation failed: {exc}") from exc


