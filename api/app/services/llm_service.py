from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.services.llm_input_transformer import build_llm_input_summary


DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 4200
MIN_MAX_OUTPUT_TOKENS = 3000
MAX_MAX_OUTPUT_TOKENS = 7000
DEFAULT_TEMPERATURE = 0.9

FALLBACK_MARKER = "LLM structured output failed; fallback narrative used."
RAW_PREVIEW_CHARS = 400
PROMPT_PREVIEW_CHARS = 700
MIN_SECTION_CHARS = 300
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
    "톤:",
    "증거 하나",
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
    "Section 01: 이번 리포트의 핵심 한 줄 진단과 경기 리듬 요약.",
    "Section 02: 최근 흐름에서 좋아진 점/흔들린 점을 시간대 중심으로 분석.",
    "Section 03: 교전 진입, 위험관리, 의사결정 속도를 플레이 성향 관점에서 해부.",
    "Section 04: 챔피언 풀/포지션 고정도와 숙련 패턴을 설명하고 선택 전략 제시.",
    "Section 05: 다음 경기에서 승률을 올릴 실행 전략 3가지를 우선순위로 제시.",
    "Section 06: 다음 3판에서 바로 점검 가능한 체크리스트형 미션 제시.",
)

FALLBACK_OPENERS = (
    "지금 이 리포트의 핵심은 흐름의 일관성을 되찾는 것이다.",
    "최근 경기 로그를 시간대별로 보면 강점과 손실 구간이 분명히 갈린다.",
    "플레이 템포를 뜯어보면 진입 타이밍과 리스크 관리가 승패를 크게 가른다.",
    "챔피언과 포지션 선택은 이미 장점이 있으나 운영 디테일에서 차이가 난다.",
    "승률을 올리려면 새로운 기술보다 반복 가능한 선택 규칙이 먼저 필요하다.",
    "다음 세 판은 연습이 아니라 개선 효과를 검증하는 실전 구간이다.",
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


def _system_prompt(strict_json_retry: bool) -> str:
    retry_line = (
        "이전 출력이 규칙을 위반했으니 이번에는 반드시 규칙을 지켜라.\n"
        "각 섹션의 첫 문장은 서로 달라야 하고 문장 패턴 재사용을 금지한다.\n"
        if strict_json_retry
        else ""
    )
    return (
        "You are a Korean esports storytelling analyst.\n"
        "You do not write statistical reports.\n"
        "You write long, immersive, witty coaching narratives.\n"
        "Never expose raw field names.\n"
        "Never use repetitive template labels like: '톤:', '증거 하나', '근거', '전략 1'.\n"
        "Do not start sections with the same opener sentence.\n"
        "Avoid two-decimal percentages.\n"
        "No profanity.\n"
        "Each section must contain at least one creative metaphor, 2~3 actionable suggestions, and a strong closing sentence.\n"
        "Write in Korean only.\n"
        "Do not be short.\n"
        f"{retry_line}"
        "Output must be one strict JSON object only, and follow the provided schema exactly.\n"
    )


def _user_prompt(llm_input_summary: dict[str, Any], tone: str, language: str) -> str:
    summary_text = json.dumps(llm_input_summary, ensure_ascii=False, indent=2)
    purpose_lines = "\n".join(
        f"{idx + 1}. {title} — {purpose}"
        for idx, (title, purpose) in enumerate(zip(SECTION_TITLES, SECTION_PURPOSES))
    )
    return (
        f"language={language}\n"
        f"tone={tone}\n"
        "아래는 내부 집계 요약 JSON이다.\n"
        f"{summary_text}\n\n"
        "다음 6개 섹션 제목을 정확히 이 순서로 유지하라:\n"
        "1. 한 줄 요약\n"
        "2. 최근 흐름 분석\n"
        "3. 플레이 스타일 정밀 해부\n"
        "4. 챔피언/포지션 성향 분석\n"
        "5. 승률 개선 전략 로드맵\n"
        "6. 다음 3판 구체 미션\n\n"
        "섹션별 목적(구조 차별화 필수):\n"
        f"{purpose_lines}\n\n"
        f"각 섹션은 한국어 기준 최소 {TARGET_SECTION_CHARS}자 이상으로 길게 써라.\n"
        "설명은 풍부하게 확장하고, 문장 패턴 반복을 피하라.\n"
        "각 섹션 첫 문장을 서로 다르게 시작하라.\n"
        "금지 시작 문구: '최근 20판'.\n"
        "내부 통계 키 이름을 본문에 노출하지 마라.\n"
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


def _validate_style_requirements(
    sections: list[dict[str, str]],
    tone: str,
    enforce_tone_cue: bool = True,
) -> None:
    del tone  # tone still exists in signature for compatibility with existing callers/tests

    normalized_contents: list[str] = []
    opening_sentences: list[str] = []

    for section in sections:
        title = section.get("title", "")
        content = section.get("content_markdown", "").strip()
        if len(content) < MIN_SECTION_CHARS:
            raise LlmStructuredOutputError(f"section too short (<{MIN_SECTION_CHARS} chars): {title}")

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

        normalized_contents.append(_normalize_content_for_dup(content))
        opening_sentences.append(opening.lower())

    if len(set(normalized_contents)) != len(normalized_contents):
        raise LlmStructuredOutputError("duplicate section content detected")
    if len(set(opening_sentences)) != len(opening_sentences):
        raise LlmStructuredOutputError("duplicate section opener detected")


def _parse_sections_from_response(response: Any, tone: str) -> list[dict[str, str]]:
    for source, parsed in _extract_structured_candidates(response):
        logger.info("llm structured candidate source=%s preview=%s", source, _short_preview(parsed))
        try:
            sections = _validate_parsed_sections(parsed)
            _validate_style_requirements(sections, tone=tone, enforce_tone_cue=True)
            return sections
        except LlmStructuredOutputError as exc:
            logger.error("structured candidate rejected source=%s error=%s", source, exc)

    errors: list[str] = []
    for source, raw_text in _extract_text_candidates(response):
        logger.info("llm text candidate source=%s preview=%s", source, _short_preview(raw_text))
        try:
            parsed = json.loads(raw_text)
            sections = _validate_parsed_sections(parsed)
            _validate_style_requirements(sections, tone=tone, enforce_tone_cue=True)
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
    user_prompt = _user_prompt(llm_input_summary=llm_input_summary, tone=tone, language=language)
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


def _fallback_section(title: str, llm_input_summary: dict[str, Any], tone: str, section_index: int) -> dict[str, str]:
    sample_size = int(llm_input_summary.get("sample_size", 0))
    main_position = llm_input_summary.get("main_position", "UNKNOWN")
    main_champion = llm_input_summary.get("main_champion", "Unknown")
    avg_deaths = llm_input_summary.get("avg_deaths", 0.0)
    cs_per_min = llm_input_summary.get("cs_per_min", 0.0)
    avg_duration = llm_input_summary.get("avg_game_duration", 0.0)

    tone_line = {
        "funny": "마치 예능 편집점처럼, 잘한 장면과 아쉬운 장면이 번갈아 튀어나오는데 그 흐름이 오히려 성장 포인트를 선명하게 보여준다.",
        "roast": "마치 브레이크가 늦은 레이싱카처럼, 속도는 충분한데 멈춰야 할 타이밍을 놓치면서 손해를 키우는 구간이 보인다.",
        "sweet": "마치 천천히 가열되는 엔진처럼, 기본 체급은 분명하고 작은 습관 수정만으로도 체감이 크게 올라갈 여지가 있다.",
    }.get(tone, "마치 긴 항해를 준비하는 선장처럼, 방향은 맞지만 세부 조정이 승률의 체감 차이를 만든다.")

    section_focus = {
        "한 줄 요약": "이번 판수의 핵심 흐름을 한 문장으로 압축하고 우선순위를 제시한다.",
        "최근 흐름 분석": "초중후반 시간대별로 손익이 갈리는 순간을 짚고 변곡점을 설명한다.",
        "플레이 스타일 정밀 해부": "교전 개시 타이밍과 위험관리 습관이 결과에 미치는 영향을 분해한다.",
        "챔피언/포지션 성향 분석": "챔피언 숙련도와 포지션 일관성의 장단점을 운영 관점으로 정리한다.",
        "승률 개선 전략 로드맵": "다음 경기에서 즉시 적용 가능한 행동 규칙을 우선순위로 제안한다.",
        "다음 3판 구체 미션": "3판 체크리스트를 통해 개선 효과를 검증하는 실행 계획을 만든다.",
    }.get(title, "현재 구간에서 가장 먼저 고쳐야 할 선택 기준을 제안한다.")

    opener = FALLBACK_OPENERS[section_index % len(FALLBACK_OPENERS)]

    content = (
        f"{opener} {tone_line}\n"
        f"{section_focus}\n"
        f"최근 표본은 {sample_size}판이고, 평균 데스는 약 {avg_deaths:.1f}회이며 분당 CS는 {cs_per_min:.1f} 수준이다. "
        f"주 포지션은 {main_position}, 대표 챔피언은 {main_champion}으로 보이며 평균 경기 시간은 {avg_duration:.1f}분이다. "
        "이 조합은 한타 집중력과 라인 관리의 균형이 맞으면 파급력이 커지고, 반대로 첫 실수가 길어지면 손실이 연쇄적으로 번지는 성격을 가진다. "
        "핵심은 숫자를 외우는 것이 아니라 장면의 질서를 잡는 것이다.\n"
        "- 제안 1: 첫 8분에는 교전 각보다 라인 상태와 시야 선점을 먼저 정리해, 불리한 싸움을 시작하지 않는 습관을 고정한다.\n"
        "- 제안 2: 킬각이 열려도 스킬 한 번 더 확인하고 진입해, 이득 교환을 만들고 손해 교환을 줄이는 판단 템포를 만든다.\n"
        "- 제안 3: 중반 이후에는 오브젝트 타이밍 45초 전에 라인을 정리해, 팀 합류 속도에서 먼저 이기는 흐름을 만든다.\n"
        "지금 필요한 건 새로운 기교가 아니라, 이미 가진 장점을 경기 내내 끊기지 않게 이어 붙이는 운영의 일관성이다."
    )

    if len(content) < MIN_SECTION_CHARS:
        content += " 마지막으로, 한 판 한 판의 결과보다 의사결정의 재현성을 지키면 승률은 뒤에서 반드시 따라온다."
    return {"title": title, "content_markdown": content}


def _dummy_sections(summary_json: dict[str, Any], tone: str, language: str) -> list[dict[str, str]]:
    del language
    llm_input_summary = build_llm_input_summary(summary_json)
    sections = [
        _fallback_section(title, llm_input_summary, tone, section_index=index)
        for index, title in enumerate(SECTION_TITLES)
    ]
    sections[-1]["content_markdown"] = (
        "마치 짧은 스크림 루틴처럼, 다음 3판은 실험이 아니라 검증이다.\n"
        "- 미션 1: 10분 이전 데스를 판당 1회 이하로 제한하고, 첫 귀환 전 라인 손실을 최소화한다.\n"
        "- 미션 2: 오브젝트 45초 전 라인 정리와 시야 확보를 루틴화해, 팀 합류 타이밍을 먼저 잡는다.\n"
        "- 미션 3: 대표 챔피언군 위주로 포지션을 고정해 판단 변수부터 줄이고, 전투 시작 전 2초 체크를 반드시 수행한다.\n"
        "세 판만 집중해도 경기 체감은 눈에 띄게 달라지고, 그 변화가 다음 주 전체 성적의 방향을 바꾼다."
    )
    return sections


def generate_sections(summary_json: dict[str, Any], tone: str, language: str, openai_api_key: str) -> list[dict[str, str]]:
    llm_input_summary = build_llm_input_summary(summary_json)
    logger.info("llm input summary=%s", _short_preview(llm_input_summary))

    if not openai_api_key:
        sections = _dummy_sections(summary_json, tone, language)
        _validate_style_requirements(sections, tone=tone, enforce_tone_cue=False)
        logger.info("section distinctness check passed source=fallback_no_key count=%s", len(sections))
        return sections

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
                logger.info("section distinctness check passed source=llm count=%s", len(sections))
                return sections
            except LlmStructuredOutputError as exc:
                last_error = str(exc)
                logger.error("llm output validation failed attempt=%s error=%s", attempt, exc)
                strict_json_retry = True
                continue

        logger.error("llm fallback after retries; last_error=%s", last_error)
        sections = _dummy_sections(summary_json, tone, language)
        _validate_style_requirements(sections, tone=tone, enforce_tone_cue=False)
        logger.info("section distinctness check passed source=fallback_retry count=%s", len(sections))
        return sections
    except Exception as exc:
        logger.exception("llm report generation failed")
        raise LlmGenerationError(f"LLM report generation failed: {exc}") from exc
