import json
from typing import Any

from openai import OpenAI


SECTION_TITLES = [
    "한 줄 요약",
    "최근 흐름",
    "플레이 스타일",
    "챔프/포지션 성향",
    "한 줄 처방 + 다음 3판 미션",
]


class LlmGenerationError(Exception):
    pass


def _system_prompt() -> str:
    return (
        "You are a League of Legends report writer.\n"
        "Write exactly 5 markdown sections in Korean or English based on language input.\n"
        "Each section must follow: evidence(metrics) -> interpretation -> narrative.\n"
        "If uncertain, explicitly say '추정' (or 'estimated').\n"
        "No hateful, discriminatory, or violent content.\n"
        "Keep each section concise, around 8-14 lines.\n"
        "The final section must include exactly 3 bullet missions for next 3 games.\n"
        "Return strict JSON: {\"sections\":[{\"title\":\"...\",\"content_markdown\":\"...\"}, ...]}"
    )


def _user_prompt(summary_json: dict[str, Any], tone: str, language: str) -> str:
    return (
        f"tone={tone}\n"
        f"language={language}\n"
        f"required_titles={SECTION_TITLES}\n"
        "summary_json=\n"
        f"{json.dumps(summary_json, ensure_ascii=False, indent=2)}\n"
    )


def _dummy_sections(summary_json: dict[str, Any], tone: str, language: str) -> list[dict[str, str]]:
    win_rate = summary_json.get("win_rate", 0)
    champ = summary_json.get("most_played_champ", "Unknown")
    role = summary_json.get("main_role", "UNKNOWN")
    missions = (
        "- 첫 10분 데스 1회 이하 유지\n"
        "- 시야 점수/오브젝트 관여를 한 번 더 의식하기\n"
        "- 3판 중 2판은 주포지션+주챔프로 고정"
    )
    body = [
        {
            "title": "한 줄 요약",
            "content_markdown": f"- 근거: 승률 {win_rate:.1%}, 주포지션 {role}, 모스트 챔프 {champ}\n- 해석: 지금 플레이어는 '{tone}' 톤으로 보면 기복 관리형 캐릭터\n- 서사: 오늘도 협곡에서 자기 서사를 쌓는 중",
        },
        {
            "title": "최근 흐름",
            "content_markdown": "- 근거: 연승/연패 지표와 평균 데스를 기반으로 최근 흐름 파악\n- 해석: 상승/하락 구간이 보이면 멘탈 영향이 있을 수 있음(추정)\n- 서사: 흐름 탈 때는 과감하고, 끊길 때는 리셋이 핵심",
        },
        {
            "title": "플레이 스타일",
            "content_markdown": "- 근거: early_impact_proxy, deaths_per_game, avg_cs_per_min\n- 해석: 공격성과 안정성의 균형이 핵심, 과감함 뒤에 리스크 관리 필요\n- 서사: 한타 때 존재감은 확실하지만, 진입 타이밍은 한 박자만 더",
        },
        {
            "title": "챔프/포지션 성향",
            "content_markdown": "- 근거: champion_pool_size, main_role, role_consistency\n- 해석: 챔프폭과 포지션 고정도에 따라 숙련/적응형 성향이 갈림\n- 서사: 메타 추종보다 본인 시그니처를 살릴 때 퍼포먼스가 안정",
        },
        {
            "title": "한 줄 처방 + 다음 3판 미션",
            "content_markdown": f"- 한 줄 처방: '{tone}' 모드에서도 숫자 근거 중심으로 플레이 리듬을 고정하세요.\n- 다음 3판 미션:\n{missions}",
        },
    ]
    if language == "en":
        return [
            {
                "title": "One-Line Summary",
                "content_markdown": "Evidence-first summary in MVP fallback mode.",
            },
            {
                "title": "Recent Trend",
                "content_markdown": "Estimated momentum from streak and stability metrics.",
            },
            {
                "title": "Play Style",
                "content_markdown": "Aggression vs stability interpreted from proxy stats.",
            },
            {
                "title": "Champion/Role Profile",
                "content_markdown": "Pool breadth and role consistency indicate preference.",
            },
            {
                "title": "Prescription + Next 3 Games",
                "content_markdown": "- Mission 1\n- Mission 2\n- Mission 3",
            },
        ]
    return body


def generate_sections(summary_json: dict[str, Any], tone: str, language: str, openai_api_key: str) -> list[dict[str, str]]:
    if not openai_api_key:
        return _dummy_sections(summary_json, tone, language)

    client = OpenAI(api_key=openai_api_key)
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(summary_json, tone, language)},
            ],
            temperature=0.8 if tone == "funny" else 0.6,
        )
        raw_content = completion.choices[0].message.content or "{}"
        parsed = json.loads(raw_content)
        sections = parsed.get("sections", [])
        if len(sections) != 5:
            raise LlmGenerationError("LLM 출력 형식이 올바르지 않습니다.")
        clean = []
        for idx, section in enumerate(sections):
            title = section.get("title") or SECTION_TITLES[idx]
            content = section.get("content_markdown") or ""
            clean.append({"title": title, "content_markdown": content})
        return clean
    except Exception as exc:
        raise LlmGenerationError(f"LLM 리포트 생성 실패: {exc}") from exc
