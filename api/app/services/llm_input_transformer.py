from __future__ import annotations

from typing import Any, Literal


LlmEarlyPresence = Literal["low", "mid", "high"]
LlmRecentResult = Literal["W", "L"]


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _parse_avg_kda_ratio(avg_kda: Any, deaths_per_game: Any) -> float:
    if isinstance(avg_kda, str):
        parts = avg_kda.split("/")
        if len(parts) == 3:
            try:
                kills = float(parts[0].strip())
                deaths = float(parts[1].strip())
                assists = float(parts[2].strip())
                denom = deaths if deaths > 0 else max(_as_float(deaths_per_game, 1.0), 1.0)
                return round((kills + assists) / max(denom, 1.0), 2)
            except ValueError:
                pass
    return 0.0


def _recent_result(streak_state: Any, win_rate: Any) -> LlmRecentResult:
    if isinstance(streak_state, dict):
        streak_type = str(streak_state.get("type", "")).upper()
        if streak_type in {"W", "L"}:
            return streak_type  # type: ignore[return-value]
    return "W" if _as_float(win_rate) >= 0.5 else "L"


def _early_presence(early_impact_proxy: Any) -> LlmEarlyPresence:
    value = _as_float(early_impact_proxy)
    if value < 0.4:
        return "low"
    if value < 0.65:
        return "mid"
    return "high"


def build_llm_input_summary(summary_json: dict[str, Any]) -> dict[str, Any]:
    sample_size = int(summary_json.get("games_analyzed", 0) or 0)
    deaths_per_game = round(_as_float(summary_json.get("deaths_per_game")), 2)
    return {
        "sample_size": sample_size,
        "recent_result": _recent_result(summary_json.get("streak_state"), summary_json.get("win_rate")),
        "avg_kda": _parse_avg_kda_ratio(summary_json.get("avg_kda"), deaths_per_game),
        "avg_deaths": deaths_per_game,
        "cs_per_min": round(_as_float(summary_json.get("avg_cs_per_min")), 2),
        "main_position": str(summary_json.get("main_role", "UNKNOWN") or "UNKNOWN"),
        "main_champion": str(summary_json.get("most_played_champ", "Unknown") or "Unknown"),
        "champion_pool_size": int(summary_json.get("champion_pool_size", 0) or 0),
        "early_game_presence": _early_presence(summary_json.get("early_impact_proxy")),
        "avg_game_duration": round(_as_float(summary_json.get("avg_game_duration_min")), 2),
    }
