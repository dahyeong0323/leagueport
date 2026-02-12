import asyncio
from dataclasses import dataclass
from statistics import mean
from typing import Any

import httpx


REGION_MAP = {
    "KR": {"platform": "kr1", "routing": "asia"},
    "NA": {"platform": "na1", "routing": "americas"},
    "EUW": {"platform": "euw1", "routing": "europe"},
}


class RiotApiError(Exception):
    pass


@dataclass
class RiotIdentity:
    game_name: str
    tag_line: str
    puuid: str


async def _request_with_retry(client: httpx.AsyncClient, method: str, url: str, headers: dict[str, str], max_attempts: int = 5) -> httpx.Response:
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        response = await client.request(method, url, headers=headers)
        if response.status_code != 429:
            return response

        retry_after = response.headers.get("Retry-After")
        wait = float(retry_after) if retry_after else delay
        await asyncio.sleep(wait)
        delay = min(delay * 2, 8.0)
    return response


def _split_riot_id(riot_id: str) -> tuple[str, str]:
    if "#" in riot_id:
        game_name, tag = riot_id.split("#", 1)
        return game_name.strip(), tag.strip()
    return riot_id.strip(), "KR1"


def _kda_to_str(kills: float, deaths: float, assists: float) -> str:
    return f"{kills:.1f}/{deaths:.1f}/{assists:.1f}"


def _calc_streak(wins: list[bool]) -> dict[str, Any]:
    if not wins:
        return {"type": "NONE", "length": 0}
    first = wins[0]
    length = 1
    for w in wins[1:]:
        if w == first:
            length += 1
        else:
            break
    return {"type": "W" if first else "L", "length": length}


async def fetch_riot_summary(riot_id: str, region: str, riot_api_key: str, max_games: int = 20) -> dict[str, Any]:
    if not riot_api_key:
        raise RiotApiError("RIOT_API_KEY가 설정되지 않았습니다.")

    region_upper = region.upper()
    if region_upper not in REGION_MAP:
        raise RiotApiError("지원하지 않는 region입니다. KR/NA/EUW만 지원합니다.")
    region_info = REGION_MAP[region_upper]
    headers = {"X-Riot-Token": riot_api_key}
    timeout = httpx.Timeout(20.0, read=20.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        game_name, tag_line = _split_riot_id(riot_id)
        account_url = f"https://{region_info['routing']}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        account_res = await _request_with_retry(client, "GET", account_url, headers=headers)
        if account_res.status_code == 404:
            raise RiotApiError("Riot ID를 찾을 수 없습니다. 입력값을 확인해주세요.")
        if account_res.status_code >= 400:
            raise RiotApiError("Riot 계정 조회에 실패했습니다. 잠시 후 다시 시도해주세요.")

        account = account_res.json()
        identity = RiotIdentity(game_name=account.get("gameName", game_name), tag_line=account.get("tagLine", tag_line), puuid=account["puuid"])

        match_ids_url = (
            f"https://{region_info['routing']}.api.riotgames.com/lol/match/v5/matches/by-puuid/"
            f"{identity.puuid}/ids?start=0&count={max_games}"
        )
        match_ids_res = await _request_with_retry(client, "GET", match_ids_url, headers=headers)
        if match_ids_res.status_code >= 400:
            raise RiotApiError("매치 목록 조회에 실패했습니다. 잠시 후 다시 시도해주세요.")
        match_ids = match_ids_res.json()
        if not match_ids:
            raise RiotApiError("최근 매치가 없어 리포트를 생성할 수 없습니다.")

        sem = asyncio.Semaphore(4)

        async def fetch_match(match_id: str) -> dict[str, Any] | None:
            async with sem:
                detail_url = f"https://{region_info['routing']}.api.riotgames.com/lol/match/v5/matches/{match_id}"
                res = await _request_with_retry(client, "GET", detail_url, headers=headers)
                if res.status_code >= 400:
                    return None
                return res.json()

        matches = [m for m in await asyncio.gather(*(fetch_match(mid) for mid in match_ids)) if m]
        if not matches:
            raise RiotApiError("매치 상세 조회에 실패했습니다. 잠시 후 다시 시도해주세요.")

    participants: list[dict[str, Any]] = []
    for match in matches:
        target = next((p for p in match.get("info", {}).get("participants", []) if p.get("puuid") == identity.puuid), None)
        if target:
            participants.append(target)

    if not participants:
        raise RiotApiError("분석 가능한 매치를 찾지 못했습니다.")

    games_analyzed = len(participants)
    wins = [bool(p.get("win", False)) for p in participants]
    role_counts: dict[str, int] = {}
    champ_counts: dict[str, int] = {}
    kills = [float(p.get("kills", 0)) for p in participants]
    deaths = [float(p.get("deaths", 0)) for p in participants]
    assists = [float(p.get("assists", 0)) for p in participants]
    durations = [float(p.get("timePlayed", 0)) / 60 for p in participants if p.get("timePlayed")]
    cs_per_min = []
    early_proxy = []
    vision_proxy = []
    objective_proxy = []

    for p in participants:
        role = (p.get("teamPosition") or p.get("individualPosition") or "UNKNOWN").upper()
        role_counts[role] = role_counts.get(role, 0) + 1
        champ = p.get("championName", "Unknown")
        champ_counts[champ] = champ_counts.get(champ, 0) + 1

        min_played = max(float(p.get("timePlayed", 1)) / 60, 1)
        total_cs = float(p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0))
        cs_per_min.append(total_cs / min_played)

        early_proxy.append(float(p.get("kills", 0) + p.get("assists", 0) * 0.7) / min_played)
        vision_proxy.append(float(p.get("visionScore", 0)) / min_played)
        objective_proxy.append(float(p.get("damageDealtToObjectives", 0)) / 10000)

    main_role = max(role_counts, key=role_counts.get)
    most_played_champ = max(champ_counts, key=champ_counts.get)

    summary = {
        "games_analyzed": games_analyzed,
        "win_rate": round(sum(1 for x in wins if x) / games_analyzed, 3),
        "main_role": main_role,
        "role_consistency": round(role_counts.get(main_role, 0) / games_analyzed, 3),
        "champion_pool_size": len(champ_counts),
        "most_played_champ": most_played_champ,
        "avg_kda": _kda_to_str(mean(kills), mean(deaths), mean(assists)),
        "deaths_per_game": round(mean(deaths), 2),
        "avg_cs_per_min": round(mean(cs_per_min), 2),
        "avg_game_duration_min": round(mean(durations), 2) if durations else 0.0,
        "streak_state": _calc_streak(wins),
        "early_impact_proxy": round(mean(early_proxy), 3),
        "vision_proxy": round(mean(vision_proxy), 3),
        "objective_proxy": round(mean(objective_proxy), 3),
    }
    return summary
