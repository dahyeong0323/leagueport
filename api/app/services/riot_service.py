import asyncio
from dataclasses import dataclass
import logging
from statistics import mean
from typing import Any

import httpx


@dataclass(frozen=True)
class RiotRegionRouting:
    platform: str
    routing: str


RIOT_REGION_MAP: dict[str, RiotRegionRouting] = {
    "EUW": RiotRegionRouting(platform="euw1", routing="europe"),
    "KR": RiotRegionRouting(platform="kr1", routing="asia"),
}
ALLOWED_REGIONS = tuple(RIOT_REGION_MAP.keys())
ALLOWED_REGIONS_TEXT = ", ".join(ALLOWED_REGIONS)


class RiotApiError(Exception):
    pass


class RiotIdParseError(RiotApiError):
    status_code = 400


class RiotUserInputError(RiotApiError):
    status_code = 400


class RiotUpstreamError(RiotApiError):
    status_code = 502


@dataclass
class RiotIdentity:
    game_name: str
    tag_line: str
    puuid: str


logger = logging.getLogger("lol-report-riot")


def _masked_api_key(api_key: str) -> str:
    if not api_key:
        return "missing"
    return f"{api_key[:6]}..."


def normalize_region(region: str) -> str:
    region_upper = region.strip().upper()
    if region_upper not in RIOT_REGION_MAP:
        raise RiotUserInputError(f"Unsupported region. Allowed: {ALLOWED_REGIONS_TEXT}.")
    return region_upper


def get_region_routing(region: str) -> RiotRegionRouting:
    return RIOT_REGION_MAP[normalize_region(region)]


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    max_attempts: int = 5,
) -> httpx.Response:
    delay = 1.0
    last_response: httpx.Response | None = None
    for attempt in range(1, max_attempts + 1):
        logger.info("riot request attempt=%s method=%s url=%s", attempt, method, url)
        try:
            response = await client.request(method, url, headers=headers)
        except Exception as exc:
            logger.exception("riot request exception attempt=%s method=%s url=%s", attempt, method, url)
            logger.error("metric=riot_upstream_failure kind=request_exception method=%s url=%s", method, url)
            raise RiotUpstreamError(
                f"Riot request exception (method={method}, url={url}, error={exc})"
            ) from exc
        last_response = response
        logger.info("riot response status=%s url=%s", response.status_code, url)
        if response.status_code != 200:
            logger.error(
                "riot non-200 response method=%s url=%s status=%s body=%s",
                method,
                url,
                response.status_code,
                response.text,
            )
        if response.status_code != 429:
            return response

        retry_after = response.headers.get("Retry-After")
        wait = float(retry_after) if retry_after else delay
        await asyncio.sleep(wait)
        delay = min(delay * 2, 8.0)
    return last_response if last_response else await client.request(method, url, headers=headers)


def _upstream_error(label: str, response: httpx.Response) -> RiotUpstreamError:
    logger.error(
        "metric=riot_upstream_failure kind=non_200 label=%s status=%s url=%s",
        label,
        response.status_code,
        response.request.url,
    )
    return RiotUpstreamError(
        f"{label} failed (status={response.status_code}, url={response.request.url}, body={response.text})"
    )


def _split_riot_id(riot_id: str) -> tuple[str, str]:
    if "#" not in riot_id:
        raise RiotIdParseError("Riot ID must use name#tag format. Example: Hide on bush#KR1")
    game_name, tag = riot_id.split("#", 1)
    game_name = game_name.strip()
    tag = tag.strip()
    if not game_name or not tag:
        raise RiotIdParseError("Riot ID parsing failed: empty name or tag.")
    return game_name, tag


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
        raise RiotUpstreamError("RIOT_API_KEY is missing.")

    logger.info("riot key loaded=%s", _masked_api_key(riot_api_key))
    logger.info("riot fetch start riot_id=%s region=%s", riot_id, region)

    region_upper = normalize_region(region)
    region_info = RIOT_REGION_MAP[region_upper]
    logger.info(
        "riot region mapping region=%s platform=%s routing=%s",
        region_upper,
        region_info.platform,
        region_info.routing,
    )

    headers = {"X-Riot-Token": riot_api_key}
    timeout = httpx.Timeout(20.0, read=20.0)

    try:
        game_name, tag_line = _split_riot_id(riot_id)
        async with httpx.AsyncClient(timeout=timeout) as client:
            account_url = (
                f"https://{region_info.routing}.api.riotgames.com/"
                f"riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
            )
            account_res = await _request_with_retry(client, "GET", account_url, headers=headers)
            if account_res.status_code == 404:
                raise RiotUserInputError("Riot ID not found.")
            if account_res.status_code >= 400:
                raise _upstream_error("Account lookup", account_res)

            account = account_res.json()
            identity = RiotIdentity(
                game_name=account.get("gameName", game_name),
                tag_line=account.get("tagLine", tag_line),
                puuid=account["puuid"],
            )
            logger.info(
                "riot account resolved game_name=%s tag_line=%s puuid=%s",
                identity.game_name,
                identity.tag_line,
                identity.puuid,
            )

            match_ids_url = (
                f"https://{region_info.routing}.api.riotgames.com/lol/match/v5/matches/"
                f"by-puuid/{identity.puuid}/ids?start=0&count={max_games}"
            )
            match_ids_res = await _request_with_retry(client, "GET", match_ids_url, headers=headers)
            if match_ids_res.status_code >= 400:
                raise _upstream_error("Match ids lookup", match_ids_res)
            match_ids = match_ids_res.json()
            if not match_ids:
                raise RiotUserInputError("No recent matches found for this account.")

            sem = asyncio.Semaphore(4)

            async def fetch_match(match_id: str) -> dict[str, Any] | None:
                async with sem:
                    detail_url = f"https://{region_info.routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"
                    res = await _request_with_retry(client, "GET", detail_url, headers=headers)
                    if res.status_code >= 400:
                        logger.error(
                            "riot match detail failed match_id=%s url=%s status=%s body=%s",
                            match_id,
                            detail_url,
                            res.status_code,
                            res.text,
                        )
                        logger.error(
                            "metric=riot_upstream_failure kind=match_detail_non_200 match_id=%s status=%s",
                            match_id,
                            res.status_code,
                        )
                        return None
                    return res.json()

            matches = [m for m in await asyncio.gather(*(fetch_match(mid) for mid in match_ids)) if m]
            if not matches:
                raise RiotUpstreamError(f"Match detail fetch failed. requested={len(match_ids)} success=0")
    except Exception as exc:
        logger.exception("riot fetch error riot_id=%s region=%s error=%s", riot_id, region, exc)
        raise

    participants: list[dict[str, Any]] = []
    for match in matches:
        target = next(
            (p for p in match.get("info", {}).get("participants", []) if p.get("puuid") == identity.puuid),
            None,
        )
        if target:
            participants.append(target)

    if not participants:
        raise RiotUserInputError("No participant data found for this account.")

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
        "data_source": "riot",
        "riot_error": None,
        "matches_fetched": len(matches),
        "puuid": identity.puuid,
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
    logger.info(
        "riot summary built riot_id=%s region=%s games_analyzed=%s matches_fetched=%s",
        riot_id,
        region,
        games_analyzed,
        len(matches),
    )
    return summary
