"""fetch_data.py — thin client for the official Fantasy Premier League API.

No authentication is required for any of these endpoints. Everything is pulled
*live* and cached to ./data/*.json so the optimizer can be re-run offline (or
when the FPL site is briefly down) without re-hitting the API.

Endpoints used
--------------
- bootstrap-static : every player (price, position, form, news, etc.) + teams
- fixtures         : full fixture list with per-team fixture difficulty (FDR)
- entry/{id}       : your overall entry info (bank, rank...) — used in Phase 2
- entry/{id}/event/{gw}/picks : your actual squad for a gameweek — Phase 2
- element-summary/{player_id} : per-player GW history & upcoming fixtures

Run directly to refresh the on-disk cache:

    python fetch_data.py            # refresh bootstrap-static + fixtures
    python fetch_data.py --entry    # also refresh your entry info
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import io
import json
import os
import sys
import time
from typing import Any

import requests

BASE_URL = "https://fantasy.premierleague.com/api"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# A browser-ish UA — the FPL API occasionally 403s the default requests UA.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def default_team_id() -> int:
    """Team id from team_id.txt if present, else the project default (5156799)."""
    path = os.path.join(HERE, "team_id.txt")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return 5156799


def _cache_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.json")


def _load_cache(name: str) -> Any | None:
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_cache(name: str, payload: Any) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(_cache_path(name), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _get_json(endpoint: str, *, params: dict | None = None, retries: int = 3,
              timeout: int = 30) -> Any:
    """GET {BASE_URL}/{endpoint}/ with a short exponential backoff on failure."""
    url = f"{BASE_URL}/{endpoint}/"
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:  # network / HTTP error
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise RuntimeError(
        f"Failed to fetch {url} after {retries} attempts: {last_exc}\n"
        "If you are behind a corporate/egress firewall the FPL host may be "
        "blocked — run somewhere the site is reachable, or use a previously "
        "cached copy with the --offline flag on the optimizer."
    ) from last_exc


def get_bootstrap(*, refresh: bool = True, offline: bool = False) -> dict:
    """All players + teams + events. Fetches live unless offline is requested."""
    name = "bootstrap"
    if offline:
        cached = _load_cache(name)
        if cached is None:
            raise RuntimeError(
                "No cached bootstrap data found. Run once online first "
                "(python fetch_data.py) to populate ./data/."
            )
        return cached
    if not refresh:
        cached = _load_cache(name)
        if cached is not None:
            return cached
    data = _get_json("bootstrap-static")
    _save_cache(name, data)
    return data


def get_fixtures(*, refresh: bool = True, offline: bool = False) -> list[dict]:
    """Full fixture list with per-team fixture-difficulty ratings."""
    name = "fixtures"
    if offline:
        cached = _load_cache(name)
        if cached is None:
            raise RuntimeError(
                "No cached fixtures found. Run once online first "
                "(python fetch_data.py) to populate ./data/."
            )
        return cached
    if not refresh:
        cached = _load_cache(name)
        if cached is not None:
            return cached
    data = _get_json("fixtures")
    _save_cache(name, data)
    return data


def get_entry(team_id: int | None = None, *, refresh: bool = True,
              offline: bool = False) -> dict:
    """Your overall entry info (bank, rank...). Mostly useful in Phase 2."""
    team_id = team_id or default_team_id()
    name = f"entry_{team_id}"
    if offline:
        cached = _load_cache(name)
        if cached is None:
            raise RuntimeError(f"No cached entry for {team_id}.")
        return cached
    if not refresh:
        cached = _load_cache(name)
        if cached is not None:
            return cached
    data = _get_json(f"entry/{team_id}")
    _save_cache(name, data)
    return data


def get_entry_picks(gw: int, team_id: int | None = None) -> dict:
    """Your actual squad for a gameweek (only populated once the GW locks)."""
    team_id = team_id or default_team_id()
    return _get_json(f"entry/{team_id}/event/{gw}/picks")


def get_entry_history(team_id: int | None = None) -> dict:
    """Your season history: per-GW points/bank/value + chips used."""
    team_id = team_id or default_team_id()
    return _get_json(f"entry/{team_id}/history")


OVERALL_LEAGUE_ID = 314  # the classic league every FPL manager is in


def get_top_manager_ids(n: int = 50, league_id: int = OVERALL_LEAGUE_ID
                        ) -> list[int]:
    """Entry ids of the top `n` managers in a classic league (default: overall).
    Standings paginate 50 per page."""
    ids: list[int] = []
    page = 1
    while len(ids) < n:
        data = _get_json(f"leagues-classic/{league_id}/standings",
                         params={"page_standings": page})
        standings = data.get("standings", {})
        results = standings.get("results", [])
        if not results:
            break
        ids.extend(r["entry"] for r in results)
        if not standings.get("has_next"):
            break
        page += 1
    return ids[:n]


def get_manager_picks_bulk(manager_ids, gw: int, workers: int = 8
                           ) -> dict[int, dict]:
    """{entry_id: picks_payload} for many managers at a gameweek, threaded.
    Failures are skipped."""
    def _one(mid):
        try:
            return mid, get_entry_picks(gw, mid)
        except RuntimeError:
            return mid, None
    out: dict[int, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for mid, picks in ex.map(_one, manager_ids):
            if picks:
                out[mid] = picks
    return out


def current_and_next_gw(bootstrap: dict) -> tuple[int | None, int | None]:
    """From bootstrap events, the current (last locked) and next gameweek ids."""
    current = next_gw = None
    for ev in bootstrap.get("events", []):
        if ev.get("is_current"):
            current = ev["id"]
        if ev.get("is_next"):
            next_gw = ev["id"]
    if next_gw is None:  # end of season, or pre-season before any 'next'
        upcoming = [ev["id"] for ev in bootstrap.get("events", [])
                    if not ev.get("finished")]
        next_gw = min(upcoming) if upcoming else current
    return current, next_gw


def get_element_summary(player_id: int) -> dict:
    """Per-player gameweek history + upcoming fixtures (used sparingly)."""
    return _get_json(f"element-summary/{player_id}")


def save_squad(player_ids, team_id: int | None = None) -> None:
    """Remember a proposed 15 (from the optimizer) so benchmark.py can compare
    it before the season starts and your real team exists."""
    team_id = team_id or default_team_id()
    _save_cache(f"squad_{team_id}", {"ids": [int(i) for i in player_ids]})


def load_squad(team_id: int | None = None) -> list[int] | None:
    team_id = team_id or default_team_id()
    data = _load_cache(f"squad_{team_id}")
    return data.get("ids") if data else None


CLUBELO_URL = "https://api.clubelo.com"


def get_club_elo(*, refresh: bool = True, offline: bool = False,
                 on_date: str | None = None) -> dict[str, float]:
    """Current club Elo ratings from ClubElo (free, no key). Maps club name ->
    Elo. Forward-looking team strength that reflects current form/trajectory.

    Cached to ./data/elo.json. Returns {} (graceful) if unreachable, so the
    optimizer just falls back to FPL strength ratings.
    """
    name = "elo"
    if offline:
        return _load_cache(name) or {}
    if not refresh:
        cached = _load_cache(name)
        if cached is not None:
            return cached
    date = on_date or datetime.date.today().isoformat()
    ua = {"User-Agent": HEADERS["User-Agent"]}  # no JSON Accept — it returns CSV
    text, errors = None, []
    for scheme in ("https", "http"):  # ClubElo is http-first; try both
        try:
            resp = requests.get(f"{scheme}://api.clubelo.com/{date}", headers=ua,
                                timeout=30)
            resp.raise_for_status()
            text = resp.text
            break
        except requests.RequestException as exc:
            errors.append(f"{scheme}: {exc}")
    if text is None:
        print(f"ClubElo unavailable ({' | '.join(errors)}) — falling back to "
              "FPL/neutral team strength.", file=sys.stderr)
        return _load_cache(name) or {}
    out: dict[str, float] = {}
    for row in csv.DictReader(io.StringIO(text)):
        club, elo = row.get("Club"), row.get("Elo")
        if club and elo:
            try:
                out[club] = float(elo)
            except ValueError:
                pass
    if out:
        _save_cache(name, out)
    else:
        print(f"ClubElo returned no parseable rows for {date} "
              f"({len(text)} bytes). Columns seen: {text.splitlines()[:1]}",
              file=sys.stderr)
    return out


def _hist_float(value) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _fetch_one_history(player_id: int) -> tuple[int, dict | None]:
    """Last completed season's totals for one player (via history_past)."""
    try:
        data = _get_json(f"element-summary/{player_id}", retries=2, timeout=20)
    except RuntimeError:
        return player_id, None
    past = data.get("history_past") or []
    if not past:
        return player_id, None
    s = past[-1]  # most recent prior season
    minutes = _hist_float(s.get("minutes"))
    return player_id, {
        "season": s.get("season_name"),
        "minutes": minutes,
        # Prefer expected stats; fall back to actual goals/assists for older
        # seasons that predate xG in the API.
        "xg": _hist_float(s.get("expected_goals")) or _hist_float(s.get("goals_scored")),
        "xa": _hist_float(s.get("expected_assists")) or _hist_float(s.get("assists")),
        "cs": _hist_float(s.get("clean_sheets")),
    }


def get_last_season_stats(player_ids, *, refresh: bool = True,
                          offline: bool = False, workers: int = 6,
                          progress=None) -> dict[int, dict]:
    """Map player_id -> last-season {minutes, xg, xa, cs} via element-summary.

    Cached to ./data/last_season.json and fetched incrementally (only ids not
    already cached). One HTTP call per player, so it is threaded. Failures are
    skipped silently — those players just fall back to the price baseline.
    """
    name = "last_season"
    cache_raw = _load_cache(name) or {}
    cache = {int(k): v for k, v in cache_raw.items()}
    if offline:
        return cache

    to_fetch = [pid for pid in player_ids if pid not in cache] if refresh \
        else []
    if to_fetch:
        if progress:
            progress(f"Fetching last-season stats for {len(to_fetch)} players "
                     "(one-off, cached)...")
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for pid, rec in ex.map(_fetch_one_history, to_fetch):
                cache[pid] = rec  # rec may be None (no PL history) — remembered
                done += 1
                if progress and done % 100 == 0:
                    progress(f"  ...{done}/{len(to_fetch)}")
        _save_cache(name, {str(k): v for k, v in cache.items()})
    # Drop the None sentinels before returning usable stats.
    return {pid: rec for pid, rec in cache.items() if rec}


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the FPL data cache.")
    parser.add_argument("--entry", action="store_true",
                        help="also refresh your entry info")
    parser.add_argument("--team-id", type=int, default=None,
                        help="override team id (default from team_id.txt)")
    args = parser.parse_args(argv)

    print("Fetching bootstrap-static ...", file=sys.stderr)
    boot = get_bootstrap(refresh=True)
    print(f"  {len(boot['elements'])} players, {len(boot['teams'])} teams",
          file=sys.stderr)

    print("Fetching fixtures ...", file=sys.stderr)
    fixtures = get_fixtures(refresh=True)
    print(f"  {len(fixtures)} fixtures", file=sys.stderr)

    if args.entry:
        team_id = args.team_id or default_team_id()
        print(f"Fetching entry {team_id} ...", file=sys.stderr)
        get_entry(team_id, refresh=True)

    print(f"Cache written to {DATA_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
