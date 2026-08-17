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


def _get_json(endpoint: str, *, retries: int = 3, timeout: int = 30) -> Any:
    """GET {BASE_URL}/{endpoint}/ with a short exponential backoff on failure."""
    url = f"{BASE_URL}/{endpoint}/"
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
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


def get_element_summary(player_id: int) -> dict:
    """Per-player gameweek history + upcoming fixtures (used sparingly)."""
    return _get_json(f"element-summary/{player_id}")


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
