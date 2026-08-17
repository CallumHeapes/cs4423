# FPL Squad Builder & Weekly Tracker

CLI tools that pull **live** data from the official Fantasy Premier League API
to build and manage the *Corrib Athletic* squad (team id `5156799`).

```
fpl-tool/
  team_id.txt        # your FPL entry id (gitignored; code defaults to 5156799)
  fetch_data.py      # thin FPL API client + on-disk cache (./data)
  optimize_squad.py  # Phase 1 — pre-season 15-man squad optimizer
  weekly_digest.py   # Phase 2 — in-season weekly report (TODO, after GW1)
  requirements.txt
```

## Install

```bash
cd fpl-tool
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Dependencies are minimal: `requests` (HTTP) and `pulp` (bundles the CBC
integer-programming solver).

## Phase 1 — pre-season squad optimizer

```bash
python optimize_squad.py                    # fetch live data, print squad
python optimize_squad.py --offline          # reuse the last ./data snapshot
python optimize_squad.py --max-player-cost 13.0 --budget 100.0 --horizon 5
```

What it does:

1. Fetches `bootstrap-static` (every player: live `now_cost`,
   `chance_of_playing_next_round`, injury/news flags) and `fixtures` fresh.
2. Computes a fixture-difficulty score per team over the next `--horizon`
   gameweeks (default 5), handling double gameweeks and blanks.
3. Scores every player on a blend of expected output (`ep_next`, `form`,
   `points_per_game`, a low-weighted slice of last season's `total_points`,
   and a price-implied floor for deep pre-season), then adjusts for fixtures
   and minutes security (anyone < 75% to play is flagged and down-weighted).
4. Runs a constrained integer program that **maximises total score** under the
   FPL rules — 15 players (2/5/5/3), £100.0m, max 3 per club — **plus a hard
   per-player price cap (~13.5% of budget)** so no single £15m+ superstar can
   anchor the squad. Positions come from the API's `element_type`, so FPL's
   classification (e.g. João Pedro = FWD) is always respected.
5. Prints a markdown report: full 15, best-formation starting XI, bench order,
   captain/vice suggestion, and a short strategy explanation.

> **Network note:** the FPL API is unauthenticated but must be reachable from
> wherever you run this. On a locked-down corporate network the host may be
> blocked — run it on a personal machine, or populate `./data` once somewhere
> it *is* reachable (`python fetch_data.py`) and then use `--offline`.

Re-run right before the **Friday 18:30 BST** deadline to catch late price
changes and team news.

## Refresh the cache manually

```bash
python fetch_data.py            # bootstrap-static + fixtures -> ./data
python fetch_data.py --entry    # also your entry info (bank, rank)
```

## Phase 2 — weekly in-season digest

`weekly_digest.py` (to be built after the GW1 deadline locks) will pull your
actual squad, bank, free transfers and chips, run a fixture-difficulty check
over the next 4-6 GWs, flag players on bad runs / form dips / price drops,
suggest budget-matched transfers, and recommend a captain — output as a single
markdown digest.
