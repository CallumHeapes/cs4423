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
python optimize_squad.py --horizon 6        # opponent look-ahead in GWs
python optimize_squad.py --differentials 2  # force >=2 sub-10%-owned picks
python optimize_squad.py --max-player-cost 13.0 --budget 100.0
```

What it does:

1. Fetches `bootstrap-static` (every player: live `now_cost`, per-90 expected
   stats, `chance_of_playing_next_round`, set-piece/penalty order, injury/news
   flags) and `fixtures` fresh.
2. **Opponent analysis** per team over the next `--horizon` gameweeks
   (default 5): each fixture is weighted by the *opponent's* strength — attack
   ease (weak opponent defences → more goals) and clean-sheet ease (weak
   opponent attacks → more clean sheets) — plus FDR. Double gameweeks and
   blanks fall out naturally.
3. **Scores every player on expected FPL points**, not just price:
   - Attacking threat: `expected_goals_per_90` + `expected_assists_per_90`
     converted to points at the position's goal value + 3/assist, with a bump
     for penalty and set-piece takers.
   - Clean sheets: defenders and keepers are ranked further on a clean-sheet
     probability (team defensive strength, or `clean_sheets_per_90` in-season),
     worth 4 pts (GK/DEF) / 1 pt (MID); keepers also get save points.
   - A convex price base so that pre-season — when the per-90 stats are still 0
     — the optimizer still prefers a few genuine premium anchors over a flat
     spread. Minutes security down-weights anyone < 75% to play.
4. Runs a constrained integer program that **maximises total expected points**
   under the FPL rules — 15 players (2/5/5/3), £100.0m, max 3 per club —
   **plus a per-player price cap** (default £13.0m) so no single £15m+ superstar
   anchors the squad, and an optional **minimum number of differentials**
   (`--differentials N`). Positions come from `element_type`, so FPL's
   classification (e.g. João Pedro = FWD) is always respected.
5. Picks a **legal starting XI that never starts 5 defenders** (always 2-3
   forwards and 3-5 midfielders), a bench order, and a captain/vice based on
   attacking threat + fixtures.
6. Prints a markdown report: full 15, starting XI, bench, opponent outlook per
   club, captaincy, a **differentials-to-consider** shortlist, a strategy note,
   and a **transfer-rules heads-up** (1 free transfer/GW, bankable to 5, -4 per
   extra hit) that Phase 2 will enforce.

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
