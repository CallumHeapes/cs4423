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
python optimize_squad.py --no-history       # skip the last-season pull (faster)
```

Squad-shape levers (all have sensible defaults):

| Flag | Default | Effect |
|------|---------|--------|
| `--min-bank` / `--max-bank` | 2.0 / 5.0 | keep £2–5m unspent (flexibility reserve) |
| `--min-premiums` | 2 | require ≥2 attacking (MID/FWD) picks at `--premium-cost`+ |
| `--premium-cost` | 9.0 | price (£m) that counts as an attacking premium |
| `--bench-gk-max` | 4.5 | force a cheap bench keeper so no budget is wasted on a non-playing #2 |
| `--max-player-cost` | 13.0 | per-player cap (bars £14m+ superstars) |
| `--differentials` | 0 | force ≥N sub-10%-owned picks |

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
   - **Last season pooled with current** (via each player's `history_past`):
     pre-season the goals/assists/clean-sheet rates come from last season's real
     sample (with last season's minutes driving how much they're trusted), and
     the current season takes over as games accrue. Promoted-club and overseas
     players with no PL history fall back to the price baseline — a deliberate,
     accepted gap. Skip this slower per-player pull with `--no-history`.
   - Attacking threat: pooled `expected_goals` + `expected_assists` per 90
     converted to points at the position's goal value + 3/assist, with a bump
     for penalty and set-piece takers, and minutes-based shrinkage so freak
     small-sample rates can't dominate.
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

```bash
python weekly_digest.py                    # your team, next deadline
python weekly_digest.py --free-transfers 2 # tell it how many FTs you have
python weekly_digest.py --horizon 6
```

`weekly_digest.py` pulls your actual squad (`/entry/5156799/event/{gw}/picks`),
bank, chips, and current captain, then produces one markdown digest:

1. Fixture outlook over the next `--horizon` GWs for every club you own.
2. **Flags** each owned player for bad fixture runs, form dips, price-drop risk
   (heavy net transfers out), or availability (injury/suspension/doubt).
3. **Budget-matched transfer suggestions** (1–2 per flagged player) — same
   position, affordable from your sale value + bank, rule-legal (max 3/club),
   ranked by projected-points gain.
4. A **transfer plan** that respects your free transfers and only advises a
   −4 hit when the gain clearly beats it (roll otherwise).
5. **Captain / vice** for the upcoming GW on attacking threat + fixtures.

It reuses the exact scoring model from `optimize_squad.py`, but the last-season
weight **decays with the gameweek** (`weight_for_gw`: 0.8 at GW1 → 0 by ~GW20),
so last year's data phases out on its own and current form takes over — no
manual switchover.

> The public API can't reliably report your free-transfer count, so pass
> `--free-transfers N` (default 1).

## One-tap Colab notebook

`FPL_Corrib_Athletic.ipynb` — open it in [Google Colab](https://colab.research.google.com)
(File → Open notebook → GitHub, or upload it), then run the **Setup** cell and
whichever phase you need. It clones the latest code and fetches fresh data each
run, so you're always current. Ideal for running from a phone before a deadline.
