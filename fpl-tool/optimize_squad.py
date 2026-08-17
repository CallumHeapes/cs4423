"""optimize_squad.py — Phase 1 pre-season FPL squad optimizer.

Builds an optimal 15-man Gameweek-1 squad from *live* FPL data under the
standard rules (2/5/5/3, £100.0m, max 3 per club) while deliberately spreading
spend rather than anchoring on a single £15m+ superstar.

Usage
-----
    python optimize_squad.py                 # fetch live data, optimize
    python optimize_squad.py --offline       # reuse ./data cache
    python optimize_squad.py --max-player-cost 13.5 --budget 100.0
    python optimize_squad.py --horizon 5     # fixture look-ahead in GWs

The model
---------
Each player gets an *expected points over the next `horizon` gameweeks* score:

    score = quality_pp_gw * minutes_multiplier * Σ_fixtures fixture_multiplier

- quality_pp_gw : expected points per game, blended from the FPL model's own
  `ep_next`, recent `form`, and `points_per_game`. Pre-season these are often
  zero, so we fall back to a price-implied expectation (FPL prices players by
  expected output) and always blend a little price signal for stability.
  Last-season cumulative `total_points` is included but weighted low, as the
  brief asks — pre-season it is unreliable.
- minutes_multiplier : discounts rotation / injury risk using
  `chance_of_playing_next_round` and `status`. Anyone < 75% is flagged risky.
- fixture_multiplier : per upcoming fixture, easier (low FDR) scores higher.
  Summing over the horizon's fixtures naturally rewards double gameweeks and
  penalises blanks.

A PuLP integer program then MAXIMISES total squad score subject to the FPL
rules PLUS a hard per-player price cap (~13-14% of budget) so no single
talisman can anchor the squad.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Iterable

import pulp

import fetch_data

# ---------------------------------------------------------------------------
# Tunable model constants (documented so they can be adjusted with intent).
# ---------------------------------------------------------------------------

# How much each performance signal adds ON TOP of the price-implied base.
W_EP_NEXT = 0.50     # FPL's own expected points for next GW (best pre-season)
W_FORM = 0.15        # recent points-per-game form
W_PPG = 0.15         # season points-per-game
W_TOTAL = 0.05       # last-season cumulative total (weighted LOW on purpose)

# Price-implied expectation is the pre-season BASE (performance stats are ~0
# before a ball is kicked). It is deliberately CONVEX in price:
#   base_ppg = PRICE_CONVEX_C * (cost_m ** PRICE_EXP)
# so premiums carry disproportionately more ceiling and the optimizer prefers a
# few genuine premium anchors funded by cheap enablers (a "barbell") rather than
# a flat mid-price spread. PRICE_EXP = 1.0 would make points-per-£ constant and
# leave the optimizer indifferent to premiums (the naive v1 behaviour); > 1.0
# tips it toward anchors. The per-player price cap still bars true superstars.
PRICE_EXP = 1.3
PRICE_CONVEX_C = 0.18

# Fixture sensitivity. FDR is 1 (easy) .. 5 (hard); difficulty 3 is neutral.
# multiplier = 1 + FIX_SENSITIVITY * (3 - difficulty), clamped to [0.6, 1.4].
FIX_SENSITIVITY = 0.15

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}  # GK, DEF, MID, FWD
SQUAD_SIZE = 15

# Starting XI must have exactly 1 GK and these outfield bounds (total 11).
XI_BOUNDS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}


# ---------------------------------------------------------------------------
# Player model
# ---------------------------------------------------------------------------


@dataclass
class Player:
    id: int
    name: str
    team_id: int
    team_short: str
    position: int          # element_type: 1 GK, 2 DEF, 3 MID, 4 FWD
    cost: int              # now_cost in tenths of a million (125 => £12.5m)
    quality_ppg: float     # blended expected points per game
    minutes_mult: float    # rotation/injury discount in (0, 1]
    fixture_sum: float      # Σ fixture_multiplier over the horizon
    n_fixtures: int
    avg_fdr: float | None
    chance: int | None     # chance_of_playing_next_round (None == assumed fit)
    status: str            # a/d/i/s/u/n
    news: str
    selected_by: float
    score: float = 0.0
    risky: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def position_name(self) -> str:
        return POSITIONS[self.position]

    @property
    def cost_m(self) -> float:
        return self.cost / 10.0


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Fixture difficulty over the horizon
# ---------------------------------------------------------------------------


def compute_team_fixtures(fixtures: list[dict], horizon: int
                          ) -> dict[int, list[int]]:
    """Map team_id -> list of FDR difficulties for its next `horizon` GWs.

    Uses the earliest `horizon` scheduled gameweeks. Handles double gameweeks
    (multiple fixtures in one GW) and blanks (a GW absent from a team's list).
    """
    # Which gameweeks are still to come? Pick the first `horizon` events that
    # have at least one unfinished fixture.
    upcoming_events = sorted(
        {f["event"] for f in fixtures
         if f["event"] is not None and not f.get("finished", False)}
    )[:horizon]
    horizon_events = set(upcoming_events)

    per_team: dict[int, list[int]] = {}
    for f in fixtures:
        if f["event"] not in horizon_events:
            continue
        home, away = f["team_h"], f["team_a"]
        per_team.setdefault(home, []).append(f["team_h_difficulty"])
        per_team.setdefault(away, []).append(f["team_a_difficulty"])
    return per_team


def _fixture_multiplier(difficulty: int) -> float:
    mult = 1.0 + FIX_SENSITIVITY * (3 - difficulty)
    return max(0.6, min(1.4, mult))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _minutes_multiplier(chance: int | None, status: str) -> tuple[float, bool,
                                                                   list[str]]:
    """Discount for rotation/injury risk. Returns (multiplier, risky, flags)."""
    flags: list[str] = []
    # Hard unavailability: injured / suspended / unavailable / on loan.
    if status in ("i", "s", "u"):
        label = {"i": "injured", "s": "suspended", "u": "unavailable"}[status]
        flags.append(label)
        return 0.05, True, flags
    if status == "n":  # not in squad / ineligible
        flags.append("not in squad")
        return 0.05, True, flags

    if chance is None:
        # No news == assumed fully fit.
        return 1.0, False, flags

    if chance >= 75:
        if chance < 100:
            flags.append(f"{chance}% fit")
        return 1.0, chance < 100, flags
    if chance >= 50:
        flags.append(f"doubt {chance}%")
        return 0.65, True, flags
    if chance >= 25:
        flags.append(f"major doubt {chance}%")
        return 0.35, True, flags
    flags.append(f"unlikely {chance}%")
    return 0.10, True, flags


def _quality_ppg(el: dict) -> float:
    """Expected points-per-game: a convex price base + performance signals.

    Pre-season the performance terms are ~0, so quality is driven by the convex
    price base — which rewards premium ceiling and produces a barbell squad.
    Once the season starts, ep_next/form/ppg add real differentiation on top.
    """
    ep_next = _to_float(el.get("ep_next"))
    form = _to_float(el.get("form"))
    ppg = _to_float(el.get("points_per_game"))
    total = _to_float(el.get("total_points"))
    cost_m = el["now_cost"] / 10.0

    base_ppg = PRICE_CONVEX_C * (cost_m ** PRICE_EXP)
    # Normalise last-season total into a per-game-ish figure (~38 games).
    total_ppg = total / 38.0

    return (
        base_ppg
        + W_EP_NEXT * ep_next
        + W_FORM * form
        + W_PPG * ppg
        + W_TOTAL * total_ppg
    )


def build_players(bootstrap: dict, fixtures: list[dict], horizon: int
                  ) -> list[Player]:
    teams = {t["id"]: t for t in bootstrap["teams"]}
    team_fixtures = compute_team_fixtures(fixtures, horizon)

    players: list[Player] = []
    for el in bootstrap["elements"]:
        team_id = el["team"]
        difficulties = team_fixtures.get(team_id, [])
        fixture_sum = sum(_fixture_multiplier(d) for d in difficulties)
        avg_fdr = (sum(difficulties) / len(difficulties)) if difficulties else None

        chance = el.get("chance_of_playing_next_round")
        minutes_mult, risky, flags = _minutes_multiplier(chance, el.get("status", "a"))

        quality = _quality_ppg(el)
        score = quality * minutes_mult * fixture_sum

        news = (el.get("news") or "").strip()
        if news and news not in " ".join(flags):
            # Surface the raw news blurb too (kept short).
            flags.append(news if len(news) <= 48 else news[:45] + "...")

        players.append(Player(
            id=el["id"],
            name=el.get("web_name", f"#{el['id']}"),
            team_id=team_id,
            team_short=teams.get(team_id, {}).get("short_name", "?"),
            position=el["element_type"],
            cost=el["now_cost"],
            quality_ppg=quality,
            minutes_mult=minutes_mult,
            fixture_sum=fixture_sum,
            n_fixtures=len(difficulties),
            avg_fdr=avg_fdr,
            chance=chance,
            status=el.get("status", "a"),
            news=news,
            selected_by=_to_float(el.get("selected_by_percent")),
            score=score,
            risky=risky,
            flags=flags,
        ))
    return players


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------


class OptimizationError(RuntimeError):
    pass


def optimize_squad(players: list[Player], *, budget: int, max_player_cost: int,
                   max_per_club: int = 3) -> list[Player]:
    """Select the 15 that maximise total score under all FPL constraints.

    budget / max_player_cost are in tenths of a million (to match now_cost).
    """
    # Only players who could ever be picked (price cap + not hard-out).
    pool = [p for p in players if p.cost <= max_player_cost and p.minutes_mult > 0.05]

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    pick = {p.id: pulp.LpVariable(f"pick_{p.id}", cat="Binary") for p in pool}
    by_id = {p.id: p for p in pool}

    # Objective: maximise total expected score.
    prob += pulp.lpSum(p.score * pick[p.id] for p in pool)

    # Squad size + exact position quotas.
    prob += pulp.lpSum(pick.values()) == SQUAD_SIZE
    for pos, quota in SQUAD_QUOTA.items():
        prob += pulp.lpSum(pick[p.id] for p in pool if p.position == pos) == quota

    # Budget.
    prob += pulp.lpSum(p.cost * pick[p.id] for p in pool) <= budget

    # Max players per real-world club.
    club_ids = {p.team_id for p in pool}
    for club in club_ids:
        prob += pulp.lpSum(
            pick[p.id] for p in pool if p.team_id == club
        ) <= max_per_club

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise OptimizationError(
            f"Solver returned status '{pulp.LpStatus[status]}'. Try relaxing "
            "--max-player-cost or --budget."
        )

    chosen = [by_id[pid] for pid, var in pick.items() if var.value() > 0.5]
    return chosen


def pick_starting_xi(squad: list[Player]) -> list[Player]:
    """Choose the best legal 11 from the 15 (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD)."""
    prob = pulp.LpProblem("fpl_xi", pulp.LpMaximize)
    start = {p.id: pulp.LpVariable(f"start_{p.id}", cat="Binary") for p in squad}
    by_id = {p.id: p for p in squad}

    prob += pulp.lpSum(p.score * start[p.id] for p in squad)
    prob += pulp.lpSum(start.values()) == 11
    for pos, (lo, hi) in XI_BOUNDS.items():
        pos_sum = pulp.lpSum(start[p.id] for p in squad if p.position == pos)
        prob += pos_sum >= lo
        prob += pos_sum <= hi

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return [by_id[pid] for pid, var in start.items() if var.value() > 0.5]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _formation(xi: list[Player]) -> str:
    counts = {pos: sum(1 for p in xi if p.position == pos) for pos in POSITIONS}
    return f"{counts[2]}-{counts[3]}-{counts[4]}"


def _flag_str(p: Player) -> str:
    return "; ".join(p.flags) if p.flags else "—"


def _squad_table(players: Iterable[Player]) -> str:
    rows = ["| Pos | Player | Club | £m | Score | Fixtures (avg FDR) | Notes |",
            "|-----|--------|------|----|-------|--------------------|-------|"]
    order = {1: 0, 2: 1, 3: 2, 4: 3}
    for p in sorted(players, key=lambda x: (order[x.position], -x.score)):
        fdr = f"{p.n_fixtures} ({p.avg_fdr:.1f})" if p.avg_fdr else f"{p.n_fixtures}"
        rows.append(
            f"| {p.position_name} | {p.name} | {p.team_short} | "
            f"{p.cost_m:.1f} | {p.score:.1f} | {fdr} | {_flag_str(p)} |"
        )
    return "\n".join(rows)


def build_report(squad: list[Player], xi: list[Player], *, budget: int,
                 max_player_cost: int, horizon: int) -> str:
    xi_ids = {p.id for p in xi}
    bench = [p for p in squad if p.id not in xi_ids]
    # Bench order: reserve GK first, then outfielders by descending score.
    bench_gk = [p for p in bench if p.position == 1]
    bench_out = sorted((p for p in bench if p.position != 1),
                       key=lambda p: -p.score)
    bench_ordered = bench_gk + bench_out

    captain = max(xi, key=lambda p: p.score)
    # Prefer a vice from a different club so a single blank/postponement can't
    # sink both armband picks; fall back to best remaining if none.
    other_club = [p for p in xi if p.team_id != captain.team_id]
    vice_pool = other_club or [p for p in xi if p.id != captain.id]
    vice = max(vice_pool, key=lambda p: p.score)

    total_cost = sum(p.cost for p in squad)
    bank = budget - total_cost

    # Premium spread: how many £8m+ assets, and the most expensive pick.
    premiums = sorted((p for p in squad if p.cost >= 80),
                      key=lambda p: -p.cost)
    most_expensive = max(squad, key=lambda p: p.cost)

    out = []
    out.append("# FPL Gameweek 1 — Optimised 15-Man Squad")
    out.append("")
    out.append(f"**Budget used:** £{total_cost/10:.1f}m of £{budget/10:.1f}m "
               f"(£{bank/10:.1f}m in the bank)  ")
    out.append(f"**Per-player price cap:** £{max_player_cost/10:.1f}m "
               f"(~{100*max_player_cost/budget:.0f}% of budget)  ")
    out.append(f"**Fixture horizon:** next {horizon} gameweeks  ")
    out.append(f"**Starting XI formation:** {_formation(xi)}")
    out.append("")

    out.append("## Full squad (15)")
    out.append("")
    out.append(_squad_table(squad))
    out.append("")

    out.append(f"## Suggested Starting XI ({_formation(xi)})")
    out.append("")
    out.append(_squad_table(xi))
    out.append("")

    out.append("## Bench (in priority order)")
    out.append("")
    rows = ["| Order | Player | Pos | Club | £m | Score |",
            "|-------|--------|-----|------|----|-------|"]
    for i, p in enumerate(bench_ordered, start=1):
        label = "GK" if p.position == 1 else str(i - len(bench_gk))
        rows.append(f"| {label} | {p.name} | {p.position_name} | "
                    f"{p.team_short} | {p.cost_m:.1f} | {p.score:.1f} |")
    out.append("\n".join(rows))
    out.append("")

    out.append("## Captaincy")
    out.append("")
    out.append(f"- **Captain: {captain.name} ({captain.team_short})** — "
               f"top projected scorer in the XI ({captain.score:.1f} over "
               f"{horizon} GWs, {captain.n_fixtures} fixtures, avg FDR "
               f"{captain.avg_fdr:.1f})." if captain.avg_fdr
               else f"- **Captain: {captain.name}** — top projected scorer.")
    out.append(f"- **Vice-captain: {vice.name} ({vice.team_short})** — "
               f"next-best projected scorer ({vice.score:.1f}), a differently "
               "fixtured backup so you're covered if the captain is rotated.")
    out.append("")

    # Strategy narrative.
    out.append("## Strategy")
    out.append("")
    prem_desc = ", ".join(f"{p.name} (£{p.cost_m:.1f}m)" for p in premiums[:4])
    out.append(
        f"Spend is spread across **{len(premiums)} premium/near-premium assets "
        f"(£8.0m+)** — {prem_desc} — rather than anchoring on one £15m+ "
        f"talisman. The most expensive pick is **{most_expensive.name} "
        f"(£{most_expensive.cost_m:.1f}m)**, kept at or under the "
        f"£{max_player_cost/10:.1f}m cap so no single player dominates the "
        "budget."
    )
    n_risky = sum(1 for p in squad if p.risky)
    out.append("")
    if n_risky == 0:
        minutes_line = ("Minutes security was enforced: every pick is currently "
                        "expected to play (no injury or rotation flags). ")
    else:
        noun = "pick carries" if n_risky == 1 else "picks carry"
        minutes_line = (f"Minutes security was enforced: {n_risky} {noun} an "
                        "injury/rotation flag (see the Notes column) — anyone "
                        "below 75% projected to play was down-weighted. ")
    out.append(
        minutes_line
        + "Value picks fund the premiums, and fixture difficulty over the next "
        f"{horizon} gameweeks is baked into every score (easier runs rank higher)."
    )
    out.append("")
    out.append("> Prices and availability are pulled live at run time — "
               "re-run right before the deadline to catch late price changes "
               "and team news.")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(*, budget_m: float, max_player_cost_m: float, horizon: int,
        offline: bool) -> str:
    bootstrap = fetch_data.get_bootstrap(refresh=not offline, offline=offline)
    fixtures = fetch_data.get_fixtures(refresh=not offline, offline=offline)

    players = build_players(bootstrap, fixtures, horizon)
    budget = int(round(budget_m * 10))
    max_player_cost = int(round(max_player_cost_m * 10))

    squad = optimize_squad(players, budget=budget, max_player_cost=max_player_cost)
    xi = pick_starting_xi(squad)
    return build_report(squad, xi, budget=budget,
                        max_player_cost=max_player_cost, horizon=horizon)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1 — build an optimal 15-man FPL GW1 squad.")
    parser.add_argument("--budget", type=float, default=100.0,
                        help="total budget in £m (default 100.0)")
    parser.add_argument("--max-player-cost", type=float, default=13.0,
                        help="hard cap on any single player's price in £m "
                             "(default 13.0 — lets £11-12m premiums anchor "
                             "while still barring the £14m+ superstars)")
    parser.add_argument("--horizon", type=int, default=5,
                        help="fixture look-ahead in gameweeks (default 5)")
    parser.add_argument("--offline", action="store_true",
                        help="use the cached ./data snapshot instead of "
                             "fetching live")
    args = parser.parse_args(argv)

    try:
        report = run(budget_m=args.budget,
                     max_player_cost_m=args.max_player_cost,
                     horizon=args.horizon, offline=args.offline)
    except (RuntimeError, OptimizationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
