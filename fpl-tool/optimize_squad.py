"""optimize_squad.py — Phase 1 pre-season FPL squad optimizer.

Builds an optimal 15-man Gameweek-1 squad from *live* FPL data under the
standard rules (2/5/5/3, £100.0m, max 3 per club) while deliberately spreading
spend rather than anchoring on a single £15m+ superstar.

Usage
-----
    python optimize_squad.py                 # fetch live data, optimize
    python optimize_squad.py --offline       # reuse ./data cache
    python optimize_squad.py --horizon 6     # opponent look-ahead in GWs
    python optimize_squad.py --differentials 2   # force >=2 sub-10%-owned picks
    python optimize_squad.py --max-player-cost 13.0

The model (what each player is scored on)
-----------------------------------------
Every player earns an *expected FPL points* score built from concrete returns,
not just price:

- Attacking threat: expected goals + expected assists per 90 (`expected_goals_
  per_90`, `expected_assists_per_90`), converted to points at the position's
  goal value (GK/DEF 6, MID 5, FWD 4) + 3 per assist. Penalty takers
  (`penalties_order == 1`) and set-piece takers get a threat bump.
- Clean sheets (defenders & keepers ranked further on this): a clean-sheet
  probability from the team's defensive strength (or `clean_sheets_per_90`
  once the season is underway), worth 4 pts for GK/DEF and 1 for MID.
- Keeper saves (`saves_per_90`) and a nailed-starter appearance floor.
- Opponent analysis over the next `--horizon` gameweeks: each fixture is
  weighted by the *opponent's* strength — attacking returns scale with how weak
  the opponent's defence is, clean-sheet returns with how weak their attack is
  (plus FDR). Double gameweeks and blanks fall out naturally.
- A convex price base so that pre-season (when all the per-90 stats are still 0)
  the optimizer still prefers a few genuine premium anchors over a flat spread.

A PuLP integer program then MAXIMISES total squad score subject to the FPL
rules PLUS a per-player price cap so no single talisman anchors the squad, and
an optional minimum number of low-ownership *differential* picks.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Iterable

import pulp

import fetch_data

# ---------------------------------------------------------------------------
# FPL scoring constants (2024/25+ rules)
# ---------------------------------------------------------------------------
GOAL_PTS = {1: 6, 2: 6, 3: 5, 4: 4}   # points per goal by position
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}     # points per clean sheet by position
ASSIST_PTS = 3
SAVE_PTS = 1 / 3                       # 1 pt per 3 keeper saves
APPEAR_PTS = 2.0                       # nailed starter appearance floor (60'+)

# ---------------------------------------------------------------------------
# Model weights & tunables (documented so they can be adjusted with intent).
# ---------------------------------------------------------------------------

# Performance signals add ON TOP of the convex price base.
W_ATTACK = 1.0       # xG/xA-derived expected attacking points (full value)
W_CS = 1.0           # clean-sheet expected points (full value)
W_EP_NEXT = 0.30     # FPL's own expected points for next GW
W_FORM = 0.20        # recent points-per-game form
W_PPG = 0.20         # season points-per-game
W_TOTAL = 0.05       # last-season cumulative total (weighted LOW on purpose)

# Convex price base: pre-season every per-90 stat is 0, so quality is driven by
# this term. Convexity (PRICE_EXP > 1) rewards premium ceiling so the optimizer
# prefers a few genuine premium anchors funded by cheap enablers (a "barbell").
PRICE_EXP = 1.3
PRICE_CONVEX_C = 0.18

# Threat bumps for set-piece duty (extra xG/xA per game).
PEN_XG_BONUS = 0.20        # first-choice penalty taker
SETPIECE_XA_BONUS = 0.05   # first-choice corner / free-kick taker

# Clean-sheet baseline probability (league average) before team/opponent
# adjustment; clamped to a sane range.
CS_BASE_PROB = 0.32
CS_PROB_CLAMP = (0.05, 0.70)

# Fixture ease multipliers are clamped so no single fixture dominates.
EASE_CLAMP = (0.70, 1.30)
FIX_SENSITIVITY = 0.15     # FDR-based general multiplier sensitivity

# A pick is a "differential" below this effective-ownership %.
DIFF_OWNERSHIP = 10.0

# Minutes-based reliability. Per-90 rates from tiny samples (pre-season cameos,
# bit-part players) are noise — a bit-part player with one goal involvement in a
# 200' cameo shows a wild per-90. Trust the rate in proportion to the minutes
# behind it (≈50% trust at MINUTES_ANCHOR minutes) and shrink the rest toward a
# modest price/position baseline. Also clamp per-90 to realistic ceilings so no
# freak cameo can dominate even before shrinkage.
MINUTES_ANCHOR = 900       # ~10 full matches for ~50% trust
XG90_CAP = 1.10            # elite ~0.7; leaves headroom
XA90_CAP = 0.70
ATTACK_BASELINE_C = 0.20   # baseline attacking pts/game ≈ C * £m * pos factor
ATTACK_POS_FACTOR = {1: 0.0, 2: 0.30, 3: 0.80, 4: 1.0}

# Last season's totals (minutes, xG, xA, clean sheets — from element-summary's
# history_past) are POOLED with the current season so that pre-season we score
# on a real sample instead of noise, and the current season takes over as games
# accrue. Last season is down-weighted (players change), and promoted-club or
# overseas players with no PL history just fall back to the price baseline.
LAST_SEASON_WEIGHT = 0.80

# Flag players whose last season had few minutes (known fringe/rotation/backup)
# so a cheap non-starter — e.g. a #2 keeper — is visible for the eye-test. Only
# applies to players with PL history; new signings have no history to judge.
NAILED_FLAG_MINUTES = 1000

# Nailed-minutes guard. The optimizer loves cheap enablers, but a cheap player
# who barely featured last season is usually a backup the model can't see is
# second-choice. Scale each score by how nailed they look: full trust at
# NAILED_FULL_MINUTES, down to NAILED_FLOOR for a near-zero-minute player. This
# directly serves the "prefer secure minutes, especially in defence" preference
# — a nailed £4.5m keeper now beats a 90-minute #3. Players with NO PL history
# (new signings, promoted clubs) can't be judged, so they get a mild neutral
# discount rather than the full penalty.
NAILED_FULL_MINUTES = 2000
NAILED_FLOOR = 0.50
NO_HISTORY_NAILED = 0.85

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}  # GK, DEF, MID, FWD (hard FPL squad rule)
SQUAD_SIZE = 15

# Starting XI: exactly 1 GK, never 5 at the back, always 2-3 up top and 3-5 in
# midfield (user preference). Total 11.
XI_BOUNDS = {1: (1, 1), 2: (3, 4), 3: (3, 5), 4: (2, 3)}

# ---------------------------------------------------------------------------
# FPL transfer / chip rules the wider tool is aware of (used fully in Phase 2).
# Phase 1 is a fresh build, so transfers are effectively UNLIMITED before the
# first deadline — see the note printed in the report.
# ---------------------------------------------------------------------------
TRANSFER_RULES = {
    "free_transfers_per_gw": 1,
    "max_saved_free_transfers": 5,   # FTs can be banked up to 5
    "extra_transfer_cost": 4,        # points hit per transfer beyond your FTs
    "gw1_unlimited_until_deadline": True,
    "chips": ["wildcard", "free_hit", "bench_boost", "triple_captain"],
}


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
    cost: int              # now_cost in tenths (125 => £12.5m)
    score: float           # expected points over the horizon
    attack_pts: float      # per-game expected attacking points (xG+xA)
    cs_pts: float          # per-game expected clean-sheet points
    xgi90: float           # expected goal involvements per 90 (goals+assists)
    n_fixtures: int
    avg_fdr: float | None
    attack_ease: float
    cs_ease: float
    chance: int | None
    status: str
    news: str
    selected_by: float
    is_pen_taker: bool = False
    is_setpiece: bool = False
    risky: bool = False
    differential: bool = False
    opponents: list[str] = field(default_factory=list)
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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Opponent analysis over the horizon
# ---------------------------------------------------------------------------


@dataclass
class TeamOutlook:
    n_fixtures: int
    avg_fdr: float | None
    attack_ease: float     # >1 == weak opponent defences (good for attackers)
    cs_ease: float         # >1 == weak opponent attacks (good for clean sheets)
    gen_ease: float        # FDR-based general multiplier
    opponents: list[str]   # e.g. ["ARS (H)", "bou (A)", ...]


def _fdr_multiplier(difficulty: int) -> float:
    return _clamp(1.0 + FIX_SENSITIVITY * (3 - difficulty), 0.6, 1.4)


def compute_team_outlook(fixtures: list[dict], teams: dict[int, dict],
                         horizon: int) -> dict[int, TeamOutlook]:
    """For each team, analyse its next `horizon` gameweeks vs opponent strength.

    Returns per-team attacking ease (weak opponent defences), clean-sheet ease
    (weak opponent attacks) and a general FDR multiplier, averaged over the
    fixtures in the horizon (double GWs add fixtures, blanks remove them).
    """
    # League-average strengths, used to normalise ease into a multiplier ~1.0.
    # Treat 0 as "not published yet" (FPL leaves strengths at 0 deep pre-season)
    # so we don't divide by zero — in that case ease falls back to a neutral 1.0
    # and only FDR drives the fixture weighting.
    def _avg(side_keys):
        vals = [t[k] for t in teams.values() for k in side_keys if t.get(k)]
        return sum(vals) / len(vals) if vals else 0.0

    avg_def = _avg(("strength_defence_home", "strength_defence_away"))
    avg_att = _avg(("strength_attack_home", "strength_attack_away"))

    def _safe_ease(league_avg: float, opp_strength: float) -> float:
        if not league_avg or not opp_strength:
            return 1.0
        return _clamp(league_avg / opp_strength, *EASE_CLAMP)

    upcoming = sorted({f["event"] for f in fixtures
                       if f["event"] is not None and not f.get("finished")})[:horizon]
    horizon_events = set(upcoming)

    acc: dict[int, dict] = {}
    for f in fixtures:
        if f["event"] not in horizon_events:
            continue
        for team_id, opp_id, is_home, diff in (
            (f["team_h"], f["team_a"], True, f["team_h_difficulty"]),
            (f["team_a"], f["team_h"], False, f["team_a_difficulty"]),
        ):
            opp = teams.get(opp_id, {})
            # Opponent plays the opposite venue to us.
            opp_def = opp.get("strength_defence_away" if is_home
                              else "strength_defence_home") or avg_def
            opp_att = opp.get("strength_attack_away" if is_home
                              else "strength_attack_home") or avg_att
            attack_ease = _safe_ease(avg_def, opp_def)
            cs_ease = _safe_ease(avg_att, opp_att)
            rec = acc.setdefault(team_id, {"fdr": [], "att": [], "cs": [],
                                           "gen": [], "opp": []})
            rec["fdr"].append(diff)
            rec["att"].append(attack_ease)
            rec["cs"].append(cs_ease)
            rec["gen"].append(_fdr_multiplier(diff))
            venue = "H" if is_home else "A"
            rec["opp"].append(f"{opp.get('short_name', '?')} ({venue})")

    outlook: dict[int, TeamOutlook] = {}
    for team_id, rec in acc.items():
        n = len(rec["fdr"])
        outlook[team_id] = TeamOutlook(
            n_fixtures=n,
            avg_fdr=sum(rec["fdr"]) / n if n else None,
            attack_ease=sum(rec["att"]) / n if n else 1.0,
            cs_ease=sum(rec["cs"]) / n if n else 1.0,
            gen_ease=sum(rec["gen"]) / n if n else 1.0,
            opponents=rec["opp"],
        )
    return outlook


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _minutes_multiplier(chance: int | None, status: str) -> tuple[float, bool,
                                                                   list[str]]:
    """Discount for rotation/injury risk. Returns (multiplier, risky, flags)."""
    flags: list[str] = []
    if status in ("i", "s", "u", "n"):
        flags.append({"i": "injured", "s": "suspended", "u": "unavailable",
                      "n": "not in squad"}[status])
        return 0.05, True, flags
    if chance is None:
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


def _nailed_factor(hist: dict | None) -> float:
    """Down-weight players who barely featured last season (likely backups).

    New signings / promoted-club players have no PL history to judge, so they
    get a mild neutral discount rather than the full penalty.
    """
    if hist is None:
        return NO_HISTORY_NAILED
    mins = hist.get("minutes", 0)
    return _clamp(NAILED_FLOOR + (1 - NAILED_FLOOR) * (mins / NAILED_FULL_MINUTES),
                  NAILED_FLOOR, 1.0)


def _effective_rates(el: dict, hist: dict | None
                     ) -> tuple[float, float, float, float]:
    """Pool current-season totals with last season into per-90 rates.

    Returns (xg90, xa90, cs90, effective_minutes). Pre-season the current-season
    totals are ~0 so the rates come from last season; the effective-minutes
    figure (pooled) drives how much we trust them.
    """
    cur_min = _to_float(el.get("minutes"))
    last_min = hist["minutes"] if hist else 0.0
    eff_min = cur_min + LAST_SEASON_WEIGHT * last_min
    if eff_min <= 0:
        return 0.0, 0.0, 0.0, 0.0

    def pooled(cur_key: str, last_key: str) -> float:
        cur = _to_float(el.get(cur_key))
        last = hist[last_key] if hist else 0.0
        return (cur + LAST_SEASON_WEIGHT * last) / (eff_min / 90.0)

    xg90 = min(pooled("expected_goals", "xg"), XG90_CAP)
    xa90 = min(pooled("expected_assists", "xa"), XA90_CAP)
    cs90 = pooled("clean_sheets", "cs")
    return xg90, xa90, cs90, eff_min


def _clean_sheet_prob(el: dict, team: dict, avg_def_strength: float,
                      cs90: float, eff_min: float) -> float:
    """Per-game clean-sheet probability, from a real per-90 sample when we have
    one, else the team's defensive strength, else the league baseline."""
    if eff_min > 0 and cs90 > 0:
        return _clamp(cs90, *CS_PROB_CLAMP)
    team_def = ((team.get("strength_defence_home") or 0)
                + (team.get("strength_defence_away") or 0)) / 2
    if not avg_def_strength or not team_def:
        return CS_BASE_PROB
    return _clamp(CS_BASE_PROB * (team_def / avg_def_strength), *CS_PROB_CLAMP)


def _attacking_points(el: dict, pos: int, xg90: float, xa90: float,
                      eff_min: float) -> tuple[float, float, bool, bool]:
    """Per-game expected attacking points from pooled xG/xA, with minutes-based
    shrinkage toward a price/position baseline. Returns
    (attack_pts, effective_xgi90, is_pen_taker, is_setpiece)."""
    is_pen = el.get("penalties_order") == 1
    is_sp = (el.get("direct_freekicks_order") == 1
             or el.get("corners_and_indirect_freekicks_order") == 1)
    if is_pen:
        xg90 += PEN_XG_BONUS
    if is_sp:
        xa90 += SETPIECE_XA_BONUS

    raw = xg90 * GOAL_PTS[pos] + xa90 * ASSIST_PTS
    # Trust the rate in proportion to the (pooled) minutes behind it; shrink the
    # rest toward a modest price/position baseline.
    reliability = eff_min / (eff_min + MINUTES_ANCHOR) if eff_min > 0 else 0.0
    cost_m = el["now_cost"] / 10.0
    baseline = ATTACK_BASELINE_C * cost_m * ATTACK_POS_FACTOR[pos]
    attack_pts = reliability * raw + (1 - reliability) * baseline
    return attack_pts, min(xg90 + xa90, XG90_CAP + XA90_CAP), is_pen, is_sp


def build_players(bootstrap: dict, fixtures: list[dict], horizon: int,
                  last_season: dict[int, dict] | None = None) -> list[Player]:
    teams = {t["id"]: t for t in bootstrap["teams"]}
    outlook = compute_team_outlook(fixtures, teams, horizon)
    last_season = last_season or {}
    avg_def_strength = (
        sum((t.get("strength_defence_home", 0) + t.get("strength_defence_away", 0)) / 2
            for t in teams.values()) / max(1, len(teams))
    ) or 1.0

    players: list[Player] = []
    for el in bootstrap["elements"]:
        pos = el["element_type"]
        team = teams.get(el["team"], {})
        ot = outlook.get(el["team"])
        n_fix = ot.n_fixtures if ot else 0
        attack_ease = ot.attack_ease if ot else 1.0
        cs_ease = ot.cs_ease if ot else 1.0
        gen_ease = ot.gen_ease if ot else 1.0

        chance = el.get("chance_of_playing_next_round")
        minutes_mult, risky, flags = _minutes_multiplier(chance, el.get("status", "a"))

        hist = last_season.get(el["id"])
        xg90, xa90, cs90, eff_min = _effective_rates(el, hist)
        attack_pts, xgi90, is_pen, is_sp = _attacking_points(el, pos, xg90, xa90, eff_min)
        cs_prob = _clean_sheet_prob(el, team, avg_def_strength, cs90, eff_min)
        cs_pts = cs_prob * CS_PTS[pos]

        cost_m = el["now_cost"] / 10.0
        base = PRICE_CONVEX_C * (cost_m ** PRICE_EXP)
        saves_pts = _to_float(el.get("saves_per_90")) * SAVE_PTS if pos == 1 else 0.0

        generic = (base + APPEAR_PTS
                   + W_EP_NEXT * _to_float(el.get("ep_next"))
                   + W_FORM * _to_float(el.get("form"))
                   + W_PPG * _to_float(el.get("points_per_game"))
                   + W_TOTAL * _to_float(el.get("total_points")) / 38.0)

        # Per-game expected points, fixture-adjusted by opponent strength.
        per_game = (generic * gen_ease
                    + W_ATTACK * attack_pts * attack_ease
                    + W_CS * cs_pts * cs_ease
                    + saves_pts)
        score = per_game * minutes_mult * n_fix * _nailed_factor(hist)

        selected_by = _to_float(el.get("selected_by_percent"))
        news = (el.get("news") or "").strip()
        if is_pen:
            flags.append("pen taker")
        if is_sp:
            flags.append("set pieces")
        if hist and hist["minutes"] < NAILED_FLAG_MINUTES:
            flags.append(f"{int(hist['minutes'])}m last yr")
        if news and not any(news[:20] in f for f in flags):
            flags.append(news if len(news) <= 44 else news[:41] + "...")

        players.append(Player(
            id=el["id"], name=el.get("web_name", f"#{el['id']}"),
            team_id=el["team"], team_short=team.get("short_name", "?"),
            position=pos, cost=el["now_cost"], score=score,
            attack_pts=attack_pts, cs_pts=cs_pts, xgi90=xgi90,
            n_fixtures=n_fix, avg_fdr=ot.avg_fdr if ot else None,
            attack_ease=attack_ease, cs_ease=cs_ease,
            chance=chance, status=el.get("status", "a"), news=news,
            selected_by=selected_by, is_pen_taker=is_pen, is_setpiece=is_sp,
            risky=risky, differential=selected_by < DIFF_OWNERSHIP,
            opponents=(ot.opponents if ot else []), flags=flags,
        ))
    return players


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------


class OptimizationError(RuntimeError):
    pass


def optimize_squad(players: list[Player], *, budget: int, max_player_cost: int,
                   max_per_club: int = 3, min_differentials: int = 0
                   ) -> list[Player]:
    """Select the 15 that maximise total score under all FPL constraints."""
    pool = [p for p in players if p.cost <= max_player_cost and p.score > 0]

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    pick = {p.id: pulp.LpVariable(f"pick_{p.id}", cat="Binary") for p in pool}
    by_id = {p.id: p for p in pool}

    prob += pulp.lpSum(p.score * pick[p.id] for p in pool)
    prob += pulp.lpSum(pick.values()) == SQUAD_SIZE
    for pos, quota in SQUAD_QUOTA.items():
        prob += pulp.lpSum(pick[p.id] for p in pool if p.position == pos) == quota
    prob += pulp.lpSum(p.cost * pick[p.id] for p in pool) <= budget
    for club in {p.team_id for p in pool}:
        prob += pulp.lpSum(pick[p.id] for p in pool if p.team_id == club) <= max_per_club
    if min_differentials > 0:
        prob += pulp.lpSum(pick[p.id] for p in pool if p.differential) >= min_differentials

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise OptimizationError(
            f"Solver returned '{pulp.LpStatus[status]}'. Try relaxing "
            "--max-player-cost, --budget, or --differentials.")

    return [by_id[pid] for pid, var in pick.items() if var.value() > 0.5]


def pick_starting_xi(squad: list[Player]) -> list[Player]:
    """Best legal 11 from the 15 (1 GK, 3-4 DEF, 3-5 MID, 2-3 FWD)."""
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
    rows = ["| Pos | Player | Club | £m | Score | xGI/90 | Own% | Fix (FDR) | Notes |",
            "|-----|--------|------|----|-------|--------|------|-----------|-------|"]
    order = {1: 0, 2: 1, 3: 2, 4: 3}
    for p in sorted(players, key=lambda x: (order[x.position], -x.score)):
        fdr = f"{p.n_fixtures} ({p.avg_fdr:.1f})" if p.avg_fdr else f"{p.n_fixtures}"
        rows.append(
            f"| {p.position_name} | {p.name} | {p.team_short} | {p.cost_m:.1f} | "
            f"{p.score:.1f} | {p.xgi90:.2f} | {p.selected_by:.1f} | {fdr} | "
            f"{_flag_str(p)} |")
    return "\n".join(rows)


def build_report(squad: list[Player], xi: list[Player], all_players: list[Player],
                 *, budget: int, max_player_cost: int, horizon: int,
                 min_differentials: int, n_history: int = 0) -> str:
    xi_ids = {p.id for p in xi}
    bench = [p for p in squad if p.id not in xi_ids]
    bench_gk = [p for p in bench if p.position == 1]
    bench_out = sorted((p for p in bench if p.position != 1), key=lambda p: -p.score)
    bench_ordered = bench_gk + bench_out

    # Captain/vice: attacking ceiling only (goals & assists win armbands), and
    # only midfielders/forwards — you never captain a defender or keeper.
    def _cap_key(p: Player) -> float:
        return p.attack_pts * p.attack_ease * p.n_fixtures
    cap_pool = [p for p in xi if p.position in (3, 4)] or xi
    captain = max(cap_pool, key=_cap_key)
    other_club = [p for p in cap_pool if p.team_id != captain.team_id]
    vice = max(other_club or [p for p in cap_pool if p.id != captain.id],
               key=_cap_key)

    total_cost = sum(p.cost for p in squad)
    bank = budget - total_cost
    premiums = sorted((p for p in squad if p.cost >= 80), key=lambda p: -p.cost)
    most_expensive = max(squad, key=lambda p: p.cost)

    out = ["# FPL Gameweek 1 — Optimised 15-Man Squad", ""]
    out.append(f"**Budget used:** £{total_cost/10:.1f}m of £{budget/10:.1f}m "
               f"(£{bank/10:.1f}m in the bank)  ")
    out.append(f"**Per-player price cap:** £{max_player_cost/10:.1f}m  ")
    out.append(f"**Opponent horizon:** next {horizon} gameweeks  ")
    if n_history:
        out.append(f"**Goals/assists basis:** last season pooled with current "
                   f"for {n_history} players  ")
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
        rows.append(f"| {label} | {p.name} | {p.position_name} | {p.team_short} | "
                    f"{p.cost_m:.1f} | {p.score:.1f} |")
    out.append("\n".join(rows))
    out.append("")

    # Opponent outlook per club in the squad.
    out.append(f"## Opponent outlook (next {horizon} GWs)")
    out.append("")
    rows = ["| Club | Fixtures | Avg FDR | Attack ease | CS ease | Opponents |",
            "|------|----------|---------|-------------|---------|-----------|"]
    seen = {}
    for p in squad:
        seen.setdefault(p.team_short, p)
    for short, p in sorted(seen.items(), key=lambda kv: -(kv[1].attack_ease)):
        fdr = f"{p.avg_fdr:.1f}" if p.avg_fdr else "—"
        opps = ", ".join(p.opponents) if p.opponents else "—"
        rows.append(f"| {short} | {p.n_fixtures} | {fdr} | {p.attack_ease:.2f}× | "
                    f"{p.cs_ease:.2f}× | {opps} |")
    out.append("\n".join(rows))
    out.append("")
    out.append("_Attack ease > 1.0 = soft opponent defences (good for goals); "
               "CS ease > 1.0 = soft opponent attacks (good for clean sheets)._")
    out.append("")

    out.append("## Captaincy")
    out.append("")
    cap_fix = f", avg FDR {captain.avg_fdr:.1f}" if captain.avg_fdr else ""
    out.append(f"- **Captain: {captain.name} ({captain.team_short})** — highest "
               f"attacking threat in the XI (xGI/90 {captain.xgi90:.2f}, "
               f"{captain.n_fixtures} fixtures{cap_fix}, attack ease "
               f"{captain.attack_ease:.2f}×)"
               + (" — on penalties." if captain.is_pen_taker else "."))
    out.append(f"- **Vice-captain: {vice.name} ({vice.team_short})** — next-best "
               "threat from a different club, so a single blank can't sink both.")
    out.append("")

    # Differentials.
    owned_ids = {p.id for p in squad}
    diffs = sorted((p for p in all_players
                    if p.differential and p.score > 0 and p.cost <= max_player_cost),
                   key=lambda p: -p.score)[:8]
    out.append("## Differentials to consider")
    out.append("")
    if min_differentials:
        n_in = sum(1 for p in squad if p.differential)
        out.append(f"_Squad currently holds {n_in} sub-{DIFF_OWNERSHIP:.0f}%-owned "
                   f"differential(s) (min requested: {min_differentials})._")
        out.append("")
    rows = ["| Player | Pos | Club | £m | Score | Own% | In squad? |",
            "|--------|-----|------|----|-------|------|-----------|"]
    for p in diffs:
        rows.append(f"| {p.name} | {p.position_name} | {p.team_short} | "
                    f"{p.cost_m:.1f} | {p.score:.1f} | {p.selected_by:.1f} | "
                    f"{'✅' if p.id in owned_ids else '—'} |")
    out.append("\n".join(rows))
    out.append("")
    out.append(f"_Low-ownership (< {DIFF_OWNERSHIP:.0f}%) high-projection picks. "
               "Pass `--differentials N` to force at least N into the squad._")
    out.append("")

    # Strategy.
    out.append("## Strategy")
    out.append("")
    prem_desc = ", ".join(f"{p.name} (£{p.cost_m:.1f}m)" for p in premiums[:4])
    out.append(
        f"Spend is spread across **{len(premiums)} premium/near-premium assets "
        f"(£8.0m+)** — {prem_desc} — rather than one £15m+ talisman. Priciest "
        f"pick **{most_expensive.name} (£{most_expensive.cost_m:.1f}m)**, held at "
        f"or under the £{max_player_cost/10:.1f}m cap.")
    n_risky = sum(1 for p in squad if p.risky)
    out.append("")
    if n_risky == 0:
        out.append("Minutes security was enforced: every pick is currently expected "
                   "to play (no injury or rotation flags). Attacking threat "
                   "(xG+xA), clean-sheet likelihood for defenders, penalty duty and "
                   f"opponent strength over the next {horizon} GWs are all baked "
                   "into every score.")
    else:
        noun = "pick carries" if n_risky == 1 else "picks carry"
        out.append(f"{n_risky} {noun} an injury/rotation flag (see Notes) and were "
                   "down-weighted. Attacking threat (xG+xA), clean-sheet likelihood, "
                   f"penalty duty and opponent strength over the next {horizon} GWs "
                   "are all baked into every score.")
    out.append("")

    # Transfer-rule awareness note.
    out.append("## Transfer rules (heads-up for the season)")
    out.append("")
    out.append("- **GW1 is a free build** — transfers are unlimited until the "
               "Friday 18:30 BST deadline, so re-run right before to lock in "
               "late prices/news.")
    out.append(f"- In-season you get **{TRANSFER_RULES['free_transfers_per_gw']} "
               f"free transfer per GW**, bankable up to "
               f"**{TRANSFER_RULES['max_saved_free_transfers']}**; every extra "
               f"transfer costs **-{TRANSFER_RULES['extra_transfer_cost']} pts**. "
               "Phase 2's weekly digest will respect your free-transfer count and "
               "only suggest hits when the projected gain beats the -4.")
    out.append("")
    out.append("> Prices and availability are pulled live at run time.")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(*, budget_m: float, max_player_cost_m: float, horizon: int,
        offline: bool, min_differentials: int, use_history: bool = True) -> str:
    bootstrap = fetch_data.get_bootstrap(refresh=not offline, offline=offline)
    fixtures = fetch_data.get_fixtures(refresh=not offline, offline=offline)

    last_season: dict[int, dict] = {}
    if use_history:
        ids = [el["id"] for el in bootstrap["elements"]]
        last_season = fetch_data.get_last_season_stats(
            ids, refresh=not offline, offline=offline,
            progress=lambda m: print(m, file=sys.stderr))
        if last_season:
            print(f"Last-season stats loaded for {len(last_season)} players.",
                  file=sys.stderr)

    players = build_players(bootstrap, fixtures, horizon, last_season)
    budget = int(round(budget_m * 10))
    max_player_cost = int(round(max_player_cost_m * 10))

    squad = optimize_squad(players, budget=budget, max_player_cost=max_player_cost,
                           min_differentials=min_differentials)
    xi = pick_starting_xi(squad)
    return build_report(squad, xi, players, budget=budget,
                        max_player_cost=max_player_cost, horizon=horizon,
                        min_differentials=min_differentials,
                        n_history=len(last_season))


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1 — build an optimal 15-man FPL GW1 squad.")
    parser.add_argument("--budget", type=float, default=100.0,
                        help="total budget in £m (default 100.0)")
    parser.add_argument("--max-player-cost", type=float, default=13.0,
                        help="hard cap on any single player's price in £m "
                             "(default 13.0 — lets £11-12m premiums anchor while "
                             "barring the £14m+ superstars)")
    parser.add_argument("--horizon", type=int, default=5,
                        help="opponent look-ahead in gameweeks (default 5; try 6)")
    parser.add_argument("--differentials", type=int, default=0,
                        help=f"force at least N sub-{DIFF_OWNERSHIP:.0f}%%-owned "
                             "picks into the squad (default 0)")
    parser.add_argument("--offline", action="store_true",
                        help="use the cached ./data snapshot instead of fetching live")
    parser.add_argument("--no-history", action="store_true",
                        help="skip the last-season pull (faster; weaker pre-season "
                             "goals/assists signal)")
    args = parser.parse_args(argv)

    try:
        report = run(budget_m=args.budget, max_player_cost_m=args.max_player_cost,
                     horizon=args.horizon, offline=args.offline,
                     min_differentials=args.differentials,
                     use_history=not args.no_history)
    except (RuntimeError, OptimizationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
