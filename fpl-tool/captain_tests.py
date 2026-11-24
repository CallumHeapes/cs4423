"""captain_tests.py — backtests of captaincy rules on past gameweeks.

Every rule only uses information available before the gameweek it picks for:
form, fixtures, results and top managers' captain choices, which are made before
the deadline. Used by review_history.py.

Caveats: the top-manager list is today's (past standings can't be fetched), and
penalty duty and the top-manager list are current, not historical. The weighted
search is tuned on the first half of the season and checked on the second half.
"""

from __future__ import annotations

import itertools
import json
import os
import statistics
from collections import Counter

import fetch_data

TOP_MANAGERS = 50
STARTER_MINUTES = 60

RULES = {
    "Recent points (last 3 GWs)": ("recent_pts",),
    "Recent xGI (last 3 GWs)": ("recent_xgi",),
    "Season xG per 90": ("season_xg90",),
    "Top managers' captain, then recent points": ("top_share", "recent_pts"),
    "Penalty taker, then recent points": ("penalty", "recent_pts"),
    "Weakest opponent, then recent points": ("opp_conceded", "recent_pts"),
    "Home, then recent points": ("home", "recent_pts"),
    "Starters only, then recent points": ("starter", "recent_pts"),
}

COMPOSITE_FACTORS = ("recent_pts", "season_xg90", "home", "opp_conceded",
                     "top_share", "penalty", "starter")
COMPOSITE_LEVELS = (0, 1, 2)


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_cases(team_id: int, bootstrap: dict, finished: list[int],
                log_path: str | None = None) -> list[dict]:
    pos_of = {el["id"]: el["element_type"] for el in bootstrap["elements"]}
    team_of = {el["id"]: el["team"] for el in bootstrap["elements"]}
    pen_of = {el["id"]: el.get("penalties_order") for el in bootstrap["elements"]}
    live = {gw: {e["id"]: e["stats"] for e in fetch_data.get_event_live(gw)["elements"]}
            for gw in finished}

    def stat(gw: int, i: int, key: str) -> float:
        return _f(live[gw].get(i, {}).get(key))

    matches: dict[tuple[int, int], tuple[int, bool]] = {}
    conceded: dict[tuple[int, int], int] = {}
    for f in fetch_data.get_fixtures(refresh=True):
        ev = f["event"]
        if ev is None:
            continue
        matches[(ev, f["team_h"])] = (f["team_a"], True)
        matches[(ev, f["team_a"])] = (f["team_h"], False)
        if f.get("finished") and f.get("team_h_score") is not None:
            conceded[(ev, f["team_h"])] = f["team_a_score"]
            conceded[(ev, f["team_a"])] = f["team_h_score"]

    top_ids = fetch_data.get_top_manager_ids(TOP_MANAGERS)
    logged: dict[int, int] = {}
    if log_path and os.path.exists(log_path):
        with open(log_path) as f:
            logged = {int(k): v["captain"]["id"] for k, v in json.load(f)["runs"].items()}

    cases = []
    for idx, gw in enumerate(finished):
        prior = finished[:idx]
        if not prior:
            continue
        recent = prior[-3:]
        picks = fetch_data.get_entry_picks(gw, team_id)["picks"]
        cands = [pk["element"] for pk in picks
                 if pk["position"] <= 11 and pos_of[pk["element"]] in (3, 4)]
        if not cands:
            continue
        actual = next(pk["element"] for pk in picks if pk.get("is_captain"))

        top = fetch_data.get_manager_picks_bulk(top_ids, gw)
        cap_counts = Counter(pk["element"] for p in top.values()
                             for pk in p["picks"] if pk.get("is_captain"))
        n_top = max(len(top), 1)

        feats: dict[int, dict[str, float]] = {}
        for i in cands:
            opp, home = matches.get((gw, team_of[i]), (None, False))
            conc = [conceded[(e, opp)] for e in recent if (e, opp) in conceded]
            feats[i] = {
                "recent_pts": statistics.mean(stat(g, i, "total_points") for g in recent),
                "recent_xgi": statistics.mean(
                    stat(g, i, "expected_goal_involvements") for g in recent),
                "season_xg90": (sum(stat(g, i, "expected_goal_involvements") for g in prior)
                                / max(sum(stat(g, i, "minutes") for g in prior), 1) * 90),
                "home": 1.0 if home else 0.0,
                "opp_conceded": statistics.mean(conc) if conc else 1.0,
                "top_share": cap_counts.get(i, 0) / n_top,
                "penalty": 1.0 if pen_of.get(i) == 1 else 0.0,
                "starter": 1.0 if all(stat(g, i, "minutes") >= STARTER_MINUTES
                                      for g in recent) else 0.0,
            }

        cases.append({
            "gw": gw,
            "cands": cands,
            "feats": feats,
            "pts": {i: int(stat(gw, i, "total_points")) for i in cands},
            "actual_pts": int(stat(gw, actual, "total_points")),
            "logged": logged.get(gw) if logged.get(gw) in cands else None,
        })
    return cases


def _pick(case: dict, keys: tuple[str, ...]) -> int:
    return max(case["cands"], key=lambda i: tuple(case["feats"][i][k] for k in keys))


def _zs(values: dict[int, float]) -> dict[int, float]:
    xs = list(values.values())
    if len(xs) < 2:
        return {k: 0.0 for k in values}
    mean, sd = statistics.mean(xs), statistics.pstdev(xs)
    return {k: ((v - mean) / sd if sd else 0.0) for k, v in values.items()}


def _composite_pick(case: dict, weights: tuple[int, ...]) -> int:
    return max(case["cands"],
               key=lambda i: sum(w * case["z"][f][i]
                                 for f, w in zip(COMPOSITE_FACTORS, weights) if w))


def _avg(cases: list[dict], weights: tuple[int, ...]) -> float:
    return statistics.mean(c["pts"][_composite_pick(c, weights)] for c in cases)


def _describe(weights: tuple[int, ...]) -> str:
    parts = [f"{f}×{w}" for f, w in zip(COMPOSITE_FACTORS, weights) if w]
    return ", ".join(parts) or "none"


def _rule_table(cases: list[dict]) -> list[str]:
    rows: dict[str, list[float]] = {}
    for name, keys in RULES.items():
        rows[name] = [c["pts"][_pick(c, keys)] for c in cases]
    rows["Your actual captain"] = [c["actual_pts"] for c in cases]
    rows["Random attacker in XI (average)"] = [
        statistics.mean(c["pts"].values()) for c in cases]
    rows["Best attacker in XI (hindsight)"] = [max(c["pts"].values()) for c in cases]
    logged = [c for c in cases if c["logged"] is not None]
    rows["Model captain (logged)"] = [c["pts"][c["logged"]] for c in logged]

    out = ["| Rule | Avg pts | GWs |", "|------|---------|-----|"]
    for name, vals in sorted(((n, v) for n, v in rows.items() if v),
                             key=lambda kv: statistics.mean(kv[1]), reverse=True):
        out.append(f"| {name} | {statistics.mean(vals):.1f} | {len(vals)} |")
    return out


def _composite(cases: list[dict]) -> list[str]:
    if len(cases) < 4:
        return ["_Not enough gameweeks to split into a tuning and a checking period "
                "yet — this needs at least four._"]
    for c in cases:
        c["z"] = {f: _zs({i: c["feats"][i][f] for i in c["cands"]})
                  for f in COMPOSITE_FACTORS}
    half = max(len(cases) // 2, 2)
    tune, check = cases[:half], cases[half:]
    grid = [w for w in itertools.product(COMPOSITE_LEVELS, repeat=len(COMPOSITE_FACTORS))
            if any(w)]
    best_tune = max(grid, key=lambda w: _avg(tune, w))
    best_all = max(grid, key=lambda w: _avg(cases, w))

    out = [f"Tuning period: GWs {tune[0]['gw']}–{tune[-1]['gw']} · checking period: "
           f"GWs {check[0]['gw']}–{check[-1]['gw']}. Each factor is weighted 0, 1 or 2 "
           f"and scored relative to the other attackers in your XI.", "",
           f"- **Weights chosen on the tuning period:** {_describe(best_tune)}",
           f"  - tuning average **{_avg(tune, best_tune):.1f}** pts · "
           f"checking average **{_avg(check, best_tune):.1f}** pts",
           f"- On the checking period: your actual captain averages "
           f"**{statistics.mean(c['actual_pts'] for c in check):.1f}**, a random "
           f"attacker **{statistics.mean(statistics.mean(c['pts'].values()) for c in check):.1f}**.",
           f"- Best weights over the whole season (in-sample, so optimistic): "
           f"{_describe(best_all)} at **{_avg(cases, best_all):.1f}** pts.", "",
           "_If the checking average is no better than the random baseline, the "
           "weights are fitting noise. A single good season isn't enough to trust "
           "them._"]
    return out


def report(team_id: int, bootstrap: dict, finished: list[int],
           log_path: str | None = None) -> str:
    cases = build_cases(team_id, bootstrap, finished, log_path)
    if not cases:
        return ""
    out = ["## Captain rules, backtested", "",
           f"Average captain points over {len(cases)} gameweeks (base points, before the "
           "doubling). Each rule only uses information from before that gameweek.", ""]
    out += _rule_table(cases)
    out += ["", "_Random attacker is what a random pick would earn. Best attacker "
            "(hindsight) is the ceiling no rule can reach. The model captain only "
            "covers logged gameweeks._", "",
            "## Weighted captain score", ""]
    out += _composite(cases)
    out.append("")
    return "\n".join(out)
