"""review_history.py — look back over your season, gameweek by gameweek.

    python review_history.py                                 # actual results per finished GW
    python review_history.py --log ~/.fpl_digest_log.json    # plus projected vs actual

Uses only the FPL API for the results, so any finished gameweek can be reviewed.
The projected-vs-actual section needs the log written by weekly_digest.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
from collections import Counter

import fetch_data
import optimize_squad as opt

DEFAULT_LOG = os.path.expanduser("~/.fpl_digest_log.json")


def hindsight_xi(players: list[tuple[int, int, int]]) -> tuple[int, list[int]]:
    """Best legal 11 by actual points from (id, position, points) rows."""
    best: tuple[int, list[int]] = (-1, [])
    for combo in itertools.combinations(players, 11):
        counts = Counter(pos for _, pos, _ in combo)
        if all(lo <= counts.get(pos, 0) <= hi
               for pos, (lo, hi) in opt.FPL_XI_BOUNDS.items()):
            total = sum(pts for _, _, pts in combo)
            if total > best[0]:
                best = (total, [pid for pid, _, _ in combo])
    return best


def review_results(team_id: int, bootstrap: dict, finished: list[int]) -> str:
    names = {el["id"]: el["web_name"] for el in bootstrap["elements"]}
    pos_of = {el["id"]: el["element_type"] for el in bootstrap["elements"]}
    history = {g["event"]: g for g in
               fetch_data.get_entry_history(team_id).get("current", [])}

    rows, transfers = [], []
    prev_ids: set[int] | None = None
    for gw in finished:
        if gw not in history:
            continue
        picks = fetch_data.get_entry_picks(gw, team_id)["picks"]
        live = {e["id"]: e["stats"]["total_points"]
                for e in fetch_data.get_event_live(gw)["elements"]}
        ids_now = {pk["element"] for pk in picks}
        starters = {pk["element"] for pk in picks if pk["position"] <= 11}
        cap = next(pk["element"] for pk in picks if pk.get("is_captain"))
        best_cap = max(starters, key=lambda i: live.get(i, 0))
        squad = [(i, pos_of[i], live.get(i, 0)) for i in ids_now]
        best_total, _ = hindsight_xi(squad)
        xi_actual = sum(live.get(i, 0) for i in starters)

        if prev_ids is not None:
            transfers.append((gw,
                              sorted(names[i] for i in ids_now - prev_ids),
                              sorted(names[i] for i in prev_ids - ids_now)))
        prev_ids = ids_now

        h = history[gw]
        rows.append({
            "gw": gw, "pts": h["points"], "hit": h.get("event_transfers_cost", 0),
            "bench": h.get("points_on_bench", 0), "rank": h.get("overall_rank"),
            "cap": names[cap], "cap_pts": live.get(cap, 0),
            "best_cap": names[best_cap], "best_cap_pts": live.get(best_cap, 0),
            "xi_lost": best_total - xi_actual,
        })

    if not rows:
        return "_No finished gameweeks with picks yet._"

    out = ["## Gameweek results", "",
           "| GW | Pts | Hit | Bench | Rank | Your captain (pts) | "
           "Best starter (pts) | XI pts lost |",
           "|----|-----|-----|-------|------|--------------------|"
           "--------------------|-------------|"]
    for r in rows:
        rank = f"{r['rank']:,}" if r["rank"] else "—"
        out.append(f"| {r['gw']} | {r['pts']} | {r['hit'] or '—'} | {r['bench']} | "
                   f"{rank} | {r['cap']} ({r['cap_pts']}) | "
                   f"{r['best_cap']} ({r['best_cap_pts']}) | {r['xi_lost']} |")

    cap_lost = sum(2 * (r["best_cap_pts"] - r["cap_pts"]) for r in rows)
    best_cap_hits = sum(r["cap"] == r["best_cap"] for r in rows)
    out += ["", "## Totals", "",
            f"- Points scored: **{sum(r['pts'] for r in rows)}** · "
            f"hits taken: **{sum(r['hit'] for r in rows)}** · "
            f"bench points: **{sum(r['bench'] for r in rows)}**",
            f"- Captain was the best starter in **{best_cap_hits} of {len(rows)}** "
            f"gameweeks; captain choice left about **{cap_lost}** points "
            "(captaincy doubles the points).",
            f"- Starting XI left about **{sum(r['xi_lost'] for r in rows)}** points "
            "on the table (hindsight best legal XI from your squad; ignores autosubs).",
            ""]

    if transfers:
        out += ["## Transfers by gameweek", ""]
        for gw, tin, tout in transfers:
            out.append(f"- **GW{gw}:** in {', '.join(tin) or '—'} · "
                       f"out {', '.join(tout) or '—'}")
        out.append("")
    return "\n".join(out)


def review_projections(log_path: str, team_id: int, bootstrap: dict,
                       finished: list[int]) -> str:
    names = {el["id"]: el["web_name"] for el in bootstrap["elements"]}
    with open(log_path) as f:
        runs = json.load(f)["runs"]

    xs, ys, cap_rows = [], [], []
    for gw_key, run in sorted(runs.items(), key=lambda kv: int(kv[0])):
        gw = int(gw_key)
        if gw not in finished:
            continue
        live = {e["id"]: e["stats"]["total_points"]
                for e in fetch_data.get_event_live(gw)["elements"]}
        owned = {pk["element"] for pk in
                 fetch_data.get_entry_picks(gw, team_id)["picks"]}
        proj = {int(k): v for k, v in run["projections"].items() if int(k) in owned}
        for pid, score in proj.items():
            xs.append(score)
            ys.append(live.get(pid, 0))
        cap_id = run["captain"]["id"]
        cap_pts = live.get(cap_id, 0) if cap_id in owned else None
        best_id = max(proj, key=lambda i: live.get(i, 0)) if proj else None
        cap_rows.append((gw, run["captain"]["name"], cap_pts,
                         names.get(best_id, "—"),
                         live.get(best_id, 0) if best_id else 0))

    if not xs:
        return ("## Projected vs actual\n\n_No logged runs match finished "
                "gameweeks yet — run weekly_digest.py before each deadline._\n")

    corr = statistics.correlation(xs, ys) if len(xs) > 2 else float("nan")
    out = ["## Projected vs actual (logged digests)", "",
           f"- Players compared: **{len(xs)}** · average projected "
           f"**{statistics.mean(xs):.1f}** vs average actual "
           f"**{statistics.mean(ys):.1f}**",
           f"- Correlation between projection and actual points: **{corr:.2f}** "
           "(1.0 = perfect ranking, 0 = no link).",
           "", "| GW | Model captain (pts) | Best owned player (pts) |",
           "|----|---------------------|-------------------------|"]
    for gw, cap_name, cap_pts, best_name, best_pts in cap_rows:
        cap_txt = f"{cap_name} ({cap_pts})" if cap_pts is not None else f"{cap_name} (not owned)"
        out.append(f"| {gw} | {cap_txt} | {best_name} ({best_pts}) |")
    out.append("")
    return "\n".join(out)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review your season gameweek by gameweek.")
    parser.add_argument("--team-id", type=int, default=fetch_data.default_team_id())
    parser.add_argument("--log", default=DEFAULT_LOG,
                        help="digest log written by weekly_digest.py (for projected vs actual)")
    args = parser.parse_args(argv)

    bootstrap = fetch_data.get_bootstrap(refresh=True)
    finished = [e["id"] for e in bootstrap["events"] if e.get("finished")]
    print(f"# Corrib Athletic — season review ({len(finished)} finished GWs)\n")
    print(review_results(args.team_id, bootstrap, finished))
    if os.path.exists(args.log):
        print(review_projections(args.log, args.team_id, bootstrap, finished))
    else:
        print(f"_No digest log at {args.log} — projected vs actual starts after the "
              "next weekly_digest.py run._")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
