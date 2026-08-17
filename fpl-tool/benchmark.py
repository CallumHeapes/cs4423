"""benchmark.py — quality-check your squad against the field & elite managers.

FPL is a rank game: you win by beating other managers, so what the *best*
managers own matters more than raw projections. This pulls the top N managers
from the global league (id 314), aggregates what they own ("elite ownership"),
and shows where your team aligns, where you have a template hole (a highly-owned
pick you're missing = rank risk), and where you differ (your swing).

    python benchmark.py                 # your team vs the top 50 managers
    python benchmark.py --top 100       # vs the top 100
    python benchmark.py --team-id 5156799

Timing:
- In-season it uses your live squad and elite managers' actual picks.
- Pre-season (before GW1 locks) elite picks don't exist yet, so it compares
  your *proposed* squad (saved by optimize_squad.py) to overall ownership —
  run `python optimize_squad.py` first in the same session.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import fetch_data

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POS_ORDER = {1: 0, 2: 1, 3: 2, 4: 3}
ELITE_TEMPLATE = 0.40   # owned by >= this share of elite -> a hole if you miss it
ELITE_DIFF = 0.10       # you own, but <= this elite share -> a differential
OVERALL_TEMPLATE = 30.0  # pre-season fallback: overall selected_by %


def build_benchmark(*, team_id: int, top_n: int, offline: bool) -> str:
    boot = fetch_data.get_bootstrap(refresh=not offline, offline=offline)
    el = {e["id"]: e for e in boot["elements"]}
    teams = {t["id"]: t for t in boot["teams"]}

    def name(pid): return el.get(pid, {}).get("web_name", f"#{pid}")
    def club(pid): return teams.get(el.get(pid, {}).get("team"), {}).get("short_name", "?")
    def pos(pid): return el.get(pid, {}).get("element_type", 0)
    def price(pid): return el.get(pid, {}).get("now_cost", 0) / 10
    def overall(pid): return float(el.get(pid, {}).get("selected_by_percent") or 0)

    current_gw, _ = fetch_data.current_and_next_gw(boot)

    # --- your squad: live if the season's underway, else the proposed one ---
    my_ids, my_source = None, None
    if current_gw is not None:
        try:
            picks = fetch_data.get_entry_picks(current_gw, team_id)
            my_ids = [pk["element"] for pk in picks["picks"]]
            my_source = f"live team, GW{current_gw}"
        except RuntimeError:
            pass
    if my_ids is None:
        saved = fetch_data.load_squad(team_id)
        if saved:
            my_ids, my_source = saved, "proposed squad (from optimize_squad.py)"
    my_set = set(my_ids or [])

    # --- elite ownership (needs a locked GW so managers' picks exist) ---
    elite: dict[int, float] = {}
    cap = Counter()
    n_elite = 0
    if current_gw is not None:
        print(f"Fetching top {top_n} managers' squads...", file=sys.stderr)
        mgr_ids = fetch_data.get_top_manager_ids(top_n)
        picks_by = fetch_data.get_manager_picks_bulk(mgr_ids, current_gw)
        n_elite = len(picks_by)
        own = Counter()
        for p in picks_by.values():
            for pk in p["picks"]:
                own[pk["element"]] += 1
                if pk.get("is_captain"):
                    cap[pk["element"]] += 1
        elite = {pid: c / n_elite for pid, c in own.items()} if n_elite else {}

    if elite:
        return _render_elite(elite, cap, n_elite, my_set, my_source, current_gw,
                             name, club, pos, price, overall)
    return _render_preseason(el, my_set, my_source, name, club, pos, price, overall)


def _render_elite(elite, cap, n_elite, my_set, my_source, gw,
                  name, club, pos, price, overall) -> str:
    out = [f"# Squad benchmark — vs the top {n_elite} managers (GW{gw})", ""]
    out.append(f"**Your squad:** {my_source or 'not found'}  ")
    out.append("**Elite ownership** = share of the top managers who own a player "
               "(a sharper signal than overall ownership).")
    out.append("")

    top = sorted(elite.items(), key=lambda kv: -kv[1])[:15]
    out.append("## What the elite own (top 15)")
    out.append("")
    rows = ["| Player | Pos | Club | £m | Elite% | Overall% | You? |",
            "|--------|-----|------|----|--------|----------|------|"]
    for pid, share in top:
        rows.append(f"| {name(pid)} | {POS.get(pos(pid))} | {club(pid)} | "
                    f"{price(pid):.1f} | {share*100:.0f}% | {overall(pid):.1f}% | "
                    f"{'✅' if pid in my_set else '—'} |")
    out.append("\n".join(rows))
    out.append("")

    if cap:
        out.append("## Elite captaincy")
        out.append("")
        ct = sorted(cap.items(), key=lambda kv: -kv[1])[:5]
        out.append(", ".join(f"**{name(pid)}** {c/n_elite*100:.0f}%"
                             for pid, c in ct))
        out.append("")

    if my_set:
        held = [pid for pid, s in elite.items() if pid in my_set]
        core = [pid for pid, s in top if pid in my_set]
        holes = [(pid, s) for pid, s in elite.items()
                 if s >= ELITE_TEMPLATE and pid not in my_set]
        diffs = [(pid, elite.get(pid, 0)) for pid in my_set
                 if elite.get(pid, 0) <= ELITE_DIFF]
        holes.sort(key=lambda kv: -kv[1])
        diffs.sort(key=lambda kv: kv[1])

        out.append("## Your team vs the field")
        out.append("")
        out.append(f"- You own **{len(core)}/15** of the most-owned elite assets.")
        out.append("")
        out.append("**Template holes** (heavily elite-owned, you don't have — "
                   "rank risk if they haul):")
        if holes:
            for pid, s in holes[:8]:
                out.append(f"- {name(pid)} ({club(pid)}, £{price(pid):.1f}m) — "
                           f"{s*100:.0f}% elite-owned")
        else:
            out.append("- None — you cover every heavily-owned pick. Solid, but "
                       "low differentiation.")
        out.append("")
        out.append("**Your differentials** (you own, the elite mostly don't — "
                   "your swing for rank gains):")
        if diffs:
            for pid, s in diffs[:8]:
                out.append(f"- {name(pid)} ({club(pid)}, £{price(pid):.1f}m) — "
                           f"{s*100:.0f}% elite-owned")
        else:
            out.append("- None — you're on the template. Safe, but hard to climb.")
        out.append("")
    else:
        out.append("_No squad found to compare — run `optimize_squad.py` first, "
                   "or check the team id._")
        out.append("")
    out.append("> Elite ownership is pulled live at run time.")
    out.append("")
    return "\n".join(out)


def _render_preseason(el, my_set, my_source, name, club, pos, price, overall) -> str:
    out = ["# Squad benchmark — pre-season (overall ownership)", ""]
    out.append("Elite managers' picks don't exist until GW1 locks, so this "
               "compares against **overall ownership** (the whole field's plans). "
               "Re-run in-season for the sharper top-manager comparison.")
    out.append(f"  \n**Your squad:** {my_source or 'not found — run optimize_squad.py first'}")
    out.append("")

    ranked = sorted(el.values(), key=lambda e: -float(e.get("selected_by_percent") or 0))
    out.append("## Most-owned overall (the template, top 15)")
    out.append("")
    rows = ["| Player | Pos | Club | £m | Overall% | You? |",
            "|--------|-----|------|----|----------|------|"]
    for e in ranked[:15]:
        pid = e["id"]
        rows.append(f"| {name(pid)} | {POS.get(pos(pid))} | {club(pid)} | "
                    f"{price(pid):.1f} | {overall(pid):.1f}% | "
                    f"{'✅' if pid in my_set else '—'} |")
    out.append("\n".join(rows))
    out.append("")

    if my_set:
        template_ids = {e["id"] for e in ranked[:15]}
        core = template_ids & my_set
        holes = [e for e in ranked if e["id"] not in my_set
                 and float(e.get("selected_by_percent") or 0) >= OVERALL_TEMPLATE]
        diffs = sorted((pid for pid in my_set), key=lambda pid: overall(pid))
        out.append("## Your team vs the template")
        out.append("")
        out.append(f"- You hold **{len(core)}/15** of the most-owned players.")
        out.append("")
        out.append(f"**Template holes** (≥ {OVERALL_TEMPLATE:.0f}% owned, you don't have):")
        if holes:
            for e in holes[:8]:
                out.append(f"- {name(e['id'])} ({club(e['id'])}, £{price(e['id']):.1f}m)"
                           f" — {overall(e['id']):.1f}% owned")
        else:
            out.append("- None above the threshold.")
        out.append("")
        out.append("**Your lowest-owned picks** (your differentials):")
        for pid in diffs[:6]:
            out.append(f"- {name(pid)} ({club(pid)}, £{price(pid):.1f}m) — "
                       f"{overall(pid):.1f}% owned")
        out.append("")
    out.append("> Ownership is pulled live at run time.")
    out.append("")
    return "\n".join(out)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark your FPL squad against elite managers / the field.")
    parser.add_argument("--team-id", type=int, default=fetch_data.default_team_id())
    parser.add_argument("--top", type=int, default=50,
                        help="how many top managers to benchmark against (default 50)")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    try:
        print(build_benchmark(team_id=args.team_id, top_n=args.top,
                              offline=args.offline))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
