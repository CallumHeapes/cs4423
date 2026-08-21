"""weekly_digest.py — Phase 2 in-season weekly tracker for Corrib Athletic.

Run once before each deadline (after GW1 locks). It pulls your ACTUAL squad and
produces a single markdown digest: fixture outlook, flagged players (bad runs,
form dips, price-drop risk, availability), budget-matched transfer suggestions
that respect your free transfers and the -4 hit, and a captain pick.

    python weekly_digest.py                 # your team, next deadline
    python weekly_digest.py --free-transfers 2
    python weekly_digest.py --horizon 6 --team-id 5156799

It reuses the exact scoring model from optimize_squad.py, but with the
last-season weight decayed for the current gameweek (see optimize_squad.
weight_for_gw), so last year's data phases out automatically as the season runs.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import fetch_data
import optimize_squad as opt

# Flag thresholds.
FDR_BAD = 3.4              # avg fixture difficulty over the horizon
FORM_DIP_REL = 1.5        # form this far below season PPG = a dip
PRICE_DROP_NET = 40000    # net transfers out (this GW) that hints at a price drop
ALL_CHIPS = ["wildcard", "freehit", "bboost", "3xc"]  # 2x wildcard over a season
CHIP_LABELS = {"wildcard": "Wildcard", "freehit": "Free Hit",
               "bboost": "Bench Boost", "3xc": "Triple Captain"}

CHIP_LOOKAHEAD = 15        # gameweeks to scan for blanks / doubles
BLANK_ALERT = 4            # >= this many of your 15 blanking = a notable blank GW
FULL_SQUAD = 15


def _f(v, default: float = 0.0) -> float:
    try:
        return default if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return default


def _cap_key(p: opt.Player) -> float:
    return p.attack_pts * p.attack_ease * p.n_fixtures


def flag_reasons(p: opt.Player, el: dict, horizon: int) -> list[str]:
    """Reasons this owned player might warrant attention this week."""
    reasons: list[str] = []
    status = el.get("status", "a")
    chance = el.get("chance_of_playing_next_round")
    if status in ("i", "s", "u", "n"):
        label = {"i": "injured", "s": "suspended", "u": "unavailable",
                 "n": "not in squad"}[status]
        news = (el.get("news") or "").strip()
        reasons.append(f"availability: {news or label}")
    elif chance is not None and chance < 75:
        reasons.append(f"availability: {chance}% to play")

    if p.avg_fdr and p.avg_fdr >= FDR_BAD:
        reasons.append(f"tough run (avg FDR {p.avg_fdr:.1f} over next {horizon})")

    form, ppg = _f(el.get("form")), _f(el.get("points_per_game"))
    if ppg > 0 and form < ppg - FORM_DIP_REL:
        reasons.append(f"form dip (form {form:.1f} vs season {ppg:.1f})")

    net_out = _f(el.get("transfers_out_event")) - _f(el.get("transfers_in_event"))
    if net_out > PRICE_DROP_NET:
        reasons.append("price-drop risk (heavy net transfers out)")
    return reasons


def suggest_replacements(p: opt.Player, players: list[opt.Player], my_ids: set[int],
                         bank: int, club_counts: Counter, n: int = 2
                         ) -> list[opt.Player]:
    """Budget-matched, rule-legal upgrades for a flagged player.

    Budget = player's current price (approx sale value) + bank. Respects the
    max-3-per-club rule after accounting for the outgoing player.
    """
    budget = p.cost + bank
    out = []
    for q in players:
        if q.position != p.position or q.id in my_ids:
            continue
        if q.cost > budget or q.score <= p.score:
            continue
        # Club rule: swapping within the same club is fine; otherwise the new
        # club must currently hold < 3 of my players.
        if q.team_id != p.team_id and club_counts.get(q.team_id, 0) >= 3:
            continue
        out.append(q)
    out.sort(key=lambda q: -q.score)
    return out[:n]


def _team_gw_counts(fixtures, current_gw, lookahead):
    """team_id -> {gw: number of fixtures} for the scheduled GWs ahead.

    Only scheduled fixtures (event set) count. Blank/double gameweeks don't exist
    in the data until cup rounds are drawn mid-season, so pre-season every team
    shows exactly one fixture per GW and this correctly reports nothing notable.
    """
    hi = (current_gw or 0) + lookahead
    counts = {}
    for f in fixtures:
        ev = f["event"]
        if ev is None or (current_gw is not None and ev <= current_gw) or ev > hi:
            continue
        for tid in (f["team_h"], f["team_a"]):
            counts.setdefault(tid, {}).setdefault(ev, 0)
            counts[tid][ev] += 1
    return counts


def fixture_radar(fixtures, my_players, current_gw, lookahead=CHIP_LOOKAHEAD):
    """Per upcoming GW: which of my players blank (0 fixtures) or double (2+),
    and the total fixture load across my 15 (a Bench Boost score)."""
    counts = _team_gw_counts(fixtures, current_gw, lookahead)
    gws = sorted({f["event"] for f in fixtures if f["event"] is not None
                  and (current_gw is None or f["event"] > current_gw)
                  and f["event"] <= (current_gw or 0) + lookahead})
    radar = {}
    for gw in gws:
        blanks = [p for p in my_players if counts.get(p.team_id, {}).get(gw, 0) == 0]
        doubles = [p for p in my_players if counts.get(p.team_id, {}).get(gw, 0) >= 2]
        load = sum(counts.get(p.team_id, {}).get(gw, 0) for p in my_players)
        radar[gw] = {"blanks": blanks, "doubles": doubles, "load": load}
    return radar


def recommend_chips(radar, my_players, chips_available_labels):
    """Suggest when to play each still-available chip, from the fixture radar."""
    lines = []
    avail = set(chips_available_labels)
    if not radar:
        return lines

    def top_attackers(pool, k=3):
        return sorted((p for p in pool if p.position in (3, 4)),
                      key=lambda p: -p.attack_pts)[:k]

    # Bench Boost — the GW your 15 have the most fixtures (doubles pay double).
    if "Bench Boost" in avail:
        bb = max(radar, key=lambda g: radar[g]["load"])
        if radar[bb]["load"] > FULL_SQUAD:
            n_dbl = len(radar[bb]["doubles"])
            lines.append(f"- **Bench Boost → GW{bb}**: {n_dbl} of your 15 have a "
                         f"double ({radar[bb]['load']} total fixtures) — the most "
                         "points on the board from a full 15.")
        else:
            lines.append("- **Bench Boost**: hold — no double gameweek scheduled "
                         "yet where your squad plays twice.")

    # Triple Captain — a premium attacker with a double gameweek.
    if "Triple Captain" in avail:
        tc = None
        for gw in sorted(radar):
            dbl_att = [p for p in top_attackers(radar[gw]["doubles"], 5)]
            if dbl_att:
                tc = (gw, max(dbl_att, key=lambda p: p.attack_pts))
                break
        if tc:
            lines.append(f"- **Triple Captain → GW{tc[0]}**: {tc[1].name} "
                         f"({tc[1].team_short}) plays twice — captain haul on two "
                         "games.")
        else:
            lines.append("- **Triple Captain**: hold for a double gameweek for one "
                         "of your big attackers.")

    # Free Hit — the worst blank gameweek for you (field a one-week team).
    if "Free Hit" in avail:
        fh = max(radar, key=lambda g: len(radar[g]["blanks"]))
        n = len(radar[fh]["blanks"])
        if n >= BLANK_ALERT:
            lines.append(f"- **Free Hit → GW{fh}**: {n} of your 15 blank — field a "
                         "full temporary team instead of carrying blanks.")
        else:
            lines.append("- **Free Hit**: hold — no big blank gameweek scheduled "
                         "for your squad yet.")

    # Wildcard — restructure ahead of a blank-heavy stretch.
    if "Wildcard" in avail:
        worst = max(radar, key=lambda g: len(radar[g]["blanks"]))
        if len(radar[worst]["blanks"]) >= BLANK_ALERT:
            lines.append(f"- **Wildcard**: consider before GW{worst} to move players "
                         "off the blanking clubs (a permanent fix vs Free Hit's "
                         "one-week patch).")
        else:
            lines.append("- **Wildcard**: hold — no fixture pile-up forcing your "
                         "hand yet.")
    return lines


def _render_radar(radar, current_gw):
    notable = [(gw, r) for gw, r in sorted(radar.items())
               if r["blanks"] or r["doubles"]]
    out = ["## Fixture radar — blanks & doubles ahead", ""]
    if not notable:
        out.append("_No blank or double gameweeks scheduled for your squad yet. "
                   "These appear mid-season when cup rounds are drawn — this will "
                   "populate then._")
        out.append("")
        return "\n".join(out)
    rows = ["| GW | Your blanks | Your doubles |",
            "|----|-------------|--------------|"]
    for gw, r in notable:
        b = ", ".join(p.name for p in r["blanks"]) or "—"
        d = ", ".join(p.name for p in r["doubles"]) or "—"
        rows.append(f"| {gw} | {b} | {d} |")
    out.append("\n".join(rows))
    out.append("")
    return "\n".join(out)


def build_digest(*, team_id: int, horizon: int, free_transfers: int,
                 offline: bool) -> str:
    bootstrap = fetch_data.get_bootstrap(refresh=not offline, offline=offline)
    fixtures = fetch_data.get_fixtures(refresh=not offline, offline=offline)
    current_gw, next_gw = fetch_data.current_and_next_gw(bootstrap)

    if current_gw is None:
        return ("# Weekly digest\n\nNo gameweek has locked yet — Phase 2 works "
                "once GW1 is underway. Until then use `optimize_squad.py` to "
                "build your initial squad.")

    ids = [el["id"] for el in bootstrap["elements"]]
    last_season = fetch_data.get_last_season_stats(
        ids, refresh=not offline, offline=offline,
        progress=lambda m: print(m, file=sys.stderr))
    weight = opt.weight_for_gw(next_gw)
    elo_by_team = opt.map_elo_to_teams(
        fetch_data.get_club_elo(refresh=not offline, offline=offline),
        bootstrap["teams"])

    players = opt.build_players(bootstrap, fixtures, horizon, last_season, weight,
                               elo_by_team=elo_by_team)
    by_id = {p.id: p for p in players}
    el_by_id = {el["id"]: el for el in bootstrap["elements"]}

    picks = fetch_data.get_entry_picks(current_gw, team_id)
    entry = fetch_data.get_entry(team_id, refresh=not offline, offline=offline)
    history = fetch_data.get_entry_history(team_id)

    my_pick_ids = [pk["element"] for pk in picks["picks"]]
    my_ids = set(my_pick_ids)
    current_cap = next((pk["element"] for pk in picks["picks"] if pk.get("is_captain")), None)
    bank = picks.get("entry_history", {}).get("bank", 0)
    squad_value = picks.get("entry_history", {}).get("value", 0)
    chips_used = {c.get("name") for c in history.get("chips", [])}
    available_chips = [CHIP_LABELS[c] for c in ALL_CHIPS if c not in chips_used]

    my_players = [by_id[i] for i in my_pick_ids if i in by_id]
    club_counts = Counter(p.team_id for p in my_players)

    # Flag + suggest.
    flagged = []
    for p in my_players:
        reasons = flag_reasons(p, el_by_id.get(p.id, {}), horizon)
        if reasons:
            flagged.append((p, reasons, suggest_replacements(
                p, players, my_ids, bank, club_counts)))

    # Captain: best attacking threat among my mids/forwards.
    cap_pool = [p for p in my_players if p.position in (3, 4)] or my_players
    captain = max(cap_pool, key=_cap_key)
    vice_pool = [p for p in cap_pool if p.team_id != captain.team_id] or \
                [p for p in cap_pool if p.id != captain.id]
    vice = max(vice_pool, key=_cap_key)

    # Blank/double radar + chip predictor.
    radar = fixture_radar(fixtures, my_players, current_gw)
    radar_md = _render_radar(radar, current_gw)
    chip_lines = recommend_chips(radar, my_players, available_chips)
    chip_md = ""
    if available_chips:
        chip_md = "## Chip watch\n\n" + (
            "\n".join(chip_lines) if chip_lines else
            "_No chips available._") + "\n"

    return _render(entry, next_gw, bank, squad_value, free_transfers,
                   available_chips, my_players, flagged, captain, vice,
                   current_cap, weight, horizon, el_by_id,
                   extra=radar_md + "\n" + chip_md)


def _render(entry, next_gw, bank, squad_value, free_transfers, available_chips,
            my_players, flagged, captain, vice, current_cap, weight, horizon,
            el_by_id, extra="") -> str:
    name = entry.get("name", "My team")
    order = {1: 0, 2: 1, 3: 2, 4: 3}
    out = [f"# {name} — Gameweek {next_gw} digest", ""]
    out.append(f"**In the bank:** £{bank/10:.1f}m · **Squad value:** "
               f"£{squad_value/10:.1f}m · **Free transfers:** {free_transfers} "
               f"(extra transfers cost -4 pts)  ")
    out.append(f"**Chips available:** {', '.join(available_chips) or 'none'}  ")
    out.append(f"**Data basis:** last-season weight {weight:.2f} "
               f"(decaying to 0 by GW{opt.DECAY_GWS}) — current form takes over "
               "as the season runs.")
    out.append("")

    # Squad overview.
    out.append(f"## Your squad — next {horizon} GWs")
    out.append("")
    rows = ["| Pos | Player | Club | £m | Score | xGI/90 | Fix (FDR) | Watch |",
            "|-----|--------|------|----|-------|--------|-----------|-------|"]
    flag_map = {p.id: reasons for p, reasons, _ in flagged}
    for p in sorted(my_players, key=lambda x: (order[x.position], -x.score)):
        fdr = f"{p.n_fixtures} ({p.avg_fdr:.1f})" if p.avg_fdr else f"{p.n_fixtures}"
        watch = "⚠️" if p.id in flag_map else "—"
        rows.append(f"| {p.position_name} | {p.name} | {p.team_short} | "
                    f"{p.cost_m:.1f} | {p.score:.1f} | {p.xgi90:.2f} | {fdr} | {watch} |")
    out.append("\n".join(rows))
    out.append("")

    # Flags + transfer suggestions.
    out.append("## Flags & transfer suggestions")
    out.append("")
    if not flagged:
        out.append("No pressing concerns — every player is fit, in form, and has "
                   "a reasonable run. **Consider rolling your transfer** (bank it, "
                   "up to 5).")
    else:
        for p, reasons, repl in sorted(flagged, key=lambda t: -len(t[1])):
            out.append(f"**{p.name} ({p.team_short}, £{p.cost_m:.1f}m)** — "
                       + "; ".join(reasons))
            if repl:
                for q in repl:
                    el = el_by_id.get(q.id, {})
                    delta = q.score - p.score
                    extra = ((q.cost - p.cost) / 10)
                    money = ("free swap" if extra <= 0 else f"+£{extra:.1f}m")
                    out.append(f"  - → **{q.name} ({q.team_short}, £{q.cost_m:.1f}m)** "
                               f"— +{delta:.1f} projected over {horizon} GWs, {money}, "
                               f"form {_f(el.get('form')):.1f}, own {q.selected_by:.1f}%")
            else:
                out.append("  - _No budget-matched upgrade found — hold, or free up "
                           "funds elsewhere._")
            out.append("")

    # Transfer strategy.
    out.append("## Transfer plan")
    out.append("")
    n_flagged = len(flagged)
    if n_flagged == 0:
        out.append(f"- Nothing forced. Roll to bank a 2nd free transfer for next week.")
    else:
        out.append(f"- You have **{free_transfers} free transfer(s)**. "
                   f"{n_flagged} player(s) flagged.")
        if n_flagged > free_transfers:
            hits = n_flagged - free_transfers
            out.append(f"- Addressing all of them needs {hits} extra transfer(s) "
                       f"= **-{hits*4} pts**. Only take a hit if a suggested swap's "
                       "projected gain clearly beats 4 pts; otherwise do the single "
                       "highest-gain move and roll the rest.")
        else:
            out.append("- All flagged players can be addressed within your free "
                       "transfers — no points hit needed.")
    out.append("")

    # Captaincy.
    out.append(f"## Captaincy — GW{next_gw}")
    out.append("")
    cur = next((p for p in my_players if p.id == current_cap), None)
    if cur and cur.id != captain.id:
        out.append(f"- Your current captain is **{cur.name}**; the model prefers "
                   f"**{captain.name}** this week.")
    out.append(f"- **Captain: {captain.name} ({captain.team_short})** — top "
               f"attacking threat (xGI/90 {captain.xgi90:.2f}, {captain.n_fixtures} "
               f"fixtures"
               + (f", avg FDR {captain.avg_fdr:.1f}" if captain.avg_fdr else "")
               + (", on penalties" if captain.is_pen_taker else "") + ").")
    out.append(f"- **Vice: {vice.name} ({vice.team_short})** — backup threat from "
               "a different club.")
    out.append("")

    if extra.strip():
        out.append(extra.rstrip())
        out.append("")
    out.append("> Prices, availability and fixtures are pulled live at run time.")
    out.append("")
    return "\n".join(out)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2 — weekly in-season FPL digest for your team.")
    parser.add_argument("--team-id", type=int, default=fetch_data.default_team_id(),
                        help="FPL entry id (default from team_id.txt / 5156799)")
    parser.add_argument("--horizon", type=int, default=5,
                        help="fixture look-ahead in gameweeks (default 5; try 4-6)")
    parser.add_argument("--free-transfers", type=int, default=1,
                        help="your free transfers this week (the public API can't "
                             "report this reliably; default 1)")
    parser.add_argument("--offline", action="store_true",
                        help="use the cached ./data snapshot instead of fetching live")
    args = parser.parse_args(argv)

    try:
        digest = build_digest(team_id=args.team_id, horizon=args.horizon,
                              free_transfers=args.free_transfers, offline=args.offline)
    except (RuntimeError, opt.OptimizationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
