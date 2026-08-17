"""fpl_gw1_standalone.py — single-file Phase 1 FPL squad optimizer.

Everything in ONE file so it can be pasted into a phone environment (Google
Colab, Termux, a-Shell/Pyto on iOS, Replit...) with no sibling modules. It is
the same model as optimize_squad.py + fetch_data.py, just self-contained.

Run:
    pip install pulp requests
    python fpl_gw1_standalone.py

Colab (easiest on a phone — Google's servers reach the FPL API even if your
org network blocks it):
    Cell 1:  !pip install -q pulp requests
    Cell 2:  (paste this whole file, then press run)
"""

from __future__ import annotations
import time
import requests
import pulp

BASE_URL = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# ---- tunables ----
BUDGET_M = 100.0            # total budget
MAX_PLAYER_COST_M = 13.0    # hard per-player cap: lets £11-12m premiums anchor,
                            # still bars the £14m+ superstars (Haaland/Salah)
HORIZON = 5                 # fixture look-ahead in gameweeks

# Performance weights add ON TOP of a CONVEX price base (see quality_ppg). The
# convexity (PRICE_EXP > 1) is what makes the optimizer prefer a few premium
# anchors funded by cheap enablers instead of a flat mid-price spread.
W_EP_NEXT, W_FORM, W_PPG, W_TOTAL = 0.50, 0.15, 0.15, 0.05
PRICE_EXP, PRICE_CONVEX_C = 1.3, 0.18
FIX_SENSITIVITY = 0.15

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}
XI_BOUNDS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}


def _get(endpoint, retries=3, timeout=30):
    url = f"{BASE_URL}/{endpoint}/"
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Could not reach {url}: {last}\nIf you are on a network that blocks "
        "the FPL site, run this on mobile data or in Google Colab instead.")


def _f(v, default=0.0):
    try:
        return default if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return default


def minutes_multiplier(chance, status):
    flags = []
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


def quality_ppg(el):
    ep, form = _f(el.get("ep_next")), _f(el.get("form"))
    ppg, total = _f(el.get("points_per_game")), _f(el.get("total_points"))
    base = PRICE_CONVEX_C * ((el["now_cost"] / 10.0) ** PRICE_EXP)  # convex base
    return base + W_EP_NEXT * ep + W_FORM * form + W_PPG * ppg + W_TOTAL * total / 38.0


def fixture_multiplier(d):
    return max(0.6, min(1.4, 1.0 + FIX_SENSITIVITY * (3 - d)))


def team_fixtures(fixtures, horizon):
    events = sorted({f["event"] for f in fixtures
                     if f["event"] is not None and not f.get("finished")})[:horizon]
    events = set(events)
    per = {}
    for f in fixtures:
        if f["event"] not in events:
            continue
        per.setdefault(f["team_h"], []).append(f["team_h_difficulty"])
        per.setdefault(f["team_a"], []).append(f["team_a_difficulty"])
    return per


def build_players(boot, fixtures, horizon):
    teams = {t["id"]: t for t in boot["teams"]}
    tf = team_fixtures(fixtures, horizon)
    players = []
    for el in boot["elements"]:
        diffs = tf.get(el["team"], [])
        fsum = sum(fixture_multiplier(d) for d in diffs)
        avg_fdr = sum(diffs) / len(diffs) if diffs else None
        chance = el.get("chance_of_playing_next_round")
        mult, risky, flags = minutes_multiplier(chance, el.get("status", "a"))
        q = quality_ppg(el)
        news = (el.get("news") or "").strip()
        if news and news not in " ".join(flags):
            flags.append(news if len(news) <= 48 else news[:45] + "...")
        players.append({
            "id": el["id"], "name": el.get("web_name", f"#{el['id']}"),
            "team_id": el["team"], "club": teams.get(el["team"], {}).get("short_name", "?"),
            "pos": el["element_type"], "cost": el["now_cost"],
            "fdr": avg_fdr, "n_fix": len(diffs), "flags": flags, "risky": risky,
            "score": q * mult * fsum, "mult": mult,
        })
    return players


def solve_squad(players, budget, max_cost, max_per_club=3):
    pool = [p for p in players if p["cost"] <= max_cost and p["mult"] > 0.05]
    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    pick = {p["id"]: pulp.LpVariable(f"p{p['id']}", cat="Binary") for p in pool}
    by_id = {p["id"]: p for p in pool}
    prob += pulp.lpSum(p["score"] * pick[p["id"]] for p in pool)
    prob += pulp.lpSum(pick.values()) == 15
    for pos, q in SQUAD_QUOTA.items():
        prob += pulp.lpSum(pick[p["id"]] for p in pool if p["pos"] == pos) == q
    prob += pulp.lpSum(p["cost"] * pick[p["id"]] for p in pool) <= budget
    for club in {p["team_id"] for p in pool}:
        prob += pulp.lpSum(pick[p["id"]] for p in pool
                           if p["team_id"] == club) <= max_per_club
    if pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=0))] != "Optimal":
        raise RuntimeError("No optimal squad — relax the price cap or budget.")
    return [by_id[i] for i, v in pick.items() if v.value() > 0.5]


def solve_xi(squad):
    prob = pulp.LpProblem("xi", pulp.LpMaximize)
    st = {p["id"]: pulp.LpVariable(f"s{p['id']}", cat="Binary") for p in squad}
    by_id = {p["id"]: p for p in squad}
    prob += pulp.lpSum(p["score"] * st[p["id"]] for p in squad)
    prob += pulp.lpSum(st.values()) == 11
    for pos, (lo, hi) in XI_BOUNDS.items():
        s = pulp.lpSum(st[p["id"]] for p in squad if p["pos"] == pos)
        prob += s >= lo
        prob += s <= hi
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return [by_id[i] for i, v in st.items() if v.value() > 0.5]


def table(players):
    order = {1: 0, 2: 1, 3: 2, 4: 3}
    rows = ["| Pos | Player | Club | £m | Score | Fix (FDR) | Notes |",
            "|-----|--------|------|----|-------|-----------|-------|"]
    for p in sorted(players, key=lambda x: (order[x["pos"]], -x["score"])):
        fdr = f"{p['n_fix']} ({p['fdr']:.1f})" if p["fdr"] else f"{p['n_fix']}"
        notes = "; ".join(p["flags"]) if p["flags"] else "—"
        rows.append(f"| {POSITIONS[p['pos']]} | {p['name']} | {p['club']} | "
                    f"{p['cost']/10:.1f} | {p['score']:.1f} | {fdr} | {notes} |")
    return "\n".join(rows)


def report(squad, xi, budget, max_cost, horizon):
    xi_ids = {p["id"] for p in xi}
    bench = [p for p in squad if p["id"] not in xi_ids]
    bench = ([p for p in bench if p["pos"] == 1]
             + sorted((p for p in bench if p["pos"] != 1), key=lambda p: -p["score"]))
    cap = max(xi, key=lambda p: p["score"])
    others = [p for p in xi if p["team_id"] != cap["team_id"]] or \
             [p for p in xi if p["id"] != cap["id"]]
    vice = max(others, key=lambda p: p["score"])
    total = sum(p["cost"] for p in squad)
    prem = sorted((p for p in squad if p["cost"] >= 80), key=lambda p: -p["cost"])
    top = max(squad, key=lambda p: p["cost"])
    counts = {pos: sum(1 for p in xi if p["pos"] == pos) for pos in POSITIONS}
    formation = f"{counts[2]}-{counts[3]}-{counts[4]}"

    out = [f"# FPL GW1 — Optimised 15-Man Squad\n",
           f"**Budget:** £{total/10:.1f}m of £{budget/10:.1f}m "
           f"(£{(budget-total)/10:.1f}m bank) · **Cap:** £{max_cost/10:.1f}m "
           f"· **Horizon:** {horizon} GWs · **Formation:** {formation}\n",
           "## Full squad (15)\n", table(squad), "",
           f"## Starting XI ({formation})\n", table(xi), "",
           "## Bench (priority order)\n"]
    b = ["| # | Player | Pos | Club | £m | Score |", "|---|--------|-----|------|----|----|"]
    for i, p in enumerate(bench):
        lbl = "GK" if p["pos"] == 1 else str(i)
        b.append(f"| {lbl} | {p['name']} | {POSITIONS[p['pos']]} | {p['club']} "
                 f"| {p['cost']/10:.1f} | {p['score']:.1f} |")
    out += ["\n".join(b), "", "## Captaincy\n",
            f"- **Captain: {cap['name']} ({cap['club']})** — top projected "
            f"scorer ({cap['score']:.1f} over {horizon} GWs).",
            f"- **Vice: {vice['name']} ({vice['club']})** — best backup from a "
            "different club.", "", "## Strategy\n",
            f"Spend spread across **{len(prem)} premium/near-premium picks "
            f"(£8m+)** — " + ", ".join(f"{p['name']} (£{p['cost']/10:.1f}m)"
                                       for p in prem[:4]) +
            f" — not one talisman. Priciest pick **{top['name']} "
            f"(£{top['cost']/10:.1f}m)**, held under the £{max_cost/10:.1f}m cap.",
            "", "> Prices are live at run time — re-run before the deadline."]
    return "\n".join(out)


def main():
    print("Fetching live FPL data ...")
    boot = _get("bootstrap-static")
    fixtures = _get("fixtures")
    players = build_players(boot, fixtures, HORIZON)
    budget = int(round(BUDGET_M * 10))
    max_cost = int(round(MAX_PLAYER_COST_M * 10))
    squad = solve_squad(players, budget, max_cost)
    xi = solve_xi(squad)
    print("\n" + report(squad, xi, budget, max_cost, HORIZON))


if __name__ == "__main__":
    main()
