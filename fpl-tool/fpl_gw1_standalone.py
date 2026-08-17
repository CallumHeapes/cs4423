"""fpl_gw1_standalone.py — single-file Phase 1 FPL squad optimizer.

Everything in ONE file so it can be pasted into a phone environment (Google
Colab, Termux, a-Shell/Pyto on iOS, Replit...) with no sibling modules. Same
model as optimize_squad.py + fetch_data.py, just self-contained.

Run:
    pip install pulp requests
    python fpl_gw1_standalone.py

Colab (easiest on a phone — Google's servers reach the FPL API even if your
org network blocks it):
    Cell 1:  !pip install -q pulp requests
    Cell 2:  (paste this whole file, then press run)

Scores every player on attacking threat (xG+xA -> points), clean-sheet
likelihood (defenders/keepers ranked further on this), penalty/set-piece duty,
and opponent strength over the next HORIZON gameweeks, on top of a convex price
base. Builds a 15-man squad (2/5/5/3, budget, max 3/club, per-player cap, and
an optional minimum of low-ownership differentials), a legal XI that never
starts 5 defenders (always 2-3 fwd, 3-5 mid), a bench, and captain picks.
"""

from __future__ import annotations
import argparse
import concurrent.futures
import time
import requests
import pulp

BASE_URL = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# ---- tunables ----
BUDGET_M = 100.0            # total budget
MAX_PLAYER_COST_M = 13.0    # per-player cap: lets £11-12m premiums anchor,
                            # bars the £14m+ superstars (Haaland/Salah)
HORIZON = 5                 # opponent look-ahead in gameweeks (try 6)
MIN_DIFFERENTIALS = 0       # force >= N sub-10%-owned picks into the squad
MIN_BANK_M = 2.0            # keep at least this much £m unspent
MAX_BANK_M = 5.0            # don't leave more than this much £m unspent
MIN_PREMIUMS = 2            # require >= N attacking (MID/FWD) picks at PREMIUM_COST_M+
PREMIUM_COST_M = 9.0        # price that counts as an attacking premium
BENCH_GK_MAX_M = 4.5        # require a cheap bench keeper at/below this £m

# FPL scoring
GOAL_PTS = {1: 6, 2: 6, 3: 5, 4: 4}
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}
ASSIST_PTS = 3
SAVE_PTS = 1 / 3
APPEAR_PTS = 2.0

# model weights (performance adds on top of a convex price base)
W_ATTACK, W_CS, W_EP_NEXT, W_FORM, W_PPG, W_TOTAL = 1.0, 1.0, 0.30, 0.20, 0.20, 0.05
PRICE_EXP, PRICE_CONVEX_C = 1.3, 0.18
PEN_XG_BONUS, SETPIECE_XA_BONUS = 0.20, 0.05
CS_BASE_PROB = 0.32
CS_PROB_CLAMP = (0.05, 0.70)
EASE_CLAMP = (0.70, 1.30)
FIX_SENSITIVITY = 0.15
DIFF_OWNERSHIP = 10.0
# Team-quality prior: temper attacking output by the player's OWN team attack
# strength vs the league average (a no-op until FPL publishes strength ratings,
# which encode the market's season expectation, then self-activates).
TEAM_ATT_CLAMP = (0.80, 1.20)
TEAM_WEAK_FLAG = 0.90

# Minutes-based reliability: shrink small-sample per-90 rates (pre-season cameos)
# toward a modest price/position baseline, and clamp per-90 to realistic ceilings.
MINUTES_ANCHOR = 900
XG90_CAP, XA90_CAP = 1.10, 0.70
ATTACK_BASELINE_C = 0.20
ATTACK_POS_FACTOR = {1: 0.0, 2: 0.30, 3: 0.80, 4: 1.0}
# Last season's totals (from element-summary history_past) are pooled with the
# current season so pre-season we score on a real sample, down-weighted since
# players change. Promoted-club / overseas players with no PL history fall back
# to the price baseline.
LAST_SEASON_WEIGHT = 0.80
USE_HISTORY = True          # set False (or --no-history) to skip the per-player pull
NAILED_FLAG_MINUTES = 1000  # flag players with few minutes last season (fringe)
# Nailed-minutes guard: down-weight cheap players who barely featured last season
# (likely backups). Full trust at NAILED_FULL_MINUTES; new signings/promoted (no
# history) get a mild neutral discount, not the full penalty.
NAILED_FULL_MINUTES, NAILED_FLOOR, NO_HISTORY_NAILED = 2000, 0.50, 0.85

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}
XI_BOUNDS = {1: (1, 1), 2: (3, 4), 3: (3, 5), 4: (2, 3)}  # never 5 DEF

TRANSFER_RULES = {"free_transfers_per_gw": 1, "max_saved_free_transfers": 5,
                  "extra_transfer_cost": 4}


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


def _fetch_one_history(pid):
    """Last completed season's totals for one player via history_past."""
    try:
        data = _get(f"element-summary/{pid}", retries=2, timeout=20)
    except RuntimeError:
        return pid, None
    past = data.get("history_past") or []
    if not past:
        return pid, None
    s = past[-1]
    return pid, {"minutes": _f(s.get("minutes")),
                 "xg": _f(s.get("expected_goals")) or _f(s.get("goals_scored")),
                 "xa": _f(s.get("expected_assists")) or _f(s.get("assists")),
                 "cs": _f(s.get("clean_sheets"))}


def get_last_season_stats(player_ids, workers=6):
    """Map player_id -> last-season {minutes, xg, xa, cs}. One call per player,
    threaded. Failures are skipped (those players use the price baseline)."""
    print(f"Fetching last-season stats for {len(player_ids)} players "
          "(one-off, ~30-60s)...")
    out, done = {}, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for pid, rec in ex.map(_fetch_one_history, player_ids):
            if rec:
                out[pid] = rec
            done += 1
            if done % 150 == 0:
                print(f"  ...{done}/{len(player_ids)}")
    print(f"Last-season stats loaded for {len(out)} players.")
    return out


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


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


def fdr_multiplier(d):
    return _clamp(1.0 + FIX_SENSITIVITY * (3 - d), 0.6, 1.4)


def team_outlook(fixtures, teams, horizon):
    """Per team: n_fix, avg_fdr, attack_ease, cs_ease, gen_ease, opponents."""
    # Treat 0 as "not published yet" (FPL leaves strengths at 0 deep pre-season)
    # so ease falls back to a neutral 1.0 and only FDR drives the weighting.
    def avg(keys):
        vals = [t[k] for t in teams.values() for k in keys if t.get(k)]
        return sum(vals) / len(vals) if vals else 0.0
    avg_def = avg(("strength_defence_home", "strength_defence_away"))
    avg_att = avg(("strength_attack_home", "strength_attack_away"))

    def ease(league_avg, opp):
        return 1.0 if (not league_avg or not opp) else _clamp(league_avg / opp, *EASE_CLAMP)

    events = sorted({f["event"] for f in fixtures
                     if f["event"] is not None and not f.get("finished")})[:horizon]
    events = set(events)
    acc = {}
    for f in fixtures:
        if f["event"] not in events:
            continue
        for tid, oid, home, diff in (
            (f["team_h"], f["team_a"], True, f["team_h_difficulty"]),
            (f["team_a"], f["team_h"], False, f["team_a_difficulty"]),
        ):
            opp = teams.get(oid, {})
            opp_def = opp.get("strength_defence_away" if home
                              else "strength_defence_home") or avg_def
            opp_att = opp.get("strength_attack_away" if home
                              else "strength_attack_home") or avg_att
            rec = acc.setdefault(tid, {"fdr": [], "att": [], "cs": [], "gen": [], "opp": []})
            rec["fdr"].append(diff)
            rec["att"].append(ease(avg_def, opp_def))
            rec["cs"].append(ease(avg_att, opp_att))
            rec["gen"].append(fdr_multiplier(diff))
            rec["opp"].append(f"{opp.get('short_name', '?')} ({'H' if home else 'A'})")
    out = {}
    for tid, rec in acc.items():
        n = len(rec["fdr"])
        out[tid] = {"n": n, "fdr": sum(rec["fdr"]) / n if n else None,
                    "att": sum(rec["att"]) / n if n else 1.0,
                    "cs": sum(rec["cs"]) / n if n else 1.0,
                    "gen": sum(rec["gen"]) / n if n else 1.0, "opp": rec["opp"]}
    return out, avg_def


def team_attack_mult(team, avg_att):
    own = ((team.get("strength_attack_home") or 0)
           + (team.get("strength_attack_away") or 0)) / 2
    if not avg_att or not own:
        return 1.0
    return _clamp(own / avg_att, *TEAM_ATT_CLAMP)


def nailed_factor(hist):
    if hist is None:
        return NO_HISTORY_NAILED
    m = hist.get("minutes", 0)
    return _clamp(NAILED_FLOOR + (1 - NAILED_FLOOR) * (m / NAILED_FULL_MINUTES),
                  NAILED_FLOOR, 1.0)


def effective_rates(el, hist):
    """Pool current-season totals with last season -> (xg90, xa90, cs90, eff_min)."""
    cur_min = _f(el.get("minutes"))
    last_min = hist["minutes"] if hist else 0.0
    eff_min = cur_min + LAST_SEASON_WEIGHT * last_min
    if eff_min <= 0:
        return 0.0, 0.0, 0.0, 0.0
    def pooled(cur_key, last_key):
        cur = _f(el.get(cur_key))
        last = hist[last_key] if hist else 0.0
        return (cur + LAST_SEASON_WEIGHT * last) / (eff_min / 90.0)
    return (min(pooled("expected_goals", "xg"), XG90_CAP),
            min(pooled("expected_assists", "xa"), XA90_CAP),
            pooled("clean_sheets", "cs"), eff_min)


def clean_sheet_prob(el, team, avg_def, cs90, eff_min):
    if eff_min > 0 and cs90 > 0:
        return _clamp(cs90, *CS_PROB_CLAMP)
    tdef = ((team.get("strength_defence_home") or 0)
            + (team.get("strength_defence_away") or 0)) / 2
    if not avg_def or not tdef:   # strengths not published yet -> baseline
        return CS_BASE_PROB
    return _clamp(CS_BASE_PROB * (tdef / avg_def), *CS_PROB_CLAMP)


def attacking_points(el, pos, xg, xa, eff_min):
    is_pen = el.get("penalties_order") == 1
    is_sp = (el.get("direct_freekicks_order") == 1
             or el.get("corners_and_indirect_freekicks_order") == 1)
    if is_pen:
        xg += PEN_XG_BONUS
    if is_sp:
        xa += SETPIECE_XA_BONUS
    raw = xg * GOAL_PTS[pos] + xa * ASSIST_PTS
    rel = eff_min / (eff_min + MINUTES_ANCHOR) if eff_min > 0 else 0.0
    baseline = ATTACK_BASELINE_C * (el["now_cost"] / 10.0) * ATTACK_POS_FACTOR[pos]
    attack = rel * raw + (1 - rel) * baseline
    return attack, min(xg + xa, XG90_CAP + XA90_CAP), is_pen, is_sp


def build_players(boot, fixtures, horizon, last_season=None):
    teams = {t["id"]: t for t in boot["teams"]}
    outlook, avg_def = team_outlook(fixtures, teams, horizon)
    last_season = last_season or {}
    att_vals = [v for t in teams.values()
                for v in (t.get("strength_attack_home"), t.get("strength_attack_away")) if v]
    avg_att = sum(att_vals) / len(att_vals) if att_vals else 0.0
    players = []
    for el in boot["elements"]:
        pos = el["element_type"]
        team = teams.get(el["team"], {})
        ot = outlook.get(el["team"], {})
        n_fix = ot.get("n", 0)
        att_ease, cs_ease, gen_ease = ot.get("att", 1.0), ot.get("cs", 1.0), ot.get("gen", 1.0)

        chance = el.get("chance_of_playing_next_round")
        mult, risky, flags = minutes_multiplier(chance, el.get("status", "a"))
        hist = last_season.get(el["id"])
        xg90, xa90, cs90, eff_min = effective_rates(el, hist)
        attack_pts, xgi, is_pen, is_sp = attacking_points(el, pos, xg90, xa90, eff_min)
        tam = team_attack_mult(team, avg_att)
        attack_pts *= tam
        cs_pts = clean_sheet_prob(el, team, avg_def, cs90, eff_min) * CS_PTS[pos]

        cost_m = el["now_cost"] / 10.0
        base = PRICE_CONVEX_C * (cost_m ** PRICE_EXP)
        saves = _f(el.get("saves_per_90")) * SAVE_PTS if pos == 1 else 0.0
        generic = (base + APPEAR_PTS + W_EP_NEXT * _f(el.get("ep_next"))
                   + W_FORM * _f(el.get("form")) + W_PPG * _f(el.get("points_per_game"))
                   + W_TOTAL * _f(el.get("total_points")) / 38.0)
        per_game = (generic * gen_ease + W_ATTACK * attack_pts * att_ease
                    + W_CS * cs_pts * cs_ease + saves)
        score = per_game * mult * n_fix * nailed_factor(hist)

        own = _f(el.get("selected_by_percent"))
        news = (el.get("news") or "").strip()
        if is_pen:
            flags.append("pen taker")
        if is_sp:
            flags.append("set pieces")
        if pos in (3, 4) and tam <= TEAM_WEAK_FLAG:
            flags.append("weak team attack")
        if hist and hist["minutes"] < NAILED_FLAG_MINUTES:
            flags.append(f"{int(hist['minutes'])}m last yr")
        if news and not any(news[:20] in x for x in flags):
            flags.append(news if len(news) <= 44 else news[:41] + "...")

        players.append({
            "id": el["id"], "name": el.get("web_name", f"#{el['id']}"),
            "team_id": el["team"], "club": team.get("short_name", "?"),
            "pos": pos, "cost": el["now_cost"], "score": score,
            "attack_pts": attack_pts, "cs_pts": cs_pts, "xgi": xgi,
            "n_fix": n_fix, "fdr": ot.get("fdr"), "att_ease": att_ease,
            "cs_ease": cs_ease, "own": own, "is_pen": is_pen,
            "diff": own < DIFF_OWNERSHIP, "opp": ot.get("opp", []),
            "flags": flags, "mult": mult, "risky": risky,
        })
    return players


def solve_squad(players, budget, max_cost, min_diff=0, max_per_club=3,
                min_bank=0, max_bank=None, min_premiums=0, premium_cost=90,
                bench_gk_max=None):
    pool = [p for p in players if p["cost"] <= max_cost and p["score"] > 0]
    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    pick = {p["id"]: pulp.LpVariable(f"p{p['id']}", cat="Binary") for p in pool}
    by_id = {p["id"]: p for p in pool}
    prob += pulp.lpSum(p["score"] * pick[p["id"]] for p in pool)
    prob += pulp.lpSum(pick.values()) == 15
    for pos, q in SQUAD_QUOTA.items():
        prob += pulp.lpSum(pick[p["id"]] for p in pool if p["pos"] == pos) == q
    total = pulp.lpSum(p["cost"] * pick[p["id"]] for p in pool)
    prob += total <= budget - min_bank
    if max_bank is not None:
        prob += total >= budget - max_bank
    for club in {p["team_id"] for p in pool}:
        prob += pulp.lpSum(pick[p["id"]] for p in pool if p["team_id"] == club) <= max_per_club
    if min_diff > 0:
        prob += pulp.lpSum(pick[p["id"]] for p in pool if p["diff"]) >= min_diff
    if min_premiums > 0:
        prob += pulp.lpSum(pick[p["id"]] for p in pool
                           if p["cost"] >= premium_cost and p["pos"] in (3, 4)) >= min_premiums
    if bench_gk_max is not None:
        prob += pulp.lpSum(pick[p["id"]] for p in pool
                           if p["pos"] == 1 and p["cost"] <= bench_gk_max) >= 1
    if pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=0))] != "Optimal":
        raise RuntimeError("No optimal squad — relax the cap, bank range, "
                           "premiums, or differentials.")
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
    rows = ["| Pos | Player | Club | £m | Score | xGI/90 | Own% | Fix (FDR) | Notes |",
            "|-----|--------|------|----|-------|--------|------|-----------|-------|"]
    for p in sorted(players, key=lambda x: (order[x["pos"]], -x["score"])):
        fdr = f"{p['n_fix']} ({p['fdr']:.1f})" if p["fdr"] else f"{p['n_fix']}"
        notes = "; ".join(p["flags"]) if p["flags"] else "—"
        rows.append(f"| {POSITIONS[p['pos']]} | {p['name']} | {p['club']} | "
                    f"{p['cost']/10:.1f} | {p['score']:.1f} | {p['xgi']:.2f} | "
                    f"{p['own']:.1f} | {fdr} | {notes} |")
    return "\n".join(rows)


def cap_key(p):
    # attacking ceiling only — you never captain a defender or keeper
    return p["attack_pts"] * p["att_ease"] * p["n_fix"]


def report(squad, xi, allp, budget, max_cost, horizon, min_diff, n_history=0):
    xi_ids = {p["id"] for p in xi}
    bench = [p for p in squad if p["id"] not in xi_ids]
    bench = ([p for p in bench if p["pos"] == 1]
             + sorted((p for p in bench if p["pos"] != 1), key=lambda p: -p["score"]))
    cap_pool = [p for p in xi if p["pos"] in (3, 4)] or xi
    cap = max(cap_pool, key=cap_key)
    others = [p for p in cap_pool if p["team_id"] != cap["team_id"]] or \
             [p for p in cap_pool if p["id"] != cap["id"]]
    vice = max(others, key=cap_key)
    total = sum(p["cost"] for p in squad)
    prem = sorted((p for p in squad if p["cost"] >= 80), key=lambda p: -p["cost"])
    top = max(squad, key=lambda p: p["cost"])
    counts = {pos: sum(1 for p in xi if p["pos"] == pos) for pos in POSITIONS}
    formation = f"{counts[2]}-{counts[3]}-{counts[4]}"

    hist_note = (f" · **G/A basis:** last season pooled for {n_history} players"
                 if n_history else "")
    out = ["# FPL Gameweek 1 — Optimised 15-Man Squad\n",
           f"**Budget:** £{total/10:.1f}m of £{budget/10:.1f}m "
           f"(£{(budget-total)/10:.1f}m bank) · **Cap:** £{max_cost/10:.1f}m "
           f"· **Horizon:** {horizon} GWs · **Formation:** {formation}{hist_note}\n",
           "## Full squad (15)\n", table(squad), "",
           f"## Starting XI ({formation})\n", table(xi), "",
           "## Bench (priority order)\n"]
    b = ["| # | Player | Pos | Club | £m | Score |", "|---|--------|-----|------|----|----|"]
    n_gk = sum(1 for p in bench if p["pos"] == 1)
    for i, p in enumerate(bench):
        lbl = "GK" if p["pos"] == 1 else str(i + 1 - n_gk)
        b.append(f"| {lbl} | {p['name']} | {POSITIONS[p['pos']]} | {p['club']} "
                 f"| {p['cost']/10:.1f} | {p['score']:.1f} |")
    out += ["\n".join(b), "", f"## Opponent outlook (next {horizon} GWs)\n"]

    seen = {}
    for p in squad:
        seen.setdefault(p["club"], p)
    o = ["| Club | Fixtures | Avg FDR | Attack ease | CS ease | Opponents |",
         "|------|----------|---------|-------------|---------|-----------|"]
    for club, p in sorted(seen.items(), key=lambda kv: -kv[1]["att_ease"]):
        fdr = f"{p['fdr']:.1f}" if p["fdr"] else "—"
        opps = ", ".join(p["opp"]) if p["opp"] else "—"
        o.append(f"| {club} | {p['n_fix']} | {fdr} | {p['att_ease']:.2f}× "
                 f"| {p['cs_ease']:.2f}× | {opps} |")
    out += ["\n".join(o), "",
            "_Attack ease > 1.0 = soft opponent defences (goals); CS ease > 1.0 "
            "= soft opponent attacks (clean sheets)._", "", "## Captaincy\n",
            f"- **Captain: {cap['name']} ({cap['club']})** — top attacking threat "
            f"(xGI/90 {cap['xgi']:.2f}, {cap['n_fix']} fixtures, attack ease "
            f"{cap['att_ease']:.2f}×)" + (" — on penalties." if cap["is_pen"] else "."),
            f"- **Vice: {vice['name']} ({vice['club']})** — next-best threat from "
            "a different club.", "", "## Differentials to consider\n"]

    owned = {p["id"] for p in squad}
    diffs = sorted((p for p in allp if p["diff"] and p["score"] > 0
                    and p["cost"] <= max_cost), key=lambda p: -p["score"])[:8]
    if min_diff:
        n_in = sum(1 for p in squad if p["diff"])
        out.append(f"_Squad holds {n_in} sub-{DIFF_OWNERSHIP:.0f}%-owned "
                   f"differential(s) (min requested {min_diff})._\n")
    d = ["| Player | Pos | Club | £m | Score | Own% | In squad? |",
         "|--------|-----|------|----|-------|------|-----------|"]
    for p in diffs:
        d.append(f"| {p['name']} | {POSITIONS[p['pos']]} | {p['club']} | "
                 f"{p['cost']/10:.1f} | {p['score']:.1f} | {p['own']:.1f} | "
                 f"{'yes' if p['id'] in owned else '—'} |")
    out += ["\n".join(d), "",
            f"_Low-ownership (< {DIFF_OWNERSHIP:.0f}%) high-projection picks. Set "
            "MIN_DIFFERENTIALS or pass --differentials N to force some in._", "",
            "## Strategy\n",
            f"Spend spread across **{len(prem)} premium/near-premium picks (£8m+)** "
            "— " + ", ".join(f"{p['name']} (£{p['cost']/10:.1f}m)" for p in prem[:4])
            + f" — not one talisman. Priciest **{top['name']} "
            f"(£{top['cost']/10:.1f}m)**, under the £{max_cost/10:.1f}m cap.", "",
            f"By design: **£{(budget-total)/10:.1f}m held in reserve**, a cheap "
            f"**£{min(p['cost'] for p in squad if p['pos']==1)/10:.1f}m bench "
            "keeper** (no budget wasted on a non-playing #2), and "
            f"**{sum(1 for p in squad if p['cost']>=90 and p['pos'] in (3,4))} "
            "attacking premium(s) (£9.0m+)** for ceiling.", "",
            "## Transfer rules (season heads-up)\n",
            "- **GW1 is a free build** — unlimited transfers until the Fri 18:30 "
            "BST deadline; re-run right before to lock in late prices/news.",
            f"- In-season: **{TRANSFER_RULES['free_transfers_per_gw']} free "
            f"transfer/GW**, bankable up to **{TRANSFER_RULES['max_saved_free_transfers']}**; "
            f"extras cost **-{TRANSFER_RULES['extra_transfer_cost']} pts**. Phase 2 "
            "will respect this and only suggest hits that beat the -4.", "",
            "> Prices and availability are pulled live at run time."]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Phase 1 FPL squad optimizer (single file).")
    ap.add_argument("--budget", type=float, default=BUDGET_M)
    ap.add_argument("--max-player-cost", type=float, default=MAX_PLAYER_COST_M)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--differentials", type=int, default=MIN_DIFFERENTIALS)
    ap.add_argument("--min-bank", type=float, default=MIN_BANK_M)
    ap.add_argument("--max-bank", type=float, default=MAX_BANK_M)
    ap.add_argument("--min-premiums", type=int, default=MIN_PREMIUMS)
    ap.add_argument("--premium-cost", type=float, default=PREMIUM_COST_M)
    ap.add_argument("--bench-gk-max", type=float, default=BENCH_GK_MAX_M)
    ap.add_argument("--max-per-club", type=int, default=3,
                    help="max players from one club (set 2 to avoid over-loading a team)")
    ap.add_argument("--no-history", action="store_true",
                    help="skip the last-season pull (faster, weaker pre-season signal)")
    args, _ = ap.parse_known_args()  # ignore Colab/Jupyter's own argv

    print("Fetching live FPL data ...")
    boot = _get("bootstrap-static")
    fixtures = _get("fixtures")

    last_season = {}
    if USE_HISTORY and not args.no_history:
        last_season = get_last_season_stats([el["id"] for el in boot["elements"]])

    players = build_players(boot, fixtures, args.horizon, last_season)
    budget = int(round(args.budget * 10))
    max_cost = int(round(args.max_player_cost * 10))
    squad = solve_squad(players, budget, max_cost, args.differentials,
                        max_per_club=args.max_per_club,
                        min_bank=int(round(args.min_bank * 10)),
                        max_bank=int(round(args.max_bank * 10)),
                        min_premiums=args.min_premiums,
                        premium_cost=int(round(args.premium_cost * 10)),
                        bench_gk_max=int(round(args.bench_gk_max * 10)))
    xi = solve_xi(squad)
    print("\n" + report(squad, xi, players, budget, max_cost, args.horizon,
                        args.differentials, len(last_season)))


if __name__ == "__main__":
    main()
