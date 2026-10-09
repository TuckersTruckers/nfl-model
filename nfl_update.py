#!/usr/bin/env python3
"""
nfl_update.py - downloads free nflverse data and writes the CSV files the NFL Model web app imports.
Core (must succeed): teams, games, players, weekly, dvp, tds.
Extras (best effort, never block the core files): player_games, injuries, history_weekly, history_games,
history_player_games (past seasons, built once and then reused).

Usage:   python nfl_update.py                 (current season, output to ./nfl_output)
         python nfl_update.py --season 2025   (use a different season)
         python nfl_update.py --out my_folder
         python nfl_update.py --history 3     (also build the previous 3 seasons once; 0 = skip)

Safety: files are only written if EVERY download and validation step succeeds.
If anything fails, your previous CSVs are left untouched.
Requires: Python 3.9+ and `pip install pandas`
"""
import argparse, datetime, os, sys, tempfile
import numpy as np
import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
TEAM_URLS = ["https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{s}.csv"]
PLAYER_URLS = [
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{s}.csv",
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{s}.csv",
]

# Same team list/aliases the web app validates against
TEAMS = set("ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GNB HOU IND JAX KAN LVR LAC LAR MIA MIN NWE NOR NYG NYJ PHI PIT SFO SEA TAM TEN WAS".split())
ALIAS = {"GB": "GNB", "KC": "KAN", "LV": "LVR", "SF": "SFO", "TB": "TAM", "NE": "NWE", "NO": "NOR", "JAC": "JAX", "LA": "LAR", "WSH": "WAS"}
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{s}.csv.gz"
PBP_COLS = {"game_id", "play_id", "week", "season_type", "home_team", "away_team", "posteam", "td_team", "td_player_id",
            "td_player_name", "touchdown", "pass_touchdown", "rush_touchdown", "return_touchdown", "yards_gained",
            "passer_player_name", "qtr", "time", "play_deleted"}
INJ_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{s}.csv"
ROUND = {"WC": "wildcard", "DIV": "division", "CON": "confchamp", "SB": "superbowl"}


def fail(msg):
    sys.exit("\nFAILED: " + msg + "\nNo files were changed.")


def download(urls, label, season):
    errs = []
    for u in urls:
        u = u.format(s=season)
        try:
            df = pd.read_csv(u, low_memory=False)
            print(f"  downloaded {label}: {len(df)} rows")
            return df
        except Exception as e:
            errs.append(f"  {u}\n    -> {e}")
    fail(f"could not download {label}. Tried:\n" + "\n".join(errs))


def need(df, cols, label):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        fail(f"{label} is missing expected columns {miss}. nflverse may have changed its format; found: {list(df.columns)[:40]}")


def code(x):
    x = str(x).strip().upper()
    return ALIAS.get(x, x)


def default_season():
    t = datetime.date.today()
    return t.year if t.month >= 3 else t.year - 1


def qb_starts(g_all):
    """Career starts BEFORE each game for each side's starting QB (counted across every season in the file)."""
    cols = ["game_id", "gameday", "home_qb_name", "away_qb_name", "home_score", "away_score"]
    if not all(c in g_all.columns for c in cols):
        return None
    x = g_all[cols].sort_values(["gameday", "game_id"])
    counts, hs, as_ = {}, [], []
    for r in x.itertuples():
        hq, aq = r.home_qb_name, r.away_qb_name
        hs.append(counts.get(hq, 0) if isinstance(hq, str) else np.nan)
        as_.append(counts.get(aq, 0) if isinstance(aq, str) else np.nan)
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            if isinstance(hq, str):
                counts[hq] = counts.get(hq, 0) + 1
            if isinstance(aq, str):
                counts[aq] = counts.get(aq, 0) + 1
    x = x.assign(home_qb_starts=hs, away_qb_starts=as_).set_index("game_id")
    return x[["home_qb_starts", "away_qb_starts"]]


def build_games(g, season, qs=None):
    need(g, ["season", "game_type", "week", "away_team", "home_team", "away_score", "home_score", "spread_line", "total_line"], "games.csv")
    g = g[g.season == season].copy()
    if g.empty:
        fail(f"no games found for season {season}")
    g["wk"] = g.apply(lambda r: ROUND.get(r.game_type, str(int(r.week))), axis=1)
    out = pd.DataFrame({
        "week": g.wk, "away": g.away_team, "home": g.home_team,
        "away_score": g.away_score.astype("Int64"), "home_score": g.home_score.astype("Int64"),
        # nflverse: positive spread_line = home favored. App: negative = home favored.
        "spread": -g.spread_line, "total": g.total_line,
    })
    # optional extras (only if the source has them): market odds and starting QBs
    for src, dst in [("home_moneyline", "home_ml"), ("away_moneyline", "away_ml"), ("home_spread_odds", "home_spread_odds"),
                     ("away_spread_odds", "away_spread_odds"), ("over_odds", "over_odds"), ("under_odds", "under_odds")]:
        if src in g.columns:
            out[dst] = pd.to_numeric(g[src], errors="coerce")
    for src, dst in [("home_qb_name", "home_qb"), ("away_qb_name", "away_qb")]:
        if src in g.columns:
            out[dst] = g[src].fillna("").astype(str).str.replace(",", "", regex=False)
    if qs is not None and "game_id" in g.columns:
        q = qs.reindex(g.game_id)
        out["home_qb_starts"] = q.home_qb_starts.values
        out["away_qb_starts"] = q.away_qb_starts.values
    if out.duplicated(["week", "away", "home"]).any():
        fail("duplicate games found in schedule")
    return g, out


def build_teams(g, ts, pl, season):
    played = g[(g.game_type == "REG") & g.home_score.notna() & g.away_score.notna()]
    if played.empty:
        fail(f"no completed regular-season games yet for {season}. Early in the season, run with --season {season - 1} "
             f"and import only teams.csv and players.csv (skip games.csv).")
    need(ts, ["team", "opponent_team", "week", "attempts", "carries", "passing_yards", "rushing_yards", "passing_tds", "rushing_tds"], "team stats")
    if "season_type" in ts.columns:
        ts = ts[ts.season_type == "REG"]
    pts = pd.concat([
        pd.DataFrame({"team": played.away_team, "pf": played.away_score, "pa": played.home_score}),
        pd.DataFrame({"team": played.home_team, "pf": played.home_score, "pa": played.away_score}),
    ]).groupby("team")[["pf", "pa"]].mean()
    off = ts.groupby("team").agg(n=("week", "size"), att=("attempts", "sum"), car=("carries", "sum"),
                                 py=("passing_yards", "sum"), ry=("rushing_yards", "sum"),
                                 ptd=("passing_tds", "sum"), rtd=("rushing_tds", "sum"))
    alw = ts.groupby("opponent_team").agg(py_a=("passing_yards", "sum"), ry_a=("rushing_yards", "sum"),
                                          ptd_a=("passing_tds", "sum"), rtd_a=("rushing_tds", "sum"))
    t = pts.join(off).join(alw)
    out = pd.DataFrame({
        "pf": t.pf, "pa": t.pa, "pass": t.py / t.n, "rush": t.ry / t.n,
        "pass_a": t.py_a / t.n, "rush_a": t.ry_a / t.n, "ypa": t.py / t.att, "ypc": t.ry / t.car,
        "ptd": t.ptd / t.n, "rtd": t.rtd / t.n, "ptd_a": t.ptd_a / t.n, "rtd_a": t.rtd_a / t.n,
    })
    # Defense vs position: PPR fantasy points allowed per game
    need(pl, ["position", "opponent_team"], "player stats")
    fp = "fantasy_points_ppr" if "fantasy_points_ppr" in pl.columns else "fantasy_points"
    p = pl[pl.position.isin(["QB", "RB", "WR", "TE"])]
    if "season_type" in p.columns:
        p = p[p.season_type == "REG"]
    dv = p.groupby(["opponent_team", "position"])[fp].sum().unstack(fill_value=0)
    for pos in ["QB", "RB", "WR", "TE"]:
        out["dvp_" + pos.lower()] = (dv[pos] / t.n) if pos in dv.columns else float("nan")
    out.index = [code(x) for x in out.index]
    out.index.name = "team"
    out = out.round(3)
    if set(out.index) != TEAMS or out.isna().any().any():
        bad = sorted(set(out.index) ^ TEAMS)
        fail(f"team table failed validation (team mismatch: {bad}, missing values: {int(out.isna().sum().sum())}). "
             f"Usually means some team has not played a game yet.")
    return out.reset_index(), off


def build_players(pl, off):
    tcol = "team" if "team" in pl.columns else "recent_team"
    ncol = "player_display_name" if "player_display_name" in pl.columns else "player_name"
    need(pl, [tcol, ncol, "player_id", "position"], "player stats")
    if "season_type" in pl.columns:
        pl = pl[pl.season_type == "REG"]
    num = ["completions", "attempts", "passing_yards", "passing_tds", "carries", "rushing_yards", "rushing_tds",
           "receptions", "targets", "receiving_yards", "receiving_tds"]
    need(pl, num, "player stats")
    icol = "passing_interceptions" if "passing_interceptions" in pl.columns else ("interceptions" if "interceptions" in pl.columns else None)
    pl = pl.copy()
    pl[num] = pl[num].fillna(0)
    pl["ints"] = pl[icol].fillna(0) if icol else 0
    a = pl.groupby(["player_id", tcol]).agg(name=(ncol, "last"), pos=("position", "last"), ints=("ints", "sum"), gp=("week", "nunique"),
                                            **{c: (c, "sum") for c in num}).reset_index()
    a["team"] = a[tcol].map(code)
    off = off.copy()
    off.index = [code(x) for x in off.index]
    a["tm_att"] = a.team.map(off.att)
    a["tm_car"] = a.team.map(off.car)
    a["name"] = a.name.astype(str).str.replace(",", "", regex=False)
    a["pos"] = a.pos.replace({"FB": "RB"})
    cols = ["player", "team", "pos", "tgt_share", "catch_rate", "ypt", "carry_share", "ypc", "td_rate",
            "att_share", "ypa", "int_rate", "rush_td_rate", "rec_td_rate", "n_att", "n_car", "n_tgt", "gp"]
    rows = []
    div = lambda x, y: (x / y) if y else None
    # every QB with a pass attempt or a carry (backups and rushing QBs included)
    for r in a[(a.pos == "QB") & ((a.attempts > 0) | (a.carries > 0))].itertuples():
        rows.append([r.name, r.team, "QB", None, div(r.completions, r.attempts), None, div(r.carries, r.tm_car),
                     div(r.rushing_yards, r.carries), div(r.passing_tds, r.attempts),
                     div(r.attempts, r.tm_att), div(r.passing_yards, r.attempts), div(r.ints, r.attempts),
                     div(r.rushing_tds, r.carries), None, r.attempts, r.carries, r.targets, r.gp])
    s = a[a.pos.isin(["RB", "WR", "TE"])].copy()
    s["touch"] = s.targets + s.carries
    s = s[(s.targets >= 2) | (s.carries >= 2)].sort_values("touch", ascending=False).groupby("team").head(18)
    for r in s.itertuples():
        touches = r.carries + r.receptions
        rows.append([r.name, r.team, r.pos, div(r.targets, r.tm_att), div(r.receptions, r.targets), div(r.receiving_yards, r.targets),
                     div(r.carries, r.tm_car), div(r.rushing_yards, r.carries),
                     div(r.rushing_tds + r.receiving_tds, touches),
                     None, None, None, div(r.rushing_tds, r.carries), div(r.receiving_tds, r.receptions),
                     r.attempts, r.carries, r.targets, r.gp])
    out = pd.DataFrame(rows, columns=cols).round(4)
    if out.empty:
        fail("no players passed the filters")
    return out


def build_dvp(pl, off):
    """Per-game stats each defense allows to QBs, RBs, WRs and TEs (for the matchup panels)."""
    if "season_type" in pl.columns:
        pl = pl[pl.season_type == "REG"]
    spec = {
        "QB": {"att": "attempts", "cmp": "completions", "yds": "passing_yards", "td": "passing_tds",
               "ratt": "carries", "ryds": "rushing_yards", "rtd": "rushing_tds"},
        "RB": {"att": "carries", "yds": "rushing_yards", "td": "rushing_tds", "tgt": "targets",
               "rec": "receptions", "ryds": "receiving_yards", "rtd": "receiving_tds"},
        "WR": {"tgt": "targets", "rec": "receptions", "yds": "receiving_yards", "td": "receiving_tds"},
        "TE": {"tgt": "targets", "rec": "receptions", "yds": "receiving_yards", "td": "receiving_tds"},
    }
    need(pl, ["position", "opponent_team"] + sorted({c for m in spec.values() for c in m.values()}), "player stats (dvp)")
    out = pd.DataFrame(index=off.index)
    for pos, m in spec.items():
        g = pl[pl.position == pos].groupby("opponent_team")[list(m.values())].sum()
        for k, c in m.items():
            out[f"{pos.lower()}_{k}"] = (g[c] / off.n).reindex(out.index).fillna(0)
    out.index = [code(x) for x in out.index]
    out.index.name = "team"
    if set(out.index) != TEAMS:
        fail("defense-vs-position table does not have exactly the 32 known teams")
    return out.round(3).reset_index()


def build_tds(pl, season):
    """One row per touchdown. Uses play-by-play when it downloads (adds type, yards, quarter, first TD of the game);
    otherwise falls back to weekly player stats (who scored vs which defense, no first-TD flag)."""
    tcol = "team" if "team" in pl.columns else "recent_team"
    ncol = "player_display_name" if "player_display_name" in pl.columns else "player_name"
    pp = pl.drop_duplicates("player_id", keep="last").set_index("player_id")
    posmap = pp.position.replace({"FB": "RB"})
    posmap = posmap.where(posmap.isin(["QB", "RB", "WR", "TE"]), "DEF")
    names = pp[ncol]
    cols = ["scorer", "pos", "type", "yards", "passer", "team", "defense", "week", "game", "qtr", "clock", "first"]
    try:
        pbp = pd.read_csv(PBP_URL.format(s=season), low_memory=False, usecols=lambda c: c in PBP_COLS)
        need(pbp, ["game_id", "play_id", "week", "td_team", "td_player_id", "touchdown", "home_team", "away_team"], "play-by-play")
        t = pbp[(pbp.touchdown == 1) & pbp.td_team.notna()]
        if "season_type" in t.columns:
            t = t[t.season_type == "REG"]
        if "play_deleted" in t.columns:
            t = t[t.play_deleted.fillna(0) != 1]
        t = t.sort_values(["game_id", "play_id"]).copy()
        t["first"] = (t.groupby("game_id").cumcount() == 0).astype(int)
        t = t.sort_values(["week", "game_id", "play_id"])
        flag = lambda c: (t[c].fillna(0) == 1) if c in t.columns else pd.Series(False, index=t.index)
        typ = np.where(flag("pass_touchdown"), "Passing", np.where(flag("rush_touchdown"), "Rushing", "Return/Defensive"))
        col = lambda c: t[c] if c in t.columns else pd.Series(np.nan, index=t.index)
        out = pd.DataFrame({
            "scorer": t.td_player_id.map(names).fillna(col("td_player_name")).fillna("Unknown"),
            "pos": t.td_player_id.map(posmap).fillna("DEF"),
            "type": typ, "yards": col("yards_gained"),
            "passer": np.where(typ == "Passing", col("passer_player_name").fillna(""), ""),
            "team": t.td_team, "defense": np.where(t.td_team == t.home_team, t.away_team, t.home_team),
            "week": t.week, "game": t.away_team + " @ " + t.home_team, "qtr": col("qtr"), "clock": col("time"), "first": t["first"],
        })
        print(f"  built {len(out)} touchdown events from play-by-play")
    except (Exception, SystemExit) as e:
        print(f"  play-by-play unavailable ({str(e).splitlines()[0][:100]}); using weekly player stats for TD events (no first-TD flag)")
        q = pl[pl.season_type == "REG"] if "season_type" in pl.columns else pl
        rows = []
        for c, typ in (("rushing_tds", "Rushing"), ("receiving_tds", "Passing")):
            for _, r in q[q[c].fillna(0) > 0].iterrows():
                pos = posmap.get(r["player_id"], "DEF")
                rows += [[r[ncol], pos, typ, "", "", r[tcol], r["opponent_team"], r["week"], "", "", "", ""]] * int(r[c])
        out = pd.DataFrame(rows, columns=cols).sort_values("week", kind="stable")
    for c in ("scorer", "passer"):
        out[c] = out[c].astype(str).str.replace(",", "", regex=False)
    for c in ("team", "defense"):
        out[c] = out[c].map(code)
    bad = set(out.team) | set(out.defense)
    if not bad <= TEAMS:
        fail(f"touchdown table has unknown team codes: {sorted(bad - TEAMS)}")
    return out[cols]


def build_player_games(pl, season):
    """One row per player per game (QB/RB/WR/TE with a pass attempt, carry or target) for boom/bust and variance."""
    tcol = "team" if "team" in pl.columns else "recent_team"
    ncol = "player_display_name" if "player_display_name" in pl.columns else "player_name"
    need(pl, [tcol, ncol, "position", "week", "opponent_team", "attempts", "completions", "passing_yards", "passing_tds",
              "carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds"], "player logs")
    q = pl[pl.season_type == "REG"] if "season_type" in pl.columns else pl
    q = q.copy()
    q["position"] = q.position.replace({"FB": "RB"})
    q = q[q.position.isin(["QB", "RB", "WR", "TE"])]
    icol = "passing_interceptions" if "passing_interceptions" in q.columns else ("interceptions" if "interceptions" in q.columns else None)
    n = lambda c: q[c].fillna(0)
    ints = n(icol) if icol else 0
    fp = q["fantasy_points_ppr"].fillna(0) if "fantasy_points_ppr" in q.columns else (
        0.04 * n("passing_yards") + 4 * n("passing_tds") - 2 * ints + 0.1 * n("rushing_yards") + 6 * n("rushing_tds")
        + 0.1 * n("receiving_yards") + n("receptions") + 6 * n("receiving_tds"))
    out = pd.DataFrame({
        "season": season, "week": q.week.astype(int), "player": q[ncol].astype(str).str.replace(",", "", regex=False),
        "team": q[tcol].map(code), "opp": q.opponent_team.map(code), "pos": q.position,
        "att": n("attempts"), "cmp": n("completions"), "pyds": n("passing_yards"), "ptd": n("passing_tds"), "int": ints,
        "car": n("carries"), "ryds": n("rushing_yards"), "rtd": n("rushing_tds"), "tgt": n("targets"), "rec": n("receptions"),
        "recyds": n("receiving_yards"), "rectd": n("receiving_tds"), "fp": fp.round(2),
    })
    out = out[(out.att >= 3) | (out.car >= 1) | (out.tgt >= 1)]
    out = out[out.team.isin(TEAMS) & out.opp.isin(TEAMS)]
    for c in ["att", "cmp", "pyds", "ptd", "int", "car", "ryds", "rtd", "tgt", "rec", "recyds", "rectd"]:
        out[c] = out[c].astype(int)
    return out.sort_values(["week", "team", "player"])


def build_injuries(season):
    """Latest posted injury report (Out / Doubtful / Questionable) for the season. Returns None if unavailable."""
    df = pd.read_csv(INJ_URL.format(s=season), low_memory=False)
    stc = "report_status" if "report_status" in df.columns else ("practice_status" if "practice_status" in df.columns else None)
    if stc is None or not {"team", "week", "full_name", "position"} <= set(df.columns):
        raise ValueError("injury file format not recognised")
    if "game_type" in df.columns:
        df = df[df.game_type == "REG"]
    df = df[df[stc].isin(["Out", "Doubtful", "Questionable"])]
    if df.empty:
        return None
    df = df[df.week == df.week.max()]
    inj = df["report_primary_injury"].fillna("") if "report_primary_injury" in df.columns else ""
    out = pd.DataFrame({"week": df.week.astype(int), "team": df.team.map(code), "player": df.full_name.astype(str).str.replace(",", "", regex=False),
                        "pos": df.position, "status": df[stc], "injury": inj})
    return out[out.team.isin(TEAMS)].drop_duplicates(["team", "player"])


def history_bundle(g_raw, season, qs):
    ts = download(TEAM_URLS, f"{season} team weekly stats", season)
    pl = download(PLAYER_URLS, f"{season} player weekly stats", season)
    g, games = build_games(g_raw, season, qs)
    weekly = build_weekly(g, ts)
    plog = build_player_games(pl, season)
    games.insert(0, "season", season)
    weekly.insert(0, "season", season)
    return weekly, games, plog


def read_existing(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def write_csv(df, out, name):
    fd, tmp = tempfile.mkstemp(dir=out, suffix=".tmp")
    os.close(fd)
    df.to_csv(tmp, index=False, na_rep="")
    os.replace(tmp, os.path.join(out, name))
    print(f"  wrote {out}/{name} ({len(df)} rows)")


def update_history(g_raw, qs, current, n, out):
    """Build the previous n seasons once. Seasons already in history_weekly.csv are reused, not downloaded again."""
    want = [current - i for i in range(1, n + 1)]
    names = {"weekly": "history_weekly.csv", "games": "history_games.csv", "plog": "history_player_games.csv"}
    old = {k: read_existing(os.path.join(out, v)) for k, v in names.items()}
    done = set(old["weekly"].season.unique()) if old["weekly"] is not None and "season" in old["weekly"].columns else set()
    todo = [s for s in want if s not in done]
    if not todo:
        print(f"  history already built for {want}")
        return
    new = {k: [] for k in names}
    for s in todo:
        try:
            w, gm, pg = history_bundle(g_raw, s, qs)
            new["weekly"].append(w), new["games"].append(gm), new["plog"].append(pg)
            print(f"  history {s}: {len(w)} team-games, {len(gm)} games, {len(pg)} player-games")
        except (Exception, SystemExit) as e:
            print(f"  history {s} skipped: {str(e).strip().splitlines()[0][:110] if str(e).strip() else type(e).__name__}")
    for k, fn in names.items():
        frames = ([old[k]] if old[k] is not None else []) + new[k]
        if new[k]:
            write_csv(pd.concat(frames, ignore_index=True), out, fn)


def build_weekly(g, ts):
    """One row per team per completed game, so the app can rebuild the stats entering any week."""
    played = g[(g.game_type == "REG") & g.home_score.notna() & g.away_score.notna()]
    pts = pd.concat([
        pd.DataFrame({"team": played.away_team, "week": played.week, "pf": played.away_score, "pa": played.home_score}),
        pd.DataFrame({"team": played.home_team, "week": played.week, "pf": played.home_score, "pa": played.away_score}),
    ])
    if "season_type" in ts.columns:
        ts = ts[ts.season_type == "REG"]
    cols = ["passing_yards", "rushing_yards", "passing_tds", "rushing_tds"]
    allowed = ts.assign(team=ts.opponent_team)[["team", "week"] + cols].rename(
        columns={"passing_yards": "pass_a", "rushing_yards": "rush_a", "passing_tds": "ptd_a", "rushing_tds": "rtd_a"})
    w = ts[["team", "opponent_team", "week"] + cols].rename(
        columns={"opponent_team": "opp", "passing_yards": "pass", "rushing_yards": "rush", "passing_tds": "ptd", "rushing_tds": "rtd"})
    w = w.merge(allowed, on=["team", "week"], how="left").merge(pts, on=["team", "week"], how="left")
    if len(w) != len(ts) or w.isna().any().any():
        fail("could not line up weekly team stats with game scores (team-code mismatch or a game missing from one source).")
    w["week"] = w.week.astype(int)
    return w[["team", "week", "opp", "pf", "pa", "pass", "rush", "pass_a", "rush_a", "ptd", "rtd", "ptd_a", "rtd_a"]].sort_values(["week", "team"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=default_season())
    ap.add_argument("--out", default="nfl_output")
    ap.add_argument("--history", type=int, default=3, help="previous seasons to build once (0 = skip)")
    a = ap.parse_args()
    print(f"Season {a.season}")
    g_raw = download([GAMES_URL], "schedule/scores/lines", a.season)
    ts = download(TEAM_URLS, "team weekly stats", a.season)
    pl = download(PLAYER_URLS, "player weekly stats", a.season)
    qs = qb_starts(g_raw)
    g, games = build_games(g_raw, a.season, qs)
    teams, off = build_teams(g, ts, pl, a.season)
    players = build_players(pl, off)
    weekly = build_weekly(g, ts)
    dvp = build_dvp(pl, off)
    tds = build_tds(pl, a.season)
    games.insert(0, "season", a.season)
    weekly.insert(0, "season", a.season)
    os.makedirs(a.out, exist_ok=True)
    for name, df in [("teams.csv", teams), ("games.csv", games), ("players.csv", players), ("weekly.csv", weekly), ("dvp.csv", dvp), ("tds.csv", tds)]:
        write_csv(df, a.out, name)
    # ---- extras: each one is optional and can never block the core files above ----
    try:
        write_csv(build_player_games(pl, a.season), a.out, "player_games.csv")
    except (Exception, SystemExit) as e:
        print("  player_games.csv skipped:", str(e).strip().splitlines()[0][:110] if str(e).strip() else type(e).__name__)
    try:
        inj = build_injuries(a.season)
        if inj is not None:
            write_csv(inj, a.out, "injuries.csv")
        else:
            print("  injuries.csv skipped: no injury report posted yet")
    except (Exception, SystemExit) as e:
        print("  injuries.csv skipped:", str(e).strip().splitlines()[0][:110] if str(e).strip() else type(e).__name__)
    if a.history > 0:
        try:
            update_history(g_raw, qs, a.season, a.history, a.out)
        except (Exception, SystemExit) as e:
            print("  history skipped:", str(e).strip().splitlines()[0][:110] if str(e).strip() else type(e).__name__)
    print("Done. Upload the core files (and any extras written above) on the app's Data tab.")


if __name__ == "__main__":
    main()
