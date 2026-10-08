#!/usr/bin/env python3
"""
nfl_update.py - downloads free nflverse data and writes the three CSV files
the NFL Model web app imports: teams.csv, games.csv, players.csv, weekly.csv

Usage:   python nfl_update.py                 (current season, output to ./nfl_output)
         python nfl_update.py --season 2025   (use a different season)
         python nfl_update.py --out my_folder

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


def build_games(g, season):
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
    num = ["completions", "attempts", "passing_tds", "carries", "rushing_yards", "rushing_tds",
           "receptions", "targets", "receiving_yards", "receiving_tds"]
    need(pl, num, "player stats")
    pl = pl.copy()
    pl[num] = pl[num].fillna(0)
    a = pl.groupby(["player_id", tcol]).agg(name=(ncol, "last"), pos=("position", "last"), **{c: (c, "sum") for c in num}).reset_index()
    a["team"] = a[tcol].map(code)
    off = off.copy()
    off.index = [code(x) for x in off.index]
    a["tm_att"] = a.team.map(off.att)
    a["tm_car"] = a.team.map(off.car)
    a["name"] = a.name.astype(str).str.replace(",", "", regex=False)
    cols = ["player", "team", "pos", "tgt_share", "catch_rate", "ypt", "carry_share", "ypc", "td_rate"]
    rows = []
    # one QB per team: most pass attempts
    q = a[(a.pos == "QB") & (a.attempts > 0)].sort_values("attempts", ascending=False).drop_duplicates("team")
    for r in q.itertuples():
        rows.append([r.name, r.team, "QB", None, r.completions / r.attempts, None, None, None, r.passing_tds / r.attempts])
    s = a[a.pos.isin(["RB", "WR", "TE"])]
    for r in s.itertuples():
        ts_ = r.targets / r.tm_att if r.tm_att else 0
        cs = r.carries / r.tm_car if r.tm_car else 0
        if ts_ < 0.05 and cs < 0.10:
            continue
        touches = r.carries + r.receptions
        rows.append([r.name, r.team, r.pos, ts_,
                     r.receptions / r.targets if r.targets else None,
                     r.receiving_yards / r.targets if r.targets else None,
                     cs, r.rushing_yards / r.carries if r.carries else None,
                     (r.rushing_tds + r.receiving_tds) / touches if touches else None])
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
    a = ap.parse_args()
    print(f"Season {a.season}")
    g_raw = download([GAMES_URL], "schedule/scores/lines", a.season)
    ts = download(TEAM_URLS, "team weekly stats", a.season)
    pl = download(PLAYER_URLS, "player weekly stats", a.season)
    g, games = build_games(g_raw, a.season)
    teams, off = build_teams(g, ts, pl, a.season)
    players = build_players(pl, off)
    weekly = build_weekly(g, ts)
    dvp = build_dvp(pl, off)
    tds = build_tds(pl, a.season)
    os.makedirs(a.out, exist_ok=True)
    for name, df in [("teams.csv", teams), ("games.csv", games), ("players.csv", players), ("weekly.csv", weekly), ("dvp.csv", dvp), ("tds.csv", tds)]:
        fd, tmp = tempfile.mkstemp(dir=a.out, suffix=".tmp")
        os.close(fd)
        df.to_csv(tmp, index=False, na_rep="")
        os.replace(tmp, os.path.join(a.out, name))
        print(f"  wrote {a.out}/{name} ({len(df)} rows)")
    print("Done. Upload the six files on the app's Data tab.")


if __name__ == "__main__":
    main()
