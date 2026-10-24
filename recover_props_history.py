"""
Rebuild props_history.csv from the old versions of props.csv stored in your GitHub repo's history.

Run it ONCE, from the same folder as nfl_update.py (your `nfl` folder), after you have
run the newest nfl_update.py at least once (so games.csv has kickoff dates):

    py recover_props_history.py

It needs internet access and the `pandas` package (you already have both).
It reads a PUBLIC repo with no login. GitHub allows 60 anonymous requests an hour, which is
plenty for a few weeks of history. If it says you are rate limited, wait an hour, or set a
GitHub token:   set GITHUB_TOKEN=your_token_here   (Windows)   /   export GITHUB_TOKEN=...   (Mac/Linux)

Options:
    --repo owner/name     default TuckersTruckers/nfl-model
    --path FILE           default nfl_output/props.csv   (the file whose history is read)
    --out FOLDER          default nfl_output             (where games.csv and props_history.csv live)
    --repo-path FOLDER    use a local clone of the repo instead of GitHub (needs git installed)

The script only ADDS to props_history.csv. It never deletes anything, and running it twice
does no harm (duplicates are removed).
"""
import argparse
import datetime
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

try:
    from nfl_update import HIST_COLS, _with_meta, read_existing, save_props_history
except ImportError:
    sys.exit("Run this from the same folder as nfl_update.py (it reuses that script's team and week mapping).")


def fetch(url):
    """GET a URL and return its text. Never prints the token."""
    headers = {"User-Agent": "nfl-model-recover"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            sys.exit("GitHub says you have made too many requests. Wait an hour, or set a GITHUB_TOKEN (see the top of this file).")
        if e.code == 404:
            return None
        sys.exit(f"GitHub returned HTTP {e.code}.")
    except Exception as e:
        sys.exit(f"Network problem ({type(e).__name__}). Check your internet connection.")


def versions_from_github(repo, path):
    """Yield (date, sha, text) for every commit that changed `path`, oldest first."""
    shas, page = [], 1
    while True:
        url = f"https://api.github.com/repos/{repo}/commits?path={path}&per_page=100&page={page}"
        txt = fetch(url)
        if txt is None:
            sys.exit(f"Could not find {path} in {repo}. Check the --repo and --path options.")
        batch = json.loads(txt)
        shas += [(c["commit"]["committer"]["date"], c["sha"]) for c in batch]
        if len(batch) < 100:
            break
        page += 1
    for date, sha in reversed(shas):
        text = fetch(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}")
        time.sleep(0.2)
        if text:
            yield date, sha, text


def versions_from_git(repo_path, path):
    """The same, from a local clone."""
    out = subprocess.run(["git", "-C", repo_path, "log", "--format=%cI %H", "--", path], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("git could not read that folder: " + out.stderr.strip()[:200])
    lines = [ln.split(" ", 1) for ln in out.stdout.strip().splitlines() if ln.strip()]
    for date, sha in reversed(lines):
        show = subprocess.run(["git", "-C", repo_path, "show", f"{sha}:{path}"], capture_output=True)
        if show.returncode == 0 and show.stdout:
            yield date, sha, show.stdout.decode("utf-8")


def season_of(commence):
    try:
        d = pd.Timestamp(commence)
        return int(d.year if d.month >= 3 else d.year - 1)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="TuckersTruckers/nfl-model")
    ap.add_argument("--path", default="nfl_output/props.csv")
    ap.add_argument("--out", default="nfl_output")
    ap.add_argument("--repo-path", default=None)
    a = ap.parse_args()

    gsched = read_existing(os.path.join(a.out, "games.csv"))
    if gsched is None or "week" not in gsched.columns:
        sys.exit(f"Could not read {a.out}/games.csv. Run nfl_update.py first so the schedule exists.")
    if "gameday" not in gsched.columns:
        print("  note: games.csv has no kickoff dates (gameday). Weeks are guessed from the schedule, which is fine unless two teams "
              "met twice. Run the newest nfl_update.py first for exact weeks.")

    source = versions_from_git(a.repo_path, a.path) if a.repo_path else versions_from_github(a.repo, a.path)
    seen, rows, n_versions = {}, [], 0
    for date, sha, text in source:                       # oldest first, so the earliest copy of each line wins
        n_versions += 1
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception:
            print(f"  skipped a version from {date[:10]} (could not read it)")
            continue
        if not {"player", "stat", "line"} <= set(df.columns):
            continue
        new = 0
        for r in df.to_dict("records"):
            key = (str(r.get("event")), str(r.get("player")), str(r.get("stat")))
            if key in seen:
                continue
            seen[key] = True
            rows.append(r)
            new += 1
        print(f"  {date[:10]} ({sha[:7]}): {len(df)} lines, {new} new")

    if not rows:
        sys.exit("No prop lines were found in the history.")

    # attach team codes, week and season to rows that predate those columns
    by_season = {}
    for r in rows:
        by_season.setdefault(season_of(r.get("commence")), []).append(r)
    for season, group in by_season.items():
        _with_meta(group, gsched, season if season is not None else datetime.datetime.now().year)

    before = read_existing(os.path.join(a.out, "props_history.csv"))
    n_before = 0 if before is None else len(before)
    save_props_history(a.out, rows, gsched, None)
    after = read_existing(os.path.join(a.out, "props_history.csv"))

    no_week = sum(1 for r in rows if r.get("week") in (None, "") or (isinstance(r.get("week"), float) and r["week"] != r["week"]))
    print(f"\nRead {n_versions} saved version(s) of {a.path}; recovered {len(rows)} distinct lines.")
    print(f"props_history.csv: {n_before} -> {len(after)} lines in {a.out}/props_history.csv")
    wk = pd.Series([r.get("week") for r in rows]).value_counts().sort_index()
    print("lines per week:", ", ".join(f"wk {int(k)}: {v}" for k, v in wk.items() if k == k))
    if no_week:
        print(f"warning: {no_week} line(s) could not be matched to a week (team names or schedule not found); they were saved without one.")
    print("Done. Upload props_history.csv to the app (Data tab box on the Claude page; commit it on GitHub).")


if __name__ == "__main__":
    main()
