#!/usr/bin/env python3
"""
Backfill props_history.csv from the GitHub history of nfl_output/props.csv.

This script replays every historical version of the file from the repo's commit history,
keeps the newest row for each (season, week, event, player, stat), and writes the
result to nfl_output/props_history.csv.

It is meant to replace the manual "look through History / click Raw" workflow.

Example:
  python rebuild_props_history.py
  python rebuild_props_history.py --owner TuckersTruckers --repo nfl-model --branch main
"""

import argparse
import io
import json
import os
import urllib.parse
import urllib.request
from typing import Iterable, List

import pandas as pd

HIST_COLS = [
    "season",
    "week",
    "event",
    "home_code",
    "away_code",
    "player",
    "stat",
    "line",
    "over_odds",
    "under_odds",
    "book_over",
    "book_under",
]


def get_json(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "nfl-model-history-rebuild",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def fetch_commit_shas(owner: str, repo: str, file_path: str, branch: str = "main") -> List[str]:
    """Return the commit SHAs for the given file, oldest-first."""
    shas: List[str] = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{owner}/{repo}/commits"
            f"?path={urllib.parse.quote(file_path)}&sha={branch}&per_page=100&page={page}"
        )
        page_data = get_json(url)
        if not page_data:
            break
        shas.extend(item["sha"] for item in page_data)
        if len(page_data) < 100:
            break
        page += 1
    return list(reversed(shas))


def fetch_raw_text(owner: str, repo: str, ref: str, file_path: str) -> str:
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{file_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-model-history-rebuild"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def normalize_record(raw: dict) -> dict:
    clean = {}
    for key, value in raw.items():
        clean[str(key).strip()] = value

    event = clean.get("event")
    player = clean.get("player")
    stat = clean.get("stat")
    if event is None:
        event = clean.get("game_id")
    if player is None:
        player = clean.get("description")
    if stat is None:
        stat = clean.get("market")

    rec = {
        "season": clean.get("season", ""),
        "week": clean.get("week", ""),
        "event": "" if event is None else str(event).strip(),
        "home_code": clean.get("home_code", clean.get("home", "")),
        "away_code": clean.get("away_code", clean.get("away", "")),
        "player": "" if player is None else str(player).replace(",", "").strip(),
        "stat": "" if stat is None else str(stat).strip(),
        "line": clean.get("line", ""),
        "over_odds": clean.get("over_odds", ""),
        "under_odds": clean.get("under_odds", ""),
        "book_over": clean.get("book_over", ""),
        "book_under": clean.get("book_under", ""),
    }

    # Some historical versions may already have the full team code columns. If not,
    # keep the raw values and let the caller decide whether to map them.
    return rec


def dedupe_and_sort(rows: Iterable[dict]) -> pd.DataFrame:
    latest = {}
    for row in rows:
        event = str(row.get("event", "")).strip()
        player = str(row.get("player", "")).strip()
        stat = str(row.get("stat", "")).strip()
        if not event or not player or not stat:
            continue
        key = (
            str(row.get("season", "")).strip(),
            str(row.get("week", "")).strip(),
            event,
            player,
            stat,
        )
        latest[key] = row

    out = pd.DataFrame(list(latest.values()), columns=HIST_COLS)
    if out.empty:
        return out

    # Keep the historical order stable but sort by season/week to make it easier to inspect.
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out = out.sort_values(["season", "week", "event", "player", "stat"], kind="stable", na_position="last")
    out = out.reset_index(drop=True)
    return out[HIST_COLS]


def rebuild_history(owner: str, repo: str, branch: str, file_path: str, out_path: str, limit: int = -1) -> int:
    commit_shas = fetch_commit_shas(owner, repo, file_path, branch)
    if limit > 0:
        commit_shas = commit_shas[:limit]

    rows = []
    for sha in commit_shas:
        try:
            csv_text = fetch_raw_text(owner, repo, sha, file_path)
        except Exception as exc:
            print(f"  skipped commit {sha}: {exc}")
            continue
        if not csv_text.strip():
            continue
        try:
            df = pd.read_csv(io.StringIO(csv_text), low_memory=False)
        except Exception as exc:
            print(f"  could not parse commit {sha}: {exc}")
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            if row.empty:
                continue
            rows.append(normalize_record(row.to_dict()))

    out_df = dedupe_and_sort(rows)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False, na_rep="")
    print(f"Wrote {len(out_df)} rows to {out_path}")
    return len(out_df)


def main():
    ap = argparse.ArgumentParser(description="Rebuild props_history.csv from GitHub file history.")
    ap.add_argument("--owner", default="TuckersTruckers", help="GitHub repo owner")
    ap.add_argument("--repo", default="nfl-model", help="GitHub repo name")
    ap.add_argument("--branch", default="main", help="Branch to inspect")
    ap.add_argument("--file", default="nfl_output/props.csv", help="File whose history should be replayed")
    ap.add_argument("--out", default="nfl_output/props_history.csv", help="Output CSV path")
    ap.add_argument("--limit", type=int, default=-1, help="Optional maximum number of historical versions to scan")
    args = ap.parse_args()

    rebuild_history(
        owner=args.owner,
        repo=args.repo,
        branch=args.branch,
        file_path=args.file,
        out_path=args.out,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
