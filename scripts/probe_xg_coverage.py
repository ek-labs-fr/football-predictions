"""One-off probe: how is xG coverage in the cached club statistics?"""

from __future__ import annotations

import pandas as pd

XG_IDS_PATH = "data/processed/xg_fixture_ids.txt"
FIXTURES_PATH = "data/processed/all_fixtures_club.csv"


def main() -> None:
    ids: set[int] = set()
    with open(XG_IDS_PATH, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                ids.add(int(line))
    print(f"Loaded xG fixture ids: {len(ids)}")

    fx = pd.read_csv(FIXTURES_PATH)
    print(f"All club fixtures: {len(fx)}")
    fx["date"] = pd.to_datetime(fx["date"], utc=True, errors="coerce")
    fx["year"] = fx["date"].dt.year
    fx["has_xg"] = fx["fixture_id"].isin(ids)

    print("\nCoverage by year (last 15):")
    yearly = (
        fx.groupby("year")["has_xg"]
        .agg(["sum", "count"])
        .assign(pct=lambda d: (d["sum"] / d["count"] * 100).round(1))
        .tail(15)
    )
    print(yearly)

    print("\nCoverage by (league_id, year) for 2022+:")
    recent = fx[fx["year"] >= 2022]
    by_league = (
        recent.groupby(["league_id", "year"])["has_xg"]
        .agg(["sum", "count"])
        .assign(pct=lambda d: (d["sum"] / d["count"] * 100).round(1))
    )
    print(by_league)


if __name__ == "__main__":
    main()
