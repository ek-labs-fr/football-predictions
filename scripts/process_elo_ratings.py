"""Process Elo ratings from Kaggle and map to API-Football team IDs.

Usage:
    uv run python scripts/process_elo_ratings.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EXTERNAL_DIR = Path("data/external")
PROCESSED_DIR = Path("data/processed")

# Elo team name -> API-Football team name mapping (where they differ)
_NAME_MAP: dict[str, str] = {
    "USA": "USA",
    "South Korea": "South Korea",
    "North Korea": "North Korea",
    "Iran": "Iran",
    "Ivory Coast": "Ivory Coast",
    "China": "China",
    "Czech Republic": "Czech Republic",
    "Czechia": "Czech Republic",
    "Swaziland": "Swaziland",
    "Eswatini": "Swaziland",
    "Turkey": "Turkey",
    "Turkiye": "Turkey",
    "Cape Verde": "Cape Verde",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Kyrgyzstan": "Kyrgyzstan",
    "Brunei": "Brunei",
    "Timor-Leste": "Timor Leste",
    "Bosnia-Herzegovina": "Bosnia And Herzegovina",
    "Bosnia and Herzegovina": "Bosnia And Herzegovina",
    "Trinidad and Tobago": "Trinidad And Tobago",
    "Antigua and Barbuda": "Antigua And Barbuda",
    "St Kitts and Nevis": "Saint Kitts And Nevis",
    "Saint Kitts and Nevis": "Saint Kitts And Nevis",
    "St Vincent and the Grenadines": "Saint Vincent And The Grenadines",
    "Saint Vincent and the Grenadines": "Saint Vincent And The Grenadines",
    "St Lucia": "Saint Lucia",
    "São Tomé and Príncipe": "Sao Tome And Principe",
    "Sao Tome and Principe": "Sao Tome And Principe",
    "Turks and Caicos Islands": "Turks And Caicos Islands",
    "Curacao": "Curacao",
    "Hong Kong": "Hong Kong",
    "Congo-Brazzaville": "Congo",
    "Congo-Kinshasa": "DR Congo",
    "Burma": "Myanmar",
    "Dahomey": "Benin",
    "Upper Volta": "Burkina Faso",
    "Zaire": "DR Congo",
    "Dutch East Indies": "Indonesia",
    "Rhodesia": "Zimbabwe",
}


def main() -> None:
    # Load Elo data
    elo_path = EXTERNAL_DIR / "elo_ratings_kaggle.csv"
    if not elo_path.exists():
        logger.error("Elo ratings file not found at %s", elo_path)
        return

    elo = pd.read_csv(elo_path)
    elo["date"] = pd.to_datetime(elo["date"], format="mixed")
    logger.info(
        "Loaded %d Elo rows (%d dates, %d teams)",
        len(elo),
        elo["date"].nunique(),
        elo["team"].nunique(),
    )

    # Load team lookup
    lookup_path = PROCESSED_DIR / "team_lookup.json"
    if not lookup_path.exists():
        logger.error("team_lookup.json not found — run bootstrap_data.py first")
        return
    team_lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    lower_lookup = {k.lower(): v for k, v in team_lookup.items()}

    def resolve(name: str) -> int | None:
        # Normalize non-breaking spaces
        name = name.replace("\xa0", " ")
        if name in team_lookup:
            return team_lookup[name]
        mapped = _NAME_MAP.get(name, name)
        if mapped in team_lookup:
            return team_lookup[mapped]
        if name.lower() in lower_lookup:
            return lower_lookup[name.lower()]
        return None

    elo["team"] = elo["team"].str.replace("\xa0", " ", regex=False)
    elo["team_id"] = elo["team"].apply(resolve)

    matched = elo[elo["team_id"].notna()].copy()
    matched["team_id"] = matched["team_id"].astype(int)
    unmatched = sorted(elo[elo["team_id"].isna()]["team"].unique())

    logger.info(
        "Mapped %d/%d rows (%.1f%%), %d unmatched teams",
        len(matched),
        len(elo),
        len(matched) / len(elo) * 100,
        len(unmatched),
    )
    if unmatched:
        logger.info("Unmatched: %s", unmatched[:20])

    # Save processed Elo ratings
    result = matched[["team_id", "team", "date", "rating"]].rename(
        columns={"team": "team_name", "date": "elo_date", "rating": "elo_rating"}
    )
    result = result.sort_values(["elo_date", "elo_rating"], ascending=[True, False]).reset_index(
        drop=True
    )

    output_path = EXTERNAL_DIR / "elo_ratings.csv"
    result.to_csv(output_path, index=False)
    logger.info(
        "Saved %d Elo rows to %s (date range: %s to %s)",
        len(result),
        output_path,
        result["elo_date"].min(),
        result["elo_date"].max(),
    )


if __name__ == "__main__":
    main()
