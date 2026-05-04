"""Quick check: generate rationales for all upcoming club fixtures and print them.

No production side-effects — only loads existing artefacts and inference data.
"""

from __future__ import annotations

import logging

from src.inference.predict import predict_upcoming

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    df = predict_upcoming("club")
    if df.empty:
        print("No upcoming fixtures.")
        return
    cols = ["date", "home_team_name", "away_team_name", "predicted_score", "rationale"]
    rows = df[cols].head(20)
    for _, r in rows.iterrows():
        date = str(r["date"])[:10]
        score = r["predicted_score"]
        home = r["home_team_name"]
        away = r["away_team_name"]
        rationale = r["rationale"] or "(no rationale)"
        print(f"{date}  {home:<22} {score:<5} {away:<22}  ->  {rationale}")


if __name__ == "__main__":
    main()
