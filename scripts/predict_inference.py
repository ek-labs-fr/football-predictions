"""Batch-predict scorelines and outcome probabilities for upcoming fixtures.

Thin CLI wrapper around src.inference.predict — runs locally when DATA_BUCKET
is unset, against S3 when it is set.

Usage:
    uv run python scripts/predict_inference.py [--mode national|club|all]
"""

from __future__ import annotations

import argparse
import logging

from src.inference.predict import CONFIGS, predict_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(modes: list[str]) -> None:
    for m in modes:
        predict_mode(m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[*CONFIGS.keys(), "all"],
        default="all",
    )
    args = parser.parse_args()
    main(list(CONFIGS.keys()) if args.mode == "all" else [args.mode])
