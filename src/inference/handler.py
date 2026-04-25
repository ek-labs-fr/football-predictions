"""AWS Lambda entrypoint for the batch inference pipeline.

Loads model artefacts (model_final_*.pkl + rho.json) and the inference
table for each requested mode from S3, runs predictions, and writes
predictions_{national_wc2026,club}.{csv,parquet} back to the same bucket.

Event shape:
    {"mode": "national" | "club" | "both"}   # default: "both"

The storage backend is selected by the ``DATA_BUCKET`` environment
variable; when unset, all I/O goes to local disk — so the same handler
body runs in pytest, in a dev shell, and in Lambda.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.features import io
from src.inference.predict import predict_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def handler(event: dict[str, Any] | None = None, context: object | None = None) -> dict[str, Any]:
    started = time.time()
    mode = (event or {}).get("mode", "both")
    backend = "s3" if io.using_s3() else "local"
    logger.info("Inference pipeline start — mode=%s backend=%s", mode, backend)

    summaries: list[dict[str, Any]] = []
    if mode in ("national", "both"):
        summaries.append(predict_mode("national"))
    if mode in ("club", "both"):
        summaries.append(predict_mode("club"))

    elapsed = round(time.time() - started, 2)
    logger.info("Inference pipeline done — elapsed=%ss summaries=%s", elapsed, summaries)
    return {
        "status": "ok",
        "mode": mode,
        "backend": backend,
        "elapsed_seconds": elapsed,
        "summaries": summaries,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(handler(), indent=2, default=str))
