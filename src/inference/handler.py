"""AWS Lambda entrypoint for the batch inference pipeline.

Runs upcoming + holdout predictions for both national and club modes,
then writes one JSON file per competition under web/data/ for the
static front-end to consume.

The storage backend is selected by the ``DATA_BUCKET`` environment
variable; when unset, all I/O goes to local disk — so the same handler
body runs in pytest, in a dev shell, and in Lambda.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.features import io
from src.inference.predict import publish_dashboard_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def handler(event: dict[str, Any] | None = None, context: object | None = None) -> dict[str, Any]:
    started = time.time()
    backend = "s3" if io.using_s3() else "local"
    logger.info("Inference pipeline start — backend=%s", backend)

    summary = publish_dashboard_json()

    elapsed = round(time.time() - started, 2)
    logger.info("Inference pipeline done — elapsed=%ss", elapsed)
    return {
        "status": "ok",
        "backend": backend,
        "elapsed_seconds": elapsed,
        **summary,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(handler(), indent=2, default=str))
