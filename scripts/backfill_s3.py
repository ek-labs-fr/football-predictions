"""One-off backfill of the local data/raw/ cache into S3.

Rewrites the hash-named local cache files into the S3 path layout the Lambda
pipeline uses, and seeds the per-domain manifest so the daily Lambda won't
re-fetch details for any fixture that is already in S3.

Layout written to S3:

    {domain}/{subdir}/{fixture_id}.json   for the 5 per-fixture detail endpoints
                                          (fixtures_events, fixtures_statistics,
                                          fixtures_lineups, odds, injuries)
    {domain}/historical/{subdir}/<filename>.json    for everything else — raw
                                          reference data (leagues, teams,
                                          standings, venues, transfers, coaches,
                                          players, players_squads,
                                          teams_statistics, fixtures per
                                          league/season, fixtures_headtohead)
    {domain}/manifests/fixtures_seen.json            manifest of processed IDs

Per-fixture files are only rewritten into the "live" layout if their JSON
carries a ``parameters.fixture`` field — anything else lands under
``historical/`` to keep the live prefix clean.

Usage:
    uv run python scripts/backfill_s3.py --bucket BUCKET [--dry-run] [--workers 20]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DOMAINS = ("club", "national")
PER_FIXTURE_SUBDIRS = (
    "fixtures_events",
    "fixtures_statistics",
    "fixtures_lineups",
    "odds",
    "injuries",
)
LOCAL_ROOT = Path("data/raw")


def _local_fixture_id(path: Path) -> int | None:
    """Extract parameters.fixture from a cached API response; None if absent."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping %s: %s", path, exc)
        return None
    params = payload.get("parameters") or {}
    fid = params.get("fixture")
    try:
        return int(fid) if fid is not None else None
    except (TypeError, ValueError):
        return None


def _plan_uploads(
    domain: str,
) -> tuple[list[tuple[Path, str]], set[int]]:
    """Compute (source_path, s3_key) pairs and the manifest fixture-ID set."""
    uploads: list[tuple[Path, str]] = []
    fixture_ids: set[int] = set()

    domain_root = LOCAL_ROOT / domain
    if not domain_root.exists():
        logger.warning("No local data for domain %s — skipping", domain)
        return [], set()

    for subdir in domain_root.iterdir():
        if not subdir.is_dir():
            continue
        subname = subdir.name

        for path in subdir.glob("*.json"):
            if subname in PER_FIXTURE_SUBDIRS:
                fid = _local_fixture_id(path)
                if fid is not None:
                    uploads.append((path, f"{domain}/{subname}/{fid}.json"))
                    fixture_ids.add(fid)
                    continue
                # Detail file without a fixture param — archive instead.
            uploads.append(
                (path, f"{domain}/historical/{subname}/{path.name}")
            )

    # Also backfill root-level metadata files like leagues.json, teams.json
    for path in domain_root.glob("*.json"):
        uploads.append((path, f"{domain}/historical/{path.name}"))

    return uploads, fixture_ids


def _upload_one(
    s3: boto3.client,  # noqa: ANN001
    bucket: str,
    path: Path,
    key: str,
) -> tuple[str, bool, str | None]:
    try:
        s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "application/json"})
        return key, True, None
    except Exception as exc:  # noqa: BLE001
        return key, False, str(exc)


def _put_manifest(s3, bucket: str, domain: str, ids: set[int]) -> None:  # noqa: ANN001
    key = f"{domain}/manifests/fixtures_seen.json"
    body = json.dumps(
        {
            "domain": domain,
            "updated_at": datetime.now(UTC).isoformat(),
            "count": len(ids),
            "fixture_ids": sorted(ids),
            "source": "backfill_s3.py",
        },
        indent=2,
    ).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    logger.info("Manifest written s3://%s/%s (%d ids)", bucket, key, len(ids))


def main(bucket: str, workers: int, dry_run: bool) -> None:
    s3 = boto3.client("s3") if not dry_run else None

    if not dry_run:
        # Sanity check — bucket must exist and we must have access.
        s3.head_bucket(Bucket=bucket)

    grand_uploads = 0
    grand_failures = 0

    for domain in DOMAINS:
        logger.info("=== Planning uploads for domain=%s ===", domain)
        uploads, fixture_ids = _plan_uploads(domain)
        logger.info(
            "Domain=%s: %d files to upload, %d distinct fixture IDs for manifest",
            domain,
            len(uploads),
            len(fixture_ids),
        )

        if dry_run:
            sample = uploads[:3]
            for src, key in sample:
                logger.info("DRY  %s -> s3://%s/%s", src, bucket, key)
            logger.info("… (dry-run — no upload)")
            grand_uploads += len(uploads)
            continue

        failures = 0
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_upload_one, s3, bucket, src, key) for src, key in uploads
            ]
            for fut in as_completed(futures):
                key, ok, err = fut.result()
                done += 1
                if not ok:
                    failures += 1
                    logger.warning("FAIL %s: %s", key, err)
                if done % 500 == 0:
                    logger.info("  progress: %d/%d uploaded", done, len(uploads))

        logger.info(
            "Domain=%s: uploaded %d/%d (failures=%d)",
            domain, done - failures, len(uploads), failures,
        )
        grand_uploads += done
        grand_failures += failures

        _put_manifest(s3, bucket, domain, fixture_ids)

    logger.info("=== Backfill complete: %d uploads, %d failures ===", grand_uploads, grand_failures)
    if grand_failures:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill local raw cache into S3")
    parser.add_argument(
        "--bucket", required=True,
        help="Target S3 bucket (from CDK output DataBucketName)",
    )
    parser.add_argument("--workers", type=int, default=20, help="Concurrent uploads")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; don't upload")
    args = parser.parse_args()
    try:
        main(bucket=args.bucket, workers=args.workers, dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(1)
