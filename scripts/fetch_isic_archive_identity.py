"""Fetch patient_id and lesion_id for a cohort directly from the ISIC Archive API.

Why this exists: every identity audit so far used third-party mirrors (the IMA++
metadata dump, derived local CSVs). One of those shipped ``patient_id`` as
``dummy_0, dummy_1, ...`` one per image, which made a leakage search return zero
by construction. Mirrors cannot be trusted for an accreditation claim.

This reads the authoritative archive (https://api.isic-archive.com) one image at
a time, records the raw response hash per image, and separates three states that
must never be collapsed:

    present   - the archive returns the field
    absent    - the archive answers and the field is not there
    unknown   - the image is not in the archive, or the request failed

``absent`` is a fact about the archive. ``unknown`` is a fact about our access.
Reporting either as "independent" is the error this script exists to prevent.

Responses are cached on disk so re-runs are cheap and the pull is resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.isic-archive.com/api/v2/images/{isic_id}/"
USER_AGENT = "dermssl-research-audit/1.0 (UNAB undergraduate thesis; identity provenance audit)"

IDENTITY_FIELDS = ("patient_id", "lesion_id")


def fetch_one(isic_id: str, timeout: float) -> dict:
    request = urllib.request.Request(
        API.format(isic_id=isic_id),
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        return {"state": "absent_from_archive" if exc.code == 404 else "request_failed",
                "http_status": exc.code}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"state": "request_failed", "error": str(exc)}

    payload = json.loads(raw)
    # The archive nests clinical metadata; identity fields appear there when present.
    clinical = payload.get("metadata", {}).get("clinical", {})
    record = {
        "state": "fetched",
        "http_status": 200,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
    }
    for field in IDENTITY_FIELDS:
        value = clinical.get(field)
        record[field] = value if value not in (None, "") else None
        record[f"{field}_state"] = "present" if record[field] else "absent_in_archive"
    return record


def load_cohort(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        return [json.loads(line)["image_id"] for line in path.open(encoding="utf-8")]
    return [line.strip() for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path, help="cases.jsonl or a plain list of ISIC ids")
    parser.add_argument("--out", type=Path, required=True, help="JSON cache / result file")
    parser.add_argument("--sleep", type=float, default=0.35, help="Seconds between requests")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    cohort = load_cohort(args.cohort)
    cache: dict[str, dict] = {}
    if args.out.is_file():
        cache = json.loads(args.out.read_text(encoding="utf-8")).get("records", {})
        print(f"resuming with {len(cache)} cached records")

    try:
        for index, isic_id in enumerate(cohort, start=1):
            if isic_id in cache and cache[isic_id].get("state") == "fetched":
                continue
            cache[isic_id] = fetch_one(isic_id, args.timeout)
            if index % 50 == 0:
                print(f"  {index}/{len(cohort)}", flush=True)
                _write(args.out, cohort, cache)
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("interrupted; writing partial cache")

    _write(args.out, cohort, cache)
    _report(cohort, cache)


def _summarize(cohort: list[str], cache: dict[str, dict]) -> dict:
    summary = {"cohort_size": len(cohort)}
    for field in IDENTITY_FIELDS:
        states: dict[str, int] = {}
        for isic_id in cohort:
            record = cache.get(isic_id, {})
            if record.get("state") != "fetched":
                key = record.get("state", "not_attempted")
            else:
                key = record[f"{field}_state"]
            states[key] = states.get(key, 0) + 1
        summary[field] = states
    return summary


def _write(path: Path, cohort: list[str], cache: dict[str, dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "source": "https://api.isic-archive.com/api/v2/images/",
                "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "summary": _summarize(cohort, cache),
                "records": cache,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _report(cohort: list[str], cache: dict[str, dict]) -> None:
    summary = _summarize(cohort, cache)
    print(f"\ncohorte: {summary['cohort_size']}")
    for field in IDENTITY_FIELDS:
        print(f"  {field}:")
        for state, count in sorted(summary[field].items(), key=lambda kv: -kv[1]):
            print(f"    {state:<22} {count:>6} ({count / summary['cohort_size']:.1%})")


if __name__ == "__main__":
    main()
