"""Map every image of the relevant ISIC Archive collections to build source-level independence.

Where identifiers are missing, source-cohort membership is the structural
substitute: if the evaluation split and the training pool draw from the same
source collection, patient overlap is plausible even when ``patient_id`` is
absent, and if they draw from disjoint collections it is implausible.

SONIC is the motivating case. It is a longitudinal nevus study, so its 9k images
are many photographs of comparatively few subjects, and it publishes neither
``patient_id`` nor ``lesion_id``. An identifier-only audit is blind to exactly
that risk; collection membership is not.

Collections of interest are the source cohorts plus the challenge partitions
that define our own splits. Collection 63 ("Challenge 2018: Task 1-2: Training")
reports 2594 images, which is the evaluation split itself.

Read-only against a public API. Stores only ids and membership.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

SEARCH = "https://api.isic-archive.com/api/v2/images/search/?collections={cid}&limit=100"
USER_AGENT = "dermssl-research-audit/1.0 (UNAB undergraduate thesis; identity provenance audit)"

# id -> label. Source cohorts first, then the challenge partitions that define our splits.
COLLECTIONS = {
    293: "SONIC",
    289: "MSK-1",
    290: "MSK-2",
    288: "MSK-3",
    287: "MSK-4",
    286: "MSK-5",
    292: "UDA-1",
    291: "UDA-2",
    212: "HAM10000",
    249: "BCN20000",
    63: "Challenge2018_Task1-2_Training",
    65: "Challenge2019_Training",
    70: "Challenge2020_Training",
}


def fetch(url: str, timeout: float, retries: int = 5) -> dict:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = exc
            time.sleep(2.5 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} attempts: {url}") from last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    membership: dict[str, list[str]] = {}
    if args.out.is_file():
        membership = json.loads(args.out.read_text(encoding="utf-8")).get("membership", {})
        print(f"resuming with {len(membership)} collections already mapped")

    for cid, label in COLLECTIONS.items():
        if label in membership:
            print(f"  {label}: cached ({len(membership[label])})")
            continue
        ids: list[str] = []
        url: str | None = SEARCH.format(cid=cid)
        reported: int | None = None
        while url:
            payload = fetch(url, args.timeout)
            reported = payload.get("count", reported)
            ids.extend(item["isic_id"] for item in payload.get("results", []) if item.get("isic_id"))
            url = payload.get("next")
            time.sleep(args.sleep)
        membership[label] = ids
        print(f"  {label}: {len(ids)} images (archive reported {reported})", flush=True)
        _write(args.out, membership)

    _write(args.out, membership)
    print(f"\nmapped {len(membership)} collections")


def _write(path: Path, membership: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source": "https://api.isic-archive.com/api/v2/images/search/?collections=",
                "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "collection_ids": COLLECTIONS,
                "counts": {k: len(v) for k, v in membership.items()},
                "membership": membership,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
