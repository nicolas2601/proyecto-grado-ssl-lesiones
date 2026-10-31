"""Snapshot patient_id and lesion_id for the whole ISIC Archive via cursor pagination.

Why the whole archive instead of per-cohort lookups: the identifier namespaces in
our local metadata do not bridge. Evaluation lesion ids pulled from the archive
are ``IL_*``; the pool's local CSV ships ``HAM_*``, ``BCN_*`` and ``MSK4_*``,
which are per-contributor systems. Comparing those strings is guaranteed to
return zero collisions and would read as independence while proving nothing.

The archive is the one namespace both sides share: spot checks returned ``IL_*``
lesion ids and ``IP_*`` patient ids for pool images as well as evaluation images.
Pulling every image once puts every exposure into that single namespace, which is
the bridge the gate requires.

One page is 100 images, so the full archive costs ~5.5k requests instead of one
per image. The snapshot is written incrementally and is resumable by cursor.

Read-only against a public API. Stores only identifiers, never pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

SEARCH_URL = "https://api.isic-archive.com/api/v2/images/search/?limit=100"
USER_AGENT = "dermssl-research-audit/1.0 (UNAB undergraduate thesis; identity provenance audit)"


def fetch_page(url: str, timeout: float, retries: int = 4) -> tuple[dict, str]:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return json.loads(raw), hashlib.sha256(raw).hexdigest()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            # The archive rate-limits under sustained paging; back off rather than
            # dropping pages, because a silently short snapshot would understate
            # coverage and bias the gate toward a false clean verdict.
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"page failed after {retries} attempts: {url}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    records: dict[str, dict] = {}
    page_hashes: list[str] = []
    url: str | None = SEARCH_URL
    total: int | None = None
    pages = 0

    if args.out.is_file():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        records = previous.get("records", {})
        page_hashes = previous.get("page_sha256", [])
        url = previous.get("next_cursor") or SEARCH_URL
        pages = previous.get("pages_fetched", 0)
        print(f"resuming at page {pages} with {len(records)} records")

    while url:
        payload, page_hash = fetch_page(url, args.timeout)
        page_hashes.append(page_hash)
        pages += 1
        total = payload.get("count", total)
        for item in payload.get("results", []):
            clinical = item.get("metadata", {}).get("clinical", {})
            isic_id = item.get("isic_id")
            if not isic_id:
                continue
            records[isic_id] = {
                "patient_id": clinical.get("patient_id") or None,
                "lesion_id": clinical.get("lesion_id") or None,
            }
        url = payload.get("next")
        if pages % 25 == 0:
            print(f"  page {pages}, {len(records)}/{total} records", flush=True)
            _write(args.out, records, page_hashes, pages, url, total)
        if args.max_pages and pages >= args.max_pages:
            print("reached --max-pages")
            break
        time.sleep(args.sleep)

    _write(args.out, records, page_hashes, pages, url, total)

    with_patient = sum(1 for v in records.values() if v["patient_id"])
    with_lesion = sum(1 for v in records.values() if v["lesion_id"])
    print(f"\nsnapshot: {len(records)} images (archive reports {total})")
    print(f"  patient_id present: {with_patient} ({with_patient / max(len(records), 1):.1%})")
    print(f"  lesion_id present:  {with_lesion} ({with_lesion / max(len(records), 1):.1%})")


def _write(
    path: Path,
    records: dict[str, dict],
    page_hashes: list[str],
    pages: int,
    next_cursor: str | None,
    total: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source": SEARCH_URL,
                "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "archive_reported_count": total,
                "pages_fetched": pages,
                "next_cursor": next_cursor,
                "complete": next_cursor is None,
                "page_sha256": page_hashes,
                "records": records,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
