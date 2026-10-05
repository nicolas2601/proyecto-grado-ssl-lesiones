"""docs/34 and cluster_env.json are generated from resultados/entorno and checked for freshness."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_cluster_env as bce  # noqa: E402

SECRET_OR_HOST = re.compile(r"(token|secret|password|netrc)\s*[=:]|https?://[^\s]*\.unab\.edu\.co|192\.168\.", re.I)


def test_committed_env_matches_its_inputs():
    assert bce.main(["--check"]) == 0


def test_no_hostnames_ips_or_tokens_in_published_files():
    for path in (bce.DOC, bce.ENV / "cluster_env.json", bce.ENV / "hardware.json"):
        assert not SECRET_OR_HOST.search(path.read_text(encoding="utf-8")), path


def test_clean_pool_is_smaller_than_the_v7_pool():
    pool = json.loads((bce.ENV / "cluster_env.json").read_text(encoding="utf-8"))["ssl_pool"]
    assert pool["clean_effective_pool_n"] < pool["effective_pool_n"]
    assert pool["clean_effective_pool_n"] >= pool["pool_n"] - pool["clean_exclusion_n"]  # some excluded ids are not in the pool
