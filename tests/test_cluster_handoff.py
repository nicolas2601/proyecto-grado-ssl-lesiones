"""The handoff package Dr. Karen receives must match what the code actually does.

Every number and dependency in cluster/ is a promise to someone who cannot ask
us a question mid-run. These tests compare those promises against the protocol
and the code, so the package can never drift into telling her to wait for jobs
that were never queued, or to install an environment where the run silently
loses its dashboard.
"""

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl.finetune import protocol  # noqa: E402

CLUSTER = REPO / "cluster"
DOCS = [CLUSTER / "finetune" / "README.md",
        CLUSTER / "finetune" / "run_all.sh",
        CLUSTER / "finetune" / "finetune_bank.sbatch"]


def test_no_document_quotes_a_job_count_the_manifest_does_not_produce():
    """The '24 jobs' wording outlived the third dataset by a week.

    ISIC 2020 left the manifest on 2026-09-04 and the bank went from 24 jobs to
    16, but two lines kept telling Dr. Karen to wait for 24 that would never
    arrive."""
    expected = len(protocol.manifest())
    stale = []
    for path in DOCS:
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for count in re.findall(r"\b(\d+)\s+(?:jobs|corridas)\b", line):
                if int(count) != expected:
                    stale.append(f"{path.name}:{n}: says {count}, manifest has {expected}")
    assert not stale, stale


@pytest.mark.parametrize("package", ["tensorboard", "timm", "torch", "scikit-learn"])
def test_the_cluster_environment_installs_what_the_runner_imports(package):
    """A tracker whose library is missing degrades to a NullTracker in silence:
    the run finishes, the JSON is written, and the dashboard never exists."""
    reqs = (CLUSTER / "requirements.txt").read_text()
    assert re.search(rf"^{re.escape(package)}[=<>~]", reqs, re.M), (
        f"{package} is imported by the runner but absent from cluster/requirements.txt")


def test_the_clone_url_is_real():
    """`git clone https://github.com/<org>/...` is not a command anyone can run."""
    for path in CLUSTER.rglob("*.md"):
        text = path.read_text()
        assert "<org>" not in text, f"{path.name} still ships the <org> placeholder"


def test_the_launcher_does_not_dead_end_when_weights_can_be_downloaded():
    """Karen's first command must not stop at a release that does not exist.

    Step 3 used to hard-fail with "copy dermssl-weights-cache.tar.gz (GitHub
    Release cluster-assets-v2)". That release was never published, so on a
    cluster with internet - the common case - the one-command launcher died
    pointing at a file nobody could fetch. With network available the run
    should fall through and let timm/torch.hub download the weights."""
    import subprocess

    out = subprocess.run(
        ["bash", str(CLUSTER / "finetune" / "run_all.sh"), "--dry-run"],
        capture_output=True, text=True, cwd=REPO,
        env={**__import__("os").environ,
             "SCRATCH": "/tmp/dermssl-handoff-test",
             "DERMSSL_ASSUME_NETWORK": "1"},
    )
    assert "weight caches missing" not in out.stdout + out.stderr
    for step in ("1/6", "2/6", "3/6", "4/6", "5/6", "6/6"):
        assert step in out.stdout, f"launcher never reached {step}:\n" + out.stdout
    assert out.returncode == 0, (
        f"a dry-run must not fail (exit {out.returncode}):\n{out.stdout}\n{out.stderr}")


def test_an_existing_weight_cache_is_respected():
    """Karen's lab may already have a shared Hugging Face cache.

    The launcher used to export HF_HOME unconditionally, so a cluster-wide
    cache was ignored and every model was downloaded again into scratch."""
    import os
    import subprocess

    shared = "/tmp/dermssl-shared-hf"
    os.makedirs(shared, exist_ok=True)
    out = subprocess.run(
        ["bash", str(CLUSTER / "finetune" / "run_all.sh"), "--dry-run"],
        capture_output=True, text=True, cwd=REPO,
        env={**os.environ, "SCRATCH": "/tmp/dermssl-handoff-test",
             "DERMSSL_ASSUME_NETWORK": "1", "HF_HOME": shared},
    )
    assert shared in out.stdout, (
        "the launcher ignored a pre-set HF_HOME:\n" + out.stdout)
