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


def test_the_clone_url_is_the_one_this_repo_actually_lives_at():
    """A clone command has to resolve, not merely look filled in.

    The first version only rejected the literal <org> placeholder, so any
    plausible-but-wrong owner - a typo in the username, an old org - sailed
    through and handed Dr. Karen a 404. It now compares against the real
    remote."""
    import re
    import subprocess

    remote = subprocess.run(["git", "remote", "get-url", "origin"],
                            capture_output=True, text=True, cwd=REPO).stdout.strip()
    slug = re.sub(r"^.*github\.com[:/]|\.git$", "", remote)
    assert "/" in slug, f"cannot read a github slug from {remote!r}"

    quoted = []
    for path in CLUSTER.rglob("*.md"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            # <> is in the class on purpose: a <org> placeholder has to be
            # captured so it can be reported as wrong, not skipped as unmatched.
            for found in re.findall(
                    r"github\.com[:/]([\w.<>-]+/[\w.<>-]+?)(?:\.git)?(?=[\s)`\"]|$)", line):
                quoted.append((path.name, n, found))
    assert quoted, "no clone URL documented anywhere in cluster/"
    wrong = [f"{f}:{n} points at {g}, repo is {slug}" for f, n, g in quoted if g != slug]
    assert not wrong, wrong


def test_the_two_weight_caches_are_separate_directories():
    """torch.hub and Hugging Face must not share a cache root.

    They are set on adjacent lines, which makes one of them a copy-paste away
    from pointing at the other's directory - a mistake that leaves both loaders
    half-working instead of failing loudly."""
    import re

    script = (CLUSTER / "finetune" / "run_all.sh").read_text()
    defaults = {}
    for var in ("HF_HOME", "TORCH_HOME"):
        m = re.search(rf'export {var}="\$\{{{var}:-([^}}]+)\}}"', script)
        assert m, f"{var} is no longer defaulted with ${{{var}:-...}}"
        defaults[var] = m.group(1)
    assert defaults["HF_HOME"] != defaults["TORCH_HOME"], (
        f"both caches default to {defaults['HF_HOME']}; the two loaders need "
        "separate roots")
