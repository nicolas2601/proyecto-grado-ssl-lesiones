"""A result is only traceable to a commit if the tree it ran from was clean."""

import subprocess

import pytest

from dermssl import run_provenance as rp


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_clean_tree_returns_head(repo):
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert rp.clean_revision(repo) == head


def test_untracked_file_refuses(repo):
    # The 2026-09-27 incident: an untracked copy of the script shadowed the commit.
    (repo / "b.py").write_text("y = 2\n")
    with pytest.raises(rp.DirtyTreeError, match="b.py"):
        rp.clean_revision(repo)


def test_modified_tracked_file_refuses(repo):
    (repo / "a.py").write_text("x = 2\n")
    with pytest.raises(rp.DirtyTreeError, match="a.py"):
        rp.clean_revision(repo)


def test_not_a_repo_refuses(tmp_path):
    with pytest.raises(rp.DirtyTreeError):
        rp.clean_revision(tmp_path / "nowhere")


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "ids.txt"
    path.write_text("ISIC_1\nISIC_2\n")
    assert rp.file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
