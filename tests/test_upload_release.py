"""Upload script: dry run by default, private token never printed, limits enforced."""

import os

import pytest

from dermssl.release import package
from scripts import upload_release as ur

SECRET = "ghp_supersecretvalue123"


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "v7"
    (root / "B_s42" / "onnx").mkdir(parents=True)
    (root / "B_s42" / "onnx" / "model.onnx").write_bytes(b"graph")
    (root / "README.md").write_text("notes")
    package.write_sha256sums(root)
    return root


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "token"
    path.write_text(SECRET + "\n")
    os.chmod(path, 0o600)
    return path


def test_token_file_must_be_private(token_file):
    assert ur.read_token(token_file) == SECRET
    os.chmod(token_file, 0o644)
    with pytest.raises(ur.UploadRefused, match="chmod 600"):
        ur.read_token(token_file)


def test_checksums_are_verified_before_anything_is_sent(release):
    assert ur.verify_checksums(release) == 2
    (release / "README.md").write_text("tampered")
    with pytest.raises(ur.UploadRefused, match="README.md"):
        ur.verify_checksums(release)


def test_unlisted_files_are_refused(release):
    (release / "extra.bin").write_bytes(b"x")
    with pytest.raises(ur.UploadRefused, match="extra.bin"):
        ur.verify_checksums(release)


def test_assets_are_flattened_and_limited(release):
    names = [name for _, name in ur.github_assets(release)]
    assert "B_s42__onnx__model.onnx" in names and "README.md" in names
    with pytest.raises(ur.UploadRefused, match="model.onnx"):
        ur.github_assets(release, limit_bytes=4)


def test_dry_run_is_the_default_and_never_prints_the_token(release, token_file, tmp_path, capsys):
    code = ur.main(["--release", str(release), "--target", "github", "--github-repo", "o/r",
                    "--token-file", str(token_file), "--staging", str(tmp_path / "stage")])
    assert code == 0
    out = capsys.readouterr().out
    assert "dry run" in out and SECRET not in out


def test_execute_passes_the_token_only_through_the_environment(release, token_file, tmp_path, capsys):
    calls = []
    args = ur.parse_args(["--release", str(release), "--target", "github", "--github-repo", "o/r",
                          "--token-file", str(token_file), "--staging", str(tmp_path / "s"),
                          "--execute"])
    command = ur.upload_github(release, args, SECRET,
                               runner=lambda cmd, **kw: calls.append((cmd, kw)))
    assert command[:4] == ["gh", "release", "create", "v7.0.0"]
    assert SECRET not in " ".join(command) and calls[0][1]["env"]["GH_TOKEN"] == SECRET
    assert SECRET not in capsys.readouterr().out


def test_hf_upload_uses_upload_folder_only_when_executing(release, tmp_path):
    seen = {}

    class FakeApi:
        def __init__(self, token):
            seen["token"] = token

        def create_repo(self, **kw):
            seen["create"] = kw

        def upload_folder(self, **kw):
            seen["upload"] = kw

    args = ur.parse_args(["--release", str(release), "--target", "hf", "--hf-repo", "o/m"])
    ur.upload_hf(release, args, None, api_factory=FakeApi)
    assert not seen
    args.execute = True
    ur.upload_hf(release, args, SECRET, api_factory=FakeApi)
    assert seen["upload"]["repo_id"] == "o/m" and seen["token"] == SECRET
    assert "Claude" not in seen["upload"]["commit_message"]


def test_execute_without_a_token_file_is_an_argument_error(release, monkeypatch):
    monkeypatch.delenv("RELEASE_TOKEN_FILE", raising=False)
    with pytest.raises(SystemExit):
        ur.parse_args(["--release", str(release), "--target", "hf", "--hf-repo", "o/m", "--execute"])
