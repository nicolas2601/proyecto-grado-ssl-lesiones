"""Checks scripts/pretrain.py makes before any GPU time, outside the training loop.

``code_revision`` came from ``GIT_COMMIT`` and silently became None when the
variable was not exported; two finals with None then passed the evaluation's
"same revision" guard as equal. A launch without a real commit is refused.
"""

from __future__ import annotations

import pytest

from dermssl.pretrain import launch_guards as lg

SHA = "f1c2053d7e9e3f3eae4e1665d6942243b02e3872"


class TestCodeRevision:
    def test_a_full_sha_that_matches_head_passes(self):
        assert lg.require_code_revision({"GIT_COMMIT": SHA}, repo_head=SHA) == SHA

    def test_a_missing_git_commit_refuses(self):
        with pytest.raises(lg.LaunchRefused, match="GIT_COMMIT"):
            lg.require_code_revision({}, repo_head=SHA)

    def test_an_empty_git_commit_refuses(self):
        with pytest.raises(lg.LaunchRefused, match="GIT_COMMIT"):
            lg.require_code_revision({"GIT_COMMIT": ""}, repo_head=SHA)

    @pytest.mark.parametrize("value", ["f1c2053", "HEAD", SHA.upper(), SHA + "0"])
    def test_anything_but_a_full_lowercase_sha_refuses(self, value):
        with pytest.raises(lg.LaunchRefused, match="40"):
            lg.require_code_revision({"GIT_COMMIT": value}, repo_head=None)

    def test_a_stale_export_that_is_not_head_refuses(self):
        with pytest.raises(lg.LaunchRefused, match="HEAD"):
            lg.require_code_revision({"GIT_COMMIT": SHA}, repo_head="0" * 40)

    def test_outside_a_git_checkout_the_sha_is_trusted(self):
        assert lg.require_code_revision({"GIT_COMMIT": SHA}, repo_head=None) == SHA


def _pretrain_args(tmp_path, *extra):
    import pretrain  # noqa: PLC0415

    return pretrain, pretrain.parse_args([
        "--method", "simsiam_dinov2", "--image-root", str(tmp_path), "--output-dir", str(tmp_path / "out"),
        *extra,
    ])


class TestPretrainEntryPoint:
    @pytest.fixture(autouse=True)
    def _scripts_on_path(self, monkeypatch):
        from pathlib import Path  # noqa: PLC0415

        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))

    def test_a_real_run_without_git_commit_is_refused(self, tmp_path):
        pretrain, args = _pretrain_args(tmp_path)
        with pytest.raises(lg.LaunchRefused, match="GIT_COMMIT"):
            pretrain.check_launch(args, env={})

    def test_a_smoke_needs_no_commit(self, tmp_path):
        pretrain, args = _pretrain_args(tmp_path, "--smoke")
        pretrain.check_launch(args, env={})
