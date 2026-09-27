"""The keepalive has to be loud when it stops working.

Its whole job is to protect a run that lasts hours, so the failure mode that
matters is not "it crashed" -- it is "it kept logging beats while the hub was
refusing every one of them".
"""

from __future__ import annotations

import pytest

from dermssl import hub_keepalive as ka

ENVIRON = {
    "JUPYTERHUB_ACTIVITY_URL": "http://hub:8081/hub/api/users/nmoreno534/activity",
    "JUPYTERHUB_API_TOKEN": "t0ken",
    "JUPYTERHUB_SERVER_NAME": "",
}


class RecordingPoster:
    def __init__(self, status: int = 200):
        self.status = status
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, url, token, payload):
        self.calls.append((url, token, dict(payload)))
        return self.status


class TestReportActivity:
    def test_posts_to_the_hub_activity_url_with_the_server_key(self):
        poster = RecordingPoster()

        stamp = ka.report_activity(ENVIRON, poster=poster, now=lambda: "2026-09-27T01:36:02Z")

        url, token, payload = poster.calls[0]
        assert url == ENVIRON["JUPYTERHUB_ACTIVITY_URL"]
        assert token == "t0ken"
        assert payload["servers"] == {"": {"last_activity": "2026-09-27T01:36:02Z"}}
        assert stamp == "2026-09-27T01:36:02Z"

    def test_named_server_goes_under_its_own_key(self):
        poster = RecordingPoster()

        ka.report_activity({**ENVIRON, "JUPYTERHUB_SERVER_NAME": "gpu"}, poster=poster)

        assert set(poster.calls[0][2]["servers"]) == {"gpu"}

    @pytest.mark.parametrize("missing", ["JUPYTERHUB_ACTIVITY_URL", "JUPYTERHUB_API_TOKEN"])
    def test_refuses_outside_a_hub_managed_server(self, missing):
        environ = {**ENVIRON, missing: ""}

        with pytest.raises(ka.KeepaliveRefused, match=missing):
            ka.report_activity(environ, poster=RecordingPoster())

    @pytest.mark.parametrize("status", [403, 404, 500])
    def test_a_rejected_report_is_an_error_not_a_warning(self, status):
        """A 403 that only printed a warning would leave the run unprotected
        while the log claimed otherwise -- the exact silent no-op family."""
        with pytest.raises(ka.KeepaliveRefused, match=str(status)):
            ka.report_activity(ENVIRON, poster=RecordingPoster(status))


class TestKeepAliveLoop:
    def test_sends_one_beat_per_interval_and_sleeps_between_them(self):
        poster = RecordingPoster()
        slept: list[float] = []

        sent = ka.keep_alive(
            30.0, environ=ENVIRON, poster=poster, sleeper=slept.append, beats=3, log=lambda _: None
        )

        assert sent == 3
        assert len(poster.calls) == 3
        # Two gaps between three beats, and no trailing sleep after the last.
        assert slept == [30.0, 30.0]

    def test_the_first_beat_fails_before_the_expensive_work_starts(self):
        poster = RecordingPoster(403)

        with pytest.raises(ka.KeepaliveRefused):
            ka.keep_alive(30.0, environ=ENVIRON, poster=poster, sleeper=lambda _: None, beats=5)

        assert len(poster.calls) == 1, "siguio latiendo despues de que el hub lo rechazara"

    def test_a_failure_halfway_through_also_raises(self):
        class FailsOnThird(RecordingPoster):
            def __call__(self, url, token, payload):
                super().__call__(url, token, payload)
                return 200 if len(self.calls) < 3 else 500

        poster = FailsOnThird()

        with pytest.raises(ka.KeepaliveRefused, match="500"):
            ka.keep_alive(
                1.0,
                environ=ENVIRON,
                poster=poster,
                sleeper=lambda _: None,
                beats=10,
                log=lambda _: None,
            )

        assert len(poster.calls) == 3

    @pytest.mark.parametrize("interval", [0, -1, -30.0])
    def test_a_non_positive_interval_is_refused(self, interval):
        """Zero would spin without waiting; negative is nonsense. Either one
        reads as "keepalive configured" and is not one."""
        with pytest.raises(ValueError, match="keepalive"):
            ka.keep_alive(interval, environ=ENVIRON, poster=RecordingPoster(), beats=1)


class TestItImportsInACleanInterpreter:
    def test_no_torch_no_wandb_needed(self):
        """It runs beside the chain on a pod, so it must not drag the stack in."""
        import subprocess
        import sys
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "src"
        result = subprocess.run(
            [sys.executable, "-c", "import dermssl.hub_keepalive"],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"},
        )

        assert result.returncode == 0, result.stderr


class TestCli:
    def test_zero_beats_is_refused_at_the_edge(self, capsys):
        with pytest.raises(SystemExit) as raised:
            ka.main(["--beats", "0"])

        assert raised.value.code == 2
        assert "no late" in capsys.readouterr().err

    def test_a_refusal_exits_nonzero_so_the_chain_can_stop(self, monkeypatch, capsys):
        def refuse(*_args, **_kwargs):
            raise ka.KeepaliveRefused("el hub respondio 403")

        monkeypatch.setattr(ka, "keep_alive", refuse)

        assert ka.main(["--beats", "1"]) == 1
        assert "403" in capsys.readouterr().out
