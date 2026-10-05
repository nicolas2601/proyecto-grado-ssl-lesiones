"""The keepalive has to be loud when it stops working.

Its whole job is to protect a run that lasts hours, so the failure mode that
matters is not "it crashed" -- it is "it kept logging beats while the hub was
refusing every one of them".
"""

from __future__ import annotations

import pytest
from pathlib import Path

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

        result = ka.keep_alive(
            30.0, environ=ENVIRON, poster=poster, sleeper=slept.append, beats=3, log=lambda _: None
        )

        assert result.beats_sent == 3
        assert len(poster.calls) == 3
        # Two gaps between three beats, and no trailing sleep after the last.
        assert slept == [30.0, 30.0]

    def test_the_first_beat_fails_before_the_expensive_work_starts(self):
        poster = RecordingPoster(403)

        with pytest.raises(ka.KeepaliveRefused):
            ka.keep_alive(30.0, environ=ENVIRON, poster=poster, sleeper=lambda _: None, beats=5)

        assert len(poster.calls) == 1, "siguio latiendo despues de que el hub lo rechazara"

    def test_the_loop_returns_a_result_not_a_bare_count(self):
        result = ka.keep_alive(
            120.0, environ=ENVIRON, poster=RecordingPoster(), sleeper=lambda _: None,
            beats=2, log=lambda _: None,
        )

        assert (result.beats_sent, result.failures, result.at_risk) == (2, 0, False)

    def test_an_isolated_failure_mid_run_does_not_kill_the_computation(self):
        """A 5xx blip must not cost six hours of GPU.

        The culler needs several minutes of silence to act, so one missed beat
        is survivable; killing the run to "protect" it is the expensive
        mistake. It still has to be counted -- see the on_failure test.
        """

        class FailsOnceOnThird(RecordingPoster):
            def __call__(self, url, token, payload):
                super().__call__(url, token, payload)
                return 500 if len(self.calls) == 3 else 200

        poster = FailsOnceOnThird()

        result = ka.keep_alive(
            120.0,
            environ=ENVIRON,
            poster=poster,
            sleeper=lambda _: None,
            beats=5,
            log=lambda _: None,
        )

        # The blip cost an attempt, not a beat: five successful reports still
        # went out, and the loop needed six attempts to get them.
        assert result.beats_sent == 5
        assert result.failures == 1
        assert len(poster.calls) == 6
        assert result.at_risk is False

    def test_every_failure_is_reported_so_it_is_never_a_silent_no_op(self):
        seen: list[str] = []

        class AlwaysFails(RecordingPoster):
            def __call__(self, url, token, payload):
                super().__call__(url, token, payload)
                return 200 if len(self.calls) == 1 else 503

        ka.keep_alive(
            120.0,
            environ=ENVIRON,
            poster=AlwaysFails(),
            sleeper=lambda _: None,
            beats=4,
            max_failures=3,
            log=lambda _: None,
            on_failure=lambda error: seen.append(str(error)),
        )

        assert len(seen) >= 3
        assert all("503" in message for message in seen)

    def test_an_endless_rejection_is_bounded_by_max_failures(self):
        """Without a bound, a hub answering 500 forever spins: `beats` counts
        successes, and there are none."""
        result = ka.keep_alive(
            120.0,
            environ=ENVIRON,
            poster=RecordingPoster(200),
            sleeper=lambda _: None,
            beats=1,
            log=lambda _: None,
        )
        assert result.beats_sent == 1  # the bounded-success path still works

        class AlwaysFailsAfterFirst(RecordingPoster):
            def __call__(self, url, token, payload):
                super().__call__(url, token, payload)
                return 200 if len(self.calls) == 1 else 500

        bounded = ka.keep_alive(
            120.0,
            environ=ENVIRON,
            poster=AlwaysFailsAfterFirst(),
            sleeper=lambda _: None,
            beats=99,
            max_failures=4,
            log=lambda _: None,
        )

        assert bounded.failures == 4
        assert bounded.beats_sent == 1

    def test_a_long_streak_of_failures_marks_the_run_at_risk_without_killing_it(self):
        raised: list[float] = []

        class AlwaysFailsAfterFirst(RecordingPoster):
            def __call__(self, url, token, payload):
                super().__call__(url, token, payload)
                return 200 if len(self.calls) == 1 else 500

        result = ka.keep_alive(
            120.0,
            environ=ENVIRON,
            poster=AlwaysFailsAfterFirst(),
            sleeper=lambda _: None,
            beats=12,
            max_failures=10,
            log=lambda _: None,
            at_risk_after_seconds=600.0,
            on_at_risk=raised.append,
        )

        assert result.at_risk is True, "10 min sin latir y nadie se entera"
        assert len(raised) == 1, "la alarma se levanta una vez, no en cada latido"
        assert raised[0] >= 600.0
        assert result.beats_sent == 1, "el computo siguio vivo igual"

    def test_a_success_resets_the_streak(self):
        """Isolated failures must not accumulate into a false alarm.

        Tuned so the reset is the ONLY thing that decides: seven failures
        spaced by successes add up past the 100 s threshold if the streak is
        never cleared, and stay at 15 s if it is.
        """
        raised: list[float] = []

        class FailsOnEveryOtherAttempt(RecordingPoster):
            def __call__(self, url, token, payload):
                super().__call__(url, token, payload)
                return 500 if len(self.calls) % 2 == 0 else 200

        result = ka.keep_alive(
            120.0,
            environ=ENVIRON,
            poster=FailsOnEveryOtherAttempt(),
            sleeper=lambda _: None,
            beats=8,
            max_failures=10,
            log=lambda _: None,
            at_risk_after_seconds=100.0,
            on_at_risk=raised.append,
        )

        assert result.failures >= 7, "el escenario tiene que producir fallos de sobra"
        assert raised == [], "fallos sueltos separados por exitos no son una racha"
        assert result.at_risk is False

    def test_a_failure_retries_sooner_than_the_normal_interval(self):
        """Backoff, but bounded by the interval: waiting the full 2 min after a
        failure wastes the window the culler is counting."""
        slept: list[float] = []

        class AlwaysFailsAfterFirst(RecordingPoster):
            def __call__(self, url, token, payload):
                super().__call__(url, token, payload)
                return 200 if len(self.calls) == 1 else 500

        ka.keep_alive(
            120.0,
            environ=ENVIRON,
            poster=AlwaysFailsAfterFirst(),
            sleeper=slept.append,
            beats=6,
            max_failures=5,
            log=lambda _: None,
            on_at_risk=lambda _: None,
        )

        retries = slept[1:]
        assert retries[0] < 120.0, "reintento inmediato al ritmo normal"
        assert retries == sorted(retries), "el backoff tiene que crecer"
        assert max(retries) <= 120.0, "nunca mas lento que el intervalo normal"

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


class TestStatusFile:
    def test_the_chain_can_read_beats_and_failures(self, tmp_path):
        status = tmp_path / "keepalive.json"

        ka.write_status(status, ka.KeepaliveResult(beats_sent=7, failures=2, at_risk=False))

        import json

        payload = json.loads(status.read_text())
        assert payload["beats_sent"] == 7
        assert payload["failures"] == 2
        assert payload["at_risk"] is False
        assert payload["finished"] is False

    def test_it_is_written_atomically(self, tmp_path):
        """A chain reading mid-write must not parse half a file and conclude
        there were zero failures."""
        status = tmp_path / "keepalive.json"
        ka.write_status(status, ka.KeepaliveResult(1, 0, False))

        seen = []
        original = Path.replace

        def spy(self, target):
            seen.append((self.name, Path(target).name))
            return original(self, target)

        import pytest as _pytest

        with _pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, "replace", spy)
            ka.write_status(status, ka.KeepaliveResult(2, 1, True))

        assert seen == [("keepalive.json.tmp", "keepalive.json")]
        assert not (tmp_path / "keepalive.json.tmp").exists()

    def test_cli_records_failures_for_the_chain(self, tmp_path, monkeypatch):
        status = tmp_path / "keepalive.json"

        def fake_keep_alive(*_args, on_failure=None, on_at_risk=None, log=None, **_kwargs):
            log("keepalive 1: actividad reportada al hub (t)")
            on_failure(ka.KeepaliveRefused("503"))
            on_at_risk(600.0)
            return ka.KeepaliveResult(beats_sent=1, failures=1, at_risk=True)

        monkeypatch.setattr(ka, "keep_alive", fake_keep_alive)

        # At risk is neither success nor failure: the chain has to tell them apart.
        assert ka.main(["--status-file", str(status)]) == 2

        import json

        payload = json.loads(status.read_text())
        assert (payload["beats_sent"], payload["failures"], payload["at_risk"]) == (1, 1, True)
        assert payload["finished"] is True
