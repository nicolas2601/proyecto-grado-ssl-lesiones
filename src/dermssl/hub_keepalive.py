"""Keeping a JupyterHub server alive for the length of a long run.

Why this exists
---------------
A C1 grid is 24 cells and runs for hours. On 2026-09-26 the pod's server was
shut down mid-chain even though the process had its own session id: detaching
from the terminal protects against a dropped connection, NOT against the hub
deciding the server is idle. The two are different failures and only one of
them is solved by ``nohup``.

What counts as activity is the server reporting it to the hub, and a process
running under ``nohup`` generates no HTTP traffic, so from the hub's point of
view a busy 6-hour run looks exactly like an empty server.

The fix is to report activity through the same endpoint the server itself
uses: ``$JUPYTERHUB_ACTIVITY_URL``. Measured on the pod (2026-09-27):

    last_activity ANTES : 2026-09-27T01:35:23.186000Z
    POST activity -> 200
    last_activity DESPUES: 2026-09-27T01:36:02.549243Z

Deliberately NOT a "ping the server's own /api/status" loop: that depends on
the server translating an incoming request into a report, which is one more
hop that can silently stop working. This writes the report directly.

The idle timeout itself lives in the hub's configuration and is not readable
with the server's token, so this does not try to guess it. The interval is
short enough (2 min by default) to sit under any plausible culler setting.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import urllib.request
from typing import Any, Callable, Mapping


class KeepaliveRefused(RuntimeError):
    """The hub did not accept the activity report.

    Raised rather than logged-and-ignored. A keepalive that fails quietly is
    worse than no keepalive: the run proceeds believing it is protected and
    dies hours later for a reason nobody recorded.
    """


ACTIVITY_URL = "JUPYTERHUB_ACTIVITY_URL"
API_TOKEN = "JUPYTERHUB_API_TOKEN"
SERVER_NAME = "JUPYTERHUB_SERVER_NAME"

DEFAULT_INTERVAL_SECONDS = 120.0


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _post(url: str, token: str, payload: Mapping[str, Any]) -> int:
    request = urllib.request.Request(url, method="POST")
    request.add_header("Authorization", f"token {token}")
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, json.dumps(payload).encode(), timeout=30) as response:
        return int(response.status)


def report_activity(
    environ: Mapping[str, str] | None = None,
    *,
    poster: Callable[[str, str, Mapping[str, Any]], int] = _post,
    now: Callable[[], str] = _utc_now,
) -> str:
    """Report one beat of activity to the hub; returns the timestamp sent.

    Raises :class:`KeepaliveRefused` when the environment does not describe a
    hub-managed server, or when the hub answers with anything but 200. Both
    are refusals rather than warnings: outside a pod there is nothing to keep
    alive, and a rejected report means the protection is not in place.
    """
    environ = os.environ if environ is None else environ

    url = environ.get(ACTIVITY_URL, "")
    token = environ.get(API_TOKEN, "")
    missing = [name for name, value in ((ACTIVITY_URL, url), (API_TOKEN, token)) if not value]
    if missing:
        raise KeepaliveRefused(
            f"falta {' y '.join(missing)} en el entorno: esto no parece un servidor "
            "gestionado por JupyterHub, asi que no hay nada que mantener vivo"
        )

    stamp = now()
    server = environ.get(SERVER_NAME, "")
    status = poster(
        url,
        token,
        {"servers": {server: {"last_activity": stamp}}, "last_activity": stamp},
    )
    if status != 200:
        raise KeepaliveRefused(
            f"el hub respondio {status} al reporte de actividad, asi que la corrida NO "
            "esta protegida del culler; abortar ahora es mas barato que descubrirlo "
            "con seis horas de GPU gastadas"
        )
    return stamp


@dataclass(frozen=True)
class KeepaliveResult:
    """What the loop actually managed to do.

    A bare count would answer "did it run" and not "was the run protected",
    which is the question that matters after the fact.
    """

    beats_sent: int
    failures: int
    at_risk: bool


#: First retry after a failure, doubling up to ``interval_seconds``. Retrying
#: at the normal cadence would burn the very window the culler is counting.
FIRST_RETRY_SECONDS = 15.0

#: A streak this long without a successful beat means the server may already
#: be on the culler's list. Ten minutes is under any plausible idle timeout.
DEFAULT_AT_RISK_AFTER_SECONDS = 600.0


def keep_alive(
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    *,
    environ: Mapping[str, str] | None = None,
    poster: Callable[[str, str, Mapping[str, Any]], int] = _post,
    sleeper: Callable[[float], None] = time.sleep,
    beats: int | None = None,
    log: Callable[[str], None] = print,
    at_risk_after_seconds: float = DEFAULT_AT_RISK_AFTER_SECONDS,
    max_failures: int | None = None,
    on_failure: Callable[[KeepaliveRefused], None] | None = None,
    on_at_risk: Callable[[float], None] | None = None,
) -> KeepaliveResult:
    """Report activity every ``interval_seconds`` until ``beats`` or killed.

    The policy is deliberately asymmetric, because the two failures cost
    different things:

    * **The first beat fails** -> raise. Nothing expensive has started, and a
      run that was never protected should not begin.
    * **A beat fails mid-run** -> count it, report it, retry sooner. Killing
      hours of GPU over one 5xx is the expensive mistake: the culler needs
      several minutes of silence to act, so a blip is survivable.
    * **A streak covers ``at_risk_after_seconds``** -> raise the alarm ONCE
      through ``on_at_risk`` and keep computing. What saves the run at that
      point is being resumable, not killing it here.

    Every failure goes through ``on_failure`` and the log, so this degrades
    into a *visible* no-op and never a silent one.

    ``beats`` counts SUCCESSFUL reports; a failed attempt is not a beat, so
    ``max_failures`` is what bounds a loop whose every attempt is rejected --
    without it, a hub that answers 500 forever keeps the loop spinning and
    ``beats`` is never reached. Reaching it RETURNS (``at_risk`` set) instead
    of raising: by then the computation is already running and killing it is
    the expensive mistake.
    """
    if not interval_seconds > 0:
        raise ValueError(
            f"interval_seconds={interval_seconds!r}: un intervalo de 0 o negativo seria "
            "un bucle sin espera o un no-op, no un keepalive"
        )
    if not at_risk_after_seconds > 0:
        raise ValueError(
            f"at_risk_after_seconds={at_risk_after_seconds!r}: una alarma que salta a los "
            "0 segundos no distingue un blip de una caida"
        )

    sent = 0
    failures = 0
    streak_seconds = 0.0
    next_retry = FIRST_RETRY_SECONDS
    at_risk = False

    while beats is None or sent < beats:
        try:
            stamp = report_activity(environ, poster=poster)
        except KeepaliveRefused as error:
            if sent == 0:
                # Nothing costly has started yet, so this is still cheap to refuse.
                raise
            failures += 1
            streak_seconds += next_retry
            log(f"ADVERTENCIA keepalive: {error} (fallo {failures}, racha {streak_seconds:.0f}s)")
            if on_failure is not None:
                on_failure(error)
            if not at_risk and streak_seconds >= at_risk_after_seconds:
                at_risk = True
                log(
                    f"ALERTA keepalive: {streak_seconds:.0f}s sin reportar actividad. La "
                    "corrida sigue, pero el servidor puede ser apagado por el culler."
                )
                if on_at_risk is not None:
                    on_at_risk(streak_seconds)
            if max_failures is not None and failures >= max_failures:
                log(
                    f"keepalive: {failures} fallos, se deja de reintentar. El computo "
                    "sigue; lo que salva la corrida desde aca es ser reanudable."
                )
                break
            sleeper(next_retry)
            next_retry = min(next_retry * 2, interval_seconds)
            continue

        sent += 1
        streak_seconds = 0.0
        next_retry = FIRST_RETRY_SECONDS
        log(f"keepalive {sent}: actividad reportada al hub ({stamp})")
        if beats is not None and sent >= beats:
            break
        sleeper(interval_seconds)

    return KeepaliveResult(beats_sent=sent, failures=failures, at_risk=at_risk)


def write_status(path: Path, result: KeepaliveResult, *, finished: bool = False) -> None:
    """Publish the keepalive's state for the chain to pick up.

    The keepalive runs as its own process, so it has no access to the chain's
    W&B tracker. Rather than opening a second run (which would put the
    keepalive's noise in its own dashboard entry where nobody looks), it
    leaves its counters here and the chain logs them next to the metrics they
    belong with.

    Written temp-then-rename: a chain that reads this file while it is being
    written must never see half a JSON object and conclude there were zero
    failures.
    """
    payload = {
        "beats_sent": result.beats_sent,
        "failures": result.failures,
        "at_risk": result.at_risk,
        "finished": finished,
        "updated": _utc_now(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    """``python -m dermssl.hub_keepalive`` -- meant to run beside a long chain.

    The first beat is sent before this returns control, so a chain that starts
    it and checks the exit code learns immediately whether it is protected.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="segundos entre reportes (por defecto %(default)s)",
    )
    parser.add_argument(
        "--beats",
        type=int,
        default=None,
        help="numero de reportes; sin esto late hasta que lo maten",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="JSON con latidos y fallos, para que la cadena los registre en W&B",
    )
    args = parser.parse_args(argv)

    # Checked at the edge: argparse accepts 0 happily, and keep_alive would
    # then raise from inside the loop instead of at the boundary.
    if args.beats is not None and args.beats < 1:
        parser.error(f"--beats={args.beats}: cero reportes es un keepalive que no late")

    status = args.status_file
    progress = KeepaliveResult(0, 0, False)

    def remember(**changes: Any) -> None:
        nonlocal progress
        progress = KeepaliveResult(
            beats_sent=changes.get("beats_sent", progress.beats_sent),
            failures=changes.get("failures", progress.failures),
            at_risk=changes.get("at_risk", progress.at_risk),
        )
        if status is not None:
            write_status(status, progress)

    def on_beat(message: str) -> None:
        print(message, flush=True)
        if message.startswith("keepalive ") and "reportada" in message:
            remember(beats_sent=progress.beats_sent + 1)

    try:
        result = keep_alive(
            args.interval,
            beats=args.beats,
            log=on_beat,
            on_failure=lambda _: remember(failures=progress.failures + 1),
            on_at_risk=lambda _: remember(at_risk=True),
        )
    except KeepaliveRefused as error:
        print(f"ERROR: {error}")
        if status is not None:
            write_status(status, progress, finished=True)
        return 1

    if status is not None:
        write_status(status, result, finished=True)
    # A run that finished at risk did not fail, but it did not run protected
    # either; the chain should be able to tell those apart.
    return 2 if result.at_risk else 0


if __name__ == "__main__":
    raise SystemExit(main())
