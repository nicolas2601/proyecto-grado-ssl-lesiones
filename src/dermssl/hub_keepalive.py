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
import json
import os
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


def keep_alive(
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    *,
    environ: Mapping[str, str] | None = None,
    poster: Callable[[str, str, Mapping[str, Any]], int] = _post,
    sleeper: Callable[[float], None] = time.sleep,
    beats: int | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Report activity every ``interval_seconds``; returns the beats sent.

    The FIRST beat is not wrapped: if the very first report fails, the caller
    is told before the expensive work starts. Later failures are also fatal --
    a keepalive that degrades into a no-op halfway through leaves the run
    exactly as exposed as it was without it, but with a log that says
    otherwise.

    ``beats`` bounds the loop (tests, and a chain that knows its own length);
    ``None`` runs until killed.
    """
    if not interval_seconds > 0:
        raise ValueError(
            f"interval_seconds={interval_seconds!r}: un intervalo de 0 o negativo seria "
            "un bucle sin espera o un no-op, no un keepalive"
        )

    sent = 0
    while beats is None or sent < beats:
        stamp = report_activity(environ, poster=poster)
        sent += 1
        log(f"keepalive {sent}: actividad reportada al hub ({stamp})")
        if beats is not None and sent >= beats:
            break
        sleeper(interval_seconds)
    return sent


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
    args = parser.parse_args(argv)

    # Checked at the edge: argparse accepts 0 happily, and keep_alive would
    # then raise from inside the loop instead of at the boundary.
    if args.beats is not None and args.beats < 1:
        parser.error(f"--beats={args.beats}: cero reportes es un keepalive que no late")

    try:
        keep_alive(args.interval, beats=args.beats)
    except KeepaliveRefused as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
