from __future__ import annotations

import asyncio
from collections import deque

from starlette.responses import JSONResponse


GUARDED_PATHS = frozenset({"/api/predict", "/api/features", "/api/analyze", "/api/simulate-tone"})
# 10 MiB image (enforced again by the handler) plus room for a mask PNG and multipart framing.
DEFAULT_MAX_BODY_BYTES = 11 * 1024 * 1024


class InferenceLimits:
    """Bound upload buffering and reject excess work before multipart parsing."""

    def __init__(self, app, max_bytes=DEFAULT_MAX_BODY_BYTES, max_inflight=2, upload_timeout=30,
                 max_requests=30):
        self.app = app
        self.max_bytes = max_bytes
        self.slots = asyncio.Semaphore(max_inflight)
        self.upload_timeout = upload_timeout
        self.max_requests = max_requests
        self.requests = deque()

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path") not in GUARDED_PATHS):
            return await self.app(scope, receive, send)
        if self.slots.locked():
            return await self._reject(scope, receive, send, 429, "INFERENCE_BUSY")
        now = asyncio.get_running_loop().time()
        while self.requests and self.requests[0] <= now - 60:
            self.requests.popleft()
        if len(self.requests) >= self.max_requests:
            return await self._reject(scope, receive, send, 429, "RATE_LIMITED")
        self.requests.append(now)
        await self.slots.acquire()
        try:
            body = bytearray()
            deadline = asyncio.get_running_loop().time() + self.upload_timeout
            while True:
                try:
                    remaining = max(0, deadline - asyncio.get_running_loop().time())
                    message = await asyncio.wait_for(receive(), timeout=remaining)
                except TimeoutError:
                    return await self._reject(scope, receive, send, 408, "UPLOAD_TIMEOUT")
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > self.max_bytes:
                    return await self._reject(scope, receive, send, 413, "IMAGE_TOO_LARGE")
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
            replayed = False

            async def replay():
                nonlocal replayed
                if replayed:
                    return await receive()
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            await self.app(scope, replay, send)
        finally:
            self.slots.release()

    @staticmethod
    async def _reject(scope, receive, send, status, code):
        response = JSONResponse(
            {"error": {"code": code, "message": "Solicitud excede los limites de la demo."}},
            status_code=status,
            headers={"Cache-Control": "no-store"},
        )
        await response(scope, receive, send)
