import asyncio
import json

from demo.backend.request_limits import InferenceLimits


def scope():
    return {"type": "http", "method": "POST", "path": "/api/predict", "headers": []}


def test_streamed_oversized_body_never_reaches_parser():
    async def check():
        async def forbidden(*args):
            raise AssertionError("parser must not see oversized body")

        middleware = InferenceLimits(forbidden, max_bytes=4)
        chunks = iter([b"abc", b"def"])
        messages = []

        async def receive():
            return {"type": "http.request", "body": next(chunks), "more_body": True}

        async def send(message):
            messages.append(message)

        await middleware(scope(), receive, send)
        assert messages[0]["status"] == 413
        assert json.loads(messages[1]["body"])["error"]["code"] == "IMAGE_TOO_LARGE"

    asyncio.run(check())


def test_slow_upload_times_out_and_releases_capacity():
    async def check():
        async def forbidden(*args):
            raise AssertionError("slow body must not reach parser")
        middleware = InferenceLimits(forbidden, upload_timeout=0.01)
        messages = []
        async def receive():
            await asyncio.sleep(1)
        async def send(message):
            messages.append(message)
        await middleware(scope(), receive, send)
        assert messages[0]["status"] == 408
        assert middleware.slots._value == 2
    asyncio.run(check())


def test_process_rate_limit_bounds_even_completed_requests():
    async def check():
        async def app(scope, receive, send):
            await receive()
        middleware = InferenceLimits(app, max_requests=1)
        messages = []
        async def receive():
            return {"type": "http.request", "body": b"image", "more_body": False}
        async def send(message):
            messages.append(message)
        await middleware(scope(), receive, send)
        await middleware(scope(), receive, send)
        assert messages[0]["status"] == 429
        assert json.loads(messages[1]["body"])["error"]["code"] == "RATE_LIMITED"
    asyncio.run(check())


def test_busy_inference_rejects_without_reading_body_and_releases_slot():
    async def check():
        started, release = asyncio.Event(), asyncio.Event()

        async def app(scope, receive, send):
            assert (await receive())["body"] == b"image"
            started.set()
            await release.wait()

        middleware = InferenceLimits(app, max_bytes=10, max_inflight=1)
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"image", "more_body": False}

        async def forbidden_receive():
            raise AssertionError("busy request body must not be read")

        async def send(message):
            messages.append(message)

        first = asyncio.create_task(middleware(scope(), receive, send))
        await started.wait()
        await middleware(scope(), forbidden_receive, send)
        assert messages[0]["status"] == 429
        release.set()
        await first
        await middleware(scope(), receive, send)

    asyncio.run(check())
