"""Bound Messages uploads and unfinished responses without occupying engine ranks."""

import asyncio
import ipaddress

from fastapi.responses import JSONResponse

from .driver import AdmissionError

MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_HTTP_REQUESTS = 32
IO_TIMEOUT = 30


async def read_body(request):
    """The complete JSON body, received within one deadline; parsing runs elsewhere."""
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise ValueError("Content-Type must be application/json")
    length = request.headers.get("content-length")
    if length is not None and (len(length) > 10 or int(length) > MAX_BODY_BYTES):
        raise AdmissionError("request body exceeds 32 MiB", status=413, kind="request_too_large")
    body = bytearray()
    try:
        async with asyncio.timeout(IO_TIMEOUT):
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_BODY_BYTES:
                    raise AdmissionError("request body exceeds 32 MiB", status=413,
                                         kind="request_too_large")
                body.extend(chunk)
    except TimeoutError:
        raise AdmissionError("request body timed out", status=408,
                             kind="invalid_request_error") from None
    return bytes(body)


def is_loopback(host):
    """Whether a bind address or Host header, with or without a port, names loopback."""
    if host.startswith("["):
        host = host[1:].partition("]")[0]
    elif host.count(":") == 1:
        host = host.partition(":")[0]
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class HTTPBounds:
    def __init__(self, app, *, loopback_only):
        self.app = app
        self.loopback_only = loopback_only
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.loopback_only:
            # A web page that rebinds its own name to 127.0.0.1 still sends that name.
            host = dict(scope["headers"]).get(b"host")
            if host is not None and not is_loopback(host.decode("latin-1")):
                response = JSONResponse(
                    {"type": "error", "error": {"type": "invalid_request_error",
                     "message": "Host must name a loopback address"}},
                    status_code=400,
                )
                return await response(scope, receive, send)
        if scope["type"] != "http" or scope["path"] not in {
            "/v1/messages", "/v1/messages/count_tokens",
        }:
            return await self.app(scope, receive, send)

        async def bounded_send(message):
            try:
                await asyncio.wait_for(send(message), IO_TIMEOUT)
            except asyncio.TimeoutError:
                # Unwind the response and cancel its job. The server protocol
                # also aborts stalled writes, including pre-header error sends
                # that Uvicorn performs outside this ASGI wrapper.
                raise OSError("response send timed out") from None

        if self.active >= MAX_HTTP_REQUESTS:
            response = JSONResponse(
                {"type": "error", "error": {"type": "overloaded_error",
                 "message": "too many unfinished HTTP requests"}},
                status_code=529, headers={"Retry-After": "1"},
            )
            return await response(scope, receive, bounded_send)
        self.active += 1
        try:
            await self.app(scope, receive, bounded_send)
        finally:
            self.active -= 1
