"""Anthropic Messages frontend over the SM100 TP8 engine queue."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
import time

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from .driver import AdmissionError, Job
from .limits import read_body
from .protocol import SamplingParams, sampling_values
from .messages_chat import Detokenizer, Renderer
from .messages_parser import OutputParser, _partial_suffix
from .schema import check_schema

PING_SECONDS = 10.0
# StopFilter rescans every held sequence after each token on the event loop.
MAX_STOP_SEQUENCES, MAX_STOP_SEQUENCE_CHARS = 16, 128
logger = logging.getLogger(__name__)


class Disconnected(Exception):
    """The request body was read, but its client left before a response."""


async def while_connected(request, operation):
    """Race work against disconnect; join both tasks before transferring receive."""
    async def disconnected():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    work = asyncio.create_task(operation)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            watcher.result()
            raise Disconnected
        return work.result()
    finally:
        work.cancel()
        watcher.cancel()
        await asyncio.gather(work, watcher, return_exceptions=True)


class MessageStream(StreamingResponse):
    """Cancel an admitted job even if the connection fails before the first frame."""

    def __init__(self, job, metrics, content, **kwargs):
        super().__init__(content, **kwargs)
        self.job, self.metrics = job, metrics

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.job.cancelled.set()
            try:
                await self.body_iterator.aclose()
            finally:
                # An unstarted generation iterator has no finally block to run.
                # record_request keeps the generator's result when it already ran.
                self.metrics.record_request(self.job, output_tokens=0, reason="cancelled",
                                            ended=time.perf_counter())


def _error(message, status=400, kind="invalid_request_error", headers=None):
    return JSONResponse(
        {"type": "error", "error": {"type": kind, "message": message}},
        status_code=status, headers=headers,
    )


def _sse(kind, **fields):
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **fields}, ensure_ascii=False, allow_nan=False)}\n\n"


def _usage(n_input, n_output, cache_read=0):
    return dict(
        input_tokens=n_input - cache_read,
        output_tokens=n_output,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=cache_read,
    )


def _text_content(content):
    if isinstance(content, str):
        return
    if not isinstance(content, list):
        raise ValueError("content must be a string or a list of text blocks")
    for block in content:
        if (
            not isinstance(block, dict)
            or block.get("type") != "text"
            or not isinstance(block.get("text"), str)
        ):
            raise ValueError("only text content is supported; images and documents are unsupported")


def validate(req, model, *, count=False):
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")
    if req.get("model") != model:
        raise ValueError(f"model must be {model!r}")
    messages = req.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    if "system" in req:
        _text_content(req["system"])
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "user",
            "assistant",
            "system",
        }:
            raise ValueError("message role must be user, assistant, or system")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("message content must be a string or a list")
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("content blocks must be objects")
            kind = block.get("type")
            if kind == "text":
                _text_content([block])
            elif kind == "thinking" and message["role"] == "assistant":
                if not isinstance(block.get("thinking"), str):
                    raise ValueError("thinking must be a string")
            elif kind == "tool_use" and message["role"] == "assistant":
                if (
                    not isinstance(block.get("input"), dict)
                    or not isinstance(block.get("name"), str)
                    or not block["name"]
                    or not isinstance(block.get("id"), str)
                    or not block["id"]
                ):
                    raise ValueError("tool_use requires id, name, and an input object")
            elif kind == "tool_result" and message["role"] == "user":
                if not isinstance(block.get("tool_use_id"), str) or not block["tool_use_id"]:
                    raise ValueError("tool_result requires tool_use_id")
                _text_content(block.get("content", ""))
                if "is_error" in block and not isinstance(block["is_error"], bool):
                    raise ValueError("tool_result is_error must be a boolean")
            else:
                raise ValueError(f"unsupported {message['role']} content block: {kind!r}")
    tools = req.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    names = set()
    for tool in tools:
        if (
            not isinstance(tool, dict)
            or tool.get("type", "custom") != "custom"
            or not isinstance(tool.get("name"), str)
            or re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", tool["name"]) is None
        ):
            raise ValueError("only named client tools are supported (no server tools)")
        if tool["name"] in names:
            raise ValueError("tool names must be unique")
        names.add(tool["name"])
        if "description" in tool and not isinstance(tool["description"], str):
            raise ValueError("tool description must be a string")
        if "strict" in tool and tool["strict"] is not False:
            raise ValueError("tool strict supports only false; constrained output is unsupported")
        schema = tool.get("input_schema")
        if not isinstance(schema, dict) or schema.get("type", "object") != "object":
            raise ValueError("tool input_schema must describe an object")
        if not isinstance(schema.get("properties", {}), dict):
            raise ValueError("tool input_schema properties must be an object")
        check_schema(schema)
    choice = req.get("tool_choice", {"type": "auto"})
    if (
        not isinstance(choice, dict)
        or choice.get("type") not in {"auto", "none"}
        or choice.get("disable_parallel_tool_use")
    ):
        raise ValueError(
            "tool_choice supports auto/none; forced or restricted tool use is unsupported"
        )
    thinking = req.get("thinking", {})
    if not isinstance(thinking, dict) or thinking.get("type", "enabled") not in {
        "enabled",
        "disabled",
        "adaptive",
    }:
        raise ValueError("thinking.type must be enabled, disabled, or adaptive")
    if "stream" in req and not isinstance(req["stream"], bool):
        raise ValueError("stream must be a boolean")
    if (not count or "max_tokens" in req) and (
        type(req.get("max_tokens")) is not int or req["max_tokens"] < 1
    ):
        raise ValueError("max_tokens must be a positive integer")
    if "budget_tokens" in thinking:
        raise ValueError("thinking.budget_tokens is unsupported; omit it and use max_tokens "
                         "to limit total output")
    for key, lo, hi in (("temperature", 0, 1), ("top_p", 0, 1)):
        if key not in req:
            continue
        value = req[key]
        if type(value) not in (int, float) or not lo <= value <= hi:
            raise ValueError(f"{key} must be a finite number between {lo} and {hi}")
    if "top_k" in req and (type(req["top_k"]) is not int or req["top_k"] < 0):
        raise ValueError("top_k must be a nonnegative integer")
    sampling_values(req)
    output = req.get("output_config", {})
    if not isinstance(output, dict) or "format" in output or "output_format" in req:
        raise ValueError("constrained output is unsupported")
    if "effort" in output and output["effort"] not in ("low", "medium", "high", "max"):
        raise ValueError("output_config.effort must be low, medium, high, or max (advisory only)")
    stops = req.get("stop_sequences", [])
    if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
        raise ValueError("stop_sequences must be a list of nonempty strings")
    if len(stops) > MAX_STOP_SEQUENCES or any(len(s) > MAX_STOP_SEQUENCE_CHARS for s in stops):
        raise ValueError(f"stop_sequences allows at most {MAX_STOP_SEQUENCES} sequences of "
                         f"up to {MAX_STOP_SEQUENCE_CHARS} characters")


def load_request(body, model, *, count=False):
    """Parse and validate a request body; nesting too deep for Python is invalid too."""
    try:
        req = json.loads(body)
        validate(req, model, count=count)
    except RecursionError:
        raise ValueError("request JSON is nested too deeply") from None
    return req


class StopFilter:
    """Hold partial stop sequences before exposing any of their text to a client."""

    def __init__(self, stops):
        self.stops, self.buf, self.hit = stops, "", None

    def feed(self, text, *, final=False):
        if self.hit is not None:
            return ""
        self.buf += text
        hits = [(self.buf.find(s), s) for s in self.stops if s in self.buf]
        if hits:
            pos, self.hit = min(hits, key=lambda pair: pair[0])
            out, self.buf = self.buf[:pos], ""
            return out
        hold = 0 if final else max((_partial_suffix(self.buf, s) for s in self.stops), default=0)
        out, self.buf = self.buf[: len(self.buf) - hold], self.buf[len(self.buf) - hold :]
        return out


class MessagesFrontend:
    def __init__(self, driver):
        self.driver = driver
        self.renderer = Renderer(driver.tok)
        self.preparation_lock = asyncio.Lock()
        self.preparing = 0

    async def offload(self, function, body):
        """One parsing and tokenizing worker, eight waiters, including abandoned work."""
        if self.preparing >= 9:
            raise AdmissionError("prompt preparation queue is full", status=529, kind="overloaded_error")
        self.preparing += 1
        abandoned = False

        async def run():
            try:
                async with self.preparation_lock:
                    if not abandoned:
                        return await asyncio.to_thread(function, body)
            finally:
                self.preparing -= 1

        task = asyncio.create_task(run())
        # A disconnected caller must not free a slot while its thread still runs.
        # Waiting abandoned requests are skipped once the active worker finishes.
        task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        try:
            return await asyncio.shield(task)
        finally:
            abandoned = True

    def prepare(self, body):
        req = load_request(body, self.driver.model_name)
        parsed_at = time.perf_counter()
        ids = self.prompt_ids(req)
        room = self.driver.max_model_len - len(ids)
        if not ids or room < 1:
            raise ValueError(f"prompt must contain 1..{self.driver.max_model_len - 1} tokens")
        params = SamplingParams(
            max_tokens=req["max_tokens"],
            stop=tuple(req.get("stop_sequences", [])),
            seed=self.driver.next_seed(),
        )
        return req, parsed_at, (ids, params, self.driver.clamp_max_new(len(ids), params.max_tokens))

    def count(self, body):
        return len(self.prompt_ids(load_request(body, self.driver.model_name, count=True)))

    def prompt_ids(self, req):
        """Token ids of a validated request; nesting too deep to render is invalid too."""
        try:
            return self.renderer.render(req)[1]
        except RecursionError:
            raise ValueError("request JSON is nested too deeply") from None

    def submit(self, req, prepared, *, received_at, parsed_at, prepared_at):
        """Admit exactly once, before HTTP response headers can be sent."""
        ids, params, max_new = prepared
        job = Job(ids, params, max_new, asyncio.get_running_loop(), "msg_" + uuid.uuid4().hex)
        job.received_at, job.parsed_at, job.prepared_at = received_at, parsed_at, prepared_at
        job.kind = "main" if req.get("tools") else "aux"
        self.driver.submit(job)
        return job

    async def generate(self, req, job):
        ids, params = job.prompt_ids, job.params
        allowed = {t["name"] for t in req.get("tools", [])}
        if req.get("tool_choice", {}).get("type") == "none":
            allowed.clear()
        parser = OutputParser(
            self.renderer.string_params(req),
            in_thinking=req.get("thinking", {}).get("type") != "disabled",
            tools_enabled=bool(allowed),
        )
        detok = Detokenizer(self.renderer)
        stop_filter = StopFilter(params.stop)
        n_output, has_tool, finish_reason = 0, False, "cancelled"
        try:
            while True:
                try:
                    event = await asyncio.wait_for(job.next_event(), PING_SECONDS)
                except asyncio.TimeoutError:
                    yield ("ping",)
                    continue
                kind = event[0]
                if kind == "err":
                    raise RuntimeError(event[1])
                if kind == "tok":
                    n_output += 1
                    raw = "" if event[1] in self.driver.eos_ids else detok.feed([event[1]])
                    parsed = parser.feed(stop_filter.feed(raw))
                    if stop_filter.hit is not None:
                        job.cancelled.set()
                else:
                    parsed = parser.feed(stop_filter.feed("", final=True), final=True)
                for item in parsed:
                    now = time.perf_counter()
                    if item[0] == "tool":
                        if item[1] not in allowed:
                            raise RuntimeError(f"model generated undeclared tool {item[1]!r}")
                        has_tool = True
                        if job.first_tool_at is None:
                            job.first_tool_at = now
                    if item[0] == "tool" or item[1]:
                        if job.first_content_at is None:
                            job.first_content_at = now
                        if item[0] in {"text", "tool"} and job.first_visible_at is None:
                            job.first_visible_at = now
                    yield item
                if kind == "done":
                    stop = stop_filter.hit
                    reason = (
                        "stop_sequence"
                        if stop
                        else ("max_tokens" if event[1] == "length" else "end_turn")
                    )
                    # A truncated response must not trigger an incomplete parallel tool batch.
                    if has_tool and reason == "end_turn":
                        reason = "tool_use"
                    finish_reason = reason
                    yield ("done", reason, stop, _usage(len(ids), n_output, job.cache_read_tokens))
                    return
        except RuntimeError:
            finish_reason = "error"
            raise
        finally:
            if not job.completed.is_set():
                job.cancelled.set()
            self.driver.metrics.record_request(job, output_tokens=n_output, reason=finish_reason,
                                               ended=time.perf_counter())

    def message(self, rid, content, reason, stop, usage):
        return dict(
            id=rid,
            type="message",
            role="assistant",
            model=self.driver.model_name,
            content=content,
            stop_reason=reason,
            stop_sequence=stop,
            usage=usage,
        )

    async def complete(self, req, job):
        blocks = []
        events = self.generate(req, job)
        try:
            async for item in events:
                kind = item[0]
                if kind == "done":
                    return self.message(job.request_id, blocks, *item[1:])
                if kind == "ping":
                    continue
                if kind == "tool":
                    blocks.append(
                        dict(
                            type="tool_use", id="toolu_" + uuid.uuid4().hex, name=item[1], input=item[2]
                        )
                    )
                elif blocks and blocks[-1]["type"] == kind:
                    blocks[-1][kind] += item[1]
                else:
                    block = {"type": kind, kind: item[1]}
                    if kind == "thinking":
                        block["signature"] = ""
                    blocks.append(block)
        finally:
            await events.aclose()

    async def stream(self, req, job):
        index, current = -1, None
        events = self.generate(req, job)
        try:
            yield _sse("message_start", message=self.message(
                job.request_id, [], None, None, _usage(len(job.prompt_ids), 0)))
            yield _sse("ping")
            async for item in events:
                kind = item[0]
                if kind == "ping":
                    yield _sse("ping")
                    continue
                if kind == "done":
                    if current is not None:
                        yield _sse("content_block_stop", index=index)
                    yield _sse(
                        "message_delta",
                        delta=dict(stop_reason=item[1], stop_sequence=item[2]),
                        usage=item[3],
                    )
                    yield _sse("message_stop")
                    return
                if kind != current or kind == "tool":
                    if current is not None:
                        yield _sse("content_block_stop", index=index)
                    index += 1
                    current = kind
                    if kind == "tool":
                        block = dict(
                            type="tool_use", id="toolu_" + uuid.uuid4().hex, name=item[1], input={}
                        )
                    else:
                        block = {"type": kind, kind: ""}
                        if kind == "thinking":
                            block["signature"] = ""
                    yield _sse("content_block_start", index=index, content_block=block)
                if kind == "tool":
                    delta = dict(
                        type="input_json_delta",
                        partial_json=json.dumps(item[2], ensure_ascii=False, allow_nan=False),
                    )
                else:
                    delta = {"type": kind + "_delta", kind: item[1]}
                yield _sse("content_block_delta", index=index, delta=delta)
        except RuntimeError as exc:
            yield _sse("error", error=dict(type="api_error", message=str(exc)))
        finally:
            # Also covers a disconnect before the generation iterator first runs.
            job.cancelled.set()
            await events.aclose()


def register_messages_routes(app, driver):
    frontend = MessagesFrontend(driver)

    @app.post("/v1/messages")
    async def messages(request: Request):
        received_at = time.perf_counter()
        try:
            body = await read_body(request)
            req, parsed_at, prepared = await while_connected(
                request, frontend.offload(frontend.prepare, body))
        except (Disconnected, ClientDisconnect):
            return Response(status_code=499)
        except AdmissionError as exc:
            return _error(str(exc), exc.status, exc.kind,
                          {"Retry-After": "1"} if exc.status in (503, 529) else None)
        except (ValueError, TypeError) as exc:
            return _error(str(exc))
        timing = dict(received_at=received_at, parsed_at=parsed_at, prepared_at=time.perf_counter())
        try:
            job = frontend.submit(req, prepared, **timing)
        except AdmissionError as exc:
            return _error(str(exc), exc.status, exc.kind, {"Retry-After": "1"})
        if req.get("stream"):
            return MessageStream(
                job, driver.metrics, frontend.stream(req, job),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            return JSONResponse(await while_connected(request, frontend.complete(req, job)))
        except (Disconnected, ClientDisconnect):
            return Response(status_code=499)
        except RuntimeError as exc:
            logger.error(
                "Messages completion failed: client=%s user_agent=%r input_tokens=%d "
                "seed=%s tools=%s error=%s",
                request.client.host if request.client else "unknown",
                request.headers.get("user-agent", "")[:256], len(prepared[0]),
                prepared[1].seed, [tool["name"] for tool in req.get("tools", [])],
                str(exc)[:512],
            )
            return _error(str(exc), 500, "api_error")
        finally:
            if not job.completed.is_set():
                job.cancelled.set()
            # Also accounts for cancellation before complete() first runs.
            driver.metrics.record_request(job, output_tokens=0, reason="cancelled",
                                          ended=time.perf_counter())

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        try:
            body = await read_body(request)
            n = await while_connected(request, frontend.offload(frontend.count, body))
            return JSONResponse({"input_tokens": n})
        except (Disconnected, ClientDisconnect):
            return Response(status_code=499)
        except AdmissionError as exc:
            return _error(str(exc), exc.status, exc.kind,
                          {"Retry-After": "1"} if exc.status in (503, 529) else None)
        except (ValueError, TypeError) as exc:
            return _error(str(exc))
