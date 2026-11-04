"""Serialize HTTP jobs across eight engine ranks with a Gloo control channel."""

from __future__ import annotations

import asyncio
import logging
import queue
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from .protocol import SamplingParams
from .metrics import Metrics


class AdmissionError(RuntimeError):
    """A retryable refusal before a request starts streaming."""

    def __init__(self, message, *, status, kind):
        super().__init__(message)
        self.status, self.kind = status, kind


@dataclass
class Job:
    prompt_ids: list[int]
    params: SamplingParams
    max_new: int
    loop: object
    request_id: str
    cache_read_tokens: int = 0
    received_at: float = field(default_factory=time.perf_counter)
    parsed_at: float = field(default_factory=time.perf_counter)
    prepared_at: float = field(default_factory=time.perf_counter)
    queued_at: float = field(default_factory=time.perf_counter)
    started_at: float = 0.
    first_token_at: float | None = None
    first_content_at: float | None = None
    first_visible_at: float | None = None
    first_tool_at: float | None = None
    kind: str = "main"
    request_recorded: bool = False
    phases: dict = field(default_factory=dict)
    cancelled: threading.Event = field(default_factory=threading.Event)
    completed: threading.Event = field(default_factory=threading.Event)
    event_lock: object = field(default_factory=threading.Lock)
    events: deque = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    wake_pending: bool = False

    def send(self, event):
        # Buffer without blocking the engine thread and keep at most one wake-up
        # queued on the event loop, so a stalled loop delays tokens but never
        # drops them. The buffer is bounded by the job's max_new tokens.
        with self.event_lock:
            self.events.append(event)
            if self.wake_pending:
                return
            self.wake_pending = True
        try:
            self.loop.call_soon_threadsafe(self.wake.set)
        except RuntimeError:
            self.cancelled.set()  # The client event loop has shut down.

    async def next_event(self):
        while True:
            with self.event_lock:
                if self.events:
                    return self.events.popleft()
                self.wake_pending = False
                self.wake.clear()
            await self.wake.wait()

    def emit(self, token_id):
        if not self.cancelled.is_set():
            self.send(("tok", token_id))

    def done(self, finish_reason):
        self.completed.set()
        self.send(("done", finish_reason))

    def fail(self, message):
        self.completed.set()
        self.send(("err", message))


class EngineDriver:
    def __init__(self, engine, tokenizer, group, *, rank, local_rank, model_name):
        self.engine, self.tok, self.group = engine, tokenizer, group
        self.rank, self.local_rank, self.model_name = rank, local_rank, model_name
        self.max_model_len = engine.max_model_len
        self.eos_ids = {tokenizer.eos_token_id}
        vocab = tokenizer.get_vocab()
        self.eos_ids.update(vocab[t] for t in ("<|im_end|>", "<|iquest_end|>") if t in vocab)
        self.eos_ids = {t for t in self.eos_ids if isinstance(t, int) and t >= 0}
        self.jobs = queue.Queue(maxsize=8)
        self.admission_lock = threading.Lock()
        self.metrics = Metrics()
        self.healthy = True
        self.stopping = threading.Event()
        self.active = None
        self.thread = None

    def next_seed(self):
        return secrets.randbits(63)

    def clamp_max_new(self, prompt_len, requested):
        room = self.max_model_len - prompt_len
        if room < 1:
            raise ValueError("prompt leaves no output space")
        # Q8 verification writes ahead before rewinding its rejected rows.
        return min(requested, max(1, room - 6))

    def submit(self, job):
        with self.admission_lock:
            if not self.healthy or self.stopping.is_set():
                raise AdmissionError("engine is unavailable", status=503, kind="api_error")
            try:
                job.queued_at = time.perf_counter()
                self.jobs.put_nowait(job)
            except queue.Full:
                raise AdmissionError("request queue is full", status=529,
                                     kind="overloaded_error") from None

    def _broadcast(self, command=None):
        import torch.distributed as dist

        data = [command]
        dist.broadcast_object_list(data, src=0, group=self.group)
        return data[0]

    def _generate(self, command, job):
        if job is not None:
            with self.admission_lock:
                job.started_at = time.perf_counter()
                # Shutdown may arrive while the generate command is broadcasting,
                # before run() has exposed this job as active.
                if self.stopping.is_set() and not job.completed.is_set():
                    job.cancelled.set()
                    job.fail("server is shutting down")
        def cancelled():
            return self._broadcast(
                self.stopping.is_set() or (job is not None and job.cancelled.is_set())
                if self.rank == 0 else None
            )

        def emit(tokens):
            if job is not None:
                if tokens and job.first_token_at is None:
                    job.first_token_at = time.perf_counter()
                for token in tokens:
                    job.emit(token)

        out, _ = self.engine.generate_mtp(
            command["ids"], command["max_new"], seed=command["seed"],
            eos_token_id=self.eos_ids, on_tokens=emit, cancelled=cancelled,
        )
        if job is not None:
            job.phases = dict(self.engine.last_request_stats)
            job.cache_read_tokens = job.phases["cache_read_tokens"]
            self.metrics.record_engine(job)
            reason = "stop" if out[-1] in self.eos_ids or job.cancelled.is_set() else "length"
            with self.admission_lock:
                if not job.completed.is_set():
                    job.done(reason)

    def run(self):
        import torch

        torch.cuda.set_device(self.local_rank)
        try:
            while True:
                job = None
                if self.rank == 0:
                    if self.stopping.is_set():
                        command = {"op": "shutdown"}
                    else:
                        try:
                            job = self.jobs.get(timeout=30)
                        except queue.Empty:
                            pass
                        if self.stopping.is_set():
                            if job is not None:
                                job.fail("server is shutting down")
                            command = {"op": "shutdown"}
                        elif job is None or job.cancelled.is_set():
                            if job is not None:
                                job.done("stop")
                            command = {"op": "idle"}
                        else:
                            command = dict(op="generate", ids=job.prompt_ids,
                                           max_new=job.max_new, seed=job.params.seed)
                else:
                    command = None
                command = self._broadcast(command)
                if command["op"] == "shutdown":
                    return
                if command["op"] == "idle":
                    continue
                self.active = job
                self._generate(command, job)
                self.active = None
        except Exception:
            self.healthy = False
            logging.exception("Fatal engine failure; all ranks must restart")
            if self.active is not None:
                self.active.fail("engine failed; restart required")
            # torchrun terminates the remaining ranks when any process fails.
            import os
            os._exit(1)

    def start(self):
        self.thread = threading.Thread(target=self.run, name="iquestrt-engine", daemon=True)
        self.thread.start()

    def stop(self):
        """Stop admission, cancel generation, and wake an idle control thread."""
        with self.admission_lock:
            if self.stopping.is_set():
                return
            self.stopping.set()
            self.healthy = False
            active = self.active
            if active is not None and not active.completed.is_set():
                active.cancelled.set()
                active.fail("server is shutting down")
            while True:
                try:
                    self.jobs.get_nowait().fail("server is shutting down")
                except queue.Empty:
                    break
            self.jobs.put_nowait(None)

    def close(self):
        self.stop()
        if self.thread is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                logging.error("Engine did not stop within 10 seconds; terminating ranks")
                import os
                os._exit(1)
