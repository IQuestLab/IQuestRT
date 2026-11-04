"""Cancel distributed generation before Uvicorn waits for HTTP requests to drain."""

import logging
import os
import threading

import uvicorn
from uvicorn.protocols.http.h11_impl import H11Protocol

from .api import create_app
from . import limits


class HTTPProtocol(H11Protocol):
    """Abort stalled writes, including Uvicorn's pre-header error responses."""

    _write_deadline = None

    def pause_writing(self):
        super().pause_writing()
        if self._write_deadline is None:
            self._write_deadline = self.loop.call_later(limits.IO_TIMEOUT, self.transport.abort)

    def _cancel_write_deadline(self):
        if self._write_deadline is not None:
            self._write_deadline.cancel()
            self._write_deadline = None

    def resume_writing(self):
        self._cancel_write_deadline()
        super().resume_writing()

    def connection_lost(self, exc):
        self._cancel_write_deadline()
        super().connection_lost(exc)


class Server(uvicorn.Server):
    def __init__(self, driver, *, host, port):
        super().__init__(uvicorn.Config(create_app(driver, host), host=host, port=port,
                                        http=HTTPProtocol, timeout_graceful_shutdown=5))
        self.driver = driver
        self._owns_runner = False
        self._deadline = None

    def run(self, sockets=None):
        # Keep the deadline armed through asyncio.run's executor teardown.
        # An embedded serve() caller owns its event loop and process lifetime.
        self._owns_runner = True
        try:
            return super().run(sockets=sockets)
        finally:
            if self._deadline is not None:
                self._deadline.cancel()
            self._owns_runner = False

    def _arm_deadline(self):
        if self._owns_runner and self._deadline is None:
            def expired():
                logging.error("Server did not exit within 15 seconds; terminating ranks")
                os._exit(1)
            self._deadline = threading.Timer(15, expired)
            self._deadline.daemon = True
            self._deadline.start()

    def handle_exit(self, sig, frame):
        self._arm_deadline()
        super().handle_exit(sig, frame)

    async def shutdown(self, sockets=None):
        self._arm_deadline()
        self.driver.stop()
        try:
            await super().shutdown(sockets)
        finally:
            # Finish the Gloo shutdown broadcast before Uvicorn re-raises SIGTERM.
            # close() bounds the join and exits on an unresponsive engine.
            self.driver.close()
