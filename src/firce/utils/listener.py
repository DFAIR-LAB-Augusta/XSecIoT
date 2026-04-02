# src/firce/listener.py
"""
Queue-backed HTTP listener for live CSV flow batches.

This module receives CSV batches over HTTP, converts them to pandas
dataframes, and places them onto a thread-safe queue for consumption by
the streaming pipeline.
"""

from __future__ import annotations

import io
import logging
import threading

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StreamingBatchServer:
    """
    Threaded HTTP server that receives CSV payloads and exposes them as a queue.

    The request body must contain CSV text. Each successfully parsed payload
    is converted to a pandas dataframe and placed on ``queue``.
    """

    host: str = '127.0.0.1'
    port: int = 2048
    max_queue_size: int = 128
    queue: Queue[pd.DataFrame] = field(init=False)
    _server: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the internal dataframe queue."""
        self.queue = Queue(maxsize=self.max_queue_size)

    def start(self) -> None:
        """
        Start the background HTTP server.

        Raises:
            RuntimeError: If the server has already been started.
        """
        if self._server is not None:
            raise RuntimeError('StreamingBatchServer is already running.')

        outer = self

        class _Handler(BaseHTTPRequestHandler):
            """Request handler bound to the outer server instance."""

            server_version = 'FIRCEBatchListener/1.0'

            def do_POST(self) -> None:
                """Handle an incoming CSV batch."""
                try:
                    content_length = int(self.headers.get('Content-Length', '0'))
                    payload = self.rfile.read(content_length).decode('utf-8')
                    dataframe = pd.read_csv(io.StringIO(payload))

                    outer.queue.put_nowait(dataframe)
                    logger.info(
                        'Received live batch with %d rows and %d columns',
                        len(dataframe),
                        len(dataframe.columns),
                    )
                    self._write_response(200, 'CSV received')
                except Exception as exc:
                    logger.exception('Failed to handle incoming CSV batch: %s', exc)
                    self._write_response(400, f'Error: {exc}')

            def log_message(self, format: str, *args: object) -> None:
                """Route HTTP server logs through the module logger."""
                logger.debug('listener: ' + format, *args)

            def _write_response(self, status_code: int, body: str) -> None:
                """Write a plain-text HTTP response."""
                encoded = body.encode('utf-8')
                self.send_response(status_code)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name='firce-live-batch-server',
            daemon=True,
        )
        self._thread.start()
        logger.info('Streaming batch server started on http://%s:%d', self.host, self.port)

    def stop(self) -> None:
        """Stop the background HTTP server."""
        if self._server is None:
            return

        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        self._server = None
        self._thread = None
        logger.info('Streaming batch server stopped')

    def iter_batches(
        self,
        poll_timeout: float = 1.0,
    ) -> Iterator[pd.DataFrame]:
        """
        Yield incoming dataframe batches forever.

        Args:
            poll_timeout: Maximum time in seconds to wait for a batch before
                polling again.

        Yields:
            Live dataframe batches as they arrive.
        """
        while True:
            try:
                yield self.queue.get(timeout=poll_timeout)
            except Empty:
                continue
