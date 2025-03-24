# src/core/rolling_csv
"""
Rolling Log Utility for Streaming Simulations

This module provides a utility class `RollingCSV` for managing a
compressed rolling log of CSV-formatted data, optimized for streaming
IoT or network flow simulations.

Key Features:
- Writes log entries in buffered batches to `.csv.gz` format.
- Maintains a fixed-size rolling window by truncating the log when `max_rows` is exceeded.
- Supports automatic header management, row counting, and efficient append/flush cycles.

Typical usage:
    logger = RollingCSV("ce_log.csv.gz", max_rows=10000, columns=[...])
    logger.append([...])
    logger.flush()
    logger.close()

Used primarily in CE simulation pipelines to log predicted network flows.
"""
import gzip
import logging
import os
import shutil
import time

from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

FLUSH_THRESHOLD = 50


class RollingCSV:
    """
    A logger that appends rows to a compressed CSV file and maintains a rolling window
    of at most `max_rows` entries. Data is written in buffered batches and truncated
    when the row limit is reached.

    Attributes:
        path (str): Path to the gzip-compressed CSV log file.
        max_rows (int): Maximum number of rows to retain.
        buffer (list): In-memory buffer to accumulate rows before writing.
        count (int): Running count of total rows in the log file.
        columns (List[str] | None): Column headers expected in each row.
    """

    path: str
    max_rows: int
    buffer: List[list]
    count: int
    columns: List[str] | None

    def __init__(
        self, 
        path: str | None = "ce_log.csv.gz", 
        max_rows: int = 10000, 
        columns: List[str] | None = None
    ) -> None:
        """
        Initialize the rolling logger and count existing rows if the log file exists.

        Args:
            path (str): File path to the compressed CSV file.
            max_rows (int): Max number of rows to keep in the file.
            columns (List[str] | None): Optional list of column headers.
        """
        self.path = path if path else "ce_log.csv.gz"
        self.max_rows = max_rows
        self.buffer = []
        self.count = 0
        self.columns = columns
        t0 = time.perf_counter()
        if os.path.exists(self.path):
            with gzip.open(self.path, 'rt') as f:
                self.count = sum(1 for _ in f)
            logger.info("Initialized RollingCSV from existing file: %d rows counted in %.4fs", self.count, time.perf_counter() - t0)  # noqa: E501
        else:
            logger.info("Initialized RollingCSV with new file")
            
    def append(self, row: List) -> None:
        """
        Add a single row to the in-memory buffer and flush to disk if the threshold is reached.
        Automatically triggers truncation if total rows exceed `max_rows`.

        Args:
            row (list): A list of values to log as a single CSV row.
        """
        self.buffer.append(row)
        self.count += 1
        if len(self.buffer) >= FLUSH_THRESHOLD:
            logger.debug("Buffer reached threshold (%d); flushing...", FLUSH_THRESHOLD)
            self.flush()

        if self.count >= self.max_rows:
            logger.debug("Max row count exceeded (%d); truncating...", self.max_rows)  # Displaying this log message can be noisy  # noqa: E501
            self._truncate_to_last_n(self.max_rows)

    def flush(self) -> None:
        """
        Write all buffered rows to the compressed CSV file and clear the buffer.

        Raises:
            ValueError: If the `columns` attribute is not set.
        """
        if not self.buffer:
            return
        t0 = time.perf_counter()

        file_exists = os.path.exists(self.path)
        write_header = not file_exists or os.path.getsize(self.path) == 0

        if not self.columns:
            raise ValueError("Logger 'columns' attribute is required to flush data with headers.")

        df = pd.DataFrame(self.buffer, columns=self.columns)

        with gzip.open(self.path, "at") as f:
            df.to_csv(f, header=write_header, index=False)

        logger.debug("Flushed %d rows to log in %.4fs", len(self.buffer), time.perf_counter() - t0)
        self.buffer = []

    def _truncate_to_last_n(self, n: int) -> None:
        """
        Truncate the CSV file to keep only the last `n` rows.

        Args:
            n (int): Number of rows to keep.
        """
        t0 = time.perf_counter()
        df = pd.read_csv(self.path, compression='gzip', low_memory=False)
        df = df.tail(n)
        tmp = self.path + ".tmp"
        df.to_csv(tmp, index=False, compression='gzip')
        shutil.move(tmp, self.path)
        self.count = len(df)
        logger.debug("Truncated log to last %d rows in %.4fs", n, time.perf_counter() - t0)  # Displaying this log message can be noisy  # noqa: E501

    def close(self) -> None:
        """
        Flush any remaining buffered rows and close the logger.
        """
        logger.debug("Closing logger and flushing remaining %d buffered rows", len(self.buffer))
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
    )
