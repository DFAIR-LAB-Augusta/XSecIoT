# src/core/circular_logger.py
"""
In-memory circular logger for simulation logs.

Implements a fixed-size rolling buffer using collections.deque.
Faster alternative to disk-based RollingCSV for simulation pipelines.
"""

from collections import deque
from typing import List, Optional

import pandas as pd


class CircularDequeLogger:
    """
    A fixed-length in-memory logger using deque.

    Attributes:
        max_rows (int): Maximum number of rows to retain.
        columns (Optional[List[str]]): Column headers for DataFrame export.
        buffer (deque): The rolling buffer.
    """

    def __init__(
        self,
        _: str | None = "ce_log.csv.gz",
        max_rows: int = 10000,
        columns: Optional[List[str]] = None
    ):
        self.max_rows = max_rows
        self.columns = columns
        self.buffer = deque(maxlen=max_rows)

    def append(self, row: List) -> None:
        """
        Append a single row to the in-memory log.

        Args:
            row (List): A list of column values.
        """
        if self.columns is not None and len(row) != len(self.columns):
            raise ValueError(
                f"[CircularDequeLogger] Row width {len(row)} != schema width {len(self.columns)}"
            )
        self.buffer.append(row)

    def flush(self) -> None:
        """No-op for in-memory; defined for interface compatibility."""
        pass

    def close(self) -> None:
        """No-op for in-memory; defined for interface compatibility."""
        self.flush()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert current buffer contents to a pandas DataFrame.

        Returns:
            pd.DataFrame: The buffer as a DataFrame with optional headers.
        """
        return pd.DataFrame(list(self.buffer), columns=self.columns)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run directly. "
    )
