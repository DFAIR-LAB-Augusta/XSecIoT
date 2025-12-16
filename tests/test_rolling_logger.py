import csv
import gzip

from src.core.rolling_csv import RollingCSV


def read_all_rows(path):
    """Helper to read all CSV rows from a gzip file."""
    with gzip.open(path, 'rt') as f:
        return list(csv.reader(f))


def test_init_no_file(tmp_path):
    path = tmp_path / 'log.csv.gz'
    logger = RollingCSV(str(path), max_rows=100)
    assert logger.count == 0
    assert logger.buffer == []
    assert not path.exists()


def test_init_with_existing_file(tmp_path):
    path = tmp_path / 'log.csv.gz'
    # Precreate 3 rows
    rows = [['a', '1'], ['b', '2'], ['c', '3']]
    with gzip.open(path, 'wt') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    logger = RollingCSV(str(path), max_rows=100)
    assert logger.count == 3


def test_truncate_direct(tmp_path):
    path = tmp_path / 'log.csv.gz'
    # Create a 20-row file
    rows = [[str(i)] for i in range(20)]
    with gzip.open(path, 'wt') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    logger = RollingCSV(str(path), max_rows=100)

    # Truncate to last 10 rows
    logger._truncate_to_last_n(10)
    # count resets to the tail length
    assert logger.count == 10
    all_rows = read_all_rows(str(path))
    # first row is a “header” (the old row-0), so drop it:
    data_rows = all_rows[1:]
    assert data_rows == [[str(i)] for i in range(10, 20)]
