import csv
import gzip

from src.core.rolling_csv import RollingCSV


def read_all_rows(path):
    """Helper to read all CSV rows from a gzip file."""
    with gzip.open(path, 'rt') as f:
        return list(csv.reader(f))


def test_init_no_file(tmp_path):
    path = tmp_path / "log.csv.gz"
    logger = RollingCSV(str(path), max_rows=100)
    assert logger.count == 0
    assert logger.buffer == []
    assert not path.exists()


def test_init_with_existing_file(tmp_path):
    path = tmp_path / "log.csv.gz"
    # Precreate 3 rows
    rows = [["a", "1"], ["b", "2"], ["c", "3"]]
    with gzip.open(path, 'wt') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    logger = RollingCSV(str(path), max_rows=100)
    assert logger.count == 3


def test_append_and_manual_flush(tmp_path):
    path = tmp_path / "log.csv.gz"
    logger = RollingCSV(str(path), max_rows=100)
    # Append 10 rows (below auto-flush threshold)
    for i in range(10):
        logger.append([f"row{i}", str(i)])
    assert not path.exists()
    assert logger.count == 10

    # Now flush manually
    logger.flush()
    assert path.exists()
    rows = read_all_rows(str(path))
    expected = [[f"row{i}", str(i)] for i in range(10)]
    assert rows == expected
    assert logger.buffer == []


def test_auto_flush_threshold(tmp_path):
    path = tmp_path / "log.csv.gz"
    logger = RollingCSV(str(path), max_rows=100)
    # Append exactly 50 rows → should auto-flush and clear buffer
    for i in range(50):
        logger.append([str(i)])
    assert logger.buffer == []
    rows = read_all_rows(str(path))
    assert rows == [[str(i)] for i in range(50)]


def test_truncate_direct(tmp_path):
    path = tmp_path / "log.csv.gz"
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


def test_close_flushes_remaining(tmp_path):
    path = tmp_path / "log.csv.gz"
    logger = RollingCSV(str(path), max_rows=100)
    # Append fewer than threshold
    for i in range(5):
        logger.append([str(i)])
    assert not path.exists()

    # close() should flush whatever is left
    logger.close()
    assert path.exists()
    rows = read_all_rows(str(path))
    assert rows == [[str(i)] for i in range(5)]
