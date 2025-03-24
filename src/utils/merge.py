import argparse
import logging
import sys

from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _validate_directory(directory: Path) -> None:
    """
    Validate that the provided path exists and is a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Provided path is not a directory: {directory}")


def _merge_and_sort_csvs(input_dir: Path) -> Path:
    """
    Merge all CSVs in the input directory and sort by 'timestamp' (descending).
    Returns the path to the output file.
    """
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    dataframes = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            dataframes.append(df)
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}", file=sys.stderr)

    if not dataframes:
        raise ValueError("No valid CSV files were loaded.")

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.sort_values(by="timestamp", ascending=False, inplace=True)

    output_file = input_dir / f"{input_dir.name}_merged.csv"
    merged_df.to_csv(output_file, index=False)

    return output_file


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Merge and sort CSV files by timestamp from a given directory."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory containing CSV files to merge."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to handle argument parsing and merge execution.
    """
    args = _parse_args()
    input_dir = Path(args.directory)

    try:
        _validate_directory(input_dir)
        output_path = _merge_and_sort_csvs(input_dir)
        print(f"Merged CSV saved to: {output_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
