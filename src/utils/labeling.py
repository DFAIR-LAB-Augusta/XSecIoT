import argparse
import logging
import re
import sys

from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _is_valid_ip(ip: str) -> bool:
    """
    Validate if the given string is a valid IPv4 address.
    """
    pattern = re.compile(
        r"^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
        r"(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$"
    )
    return pattern.match(ip) is not None


def _validate_inputs(dataset_path: Path, src_ip: str, dest_ip: str) -> None:
    """
    Perform validations on input arguments.
    Raises ValueError or FileNotFoundError as appropriate.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"File does not exist: {dataset_path}")
    if not dataset_path.is_file():
        raise ValueError(f"Provided path is not a file: {dataset_path}")
    if dataset_path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")
    if not _is_valid_ip(src_ip):
        raise ValueError(f"Invalid source IP address: {src_ip}")
    if not _is_valid_ip(dest_ip):
        raise ValueError(f"Invalid destination IP address: {dest_ip}")


def _add_binary_labels(dataset_path: Path, src_ip: str, dest_ip: str) -> Path:
    """
    Add a Bin_Label column to the dataset based on matching src/dst IPs.
    Returns the output file path.
    """
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    if "src_ip" not in df.columns or "dst_ip" not in df.columns:
        raise ValueError("CSV must contain 'src_ip' and 'dst_ip' columns.")

    df["BinLabel"] = (
        (df["src_ip"] == src_ip) & (df["dst_ip"] == dest_ip)
    ).astype(int)

    output_path = dataset_path.with_name(
        dataset_path.stem + "_labeled" + dataset_path.suffix
    )
    df.to_csv(output_path, index=False)

    return output_path


def _parse_arguments() -> argparse.Namespace:
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Label a dataset with Bin_Label column "
        "based on matching source and destination IPs."
    )
    parser.add_argument(
        "dataset_path", type=str, help="Path to the CSV dataset."
    )
    parser.add_argument("src_ip", type=str, help="Source IP address.")
    parser.add_argument("dest_ip", type=str, help="Destination IP address.")
    return parser.parse_args()


def main() -> None:
    """
    Main function to execute the labeling pipeline.
    """
    args = _parse_arguments()
    dataset_path = Path(args.dataset_path)

    try:
        _validate_inputs(dataset_path, args.src_ip, args.dest_ip)
        output_path = _add_binary_labels(dataset_path, args.src_ip, args.dest_ip)
        print(f"Labeled dataset saved to: {output_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
