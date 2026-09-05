import argparse
import logging
import re
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LABEL = 'Benign'

_IP_PATTERN = re.compile(
    r'^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
    r'(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$'
)


@dataclass(frozen=True)
class AttackMapping:
    src_ip: str
    dst_ip: str
    attack_name: str


def _is_valid_ip(ip: str) -> bool:
    """
    Validate if the given string is a valid IPv4 address.
    """
    return _IP_PATTERN.match(ip) is not None


def _parse_mapping(raw: str) -> AttackMapping:
    """
    Parse a 'src_ip,dst_ip,attack_name' string into an AttackMapping.
    Raises ValueError on malformed input.
    """
    fields = [f.strip() for f in raw.split(',')]
    if len(fields) != 3:
        raise ValueError(
            f"Mapping '{raw}' must have exactly 3 comma-separated fields: src_ip,dst_ip,attack_name"
        )
    src_ip, dst_ip, attack_name = fields
    if not _is_valid_ip(src_ip):
        raise ValueError(f'Invalid source IP address in mapping: {src_ip}')
    if not _is_valid_ip(dst_ip):
        raise ValueError(f'Invalid destination IP address in mapping: {dst_ip}')
    if not attack_name:
        raise ValueError(f"attack_name must not be empty in mapping: '{raw}'")
    return AttackMapping(src_ip=src_ip, dst_ip=dst_ip, attack_name=attack_name)


@dataclass(frozen=True)
class MCLabelConfig:
    dataset_path: Path
    mappings: Tuple[AttackMapping, ...]


def _parse_arguments(argv: List[str] | None = None) -> MCLabelConfig:
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Label a dataset with an MC_Label column by matching flows against '
            'src_ip/dst_ip pairs, each mapped to an attack name. Unmatched flows are '
            f"labeled '{DEFAULT_LABEL}'."
        )
    )
    parser.add_argument('dataset_path', type=str, help='Path to the CSV dataset.')
    parser.add_argument(
        '--mapping',
        dest='mappings',
        action='append',
        required=True,
        metavar='SRC_IP,DST_IP,ATTACK_NAME',
        help='Attack mapping as src_ip,dst_ip,attack_name. Repeat for multiple attack pairs.',
    )
    args = parser.parse_args(argv)
    mappings = tuple(_parse_mapping(raw) for raw in args.mappings)
    return MCLabelConfig(dataset_path=Path(args.dataset_path), mappings=mappings)


def _validate_inputs(dataset_path: Path, mappings: Tuple[AttackMapping, ...]) -> None:
    """
    Perform validations on input arguments.
    Raises ValueError or FileNotFoundError as appropriate.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f'File does not exist: {dataset_path}')
    if not dataset_path.is_file():
        raise ValueError(f'Provided path is not a file: {dataset_path}')
    if dataset_path.suffix.lower() != '.csv':
        raise ValueError('Only CSV files are supported.')

    seen: dict[Tuple[str, str], str] = {}
    for mapping in mappings:
        pair = (mapping.src_ip, mapping.dst_ip)
        if pair in seen and seen[pair] != mapping.attack_name:
            raise ValueError(
                f'duplicate mapping for {pair}: '
                f"'{seen[pair]}' vs '{mapping.attack_name}'"
            )
        seen[pair] = mapping.attack_name


def _add_multiclass_labels(dataset_path: Path, mappings: Tuple[AttackMapping, ...]) -> Path:
    """
    Add an MC_Label column to the dataset based on src/dst IP-pair -> attack-name
    mappings. Flows matching no mapping are labeled DEFAULT_LABEL.
    Returns the output file path.
    """
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise ValueError(f'Failed to read CSV file: {e}')

    if 'src_ip' not in df.columns or 'dst_ip' not in df.columns:
        raise ValueError("CSV must contain 'src_ip' and 'dst_ip' columns.")

    df['MC_Label'] = DEFAULT_LABEL
    for mapping in mappings:
        match = (df['src_ip'] == mapping.src_ip) & (df['dst_ip'] == mapping.dst_ip)
        df.loc[match, 'MC_Label'] = mapping.attack_name

    output_path = dataset_path.with_name(dataset_path.stem + '_mc_labeled' + dataset_path.suffix)
    df.to_csv(output_path, index=False)

    return output_path
