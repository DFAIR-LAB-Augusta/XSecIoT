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
