"""Canonical class registry and label derivation for multiclass FIRCE runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

KNOWN_CLASSES: tuple[str, ...] = (
    'Benign',
    'TCP_SYN_Flood',
    'UDP_Flood',
    'HTTP_Flood',
    'XMAS_Flood',
    'Nmap_Service_Detection',
    'SSDP_Discovery',
    'Hydra_HTTP_Default_Creds',
    'Slowloris',
    'C2_Beacon',
    'Exfil_Sim',
    'SSDP_Reflection',
    'ARP_Spoof_Gateway_MITM',
)

DEFAULT_HELD_OUT: tuple[str, ...] = (
    'HULK_HTTP_Flood',
    'Banner_Grab_HTTP',
    'Hydra_Telnet_Mirai_Creds',
    'HTTP_Fuzz',
    'C2_DNS_Beacon',
    'DNS_Tunnel_Exfil',
    'MDNS_Reflection',
    'ARP_Spoof_Bidirectional_MITM',
)

CANONICAL_CLASSES: frozenset[str] = frozenset(KNOWN_CLASSES) | frozenset(DEFAULT_HELD_OUT)

DATASET_ALIASES: dict[str, str] = {
    **{name: name for name in CANONICAL_CLASSES},
    'XMasAttack': 'XMAS_Flood',
    'TCPAttack': 'TCP_SYN_Flood',
    'UDPAttack': 'UDP_Flood',
    'HTTPAttack': 'HTTP_Flood',
}


def canonicalize_label(label: object) -> str:
    """
    Map a dataset label to its canonical class name.

    Raises:
        ValueError: If the label has no canonical mapping.
    """
    key = str(label).strip()
    try:
        return DATASET_ALIASES[key]
    except KeyError:
        raise ValueError(f'Unmapped class label: {key!r}') from None


def derive_mc_label(
    df: pd.DataFrame, label_column: str, held_out: tuple[str, ...] = (), *, drop_held_out: bool
) -> pd.DataFrame:
    """
    Return a copy of `df` with canonical `MC_Label` derived from `label_column`.

    Args:
        df: Frame holding the dataset label column.
        label_column: Name of the dataset label column.
        held_out: Canonical class names to exclude when `drop_held_out` is set.
        drop_held_out: Remove held-out rows (training/calibration) instead of keeping them (evaluation stream).

    Raises:
        ValueError: If `label_column` is missing or a label is unmapped.
    """
    if label_column not in df.columns:
        raise ValueError(f'label column {label_column!r} not found. Columns found: {df.columns.tolist()}')
    out = df.copy()
    out['MC_Label'] = out[label_column].map(canonicalize_label)
    if drop_held_out and held_out:
        out = out.loc[~out['MC_Label'].isin(held_out)].reset_index(drop=True)
    return out
