import argparse
import os
import sys

from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from firce.classes import CANONICAL_CLASSES

DEFAULT_LABEL = 'Benign'
GT_COLUMNS = ('session_id', 'device', 'target_ip', 'label', 'attempt', 'start_unix', 'end_unix')
FLOW_COLUMNS = ('src_ip', 'dst_ip', 'timestamp', 'flow_duration')
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
EXCLUDED_CAPEX_LABELS = frozenset({'Config_Tamper_Setup_CGI', 'Config_Tamper_TR064_SOAP'})


@dataclass(frozen=True)
class CapexLabelConfig:
    flows_path: Path
    gt_paths: tuple[Path, ...]
    flow_tz: str | None
    output_path: Path | None


def _parse_arguments(argv: List[str] | None = None) -> CapexLabelConfig:
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Label a CICFlowMeter flow CSV with MC_Label and MC_Partial by joining CAPEX '
            f"ground-truth attack windows on time overlap and target IP. Unmatched flows are '{DEFAULT_LABEL}'."
        )
    )
    parser.add_argument('flows_csv', type=str, help='Path to the CICFlowMeter flow CSV.')
    parser.add_argument('gt_csv', type=str, nargs='+', help='One or more CAPEX ground-truth CSVs.')
    parser.add_argument(
        '--flowTz',
        dest='flow_tz',
        type=str,
        default=None,
        help='IANA timezone of the naive flow timestamps (default: host local zone).',
    )
    parser.add_argument('--output', type=str, default=None, help='Output path (default: <flows stem>_mc_labeled.csv).')
    args = parser.parse_args(argv)
    return CapexLabelConfig(
        flows_path=Path(args.flows_csv),
        gt_paths=tuple(Path(p) for p in args.gt_csv),
        flow_tz=args.flow_tz,
        output_path=Path(args.output) if args.output else None,
    )


def _host_zone() -> tzinfo:
    """
    Return the host's DST-aware local zone, falling back to the current fixed offset with a warning.
    """
    env_tz = os.environ.get('TZ', '').lstrip(':')
    if env_tz:
        try:
            return ZoneInfo(env_tz)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    localtime = Path('/etc/localtime')
    if localtime.exists():
        parts = localtime.resolve().parts
        if 'zoneinfo' in parts:
            key = '/'.join(parts[len(parts) - parts[::-1].index('zoneinfo') :])
            try:
                return ZoneInfo(key)
            except (ZoneInfoNotFoundError, ValueError):
                pass
        try:
            with localtime.open('rb') as fh:
                return ZoneInfo.from_file(fh)
        except (OSError, ValueError):
            pass
    print(
        'Warning: could not resolve the host IANA zone; using a fixed UTC offset, DST will not be honoured.',
        file=sys.stderr,
    )
    return datetime.now().astimezone().tzinfo


def _resolve_timezone(name: str | None) -> tuple[object, str]:
    """
    Return (tzinfo, display name). Raises ValueError for an unknown zone name.
    """
    if name is None:
        tz = _host_zone()
        return tz, f'{tz} (host local)'
    try:
        return ZoneInfo(name), name
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown timezone '{name}': {exc}")


def _read_csv(path: Path, what: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'{what} file does not exist: {path}')
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise ValueError(f'Failed to read {what} CSV {path}: {e}')


def _load_ground_truth(gt_paths: tuple[Path, ...]) -> pd.DataFrame:
    """
    Load and validate CAPEX ground-truth CSVs into one frame, sorted by start_unix.
    """
    frames = []
    for path in gt_paths:
        gt = _read_csv(path, 'ground-truth')
        missing = [c for c in GT_COLUMNS if c not in gt.columns]
        if missing:
            raise ValueError(f'Ground-truth CSV {path} is missing column(s): {", ".join(missing)}')
        frames.append(gt)
    gt = pd.concat(frames, ignore_index=True)

    blank_counts = {}
    for column in ('target_ip', 'label', 'start_unix', 'end_unix'):
        values = gt[column].astype('string').str.strip()
        gt[column] = values.mask(values == '').astype(object)
        blank = int(gt[column].isna().sum())
        if blank:
            blank_counts[column] = blank
    if blank_counts:
        details = ', '.join(f'{column} ({count} row(s))' for column, count in blank_counts.items())
        raise ValueError(f'Ground-truth rows with blank required field(s): {details}')

    excluded = sorted(set(gt['label'].astype(str)) & EXCLUDED_CAPEX_LABELS)
    if excluded:
        raise ValueError(
            f'Ground-truth label(s) {", ".join(excluded)} are excluded from Direction 3 (#194); '
            'rerun the session without them or remove their rows.'
        )

    bad = sorted(set(gt['label'].astype(str)) - CANONICAL_CLASSES)
    if bad:
        raise ValueError(f'Non-canonical ground-truth label(s): {", ".join(bad)}')

    gt['start_unix'] = pd.to_numeric(gt['start_unix'], errors='raise')
    gt['end_unix'] = pd.to_numeric(gt['end_unix'], errors='raise')
    if (gt['end_unix'] < gt['start_unix']).any():
        raise ValueError('Ground-truth rows with end_unix < start_unix found.')
    return gt.sort_values('start_unix', kind='stable').reset_index(drop=True)


def _flow_start_unix(timestamps: pd.Series, tz: object) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (unix seconds, valid mask). Timestamps that are ambiguous or nonexistent in tz (a DST
    transition hour) are invalid; their unix value is NaN.
    """
    parsed = pd.to_datetime(timestamps, format=TIMESTAMP_FORMAT, errors='raise')
    localized = parsed.dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
    unix = (
        (localized.dt.tz_convert('UTC').dt.tz_localize(None) - pd.Timestamp('1970-01-01')).dt.total_seconds().to_numpy()
    )
    return unix, ~localized.isna().to_numpy()


def _label_flows(flows: pd.DataFrame, gt: pd.DataFrame, tz: object) -> tuple[pd.DataFrame, int]:
    """
    Add MC_Label and MC_Partial to flows. A flow takes the label of the matching window with the
    largest overlap (earliest start on a tie); it is partial when not fully inside that window
    or when it matched more than one window. Flows whose timestamp is ambiguous or nonexistent in tz
    are dropped; returns the labeled flows and the dropped count.
    """
    missing = [c for c in FLOW_COLUMNS if c not in flows.columns]
    if missing:
        raise ValueError(f'Flow CSV is missing column(s): {", ".join(missing)}')

    t0, valid = _flow_start_unix(flows['timestamp'], tz)
    dropped = int((~valid).sum())
    flows = flows[valid]
    t0 = t0[valid]
    t1 = t0 + pd.to_numeric(flows['flow_duration'], errors='raise').to_numpy(dtype=float)
    src = flows['src_ip'].to_numpy()
    dst = flows['dst_ip'].to_numpy()

    labels = np.full(len(flows), DEFAULT_LABEL, dtype=object)
    best = np.full(len(flows), -1.0)
    n_match = np.zeros(len(flows), dtype=int)
    inside = np.zeros(len(flows), dtype=bool)

    for row in gt.itertuples(index=False):
        start, end = float(row.start_unix), float(row.end_unix)
        hit = ((src == row.target_ip) | (dst == row.target_ip)) & (t0 <= end) & (t1 >= start)
        overlap = np.minimum(t1, end) - np.maximum(t0, start)
        better = hit & (overlap > best)
        labels[better] = row.label
        best[better] = overlap[better]
        inside[better] = ((t0 >= start) & (t1 <= end))[better]
        n_match += hit

    out = flows.copy()
    out['MC_Label'] = labels
    out['MC_Partial'] = (n_match > 0) & (~inside | (n_match > 1))
    return out, dropped


def _add_capex_labels(config: CapexLabelConfig) -> tuple[Path, str, pd.DataFrame, list[str], int]:
    gt = _load_ground_truth(config.gt_paths)
    tz, tz_name = _resolve_timezone(config.flow_tz)
    flows = _read_csv(config.flows_path, 'flow')
    labeled, dropped = _label_flows(flows, gt, tz)

    output_path = config.output_path or config.flows_path.with_name(config.flows_path.stem + '_mc_labeled.csv')
    labeled.to_csv(output_path, index=False)
    names = [DEFAULT_LABEL] + sorted(set(gt['label']) - {DEFAULT_LABEL})
    return output_path, tz_name, labeled, names, dropped


def main(argv: List[str] | None = None) -> None:
    """
    Main function to execute the CAPEX window labeling pipeline.
    """
    try:
        config = _parse_arguments(argv)
        output_path, tz_name, labeled, names, dropped = _add_capex_labels(config)
        counts = labeled['MC_Label'].value_counts()
        print(f'Labeled dataset saved to: {output_path}')
        print(f'Flow timezone: {tz_name}')
        for name in names:
            print(f'{name}: {int(counts.get(name, 0))}')
        print(f'MC_Partial: {int(labeled["MC_Partial"].sum())}')
        print(f'Dropped (DST transition): {dropped}')
        if dropped:
            print(
                f'Warning: dropped {dropped} flow(s) whose timestamp is ambiguous or nonexistent in {tz_name}; '
                'extract with TZ=UTC and pass --flowTz UTC to avoid this.',
                file=sys.stderr,
            )
    except SystemExit:
        raise
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
