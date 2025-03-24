# src.FIRE.preprocessing

import argparse
import logging
import os
import sys
import time

import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy.stats

from tqdm import tqdm

logger = logging.getLogger(__name__)
np.random.seed(42)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified preprocessing pipeline for FIRE")
    parser.add_argument("file_path", type=str,
                        help="Path to the dataset CSV file")
    parser.add_argument("--window_size", type=str,
                        default="5s", help="Window size (e.g., '5s')")
    parser.add_argument("--step_size", type=str,
                        default="1s", help="Step size (e.g., '1s')")
    parser.add_argument("--unsw", action="store_true",
                        help="Enable UNSW-specific preprocessing pipeline")
    return parser.parse_args()


def clean_data(data: pd.DataFrame, is_unsw: bool) -> pd.DataFrame:
    if is_unsw:
        required_columns = {
            'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL',
            'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'FLOW_DURATION_MILLISECONDS',
            'IN_PKTS', 'OUT_PKTS', 'IN_BYTES', 'OUT_BYTES', 'SRC_TO_DST_IAT_MIN',
            'SRC_TO_DST_IAT_MAX', 'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV',
            'DST_TO_SRC_IAT_MIN', 'DST_TO_SRC_IAT_MAX', 'DST_TO_SRC_IAT_AVG',
            'DST_TO_SRC_IAT_STDDEV'
        }
        missing_cols = required_columns - set(data.columns)
        if missing_cols:
            # raise ValueError(f"Missing expected UNSW columns: {missing_cols}")
            logger.debug(
                f"Missing expected UNSW columns when preprocessing: {missing_cols}")
            return data

        mapping = {
            'IPV4_SRC_ADDR': 'src_ip',
            'IPV4_DST_ADDR': 'dst_ip',
            'L4_SRC_PORT': 'src_port',
            'L4_DST_PORT': 'dst_port',
            'PROTOCOL': 'protocol',
            'FLOW_START_MILLISECONDS': 'start_time',
            'FLOW_END_MILLISECONDS': 'end_time',
            'FLOW_DURATION_MILLISECONDS': 'flow_duration',
            'IN_PKTS': 'tot_bwd_pkts',
            'OUT_PKTS': 'tot_fwd_pkts',
            'IN_BYTES': 'totlen_bwd_pkts',
            'OUT_BYTES': 'totlen_fwd_pkts',
            'SRC_TO_DST_IAT_MIN': 'fwd_iat_min',
            'SRC_TO_DST_IAT_MAX': 'fwd_iat_max',
            'SRC_TO_DST_IAT_AVG': 'fwd_iat_mean',
            'SRC_TO_DST_IAT_STDDEV': 'fwd_iat_std',
            'DST_TO_SRC_IAT_MIN': 'bwd_iat_min',
            'DST_TO_SRC_IAT_MAX': 'bwd_iat_max',
            'DST_TO_SRC_IAT_AVG': 'bwd_iat_mean',
            'DST_TO_SRC_IAT_STDDEV': 'bwd_iat_std'
        }
        data.rename(columns=mapping, inplace=True)
        data['start_time'] = pd.to_datetime(
            data['start_time'], unit='ms', errors='coerce')
        data['end_time'] = pd.to_datetime(
            data['end_time'], unit='ms', errors='coerce')
        data['flow_duration'] *= 1000
        data.set_index('start_time', inplace=True)
        data.sort_index(inplace=True)

        for col in ['tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts']:
            if col not in data.columns:
                raise ValueError(
                    f"Missing required column after renaming: {col}")

        data['fwd_pkt_len_mean'] = np.where(data['tot_fwd_pkts'] != 0, data['totlen_fwd_pkts'] / data['tot_fwd_pkts'], 0)  # noqa: E501
        data['bwd_pkt_len_mean'] = np.where(data['tot_bwd_pkts'] != 0, data['totlen_bwd_pkts'] / data['tot_bwd_pkts'], 0)  # noqa: E501
        total_pkts = data['tot_fwd_pkts'] + data['tot_bwd_pkts']
        data['pkt_len_mean'] = np.where(total_pkts != 0, (data['totlen_fwd_pkts'] + data['totlen_bwd_pkts']) / total_pkts, 0)  # noqa: E501
        data['flow_iat_mean'] = np.where(
            total_pkts > 1, data['flow_duration'] / (total_pkts - 1), 0)
        data['down_up_ratio'] = np.where(
            data['tot_fwd_pkts'] > 0, data['tot_bwd_pkts'] / data['tot_fwd_pkts'], 0)
        data['fwd_iat_tot'] = data['fwd_iat_mean'] * (data['tot_fwd_pkts'] - 1)
        data['bwd_iat_tot'] = data['bwd_iat_mean'] * (data['tot_bwd_pkts'] - 1)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['src_ip', 'dst_ip', 'src_port',
                    'dst_port', 'protocol', 'flow_duration'], inplace=True)
        return data
    else:
        if 'Unnamed: 0' in data.columns:
            data.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
            data.drop(columns=['id'], inplace=True, errors='ignore')
        # if 'timestamp' not in data.columns:
        #     raise ValueError("Missing 'timestamp' column in non-UNSW dataset")
        # data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce', format='%d-%m-%Y %H:%M')
        # if not pd.api.types.is_datetime64_any_dtype(data.index):
        #     data['time'] = pd.to_datetime(data['timestamp'])
        #     data.set_index('time', inplace=True)
        # Commented out for FIRCE compatibility; req for FIRE
        if not data.index.is_monotonic_increasing:
            data.sort_index(inplace=True)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        return data


def _aggregate_sessions(data: pd.DataFrame, is_unsw: bool) -> pd.DataFrame:
    """
    Perform session-based aggregation on the data.
    Branches based on dataset type:
    - UNSW: uses specific fields and includes session start/end times from cleaned data.
    - Default: aggregates standard features including subflow and FWD/BWD statistics.
    """
    if is_unsw:
        # Reset index to access start_time column
        data_reset = data.reset_index()
        session = (
            data_reset.groupby([
                'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'
            ]).agg(
                flow_duration=('flow_duration', 'sum'),
                total_forward_packets=('tot_fwd_pkts', 'sum'),
                total_backward_packets=('tot_bwd_pkts', 'sum'),
                total_bytes_forward=('totlen_fwd_pkts', 'sum'),
                total_bytes_backward=('totlen_bwd_pkts', 'sum'),
                mean_packet_length_forward=('fwd_pkt_len_mean', 'mean'),
                mean_packet_length_backward=('bwd_pkt_len_mean', 'mean'),
                packet_size_mean=('pkt_len_mean', 'mean'),
                flow_iat_mean=('flow_iat_mean', 'mean'),
                down_up_ratio=('down_up_ratio', 'mean'),
                fwd_iat_max=('fwd_iat_max', 'max'),
                fwd_iat_min=('fwd_iat_min', 'min'),
                fwd_iat_mean=('fwd_iat_mean', 'mean'),
                fwd_iat_tot=('fwd_iat_tot', 'sum'),
                bwd_iat_max=('bwd_iat_max', 'max'),
                bwd_iat_min=('bwd_iat_min', 'min'),
                bwd_iat_mean=('bwd_iat_mean', 'mean'),
                bwd_iat_tot=('bwd_iat_tot', 'sum'),
                start_time=('start_time', 'min'),
                end_time=('end_time', 'max')
            )
            .reset_index()
        )
        session['total_packets'] = (
            session['total_forward_packets'] +
            session['total_backward_packets']
        )
        session['total_bytes'] = (
            session['total_bytes_forward'] + session['total_bytes_backward']
        )
        return session
    else:
        session = (
            data.groupby([
                'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'
            ]).agg(
                flow_duration=('flow_duration', 'sum'),
                total_forward_packets=('tot_fwd_pkts', 'sum'),
                total_backward_packets=('tot_bwd_pkts', 'sum'),
                total_bytes_forward=('totlen_fwd_pkts', 'sum'),
                total_bytes_backward=('totlen_bwd_pkts', 'sum'),
                mean_packet_length_forward=('fwd_pkt_len_mean', 'mean'),
                mean_packet_length_backward=('bwd_pkt_len_mean', 'mean'),
                packet_size_mean=('pkt_len_mean', 'mean'),
                flow_iat_mean=('flow_iat_mean', 'mean'),
                down_up_ratio=('down_up_ratio', 'mean'),
                subflow_fwd_pkts=('subflow_fwd_pkts', 'sum'),
                subflow_bwd_pkts=('subflow_bwd_pkts', 'sum'),
                subflow_fwd_byts=('subflow_fwd_byts', 'sum'),
                subflow_bwd_byts=('subflow_bwd_byts', 'sum'),
                fwd_pkt_len_max=('fwd_pkt_len_max', 'max'),
                fwd_pkt_len_min=('fwd_pkt_len_min', 'min'),
                fwd_pkt_len_std=('fwd_pkt_len_std', 'std'),
                fwd_iat_max=('fwd_iat_max', 'max'),
                fwd_iat_min=('fwd_iat_min', 'min'),
                fwd_iat_tot=('fwd_iat_tot', 'sum'),
                fwd_blk_rate_avg=('fwd_blk_rate_avg', 'mean'),
                bwd_pkt_len_max=('bwd_pkt_len_max', 'max'),
                bwd_pkt_len_min=('bwd_pkt_len_min', 'min'),
                bwd_pkt_len_std=('bwd_pkt_len_std', 'std'),
                bwd_iat_max=('bwd_iat_max', 'max'),
                bwd_iat_min=('bwd_iat_min', 'min'),
                bwd_iat_tot=('bwd_iat_tot', 'sum'),
                bwd_blk_rate_avg=('bwd_blk_rate_avg', 'mean'),
                start_time=('timestamp', 'min'),
                end_time=('timestamp', 'max')
            )
        )
        session['total_packets'] = (
            session['total_forward_packets'] +
            session['total_backward_packets']
        )
        session['total_bytes'] = (
            session['total_bytes_forward'] + session['total_bytes_backward']
        )
        return session.reset_index()


def _entropy(column: pd.Series) -> float:
    """
    Calculate the entropy of a pandas Series.
    """
    counts = column.value_counts(normalize=True)
    return float(scipy.stats.entropy(counts))


def _sliding_window_aggregation(
    data: pd.DataFrame,
    window_size: pd.Timedelta,
    step_size: pd.Timedelta,
    is_unsw: bool
) -> pd.DataFrame:
    """
    Apply time-based sliding window aggregation on the data.

    For UNSW dataset, uses Dask for parallel computation and includes 5-tuple columns.
    For default dataset, simple pandas-based sliding aggregation.
    """
    if is_unsw:
        # Setup metadata schema
        meta = pd.DataFrame(columns=[
            'start_time', 'end_time',
            'total_forward_packets_window', 'total_backward_packets_window',
            'total_forward_bytes_window', 'total_backward_bytes_window',
            'average_packet_size_fwd_window', 'average_packet_size_bwd_window',
            'flow_duration_window', 'packet_count_window',
            'mean_iat_fwd_window', 'stddev_iat_fwd_window',
            'min_iat_fwd_window', 'max_iat_fwd_window',
            'mean_iat_bwd_window', 'stddev_iat_bwd_window',
            'min_iat_bwd_window', 'max_iat_bwd_window',
            'flow_rate_packets_window', 'flow_rate_bytes_window',
            'flow_direction_ratio_window', 'byte_direction_ratio_window',
            'src_ip_entropy_window', 'dst_ip_entropy_window',
            'protocol', 'src_ip', 'dst_ip', 'src_port', 'dst_port'
        ])
        meta = meta.astype({
            'start_time': 'datetime64[ns]', 'end_time': 'datetime64[ns]',
            'total_forward_packets_window': 'int64', 'total_backward_packets_window': 'int64',
            'total_forward_bytes_window': 'int64', 'total_backward_bytes_window': 'int64',
            'average_packet_size_fwd_window': 'float64', 'average_packet_size_bwd_window': 'float64',
            'flow_duration_window': 'int64', 'packet_count_window': 'int64',
            'mean_iat_fwd_window': 'float64', 'stddev_iat_fwd_window': 'float64',
            'min_iat_fwd_window': 'int64', 'max_iat_fwd_window': 'int64',
            'mean_iat_bwd_window': 'float64', 'stddev_iat_bwd_window': 'float64',
            'min_iat_bwd_window': 'int64', 'max_iat_bwd_window': 'int64',
            'flow_rate_packets_window': 'float64', 'flow_rate_bytes_window': 'float64',
            'flow_direction_ratio_window': 'float64', 'byte_direction_ratio_window': 'float64',
            'src_ip_entropy_window': 'float64', 'dst_ip_entropy_window': 'float64',
            'protocol': 'int64', 'src_ip': 'object', 'dst_ip': 'object',
            'src_port': 'int64', 'dst_port': 'int64'
        })

        def compute_agg(start_time):
            end_time = start_time + window_size
            window = data[(data.index >= start_time) & (data.index < end_time)]
            if window.empty:
                return pd.DataFrame(columns=meta.columns).astype(meta.dtypes.to_dict())
            # compute features
            duration = (window.index.max() - window.index.min()
                        ).total_seconds() + 1e-9
            agg = {
                'start_time': start_time,
                'end_time': end_time,
                'total_forward_packets_window': window['tot_fwd_pkts'].sum(),
                'total_backward_packets_window': window['tot_bwd_pkts'].sum(),
                'total_forward_bytes_window': window['totlen_fwd_pkts'].sum(),
                'total_backward_bytes_window': window['totlen_bwd_pkts'].sum(),
                'average_packet_size_fwd_window': window['fwd_pkt_len_mean'].mean(),
                'average_packet_size_bwd_window': window['bwd_pkt_len_mean'].mean(),
                'flow_duration_window': window['flow_duration'].sum(),
                'packet_count_window': len(window),
                'mean_iat_fwd_window': window['fwd_iat_mean'].mean(),
                'stddev_iat_fwd_window': window['fwd_iat_std'].mean(),
                'min_iat_fwd_window': window['fwd_iat_min'].min(),
                'max_iat_fwd_window': window['fwd_iat_max'].max(),
                'mean_iat_bwd_window': window['bwd_iat_mean'].mean(),
                'stddev_iat_bwd_window': window['bwd_iat_std'].mean(),
                'min_iat_bwd_window': window['bwd_iat_min'].min(),
                'max_iat_bwd_window': window['bwd_iat_max'].max(),
                'flow_rate_packets_window': len(window) / duration,
                'flow_rate_bytes_window': window['totlen_fwd_pkts'].sum() / duration,
                'flow_direction_ratio_window': window['tot_fwd_pkts'].sum() / (window['tot_bwd_pkts'].sum() + 1),
                'byte_direction_ratio_window': window['totlen_fwd_pkts'].sum() / (window['totlen_bwd_pkts'].sum() + 1),
                'src_ip_entropy_window': _entropy(window['src_ip']),
                'dst_ip_entropy_window': _entropy(window['dst_ip']),
                'protocol': window['protocol'].iloc[0],
                'src_ip': window['src_ip'].iloc[0],
                'dst_ip': window['dst_ip'].iloc[0],
                'src_port': window['src_port'].iloc[0],
                'dst_port': window['dst_port'].iloc[0]
            }
            return pd.DataFrame([agg])[meta.columns]

        start_times = pd.date_range(
            start=data.index.min(), end=data.index.max(), freq=step_size)
        delayed = [dask.delayed(compute_agg)(st) for st in tqdm(start_times, desc="Window Agg", file=sys.stderr)]  # type: ignore # noqa: F821
        ddf = dd.from_delayed(delayed, meta=meta)
        return ddf.compute()
    else:
        records = []
        times = pd.date_range(start=data.index.min(),
                              end=data.index.max(), freq=step_size)
        for st in times:
            et = st + window_size
            window = data[(data.index >= st) & (data.index < et)]
            if window.empty:
                continue
            window = window.copy()
            window['timestamp'] = pd.to_datetime(
                window['timestamp'], errors='coerce')
            dur = (window['timestamp'].max() -
                   window['timestamp'].min()).total_seconds() + 1e-9
            rec = {
                'start_time': st,
                'end_time': et,
                'total_forward_packets_window': window['tot_fwd_pkts'].sum(),
                'total_backward_packets_window': window['tot_bwd_pkts'].sum(),
                'total_forward_bytes_window': window['totlen_fwd_pkts'].sum(),
                'total_backward_bytes_window': window['totlen_bwd_pkts'].sum(),
                'average_packet_size_fwd_window': window['fwd_pkt_len_mean'].mean(),
                'average_packet_size_bwd_window': window['bwd_pkt_len_mean'].mean(),
                'flow_duration_window': window['flow_duration'].sum(),
                'packet_count_window': len(window),
                'mean_iat_fwd_window': window['fwd_iat_mean'].mean(),
                'stddev_iat_fwd_window': window['fwd_iat_std'].mean(),
                'min_iat_fwd_window': window['fwd_iat_min'].min(),
                'max_iat_fwd_window': window['fwd_iat_max'].max(),
                'mean_iat_bwd_window': window['bwd_iat_mean'].mean(),
                'stddev_iat_bwd_window': window['bwd_iat_std'].mean(),
                'min_iat_bwd_window': window['bwd_iat_min'].min(),
                'max_iat_bwd_window': window['bwd_iat_max'].max(),
                'flow_rate_packets_window': len(window) / dur,
                'flow_rate_bytes_window': window['totlen_fwd_pkts'].sum() / dur,
                'flow_direction_ratio_window': window['tot_fwd_pkts'].sum() / (window['tot_bwd_pkts'].sum() + 1),
                'byte_direction_ratio_window': window['totlen_fwd_pkts'].sum() / (window['totlen_bwd_pkts'].sum() + 1),
                'src_ip_entropy_window': _entropy(window['src_ip']),
                'dst_ip_entropy_window': _entropy(window['dst_ip'])
            }
            records.append(rec)
        return pd.DataFrame(records)


def _merge_aggregated_data(
    sliding_data: pd.DataFrame,
    session_data: pd.DataFrame,
    original_data: pd.DataFrame,
    is_unsw: bool
) -> pd.DataFrame:
    if is_unsw:
        print(f"[{time.strftime('%H:%M:%S')}] Starting UNSW merge...",
              file=sys.stderr)
        t0 = time.time()
        # Dask-based merge
        s_ddf = dd.from_pandas(sliding_data, npartitions=20)
        sess_ddf = dd.from_pandas(session_data, npartitions=20)
        merged_ddf = s_ddf.merge(
            sess_ddf, on=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], how='left')
        merged = merged_ddf.compute()
        print(
            f"Session merge done in {time.time() - t0:.2f}s", file=sys.stderr)
        orig = original_data.reset_index()
        subset = orig[['src_ip', 'dst_ip', 'src_port', 'dst_port',
                       'protocol', 'Label', 'Attack']].drop_duplicates()
        merged = merged.merge(
            subset, on=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], how='left')
        if 'start_time_x' in merged.columns:
            first = merged['start_time_x'].min()
            merged['timestamp_offset_seconds'] = (
                merged['start_time_x'] - first).dt.total_seconds()
        else:
            raise KeyError("Expected 'start_time_x' in merged data")
        print(
            f"DONE UNSW merge total time: {time.time() - t0:.2f}s", file=sys.stderr)
        return merged
    else:
        # Pandas-based asof merge
        merged = pd.merge_asof(
            sliding_data.sort_values('start_time'),
            session_data.sort_values('start_time'),
            left_on='start_time', right_on='start_time', direction='backward'
        )
        subset = original_data[['src_ip', 'dst_ip', 'src_port',
                                'dst_port', 'protocol', 'Label']].drop_duplicates()
        merged = merged.merge(
            subset, on=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], how='left')
        return merged


def _preprocess_pipeline(file_path: str, window_size_str: str = '5s', step_size_str: str = '1s', is_unsw: bool = False) -> pd.DataFrame:  # noqa: E501
    """
    End-to-end preprocessing pipeline:
      1. Load and clean the data.
      2. Compute session-based aggregation.
      3. Compute sliding window aggregation.
      4. Merge the aggregated results.
    """
    data = pd.read_csv(file_path)
    data = clean_data(data, is_unsw)
    if is_unsw:
        print("DONE: 1. Load and clean the UNSW data.", file=sys.stderr)
    else:
        print("DONE: 1. Load and clean the data.")
    session_data = _aggregate_sessions(data, is_unsw)
    if is_unsw:
        print("DONE: 2. Compute session-based aggregation (UNSW).", file=sys.stderr)
    else:
        print("DONE: 2. Compute session-based aggregation.")
    window_size = pd.Timedelta(window_size_str)
    step_size = pd.Timedelta(step_size_str)
    if is_unsw:
        print(
            f"Full data time range: {data.index.min()} to {data.index.max()}", file=sys.stderr)
        print(f"[{time.strftime('%H:%M:%S')}] Calling sliding_window_aggregation (UNSW)...", file=sys.stderr, flush=True)  # noqa: E501
    sliding_data = _sliding_window_aggregation(
        data, window_size, step_size, is_unsw)
    if is_unsw:
        print(f"[{time.strftime('%H:%M:%S')}] DONE: 3. Compute sliding window aggregation (UNSW).", file=sys.stderr, flush=True)  # noqa: E501
    else:
        print("DONE: 3. Compute sliding window aggregation.")
    aggregated_data = _merge_aggregated_data(
        sliding_data, session_data, data, is_unsw)
    if is_unsw:
        print("DONE: 4. Merge the aggregated results (UNSW).",
              file=sys.stderr, flush=True)
    else:
        print("DONE: 4. Merge the aggregated results.")
    return aggregated_data


def run_preprocessing(
    file_path: str,
    window_size_str: str = '5s',
    step_size_str: str = '1s',
    is_unsw: bool = False
) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline and save the output file to the dataset directory.
    The output file (aggregated_data.csv) is saved in the same folder as the input file.
    """
    aggregated_data = _preprocess_pipeline(
        file_path,
        window_size_str,
        step_size_str,
        is_unsw
    )
    output_dir = os.path.dirname(file_path)
    output_file = os.path.join(output_dir, "aggregated_data.csv")
    aggregated_data.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")
    return aggregated_data


if __name__ == '__main__':
    args = _parse_args()

    run_preprocessing(args.file_path, args.window_size,
                      args.step_size, args.unsw)
