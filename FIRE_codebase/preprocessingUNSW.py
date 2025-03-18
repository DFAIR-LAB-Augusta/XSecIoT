import os
import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm

np.random.seed(42)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the CSV file into a DataFrame.
    """
    data = pd.read_csv(file_path)
    return data

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the new UNSW dataset:
      - Rename columns to match our downstream expectations.
      - Convert FLOW_START_MILLISECONDS and FLOW_END_MILLISECONDS to datetime (with millisecond precision).
      - Convert flow_duration from milliseconds to microseconds.
      - Set the start_time as the index and sort.
      - Replace infinite values with NaN.
    """
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
    
    data['start_time'] = pd.to_datetime(data['start_time'], unit='ms', errors='coerce')
    data['end_time'] = pd.to_datetime(data['end_time'], unit='ms', errors='coerce')
    
    data['flow_duration'] = data['flow_duration'] * 1000

    data.set_index('start_time', inplace=True)
    data.sort_index(inplace=True)

    data['fwd_pkt_len_mean'] = np.where(
        data['tot_fwd_pkts'] != 0,
        data['totlen_fwd_pkts'] / data['tot_fwd_pkts'],
        0
    )

    data['bwd_pkt_len_mean'] = np.where(
        data['tot_bwd_pkts'] != 0,
        data['totlen_bwd_pkts'] / data['tot_bwd_pkts'],
        0
    )

    data['pkt_len_mean'] = np.where(
        (data['tot_fwd_pkts'] + data['tot_bwd_pkts']) != 0,
        (data['totlen_fwd_pkts'] + data['totlen_bwd_pkts']) / (data['tot_fwd_pkts'] + data['tot_bwd_pkts']),
        0
    )

    data['flow_iat_mean'] = np.where(
        (data['tot_fwd_pkts'] + data['tot_bwd_pkts'] - 1) > 0,
        data['flow_duration'] / (data['tot_fwd_pkts'] + data['tot_bwd_pkts'] - 1),
        0
    )

    data['down_up_ratio'] = np.where(
        data['tot_fwd_pkts'] > 0,
        data['tot_bwd_pkts'] / data['tot_fwd_pkts'],
        0
    )

    data['fwd_iat_tot'] = data['fwd_iat_mean'] * (data['tot_fwd_pkts'] - 1)
    data['bwd_iat_tot'] = data['bwd_iat_mean'] * (data['tot_bwd_pkts'] - 1)


    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    data.dropna(subset=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'flow_duration'], inplace=True)
    
    return data

def aggregate_sessions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform session-based aggregation on the new dataset.
    We group by the five-tuple (src_ip, dst_ip, src_port, dst_port, protocol) and compute:
      - Sum of flow durations, total packets, and bytes.
      - Mean packet sizes and download/upload ratio.
      - Start and end times for the session.
    """
    data_reset = data.reset_index()
    
    session_data = data_reset.groupby(
        ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
    ).agg(
        flow_duration=('flow_duration', 'sum'),
        total_forward_packets=('tot_fwd_pkts', 'sum'),
        total_backward_packets=('tot_bwd_pkts', 'sum'),
        total_bytes_forward=('totlen_fwd_pkts', 'sum'),
        total_bytes_backward=('totlen_bwd_pkts', 'sum'),
        mean_packet_length_forward=('fwd_pkt_len_mean', 'mean'),
        fwd_pkt_len_mean=('fwd_pkt_len_mean', 'mean'),
        flow_iat_mean=('flow_iat_mean', 'mean'),
        mean_packet_length_backward=('bwd_pkt_len_mean', 'mean'),
        packet_size_mean=('pkt_len_mean', 'mean'),
        down_up_ratio=('down_up_ratio', 'mean'),
        fwd_iat_max=('fwd_iat_max', 'max'),
        fwd_iat_min=('fwd_iat_min', 'min'),
        fwd_iat_mean=('fwd_iat_mean', 'mean'),
        fwd_iat_tot=('fwd_iat_tot', 'sum'),
        bwd_pkt_len_mean=('bwd_pkt_len_mean', 'mean'),
        bwd_iat_mean=('bwd_iat_mean', 'mean'),
        bwd_iat_max=('bwd_iat_max', 'max'),
        bwd_iat_min=('bwd_iat_min', 'min'),
        bwd_iat_tot=('bwd_iat_tot', 'sum'),
        start_time=('start_time', 'min'),
        end_time=('end_time', 'max')
    ).reset_index()

    session_data['total_packets'] = session_data['total_forward_packets'] + session_data['total_backward_packets']
    session_data['total_bytes'] = session_data['total_bytes_forward'] + session_data['total_bytes_backward']
    
    return session_data

def entropy(column: pd.Series) -> float:
    """
    Calculate the entropy of a pandas Series.
    """
    counts = column.value_counts(normalize=True)
    return scipy.stats.entropy(counts)

def sliding_window_aggregation(data: pd.DataFrame, window_size: pd.Timedelta, step_size: pd.Timedelta) -> pd.DataFrame:
    """
    Apply time-based sliding window aggregation on the UNSW dataset.
    Aggregates are computed using the preprocessed column names.
    """
    window_aggregates = []
    start_times = pd.date_range(start=data.index.min(), end=data.index.max(), freq=step_size)
    
    for start_time in tqdm(start_times, desc="Sliding Window Aggregation"):
        end_time = start_time + window_size
        window = data[(data.index >= start_time) & (data.index < end_time)]
        if window.empty:
            continue
        
        duration = (window.index.max() - window.index.min()).total_seconds() + 1e-9
        
        flow_rate_features = {
            'flow_rate_packets_window': len(window) / duration,
            'flow_rate_bytes_window': window['totlen_fwd_pkts'].sum() / duration,
        }
        
        directional_features = {
            'flow_direction_ratio_window': window['tot_fwd_pkts'].sum() / (window['tot_bwd_pkts'].sum() + 1),
            'byte_direction_ratio_window': window['totlen_fwd_pkts'].sum() / (window['totlen_bwd_pkts'].sum() + 1),
        }
        
        entropy_features = {
            'src_ip_entropy_window': entropy(window['src_ip']),
            'dst_ip_entropy_window': entropy(window['dst_ip']),
        }
        
        aggregated = {
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
        }
        aggregated.update(flow_rate_features)
        aggregated.update(directional_features)
        aggregated.update(entropy_features)
        
        window_aggregates.append(aggregated)
    
    return pd.DataFrame(window_aggregates)


def merge_aggregated_data(sliding_data: pd.DataFrame, session_data: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the sliding window aggregation and session data.
    Also merge with a subset of the original data to include additional columns (e.g., Label).
    """
    aggregated_data = pd.merge_asof(
        sliding_data.sort_values('start_time'),
        session_data.sort_values('start_time'),
        left_on='start_time',
        right_on='start_time',
        direction='backward'
    )
    
    original_reset = original_data.reset_index()
    original_subset = original_reset[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'Label', 'Attack']].drop_duplicates()
    aggregated_data = aggregated_data.merge(original_subset, on=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], how='left')
    
    first_start = aggregated_data['start_time'].min()
    aggregated_data['timestamp_offset_seconds'] = (aggregated_data['start_time'] - first_start).dt.total_seconds()
    
    return aggregated_data

def preprocess_pipeline(file_path: str, window_size_str: str = '5s', step_size_str: str = '1s') -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline for the UNSW dataset:
      1. Load and clean the data.
      2. Compute session-based aggregation.
      3. Compute sliding window aggregation.
      4. Merge the aggregated results.
    """
    data = load_data(file_path)
    data = clean_data(data)
    print("DONE: 1. Load and clean the data.")
    session_data = aggregate_sessions(data)
    print("DONE: 2. Compute session-based aggregation.")
    print(f"Full data time range: {data.index.min()} to {data.index.max()}")
    window_size = pd.Timedelta(window_size_str)
    step_size = pd.Timedelta(step_size_str)
    sliding_data = sliding_window_aggregation(data, window_size, step_size)
    print("DONE: 3. Compute sliding window aggregation.")
    aggregated_data = merge_aggregated_data(sliding_data, session_data, data)
    print("DONE: 4. Merge the aggregated results.")

    return aggregated_data

def run_preprocessingUNSW(file_path: str, window_size_str: str = '5s', step_size_str: str = '1s') -> pd.DataFrame:
    """
    Run the full preprocessing pipeline for the UNSW dataset and save the output file to the dataset directory.
    The output file (aggregated_data.csv) is saved in the same folder as the input file.
    """
    aggregated_data = preprocess_pipeline(file_path, window_size_str, step_size_str)
    
    output_dir = os.path.dirname(file_path)
    output_file = os.path.join(output_dir, "aggregated_data.csv")
    aggregated_data.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")
    
    return aggregated_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Preprocessing pipeline for UNSW dataset using FIRE_codebase framework")
    parser.add_argument("file_path", type=str, help="Path to the UNSW dataset CSV file")
    parser.add_argument("--window_size", type=str, default="5s", help="Window size (e.g., '5s')")
    parser.add_argument("--step_size", type=str, default="1s", help="Step size (e.g., '1s')")
    args = parser.parse_args()
    
    run_preprocessingUNSW(args.file_path, args.window_size, args.step_size)