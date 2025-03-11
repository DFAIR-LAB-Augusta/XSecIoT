import os
import numpy as np
import pandas as pd
import scipy.stats

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
      - Convert the FLOW_START_MILLISECONDS (renamed to 'timestamp') column from milliseconds to datetime.
      - Set the datetime column as the index.
      - Sort the index.
      - Replace infinite values with NaN.
    """
    # Mapping from UNSW dataset feature names to our internal names
    mapping = {
        'IPV4_SRC_ADDR': 'src_ip',
        'IPV4_DST_ADDR': 'dst_ip',
        'L4_SRC_PORT': 'src_port',
        'L4_DST_PORT': 'dst_port',
        'PROTOCOL': 'protocol',
        'FLOW_START_MILLISECONDS': 'timestamp',
        'FLOW_DURATION_MILLISECONDS': 'flow_duration',
        'IN_PKTS': 'tot_fwd_pkts',
        'OUT_PKTS': 'tot_bwd_pkts',
        'IN_BYTES': 'totlen_fwd_pkts',
        'OUT_BYTES': 'totlen_bwd_pkts',
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
    
    # Convert timestamp (in milliseconds) to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', errors='coerce')
    
    # Set timestamp as index and sort
    data.set_index('timestamp', inplace=True)
    data.sort_index(inplace=True)
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # (Verify w/ Bradl) Drop rows missing critical columns
    data.dropna(subset=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'flow_duration'], inplace=True)
    
    return data

def aggregate_sessions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform session-based aggregation on the new dataset.
    We group by five-tuple (src_ip, dst_ip, src_port, dst_port, protocol) and
    compute the sum of flow durations, total packets, and bytes.
    We also derive the average packet sizes and download/upload ratio.
    """
    # Reset index so that 'timestamp' becomes a column
    data_reset = data.reset_index()
    
    session_data = data_reset.groupby(
        ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
    ).agg(
        flow_duration=('flow_duration', 'sum'),
        total_forward_packets=('tot_fwd_pkts', 'sum'),
        total_backward_packets=('tot_bwd_pkts', 'sum'),
        total_bytes_forward=('totlen_fwd_pkts', 'sum'),
        total_bytes_backward=('totlen_bwd_pkts', 'sum'),
        fwd_iat_mean=('fwd_iat_mean', 'mean'),
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max')
    ).reset_index()
    
    # Derive additional session-level features.
    session_data['mean_packet_size_forward'] = session_data.apply(
        lambda row: row['total_bytes_forward'] / row['total_forward_packets']
        if row['total_forward_packets'] > 0 else np.nan,
        axis=1
    )
    session_data['mean_packet_size_backward'] = session_data.apply(
        lambda row: row['total_bytes_backward'] / row['total_backward_packets']
        if row['total_backward_packets'] > 0 else np.nan,
        axis=1
    )
    session_data['down_up_ratio'] = session_data['total_bytes_forward'] / (session_data['total_bytes_backward'] + 1)
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
    Apply time-based sliding window aggregation on the data.
    For the new dataset we aggregate using our renamed columns.
    """
    window_aggregates = []
    start_times = pd.date_range(start=data.index.min(), end=data.index.max(), freq=step_size)
    
    for start_time in start_times:
        end_time = start_time + window_size
        window = data[(data.index >= start_time) & (data.index < end_time)]
        if window.empty:
            continue
        
        # Make a copy and restore the 'timestamp' column from the index.
        window = window.copy()
        window['timestamp'] = window.index
        
        # Duration of the window in seconds.
        duration = (window['timestamp'].max() - window['timestamp'].min()).total_seconds() + 1e-9
        
        # Compute flow rate features
        flow_rate_features = {
            'flow_rate_packets_window': len(window) / duration,
            'flow_rate_bytes_window': window['totlen_fwd_pkts'].sum() / duration,
        }
        
        # Compute directional features using forward and backward packets/bytes.
        directional_features = {
            'flow_direction_ratio_window': window['tot_fwd_pkts'].sum() / (window['tot_bwd_pkts'].sum() + 1),
            'byte_direction_ratio_window': window['totlen_fwd_pkts'].sum() / (window['totlen_bwd_pkts'].sum() + 1),
        }
        
        # Entropy features of IP addresses in the window.
        entropy_features = {
            'src_ip_entropy_window': entropy(window['src_ip']),
            'dst_ip_entropy_window': entropy(window['dst_ip']),
        }
        
        tot_fwd_pkts = window['tot_fwd_pkts'].sum()
        tot_bwd_pkts = window['tot_bwd_pkts'].sum()
        tot_fwd_bytes = window['totlen_fwd_pkts'].sum()
        tot_bwd_bytes = window['totlen_bwd_pkts'].sum()
        
        aggregated = {
            'start_time': start_time,
            'end_time': end_time,
            'total_forward_packets_window': tot_fwd_pkts,
            'total_backward_packets_window': tot_bwd_pkts,
            'total_forward_bytes_window': tot_fwd_bytes,
            'total_backward_bytes_window': tot_bwd_bytes,
            'average_packet_size_fwd_window': tot_fwd_bytes / tot_fwd_pkts if tot_fwd_pkts > 0 else np.nan,
            'average_packet_size_bwd_window': tot_bwd_bytes / tot_bwd_pkts if tot_bwd_pkts > 0 else np.nan,
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
    
    # Assume the original data has a 'Label' column.
    original_reset = original_data.reset_index()
    original_subset = original_reset[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'Label']].drop_duplicates()
    aggregated_data = aggregated_data.merge(original_subset, on=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], how='left')
    
    # Create a new timestamp column based on the offset (in seconds) from the earliest sliding window start time.
    first_start = aggregated_data['start_time'].min()
    aggregated_data['timestamp'] = (aggregated_data['start_time'] - first_start).dt.total_seconds()
    
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
    session_data = aggregate_sessions(data)
    window_size = pd.Timedelta(window_size_str)
    step_size = pd.Timedelta(step_size_str)
    sliding_data = sliding_window_aggregation(data, window_size, step_size)
    aggregated_data = merge_aggregated_data(sliding_data, session_data, data)
    
    return aggregated_data

def run_preprocessing(file_path: str, window_size_str: str = '5s', step_size_str: str = '1s') -> pd.DataFrame:
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
    # For testing purposes, this module can be run directly.
    import argparse
    parser = argparse.ArgumentParser(description="Preprocessing pipeline for UNSW dataset using FIRE_codebase framework")
    parser.add_argument("file_path", type=str, help="Path to the UNSW dataset CSV file")
    parser.add_argument("--window_size", type=str, default="5s", help="Window size (e.g., '5s')")
    parser.add_argument("--step_size", type=str, default="1s", help="Step size (e.g., '1s')")
    args = parser.parse_args()
    
    run_preprocessing(args.file_path, args.window_size, args.step_size)