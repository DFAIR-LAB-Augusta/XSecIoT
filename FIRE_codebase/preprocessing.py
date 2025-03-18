import os
import numpy as np
import pandas as pd
import scipy.stats

np.random.seed(42)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the CSV file into a DataFrame.
    """
    data = pd.read_csv(file_path)
    return data

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the data:
      - Rename columns and drop unwanted columns.
      - Convert the timestamp to datetime.
      - Set the datetime column as the index.
      - Sort the index.
      - Replace infinite values with NaN.
    """
    data.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    data.drop(columns=["id"], inplace=True, errors='ignore')
    
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce', format='%d-%m-%Y %H:%M')
    
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data['time'] = pd.to_datetime(data['timestamp'])
        data.set_index('time', inplace=True)
    
    if not data.index.is_monotonic_increasing:
        data.sort_index(inplace=True)
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return data

def aggregate_sessions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform session-based aggregation on the data.
    """
    session_data = data.groupby(
        ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
    ).agg(
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
        fwd_pkt_len_mean=('fwd_pkt_len_mean', 'mean'),
        fwd_pkt_len_max=('fwd_pkt_len_max', 'max'),
        fwd_pkt_len_min=('fwd_pkt_len_min', 'min'),
        fwd_pkt_len_std=('fwd_pkt_len_std', 'std'),
        fwd_iat_mean=('fwd_iat_mean', 'mean'),
        fwd_iat_max=('fwd_iat_max', 'max'),
        fwd_iat_min=('fwd_iat_min', 'min'),
        fwd_iat_tot=('fwd_iat_tot', 'sum'),
        fwd_blk_rate_avg=('fwd_blk_rate_avg', 'mean'),
        bwd_pkt_len_mean=('bwd_pkt_len_mean', 'mean'),
        bwd_pkt_len_max=('bwd_pkt_len_max', 'max'),
        bwd_pkt_len_min=('bwd_pkt_len_min', 'min'),
        bwd_pkt_len_std=('bwd_pkt_len_std', 'std'),
        bwd_iat_mean=('bwd_iat_mean', 'mean'),
        bwd_iat_max=('bwd_iat_max', 'max'),
        bwd_iat_min=('bwd_iat_min', 'min'),
        bwd_iat_tot=('bwd_iat_tot', 'sum'),
        bwd_blk_rate_avg=('bwd_blk_rate_avg', 'mean'),
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max')
    )
    
    session_data['total_packets'] = session_data['total_forward_packets'] + session_data['total_backward_packets']
    session_data['total_bytes'] = session_data['total_bytes_forward'] + session_data['total_bytes_backward']
    
    return session_data.reset_index()

def entropy(column: pd.Series) -> float:
    """
    Calculate the entropy of a pandas Series.
    """
    counts = column.value_counts(normalize=True)
    return scipy.stats.entropy(counts)

def sliding_window_aggregation(data: pd.DataFrame, window_size: pd.Timedelta, step_size: pd.Timedelta) -> pd.DataFrame:
    """
    Apply time-based sliding window aggregation on the data.
    """
    window_aggregates = []
    start_times = pd.date_range(start=data.index.min(), end=data.index.max(), freq=step_size)
    
    for start_time in start_times:
        end_time = start_time + window_size
        window = data[(data.index >= start_time) & (data.index < end_time)]
        if window.empty:
            continue
        
        window = window.copy()
        window.loc[:, 'timestamp'] = pd.to_datetime(window['timestamp'])

        duration = (window['timestamp'].max() - window['timestamp'].min()).total_seconds() + 1e-9
        
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
    
    original_subset = original_data[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'Label']].drop_duplicates()
    aggregated_data = aggregated_data.merge(original_subset, on=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], how='left')
    
    return aggregated_data

def preprocess_pipeline(file_path: str, window_size_str: str = '5s', step_size_str: str = '1s') -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline:
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
    Run the full preprocessing pipeline and save the output file to the dataset directory.
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
    parser = argparse.ArgumentParser(description="Preprocessing pipeline for FIRE_codebase")
    parser.add_argument("file_path", type=str, help="Path to the dataset CSV file")
    parser.add_argument("--window_size", type=str, default="5s", help="Window size (e.g., '5s')")
    parser.add_argument("--step_size", type=str, default="1s", help="Step size (e.g., '1s')")
    args = parser.parse_args()
    
    run_preprocessing(args.file_path, args.window_size, args.step_size)
