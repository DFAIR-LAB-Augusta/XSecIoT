import os
import time
import argparse
import multiprocessing as mp
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb  # type: ignore
from functools import partial

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ------------------------------
# Helper Functions
# ------------------------------

def load_data(file_path):
    """Load the aggregated CSV data."""
    return pd.read_csv(file_path)

def preprocess_chunk(chunk, drop_cols):
    """
    Drop non-feature columns and fill missing values with column means.
    """
    X = chunk.drop(columns=drop_cols, errors='ignore')
    if X.isna().any().any():
        X = X.fillna(X.mean())
    return X

def get_dataset_name(file_path):
    """
    Extract the dataset name from the aggregated file path.
    For example, if file_path is "./datasets/DFAIR/aggregated_data.csv", returns "DFAIR".
    """
    return os.path.basename(os.path.dirname(file_path))

def load_simulation_objects(aggregated_file, model_type, model_variant):
    """
    Load the scaler, PCA, and model objects based on the dataset name, model type, and model variant.
    """
    dataset_name = get_dataset_name(aggregated_file)
    if model_type == 'binary':
        base_dir = os.path.join(os.getcwd(), "binary_models", dataset_name)
        scaler_file = os.path.join(base_dir, "scaler_binary.pkl")
        pca_file = os.path.join(base_dir, "pca_binary.pkl")
        model_file = os.path.join(base_dir, f"{model_variant}_model_binary.pkl")
    else:
        base_dir = os.path.join(os.getcwd(), "multi_class_models", dataset_name)
        scaler_file = os.path.join(base_dir, "scaler_multi.pkl")
        pca_file = os.path.join(base_dir, "pca_multi.pkl")
        mapping = {
            'dt': 'decision_tree_multi.pkl',
            'rf': 'random_forest_multi.pkl',
            'feedforward': 'feedforward_multi.pkl',
            'knn': 'knearest_multi.pkl',
            'svm': 'svm_multi.pkl',
            'xgb': 'xgboost_multi.pkl'
        }
        if model_variant not in mapping:
            raise ValueError(f"Model variant '{model_variant}' is not supported for multi-class models.")
        model_file = os.path.join(base_dir, mapping[model_variant])
    scaler = joblib.load(scaler_file)
    pca = joblib.load(pca_file)
    model = joblib.load(model_file)
    return scaler, pca, model

# Separated out due to pickling not liking nested functions using mp
def _process_chunk(chunk, drop_cols, scaler, pca, model, model_variant, model_type, threshold):
    """
    Process a single chunk: preprocess, scale, transform with PCA, and predict.
    """
    X_chunk = preprocess_chunk(chunk, drop_cols)
    # Select only numeric columns to ensure feature names match
    X_chunk = X_chunk.select_dtypes(include=[np.number])
    X_scaled = scaler.transform(X_chunk)
    X_pca = pca.transform(X_scaled)
    
    if model_variant.startswith('xgb'):
        # If the model is a scikit-learn wrapper, predict directly.
        if isinstance(model, xgb.XGBClassifier):
            preds = model.predict(X_pca)
        else:
            # Assume it's a Booster; get training feature names if available.
            trained_feature_names = (model.feature_names 
                                     if hasattr(model, 'feature_names') and model.feature_names is not None 
                                     else [f"f_{i}" for i in range(pca.n_components_)])
            dtest = xgb.DMatrix(X_pca, feature_names=trained_feature_names)
            preds = model.predict(dtest)
    else:
        preds = model.predict(X_pca)
        if model_variant == 'feedforward':
            preds = (preds > threshold).astype(int)
        if model_type == 'binary':
            preds = ['Attack' if int(p) == 1 else 'Benign' for p in preds]
    return preds

# ------------------------------
# Simulation Functions
# ------------------------------

def sequential_simulation(aggregated_file, model_type='binary', model_variant='dt',
                          chunk_size=1000, delay=1, threshold=0.5):
    """
    Sequential simulation that processes the aggregated data in chunks.
    """
    drop_cols = ['Label', 'BinLabel', 'src_ip', 'dst_ip', 'start_time',
                 'end_time_x', 'end_time_y', 'time_diff', 'time_diff_seconds', 'Attack']
    scaler, pca, model = load_simulation_objects(aggregated_file, model_type, model_variant)
    total_start_time = time.time()
    
    for chunk in pd.read_csv(aggregated_file, chunksize=chunk_size):
        print("\nProcessing new data chunk...")
        start_time_chunk = time.time()
        X_chunk = preprocess_chunk(chunk, drop_cols)
        # Select only numeric columns to ensure feature names match
        X_chunk = X_chunk.select_dtypes(include=[np.number])
        X_scaled = scaler.transform(X_chunk)
        X_pca = pca.transform(X_scaled)
        
        if model_variant.startswith('xgb'):
            if isinstance(model, xgb.XGBClassifier):
                predictions = model.predict(X_pca)
            else:
                trained_feature_names = (model.feature_names 
                                         if hasattr(model, 'feature_names') and model.feature_names is not None 
                                         else [f"f_{i}" for i in range(pca.n_components_)])
                dtest = xgb.DMatrix(X_pca, feature_names=trained_feature_names)
                predictions = model.predict(dtest)
        else:
            predictions = model.predict(X_pca)
            if model_variant == 'feedforward':
                predictions = (predictions > threshold).astype(int)
        
        for i, pred in enumerate(predictions):
            if model_type == 'binary':
                label = 'Attack' if int(pred) == 1 else 'Benign'
            else:
                label = pred
            print(f"Data point {i+1}: {label}")
            
        latency = time.time() - start_time_chunk
        print(f"Latency for this chunk: {latency:.4f} seconds")
        time.sleep(delay)
    
    total_time = time.time() - total_start_time
    print(f"Total processing time (sequential): {total_time:.4f} seconds")

def continuous_simulation(aggregated_file, model_type='binary', model_variant='dt',
                          chunk_size=1000, window_duration=300, delay=1, threshold=0.5):
    """
    Continuous simulation with a sliding window.
    """
    drop_cols = ['Label', 'BinLabel', 'src_ip', 'dst_ip', 'start_time',
                 'end_time_x', 'end_time_y', 'time_diff', 'time_diff_seconds', 'Attack']
    scaler, pca, model = load_simulation_objects(aggregated_file, model_type, model_variant)
    sliding_window = pd.DataFrame()
    all_true_labels = []
    all_predictions = []
    latencies = []
    total_start_time = time.time()
    
    for chunk in pd.read_csv(aggregated_file, chunksize=chunk_size):
        print("\nProcessing new data chunk...")
        start_time_chunk = time.time()
        
        sliding_window = pd.concat([sliding_window, chunk])
        sliding_window['end_time_x'] = pd.to_datetime(sliding_window['end_time_x'], errors='coerce')
        latest_time = sliding_window['end_time_x'].max()
        window_start_time = latest_time - pd.Timedelta(seconds=window_duration)
        sliding_window = sliding_window[sliding_window['end_time_x'] >= window_start_time]
        
        if model_type == 'binary':
            if 'BinLabel' not in sliding_window.columns:
                if 'Label' in sliding_window.columns:
                    sliding_window['BinLabel'] = sliding_window['Label'].apply(lambda x: 0 if x=='Benign' else 1)
                else:
                    raise KeyError("Neither 'BinLabel' nor 'Label' found in the data.")
            else:
                if sliding_window['BinLabel'].isna().any():
                    if 'Label' in sliding_window.columns:
                        sliding_window.loc[sliding_window['BinLabel'].isna(), 'BinLabel'] = \
                            sliding_window.loc[sliding_window['BinLabel'].isna(), 'Label'].apply(lambda x: 0 if x=='Benign' else 1)
        
        print("Sliding window data (first few rows):")
        print(sliding_window.head())
        
        X_chunk = sliding_window.drop(columns=drop_cols, errors='ignore')
        # Select only numeric columns to ensure feature names match
        X_chunk = X_chunk.select_dtypes(include=[np.number])
        
        if X_chunk.isna().any().any():
            print("Warning: Missing values detected in numeric features. Filling with column means.")
            X_chunk = X_chunk.fillna(X_chunk.mean())
        
        if X_chunk.isna().any().any():
            print("Warning: Still missing values after fillna. Dropping rows with missing values.")
            X_chunk = X_chunk.dropna()
        
        if X_chunk.empty:
            print("Warning: No numeric features remain after dropping columns. Skipping this chunk.")
            continue
        
        if model_type == 'binary':
            y_true = sliding_window['BinLabel']
        else:
            y_true = sliding_window['Label']
        
        X_scaled = scaler.transform(X_chunk)
        X_pca = pca.transform(X_scaled)
        
        if model_variant.startswith('xgb'):
            if isinstance(model, xgb.XGBClassifier):
                predictions = model.predict(X_pca)
            else:
                trained_feature_names = (model.feature_names 
                                         if hasattr(model, 'feature_names') and model.feature_names is not None 
                                         else [f"f_{i}" for i in range(pca.n_components_)])
                dtest = xgb.DMatrix(X_pca, feature_names=trained_feature_names)
                predictions = model.predict(dtest)
        else:
            predictions = model.predict(X_pca)
            if model_variant == 'feedforward':
                predictions = (predictions > threshold).astype(int)
        
        latency = time.time() - start_time_chunk
        latencies.append(latency)
        print(f"Processing latency for this chunk: {latency:.4f} seconds")
        
        all_true_labels.extend(y_true.tolist())
        all_predictions.extend(predictions.tolist())
        
        for i, pred in enumerate(predictions):
            if model_type == 'binary':
                label = 'Attack' if int(pred)==1 else 'Benign'
            else:
                label = pred
            print(f"Data point {i+1}: {label}")
        
        time.sleep(delay)
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    total_time = time.time() - total_start_time
    print(f"Average processing latency (continuous): {avg_latency:.4f} seconds")
    print(f"Total processing time (continuous): {total_time:.4f} seconds")
    
    return all_true_labels, all_predictions

def parallel_simulation(aggregated_file, model_type='binary', model_variant='dt',
                        chunk_size=1000, num_processes=4, threshold=0.5):
    """
    Parallel simulation that splits the data into chunks and processes them concurrently.
    """
    drop_cols = ['Label', 'BinLabel', 'src_ip', 'dst_ip', 'start_time',
                 'end_time_x', 'end_time_y', 'time_diff', 'time_diff_seconds', 'Attack']
    scaler, pca, model = load_simulation_objects(aggregated_file, model_type, model_variant)
    data = pd.read_csv(aggregated_file)
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    print(f"Processing {len(chunks)} chunks in parallel...")
    
    process_chunk_partial = partial(_process_chunk, 
                                    drop_cols=drop_cols, 
                                    scaler=scaler, 
                                    pca=pca, 
                                    model=model, 
                                    model_variant=model_variant, 
                                    model_type=model_type, 
                                    threshold=threshold)
    
    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk_partial, chunks)
    total_time = time.time() - start_time
    print(f"Total processing time with parallelization: {total_time:.4f} seconds")
    
    all_predictions = [pred for chunk_preds in results for pred in chunk_preds]
    return all_predictions

# ------------------------------
# Command-Line Interface (for testing)
# ------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run simulation pipeline for FIRE process (Step 3)")
    parser.add_argument("aggregated_file", type=str, help="Path to aggregated_data.csv")
    parser.add_argument("--mode", type=str, default="sequential", choices=["sequential", "continuous", "parallel"],
                        help="Simulation mode: sequential, continuous, or parallel (default: sequential)")
    parser.add_argument("--model_type", type=str, default="binary", choices=["binary", "multi"],
                        help="Model type: binary or multi (default: binary)")
    parser.add_argument("--model_variant", type=str, default="dt",
                        help=("Model variant. For example: 'dt' for decision tree, 'knn' for K-Nearest Neighbors, "
                              "'rf' for Random Forest, 'svm' for SVM, 'feedforward' for neural network, "
                              "'xgb' for XGBoost. (default: dt)"))
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of data points per chunk (default: 1000)")
    parser.add_argument("--delay", type=float, default=1, help="Delay (in seconds) between chunks (default: 1)")
    parser.add_argument("--window_duration", type=int, default=300, help="Window duration in seconds (for continuous mode, default: 300)")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes for parallel mode (default: 4)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for converting probabilities (if applicable, default: 0.5)")
    args = parser.parse_args()

    if args.mode == "sequential":
        sequential_simulation(args.aggregated_file,
                              model_type=args.model_type,
                              model_variant=args.model_variant,
                              chunk_size=args.chunk_size,
                              delay=args.delay,
                              threshold=args.threshold)
    elif args.mode == "continuous":
        continuous_simulation(args.aggregated_file,
                              model_type=args.model_type,
                              model_variant=args.model_variant,
                              chunk_size=args.chunk_size,
                              window_duration=args.window_duration,
                              delay=args.delay,
                              threshold=args.threshold)
    elif args.mode == "parallel":
        preds = parallel_simulation(args.aggregated_file,
                                    model_type=args.model_type,
                                    model_variant=args.model_variant,
                                    chunk_size=args.chunk_size,
                                    num_processes=args.num_processes,
                                    threshold=args.threshold)
        print("Parallel simulation predictions:")
        print(preds)
