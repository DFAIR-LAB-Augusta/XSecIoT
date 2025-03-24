# src.core.streaming_pipeline

"""
TODO: FIX lol
"""

import argparse
import logging

from typing import Any, List, cast

import pandas as pd
import xgboost as xgb

from listener import run_server
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.core.conformalEval.conformal_evaluators import ConformalEvaluator
from src.core.perf_stats import PerformanceStats
from src.core.rolling_csv import RollingCSV
from src.FIRE.preprocessing import clean_data
from src.FIRE.simulations import load_simulation_objects, preprocess_chunk, process_chunk

logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_TYPE = 'binary'
MODEL_VARIANT = 'feedforward'
CE_TYPE = 'cce'
CE_KWARGS = {'calibration_split': 0.2, 'random_state': 42}
THRESHOLD = 0.5
IS_UNSW = False

# Path for rolling log of labeled streaming data
LOG_PATH = 'streamed_labeled_data.csv.gz'

# Global holders
initialized: bool = False
scaler: StandardScaler | None = None 
pca: PCA | None = None
mode: ClassifierMixin | xgb.Booster | None = None 
ce: ConformalEvaluator | None = None 
rolling_csv: RollingCSV | None = None 

DROP_COLS: List[str] = ['Label', 'BinLabel', 'src_ip', 'dst_ip', 'start_time',
             'end_time_x', 'end_time_y', 'time_diff', 'time_diff_seconds', 'Attack']


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming pipeline for real-time data processing and model inference"
    )
    parser.add_argument(
        "--log", type=str, default="5s",
        help="Enable logging for testing purposes"
    )

    return parser.parse_args()


def _initialize_models():
    """Load scaler, PCA, and trained model"""
    global scaler, pca, model, ce, rolling_csv
    scaler, pca, model = load_simulation_objects(
        aggregated_file='',  # not used for loading path resolution
        model_type=MODEL_TYPE,
        model_variant=MODEL_VARIANT
    )
    ce = ConformalEvaluator(CE_TYPE, model, **CE_KWARGS)
    RollingCSV(LOG_PATH, max_rows=10000)


def _retrain_on_logged_data():
    """Retrain scaler, PCA, model, and recalibrate CE using the rolling log."""
    global scaler, pca, model, ce

    # Load the last max_rows entries with true labels
    df_log = pd.read_csv(LOG_PATH, compression='gzip')
    # The last two columns are our 'pred' and 'drift_flag'
    feature_cols = df_log.columns[:-2]
    data = df_log[feature_cols]

    # Clean and prepare features
    clean = clean_data(data, IS_UNSW)
    X_df = preprocess_chunk(clean, DROP_COLS).select_dtypes(include=['number'])
    y = clean['BinLabel'] if MODEL_TYPE == 'binary' else clean['Label']

    # Refit scaler and PCA
    scaler = StandardScaler().fit(X_df)
    X_scaled = scaler.transform(X_df)
    pca = PCA(n_components=0.95).fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    # Retrain model
    assert model is not None

    if hasattr(model, "fit"):
        cast('Any', model).fit(X_pca, y)
    else:
        # It’s an xgb.Booster — no .fit(), so retraining would require rebuilding via xgb.train. Skip or log a warning.
        logging.warning("Cannot call .fit() on xgb.Booster; skipping retrain for Booster models.")

    # Recreate and recalibrate CE
    ce = ConformalEvaluator(CE_TYPE, model, **CE_KWARGS)
    ce.calibrate(X_pca, y.to_numpy(), PerformanceStats())

    logging.info('Model retrained and CE recalibrated on logged data')


def _handle_batch(df: pd.DataFrame):
    """Callback for each incoming CSV batch"""
    global initialized
    # 1: Clean data
    clean = clean_data(df, IS_UNSW)

    # 2: On first batch, initialize and calibrate CE
    if not initialized:
        _initialize_models()
        assert scaler is not None
        assert pca is not None
        assert ce is not None
        # prepare features & labels for calibration
        X_df = preprocess_chunk(clean, DROP_COLS).select_dtypes(include=['number'])
        X_arr = scaler.transform(X_df) if hasattr(scaler, 'transform') else X_df.values
        X_pca = pca.transform(X_arr) if hasattr(pca, 'transform') else X_arr
        y = clean['BinLabel'] if MODEL_TYPE == 'binary' else clean['Label']
        ce.calibrate(X_pca, y.to_numpy(), PerformanceStats())
        initialized = True
        logging.info('CE calibrated on first batch')
        return
    
    # 3: Process batch: inference
    assert scaler is not None
    assert pca is not None
    assert ce is not None
    assert model is not None
    preds = process_chunk(
        clean, DROP_COLS,
        scaler, pca, model,
        MODEL_VARIANT, MODEL_TYPE, THRESHOLD
    )
    logging.info('Predictions:', preds)

    # 4: Drift detection
    X_df = preprocess_chunk(clean, DROP_COLS).select_dtypes(include=['number'])
    X_arr = scaler.transform(X_df)
    X_pca = pca.transform(X_arr)
    drift_mask = ce.detect_drift(X_pca)
    if drift_mask.any():
        logging.info('[stream][DRIFT] Concept drift detected! Retraining model...')
        _retrain_on_logged_data()
    else:
        logging.info('No drift')
    
    # 5: Label & Log new data:
    assert rolling_csv is not None, "rolling_csv should have been initialized"
    for row_vals, pred in zip(clean.itertuples(index=False, name=None), preds):
        # Convert namedtuple to list, append prediction and drift flag
        drift_flag = int(drift_mask.any())  # or per-row logic if available
        log_row = list(row_vals) + [pred, drift_flag]
        rolling_csv.append(log_row)


def run_streaming():
    """Start the listener and pass batches to _handle_batch"""
    logging.info('Starting streaming server...')
    try:
        run_server(callback=_handle_batch)
    finally:
        if rolling_csv is not None:
            rolling_csv.close()


if __name__ == '__main__':
    run_streaming()
