# src.FIRE.main

import argparse
import itertools
import logging
import os
import sys
import time

from .models import run_binary_classification, run_feature_engineering, run_multiclass_classification
from .preprocessing import run_preprocessing
from .simulations import continuous_simulation, parallel_simulation, sequential_simulation

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete FIRE process: preprocessing, modeling, simulations."
    )
    parser.add_argument(
        "dataset_path", type=str,
        help="Path to the raw dataset CSV (e.g., './datasets/DFAIR/...csv')"
    )
    parser.add_argument(
        "--window_size", type=str, default="5s",
        help="Window size for aggregation (default: 5s)"
    )
    parser.add_argument(
        "--step_size", type=str, default="1s",
        help="Step size for aggregation (default: 1s)"
    )
    parser.add_argument(
        "--unsw", action="store_true",
        help="Run in UNSW‐dataset mode"
    )
    parser.add_argument(
        "--pca", action="store_true",
        help="Enable PCA for feature reduction"
    )
    parser.add_argument(
        "--noPre", action="store_true",
        help="Skip preprocessing step"
    )
    parser.add_argument(
        "--noMod", action="store_true",
        help="Skip modeling step"
    )
    parser.add_argument(
        "--noSim", action="store_true",
        help="Skip simulations step"
    )
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    args = _parse_args()

    aggregated = os.path.join(
        os.path.dirname(args.dataset_path),
        "aggregated_data.csv"
    )

    # 1) Preprocessing
    if not args.noPre:
        print("=== Preprocessing ===")
        run_preprocessing(
            args.dataset_path,
            args.window_size,
            args.step_size,
            args.unsw
        )

    # 2) Modeling
    if not args.noMod:
        if args.noPre and not os.path.exists(aggregated):
            print(
                f"Error: {aggregated} not found (used --noPre)", file=sys.stderr)
            sys.exit(1)

        print("\n=== Modeling ===")
        print("Binary classification", file=sys.stderr)
        run_binary_classification(aggregated, args.unsw, args.pca)
        print("Multi-class classification", file=sys.stderr)
        run_multiclass_classification(aggregated, args.unsw, args.pca)
        print("Feature engineering", file=sys.stderr)
        run_feature_engineering(aggregated)

    # 3) Simulations
    if not args.noSim:
        print("\n=== Simulations ===")
        sim_funcs = {
            "sequential": sequential_simulation,
            "continuous": continuous_simulation,
            "parallel":   parallel_simulation
        }
        model_types = ["binary", "multi"]
        variants = ["dt", "knn", "rf", "feedforward", "xgb"]
        modes = ["sequential", "continuous", "parallel"]

        for model_type, variant, mode in itertools.product(model_types, variants, modes):
            print(
                f"\n>> {model_type.capitalize()} | variant={variant} | mode={mode}")
            func = sim_funcs[mode]

            # base arguments for every call
            kwargs = dict(
                aggregated_file=aggregated,
                model_type=model_type,
                model_variant=variant,
                threshold=0.5,
                isUNSW=args.unsw
            )

            # add mode-specific parameters
            if mode == "sequential":
                kwargs.update(chunk_size=1000, delay=1)
                preds = func(**kwargs)

            elif mode == "continuous":
                kwargs.update(chunk_size=1000, window_duration=300, delay=1)
                _, preds = func(**kwargs)

            else:  # parallel
                kwargs.update(chunk_size=1000, num_processes=4)
                preds = func(**kwargs)

            print(f"{mode.capitalize()} returned {len(preds)} predictions")

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f}s")


if __name__ == '__main__':
    main()

