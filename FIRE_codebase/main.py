import argparse
import os
import time
from FIRE_codebase.preprocessing import run_preprocessing
from FIRE_codebase.preprocessingUNSW import run_preprocessingUNSW
from FIRE_codebase.models import run_binary_classification, run_multiclass_classification, run_feature_engineering
from FIRE_codebase.simulations import sequential_simulation, continuous_simulation, parallel_simulation
from FIRE_codebase.simulationsunsw import sequential_simulationUNSW, continuous_simulationUNSW, parallel_simulationUNSW

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run complete FIRE process. This includes preprocessing and later steps."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Full path to the dataset CSV file (e.g., './datasets/DFAIR/combined_data_with_okpVacc_modified.csv')"
    )
    parser.add_argument(
        "--window_size",
        type=str,
        default="5s",
        help="Window size for sliding window aggregation (default: '5s')"
    )
    parser.add_argument(
        "--step_size",
        type=str,
        default="1s",
        help="Step size for sliding window aggregation (default: '1s')"
    )
    parser.add_argument(
        "--unsw",
        action="store_true",
        help="Use multiclass labels"
    )
    
    return parser.parse_args()

def main():
    # Start the timer for the full pipeline
    start_time = time.time()
    
    args = parse_args()
    
    # Step 1: Preprocessing
    print("=== Running Preprocessing ===")
    if not args.unsw:
        run_preprocessing(args.dataset_path, args.window_size, args.step_size)
    else:
        run_preprocessingUNSW(args.dataset_path, args.window_size, args.step_size)
    aggregated_data_path = os.path.join(os.path.dirname(args.dataset_path), "aggregated_data.csv")

    # Step 2: Model Training / Evaluation
    print("\n=== Running Model Training/Evaluation ===")
    run_binary_classification(aggregated_data_path, args.unsw) 
    run_multiclass_classification(args.aggregated_file, args.unsw)
    run_feature_engineering(aggregated_data_path)

    # Step 3: Simulations
    variants = ["dt", "knn", "rf", "svm", "feedforward", "xgb"]
    sim_modes = ["sequential", "continuous", "parallel"]

    # Run simulations for binary classification
    print("\n=== Running Binary Simulations ===")
    for variant in variants:
        print(f"\n--- Binary Model Variant: {variant} ---")
        for mode in sim_modes:
            print(f"\n*** Simulation Mode: {mode} ***")
            if mode == "sequential":
                if args.unsw:
                    sequential_simulationUNSW(
                        aggregated_file=aggregated_data_path,
                        model_type="binary",
                        model_variant=variant,
                        chunk_size=1000,
                        delay=1,
                        threshold=0.5
                    )
                else:
                    sequential_simulation(
                        aggregated_file=aggregated_data_path,
                        model_type="binary",
                        model_variant=variant,
                        chunk_size=1000,
                        delay=1,
                        threshold=0.5
                    )
            elif mode == "continuous":
                if args.unsw:
                    continuous_simulationUNSW(
                        aggregated_file=aggregated_data_path,
                        model_type="binary",
                        model_variant=variant,
                        chunk_size=1000,
                        window_duration=300,
                        delay=1,
                        threshold=0.5
                    )
                else:
                    continuous_simulation(
                        aggregated_file=aggregated_data_path,
                        model_type="binary",
                        model_variant=variant,
                        chunk_size=1000,
                        window_duration=300,
                        delay=1,
                        threshold=0.5
                    )
            elif mode == "parallel":
                if args.UNSW:
                    preds = parallel_simulationUNSW(
                        aggregated_file=aggregated_data_path,
                        model_type="binary",
                        model_variant=variant,
                        chunk_size=1000,
                        num_processes=4,
                        threshold=0.5
                    )
                else:
                    preds = parallel_simulation(
                        aggregated_file=aggregated_data_path,
                        model_type="binary",
                        model_variant=variant,
                        chunk_size=1000,
                        num_processes=4,
                        threshold=0.5
                    )
                print("Parallel simulation predictions:")
                print(preds)
    
    # Run simulations for multi-class classification
    print("\n=== Running Multi-Class Simulations ===")
    for variant in variants:
        print(f"\n--- Multi-Class Model Variant: {variant} ---")
        for mode in sim_modes:
            print(f"\n*** Simulation Mode: {mode} ***")
            if mode == "sequential":
                if args.unsw:
                    sequential_simulationUNSW(
                        aggregated_file=aggregated_data_path,
                        model_type="multi",
                        model_variant=variant,
                        chunk_size=1000,
                        delay=1,
                        threshold=0.5
                    )
                else:
                    sequential_simulation(
                        aggregated_file=aggregated_data_path,
                        model_type="multi",
                        model_variant=variant,
                        chunk_size=1000,
                        delay=1,
                        threshold=0.5
                    )
            elif mode == "continuous":
                if args.unsw:
                    continuous_simulationUNSW(
                        aggregated_file=aggregated_data_path,
                        model_type="multi",
                        model_variant=variant,
                        chunk_size=1000,
                        window_duration=300,
                        delay=1,
                        threshold=0.5
                    )
                else:
                    continuous_simulation(
                        aggregated_file=aggregated_data_path,
                        model_type="multi",
                        model_variant=variant,
                        chunk_size=1000,
                        window_duration=300,
                        delay=1,
                        threshold=0.5
                    )
            # elif mode == "parallel":
            #     preds = parallel_simulation(
            #         aggregated_file=aggregated_data_path,
            #         model_type="multi",
            #         model_variant=variant,
            #         chunk_size=1000,
            #         num_processes=4,
            #         threshold=0.5
            #     )
            #     print("Parallel simulation predictions:")
            #     print(preds)
    
    # Compute and print the total elapsed time for the full pipeline.
    total_time = time.time() - start_time
    print(f"\nTotal elapsed time for full pipeline: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()
