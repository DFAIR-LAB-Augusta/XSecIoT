import argparse
from FIRE_codebase.preprocessing import run_preprocessing
from FIRE_codebase.models import run_binary_classification, run_multiclass_classification


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
    
    # Future arguments for steps 2 and 3 will be added here.
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Run the preprocessing step.
    # The output file (aggregated_data.csv) will be saved in the dataset's folder by run_preprocessing.
    run_preprocessing(args.dataset_path, args.window_size, args.step_size)
    run_binary_classification(aggregated_data_path) # Add automated creation of aggregated_data_path based on args.dataset_path
    run_multiclass_classification(aggregated_data_path)

if __name__ == '__main__':
    main()
