import argparse

from pathlib import Path

from firce.config import CEType, ModelType, ModelVariant, MonitorType


def parse_sim_args() -> argparse.Namespace:
    """
    Parse command-line arguments for CE simulation. Includes flags to enable PCA, file logging, and debug mode.

    Returns:
        argparse.Namespace: Parsed CLI arguments including paths, model/CE config, and flags.
    """
    p = argparse.ArgumentParser(description='CE sim: seed log and process CE flows')
    p.add_argument('aggregated_file', type=Path)
    p.add_argument('flows_file', type=Path)
    p.add_argument('--log', type=Path, default=Path('ce_log.csv.gz'))
    p.add_argument('--chunk_size', type=int, default=1000)
    p.add_argument('--max_rows', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--runNum', type=int, default=0)
    p.add_argument('--use-pca', action='store_true', help='Enable PCA during CE simulation')
    p.add_argument('--log2File', action='store_true', help='Enable logging output to file')
    p.add_argument('--modelVariant', type=ModelVariant, choices=list(ModelVariant), default='knn')
    p.add_argument('--modelType', type=ModelType, choices=list(ModelType), default='binary')
    p.add_argument(
        '--ceType',
        type=CEType,
        choices=list(CEType),
        default='cce',
        help="Type of conformal evaluator to use; 'none' disables CE and retraining",
    )
    p.add_argument('--debug', action='store_true', help='Enable debug logging')
    p.add_argument(
        '--useCircularLogger',
        action='store_true',
        help='Use in-memory circular logger instead of disk-based RollingCSV',
    )
    p.add_argument(
        '--useASC', action='store_true', help='Use adaptive significance value controller for CE drift detection'
    )
    p.add_argument('--useSVM', action='store_true', help='Use SVM model for CEs')
    p.add_argument('--useAC', action='store_true', help='Use adaptive chunking')
    p.add_argument('--unsw', action='store_true', help='Use UNSW-NB15 dataset format (default is DFAIR 2024)')
    p.add_argument('--useMLP', action='store_true', help='Use MLP model for CEs')
    p.add_argument(
        '--monitorType',
        type=MonitorType,
        choices=list(MonitorType),
        default='ce',
        help='Runtime drift monitor backend to use',
    )
    p.add_argument(
        '--cadeDims',
        type=int,
        nargs='+',
        default=None,
        help='CADE encoder dims, e.g. --cadeDims 96 512 128 32',
    )
    p.add_argument(
        '--cadeMargin',
        type=float,
        default=10.0,
        help='CADE contrastive margin',
    )
    p.add_argument(
        '--cadeMadThreshold',
        type=float,
        default=3.5,
        help='CADE MAD-based anomaly threshold',
    )
    p.add_argument(
        '--cadeMinDriftRatio',
        type=float,
        default=0.05,
        help='CADE chunk drift ratio threshold',
    )
    p.add_argument(
        '--cadeMinDriftCount',
        type=int,
        default=1,
        help='CADE chunk drift count threshold',
    )
    p.add_argument(
        '--cadeBatchSize',
        type=int,
        default=64,
        help='CADE contrastive AE batch size',
    )
    p.add_argument(
        '--cadeEpochs',
        type=int,
        default=250,
        help='CADE contrastive AE epochs',
    )
    p.add_argument(
        '--cadeLr',
        type=float,
        default=1e-3,
        help='CADE contrastive AE learning rate',
    )
    p.add_argument(
        '--cadeLambda1',
        type=float,
        default=1e-1,
        help='CADE contrastive loss lambda_1',
    )
    p.add_argument(
        '--cadeSimilarRatio',
        type=float,
        default=0.25,
        help='CADE similar pair ratio',
    )
    p.add_argument(
        '--cadeDisplayInterval',
        type=int,
        default=10,
        help='CADE training log interval',
    )
    p.add_argument(
        '--cadeForceRetrain',
        action='store_true',
        help='Force retraining CADE weights even if a weights file exists',
    )
    p.add_argument(
        '--cadeWeightsPath',
        type=str,
        default=None,
        help='Optional path to CADE weights file',
    )
    p.add_argument(
        '--cadeDevice',
        type=str,
        default='/CPU:0',
        help='Optional device for CADE to run on',
    )
    return p.parse_args()
