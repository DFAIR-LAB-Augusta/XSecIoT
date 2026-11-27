"""Command line for the Direction 3 experiment driver: `python -m firce.experiments`."""

from __future__ import annotations

import argparse
import dataclasses

from pathlib import Path

from firce.experiments.driver import run_experiment
from firce.experiments.presets import PRESETS


def build_parser() -> argparse.ArgumentParser:
    """Build the driver's argument parser."""
    parser = argparse.ArgumentParser(prog='python -m firce.experiments', description=__doc__)
    parser.add_argument('--preset', choices=sorted(PRESETS), required=True)
    parser.add_argument('--dataPath', type=Path, default=None, help='Override the preset dataset path')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--heldOut', nargs='+', default=None, help='Held-out class names (default: the preset)')
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--modelVariant', default='rf')
    parser.add_argument('--ceType', default='ice')
    parser.add_argument('--useMLP', action='store_true', help='Use the multiclass MLP as the conformal model')
    parser.add_argument('--studyB', action='store_true', help='Explanation latency and stability')
    parser.add_argument('--studyBSamples', type=int, default=50)
    parser.add_argument('--studyC', action='store_true', help='Template (and LLM) reports')
    parser.add_argument('--noveltyLlmBackendType', default=None)
    parser.add_argument('--noveltyLlmModelPath', default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the experiment and print the run directory."""
    args = build_parser().parse_args(argv)
    preset = PRESETS[args.preset]
    if args.dataPath is not None:
        preset = dataclasses.replace(preset, path=str(args.dataPath.resolve()))
    run_dir = run_experiment(
        preset,
        args.seeds,
        args.out,
        held_out=args.heldOut,
        model_variant=args.modelVariant,
        ce_type=args.ceType,
        use_mlp=args.useMLP,
        study_b=args.studyB,
        study_c=args.studyC,
        study_b_samples=args.studyBSamples,
        llm_backend_type=args.noveltyLlmBackendType,
        llm_model_path=args.noveltyLlmModelPath,
    )
    print(run_dir)


if __name__ == '__main__':
    main()
