from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

DATASET_LABELS = {
    'DFAIR': 'DFAIR',
    'NB15': 'UNSW-NB15',
    'CIC_UNSW': 'CICIDS2018',
}

STAT_KEYS = {
    'accuracy': '[Classifier Model] Avg Accuracy',
    'precision': '[Classifier Model] Avg Precision',
    'recall': '[Classifier Model] Avg Recall',
    'f1': '[Classifier Model] Avg F1 Score',
    'runtime': 'Total simulate time',
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def extract(grouped, classifier, ce_type, model_type):
    lookup = {}

    for item in grouped:
        if item['classifier'] == classifier and item['ce_type'] == ce_type and item['model_type'] == model_type:
            key = (item['dataset'], item['stat_key'])
            lookup[key] = item

    return lookup


def get_stat(lookup, dataset, stat_key):
    item = lookup.get((dataset, stat_key))
    if item is None:
        return None, None
    return item['mean_value'], item['std_value']


def plot_performance(lookup, output_dir):
    datasets = ['DFAIR', 'NB15', 'CIC_UNSW']
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    x = range(len(datasets))
    width = 0.18

    fig, ax = plt.subplots()

    for i, metric in enumerate(metrics):
        means = []
        stds = []

        for d in datasets:
            m, s = get_stat(lookup, d, STAT_KEYS[metric])
            means.append(m if m is not None else 0)
            stds.append(s if s is not None else 0)

        positions = [xi + i * width for xi in x]
        ax.bar(positions, means, width, yerr=stds)

    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel('Score')

    fig.tight_layout()
    fig.savefig(output_dir / 'performance_bar.png', dpi=300)


def plot_runtime_vs_accuracy(lookup, output_dir):
    datasets = ['DFAIR', 'NB15', 'CIC_UNSW']

    fig, ax = plt.subplots()

    for d in datasets:
        acc, _ = get_stat(lookup, d, STAT_KEYS['accuracy'])
        runtime, _ = get_stat(lookup, d, STAT_KEYS['runtime'])

        if acc is None or runtime is None:
            continue

        ax.scatter(runtime, acc)
        ax.text(runtime, acc, DATASET_LABELS[d])

    ax.set_xlabel('Runtime (s)')
    ax.set_ylabel('Accuracy')

    fig.tight_layout()
    fig.savefig(output_dir / 'runtime_vs_accuracy.png', dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('./figures'))
    parser.add_argument('--classifier', default='feedforward')
    parser.add_argument('--ce-type', default='approx_cce')
    parser.add_argument('--model-type', default='binary')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(args.json_input)
    grouped = data['grouped_summary']

    lookup = extract(grouped, args.classifier, args.ce_type, args.model_type)

    plot_performance(lookup, args.output_dir)
    plot_runtime_vs_accuracy(lookup, args.output_dir)


if __name__ == '__main__':
    main()
