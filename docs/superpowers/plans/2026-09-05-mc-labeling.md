# Multiclass Labeling Script (#91) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scripts/mc_labeling.py`, a new script that assigns a categorical `MC_Label` column to a cicflowmeter-schema flow CSV by matching flows against a set of (src_ip, dst_ip) → attack-name mappings, defaulting unmatched flows to `'Benign'`.

**Architecture:** A standalone script mirroring the existing `scripts/labeling.py` binary labeler's structure (argparse → validate → transform → write CSV), generalized from a single src/dst-IP pair mapping to `1`/`0` into N pairs each mapping to an arbitrary attack-name string. No shared code is extracted from `labeling.py` — the IP-validation regex is duplicated (5 lines) rather than importing across sibling scripts, since `scripts/` is not a package and cross-script imports depend on fragile `sys.path[0]` behavior.

**Tech Stack:** Python 3.11+, `pandas`, `argparse`, `pytest` (existing project stack — no new dependencies).

## Global Constraints

- Follow `.ruff.toml`: single-quote strings, 120 char line length, import order (stdlib → third-party → local, blank line between groups per existing files).
- Test files go in `tests/`, named `test_*.py`, run via `pytest -q` (see `pytest.ini`).
- No new dependencies — `pandas`, `argparse`, `pathlib`, `dataclasses`, `re` are already used by `scripts/labeling.py`.
- Match the error-handling style of `scripts/labeling.py`: raise `ValueError`/`FileNotFoundError` from validation/transform functions, catch in `main()`, print to `stderr`, `sys.exit(1)`.

---

## File Structure

- Create: `scripts/mc_labeling.py` — the new script (config dataclasses, arg parsing, validation, labeling transform, `main`).
- Create: `tests/test_mc_labeling.py` — unit tests for every function in the new script.
- Modify: `Makefile` — add an `mc-label` target so the script is reachable the same way `label`/`merge` are.

---

### Task 1: `AttackMapping` parsing and validation

**Files:**
- Create: `scripts/mc_labeling.py`
- Test: `tests/test_mc_labeling.py`

**Interfaces:**
- Produces: `AttackMapping` (frozen dataclass: `src_ip: str`, `dst_ip: str`, `attack_name: str`), `_is_valid_ip(ip: str) -> bool`, `_parse_mapping(raw: str) -> AttackMapping` (raises `ValueError` on malformed input).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mc_labeling.py` with this content:

```python
import pytest

from scripts.mc_labeling import AttackMapping, _parse_mapping


def test_parse_mapping_valid():
    mapping = _parse_mapping('192.168.1.192,192.168.1.103,XMasAttack')
    assert mapping == AttackMapping(
        src_ip='192.168.1.192', dst_ip='192.168.1.103', attack_name='XMasAttack'
    )


def test_parse_mapping_strips_whitespace():
    mapping = _parse_mapping(' 192.168.1.192 , 192.168.1.103 , XMasAttack ')
    assert mapping == AttackMapping(
        src_ip='192.168.1.192', dst_ip='192.168.1.103', attack_name='XMasAttack'
    )


def test_parse_mapping_wrong_field_count():
    with pytest.raises(ValueError, match='exactly 3 comma-separated fields'):
        _parse_mapping('192.168.1.192,192.168.1.103')


def test_parse_mapping_invalid_src_ip():
    with pytest.raises(ValueError, match='Invalid source IP'):
        _parse_mapping('not-an-ip,192.168.1.103,XMasAttack')


def test_parse_mapping_invalid_dst_ip():
    with pytest.raises(ValueError, match='Invalid destination IP'):
        _parse_mapping('192.168.1.192,not-an-ip,XMasAttack')


def test_parse_mapping_empty_attack_name():
    with pytest.raises(ValueError, match='attack_name must not be empty'):
        _parse_mapping('192.168.1.192,192.168.1.103, ')
```

Since `scripts/` has no `__init__.py`, add one so it can be imported as a package from tests:

```python
# scripts/__init__.py
```

(empty file — makes `scripts` importable as `scripts.mc_labeling` from the repo root, which is where `pytest` runs per `pytest.ini`'s `testpaths = tests`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mc_labeling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.mc_labeling'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/mc_labeling.py`:

```python
import argparse
import logging
import re
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LABEL = 'Benign'

_IP_PATTERN = re.compile(
    r'^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
    r'(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$'
)


@dataclass(frozen=True)
class AttackMapping:
    src_ip: str
    dst_ip: str
    attack_name: str


def _is_valid_ip(ip: str) -> bool:
    """
    Validate if the given string is a valid IPv4 address.
    """
    return _IP_PATTERN.match(ip) is not None


def _parse_mapping(raw: str) -> AttackMapping:
    """
    Parse a 'src_ip,dst_ip,attack_name' string into an AttackMapping.
    Raises ValueError on malformed input.
    """
    fields = [f.strip() for f in raw.split(',')]
    if len(fields) != 3:
        raise ValueError(
            f"Mapping '{raw}' must have exactly 3 comma-separated fields: src_ip,dst_ip,attack_name"
        )
    src_ip, dst_ip, attack_name = fields
    if not _is_valid_ip(src_ip):
        raise ValueError(f'Invalid source IP address in mapping: {src_ip}')
    if not _is_valid_ip(dst_ip):
        raise ValueError(f'Invalid destination IP address in mapping: {dst_ip}')
    if not attack_name:
        raise ValueError(f"attack_name must not be empty in mapping: '{raw}'")
    return AttackMapping(src_ip=src_ip, dst_ip=dst_ip, attack_name=attack_name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mc_labeling.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/mc_labeling.py scripts/__init__.py tests/test_mc_labeling.py
git commit -m "feat: add AttackMapping parsing for multiclass labeling"
```

---

### Task 2: Config dataclass + CLI argument parsing

**Files:**
- Modify: `scripts/mc_labeling.py`
- Test: `tests/test_mc_labeling.py`

**Interfaces:**
- Consumes: `AttackMapping`, `_parse_mapping` (Task 1).
- Produces: `MCLabelConfig` (frozen dataclass: `dataset_path: Path`, `mappings: Tuple[AttackMapping, ...]`), `_parse_arguments(argv: List[str] | None = None) -> MCLabelConfig`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mc_labeling.py`:

```python
from pathlib import Path

from scripts.mc_labeling import MCLabelConfig, _parse_arguments


def test_parse_arguments_single_mapping():
    config = _parse_arguments([
        'data.csv',
        '--mapping', '192.168.1.192,192.168.1.103,XMasAttack',
    ])
    assert config == MCLabelConfig(
        dataset_path=Path('data.csv'),
        mappings=(AttackMapping('192.168.1.192', '192.168.1.103', 'XMasAttack'),),
    )


def test_parse_arguments_multiple_mappings():
    config = _parse_arguments([
        'data.csv',
        '--mapping', '192.168.1.192,192.168.1.103,XMasAttack',
        '--mapping', '10.0.0.1,10.0.0.2,PortScan',
    ])
    assert config.mappings == (
        AttackMapping('192.168.1.192', '192.168.1.103', 'XMasAttack'),
        AttackMapping('10.0.0.1', '10.0.0.2', 'PortScan'),
    )


def test_parse_arguments_requires_at_least_one_mapping():
    with pytest.raises(SystemExit):
        _parse_arguments(['data.csv'])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mc_labeling.py -v`
Expected: FAIL with `ImportError: cannot import name 'MCLabelConfig'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/mc_labeling.py` (after `_parse_mapping`):

```python
@dataclass(frozen=True)
class MCLabelConfig:
    dataset_path: Path
    mappings: Tuple[AttackMapping, ...]


def _parse_arguments(argv: List[str] | None = None) -> MCLabelConfig:
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Label a dataset with an MC_Label column by matching flows against '
            'src_ip/dst_ip pairs, each mapped to an attack name. Unmatched flows are '
            f"labeled '{DEFAULT_LABEL}'."
        )
    )
    parser.add_argument('dataset_path', type=str, help='Path to the CSV dataset.')
    parser.add_argument(
        '--mapping',
        dest='mappings',
        action='append',
        required=True,
        metavar='SRC_IP,DST_IP,ATTACK_NAME',
        help='Attack mapping as src_ip,dst_ip,attack_name. Repeat for multiple attack pairs.',
    )
    args = parser.parse_args(argv)
    mappings = tuple(_parse_mapping(raw) for raw in args.mappings)
    return MCLabelConfig(dataset_path=Path(args.dataset_path), mappings=mappings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mc_labeling.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/mc_labeling.py tests/test_mc_labeling.py
git commit -m "feat: add CLI argument parsing for multiclass labeling"
```

---

### Task 3: Input validation (file checks + duplicate mapping detection)

**Files:**
- Modify: `scripts/mc_labeling.py`
- Test: `tests/test_mc_labeling.py`

**Interfaces:**
- Consumes: `MCLabelConfig`, `AttackMapping` (Tasks 1-2).
- Produces: `_validate_inputs(dataset_path: Path, mappings: Tuple[AttackMapping, ...]) -> None` (raises `FileNotFoundError` or `ValueError`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mc_labeling.py`:

```python
from scripts.mc_labeling import _validate_inputs


def test_validate_inputs_missing_file(tmp_path):
    missing = tmp_path / 'nope.csv'
    with pytest.raises(FileNotFoundError):
        _validate_inputs(missing, (AttackMapping('1.2.3.4', '5.6.7.8', 'X'),))


def test_validate_inputs_non_csv(tmp_path):
    bad = tmp_path / 'data.txt'
    bad.write_text('not a csv')
    with pytest.raises(ValueError, match='Only CSV files are supported'):
        _validate_inputs(bad, (AttackMapping('1.2.3.4', '5.6.7.8', 'X'),))


def test_validate_inputs_duplicate_pair_conflicting_names(tmp_path):
    csv_path = tmp_path / 'data.csv'
    csv_path.write_text('src_ip,dst_ip\n1.2.3.4,5.6.7.8\n')
    mappings = (
        AttackMapping('1.2.3.4', '5.6.7.8', 'XMasAttack'),
        AttackMapping('1.2.3.4', '5.6.7.8', 'PortScan'),
    )
    with pytest.raises(ValueError, match='duplicate mapping'):
        _validate_inputs(csv_path, mappings)


def test_validate_inputs_accepts_valid_config(tmp_path):
    csv_path = tmp_path / 'data.csv'
    csv_path.write_text('src_ip,dst_ip\n1.2.3.4,5.6.7.8\n')
    _validate_inputs(csv_path, (AttackMapping('1.2.3.4', '5.6.7.8', 'XMasAttack'),))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mc_labeling.py -v`
Expected: FAIL with `ImportError: cannot import name '_validate_inputs'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/mc_labeling.py`:

```python
def _validate_inputs(dataset_path: Path, mappings: Tuple[AttackMapping, ...]) -> None:
    """
    Perform validations on input arguments.
    Raises ValueError or FileNotFoundError as appropriate.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f'File does not exist: {dataset_path}')
    if not dataset_path.is_file():
        raise ValueError(f'Provided path is not a file: {dataset_path}')
    if dataset_path.suffix.lower() != '.csv':
        raise ValueError('Only CSV files are supported.')

    seen: dict[Tuple[str, str], str] = {}
    for mapping in mappings:
        pair = (mapping.src_ip, mapping.dst_ip)
        if pair in seen and seen[pair] != mapping.attack_name:
            raise ValueError(
                f'duplicate mapping for {pair}: '
                f"'{seen[pair]}' vs '{mapping.attack_name}'"
            )
        seen[pair] = mapping.attack_name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mc_labeling.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/mc_labeling.py tests/test_mc_labeling.py
git commit -m "feat: add input validation for multiclass labeling"
```

---

### Task 4: Labeling transform (`_add_multiclass_labels`)

**Files:**
- Modify: `scripts/mc_labeling.py`
- Test: `tests/test_mc_labeling.py`

**Interfaces:**
- Consumes: `AttackMapping`, `DEFAULT_LABEL` (Task 1).
- Produces: `_add_multiclass_labels(dataset_path: Path, mappings: Tuple[AttackMapping, ...]) -> Path`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mc_labeling.py`:

```python
import pandas as pd

from scripts.mc_labeling import _add_multiclass_labels


def test_add_multiclass_labels_assigns_attack_and_benign(tmp_path):
    csv_path = tmp_path / 'flows.csv'
    pd.DataFrame({
        'src_ip': ['1.2.3.4', '1.2.3.4', '9.9.9.9'],
        'dst_ip': ['5.6.7.8', '5.6.7.8', '9.9.9.8'],
        'flow_duration': [1, 2, 3],
    }).to_csv(csv_path, index=False)

    mapping = AttackMapping('1.2.3.4', '5.6.7.8', 'XMasAttack')
    output_path = _add_multiclass_labels(csv_path, (mapping,))

    assert output_path == tmp_path / 'flows_mc_labeled.csv'
    result = pd.read_csv(output_path)
    assert result['MC_Label'].tolist() == ['XMasAttack', 'XMasAttack', 'Benign']


def test_add_multiclass_labels_multiple_mappings(tmp_path):
    csv_path = tmp_path / 'flows.csv'
    pd.DataFrame({
        'src_ip': ['1.2.3.4', '10.0.0.1', '9.9.9.9'],
        'dst_ip': ['5.6.7.8', '10.0.0.2', '9.9.9.8'],
    }).to_csv(csv_path, index=False)

    mappings = (
        AttackMapping('1.2.3.4', '5.6.7.8', 'XMasAttack'),
        AttackMapping('10.0.0.1', '10.0.0.2', 'PortScan'),
    )
    output_path = _add_multiclass_labels(csv_path, mappings)
    result = pd.read_csv(output_path)
    assert result['MC_Label'].tolist() == ['XMasAttack', 'PortScan', 'Benign']


def test_add_multiclass_labels_missing_ip_columns(tmp_path):
    csv_path = tmp_path / 'flows.csv'
    pd.DataFrame({'flow_duration': [1, 2, 3]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="must contain 'src_ip' and 'dst_ip'"):
        _add_multiclass_labels(csv_path, (AttackMapping('1.2.3.4', '5.6.7.8', 'X'),))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mc_labeling.py -v`
Expected: FAIL with `ImportError: cannot import name '_add_multiclass_labels'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/mc_labeling.py`:

```python
def _add_multiclass_labels(dataset_path: Path, mappings: Tuple[AttackMapping, ...]) -> Path:
    """
    Add an MC_Label column to the dataset based on src/dst IP-pair -> attack-name
    mappings. Flows matching no mapping are labeled DEFAULT_LABEL.
    Returns the output file path.
    """
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise ValueError(f'Failed to read CSV file: {e}')

    if 'src_ip' not in df.columns or 'dst_ip' not in df.columns:
        raise ValueError("CSV must contain 'src_ip' and 'dst_ip' columns.")

    df['MC_Label'] = DEFAULT_LABEL
    for mapping in mappings:
        match = (df['src_ip'] == mapping.src_ip) & (df['dst_ip'] == mapping.dst_ip)
        df.loc[match, 'MC_Label'] = mapping.attack_name

    output_path = dataset_path.with_name(dataset_path.stem + '_mc_labeled' + dataset_path.suffix)
    df.to_csv(output_path, index=False)

    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mc_labeling.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/mc_labeling.py tests/test_mc_labeling.py
git commit -m "feat: implement multiclass label assignment by IP-pair mapping"
```

---

### Task 5: `main()` entry point + end-to-end tests

**Files:**
- Modify: `scripts/mc_labeling.py`
- Test: `tests/test_mc_labeling.py`

**Interfaces:**
- Consumes: `_parse_arguments`, `_validate_inputs`, `_add_multiclass_labels` (Tasks 2-4).
- Produces: `main(argv: List[str] | None = None) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mc_labeling.py`:

```python
from scripts.mc_labeling import main


def test_main_end_to_end_success(tmp_path, capsys):
    csv_path = tmp_path / 'flows.csv'
    pd.DataFrame({
        'src_ip': ['1.2.3.4', '9.9.9.9'],
        'dst_ip': ['5.6.7.8', '9.9.9.8'],
    }).to_csv(csv_path, index=False)

    main([str(csv_path), '--mapping', '1.2.3.4,5.6.7.8,XMasAttack'])

    captured = capsys.readouterr()
    expected_output = tmp_path / 'flows_mc_labeled.csv'
    assert str(expected_output) in captured.out
    assert expected_output.exists()


def test_main_exits_nonzero_on_missing_file(tmp_path, capsys):
    missing = tmp_path / 'nope.csv'
    with pytest.raises(SystemExit) as exc_info:
        main([str(missing), '--mapping', '1.2.3.4,5.6.7.8,XMasAttack'])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert 'Error:' in captured.err


def test_main_exits_nonzero_on_invalid_mapping(tmp_path, capsys):
    csv_path = tmp_path / 'flows.csv'
    pd.DataFrame({'src_ip': ['1.2.3.4'], 'dst_ip': ['5.6.7.8']}).to_csv(csv_path, index=False)

    with pytest.raises(SystemExit) as exc_info:
        main([str(csv_path), '--mapping', 'not-an-ip,5.6.7.8,XMasAttack'])

    assert exc_info.value.code == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mc_labeling.py -v`
Expected: FAIL with `ImportError: cannot import name 'main'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/mc_labeling.py`:

```python
def main(argv: List[str] | None = None) -> None:
    """
    Main function to execute the multiclass labeling pipeline.
    """
    try:
        config = _parse_arguments(argv)
        _validate_inputs(config.dataset_path, config.mappings)
        output_path = _add_multiclass_labels(config.dataset_path, config.mappings)
        print(f'Multiclass-labeled dataset saved to: {output_path}')
    except SystemExit:
        raise
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
```

Note: `_parse_arguments` itself calls `sys.exit(2)` via `argparse` on malformed CLI usage (e.g. missing `--mapping`) — that's argparse's own behavior and is left as-is, matching `scripts/labeling.py`'s convention of not catching `SystemExit`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mc_labeling.py -v`
Expected: 19 passed

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures

- [ ] **Step 6: Commit**

```bash
git add scripts/mc_labeling.py tests/test_mc_labeling.py
git commit -m "feat: add main entry point for multiclass labeling script"
```

---

### Task 6: Makefile target

**Files:**
- Modify: `Makefile`

**Interfaces:**
- Consumes: `scripts/mc_labeling.py`'s CLI contract (Tasks 1-5): positional `dataset_path`, repeatable `--mapping SRC_IP,DST_IP,ATTACK_NAME`.

- [ ] **Step 1: Add the `mc-label` target**

In `Makefile`, add a new variable near `LABELED_DATASET_PATH`/`UNLABELED_DATASET_PATH` (around line 8-10):

```makefile
MC_MAPPING_ARGS ?=
```

Add `mc-label` to the `.PHONY` list (line 16-19), changing:

```makefile
.PHONY: help sync test clean \
        sim-bin sim-mc xseciot \
        bin-label merge \
        overall-perf overall-scrape
```

to:

```makefile
.PHONY: help sync test clean \
        sim-bin sim-mc xseciot \
        bin-label mc-label merge \
        overall-perf overall-scrape
```

Add the target itself, after the existing `label:` target (after line 69):

```makefile
mc-label:
	@if [ -z "$(UNLABELED_DATASET_PATH)" ]; then echo "ERROR: set UNLABELED_DATASET_PATH=..."; exit 1; fi
	@if [ -z "$(MC_MAPPING_ARGS)" ]; then echo "ERROR: set MC_MAPPING_ARGS='--mapping SRC_IP,DST_IP,ATTACK_NAME [--mapping ...]'"; exit 1; fi
	$(UV) run scripts/mc_labeling.py "$(UNLABELED_DATASET_PATH)" $(MC_MAPPING_ARGS)
```

Add an example to the `help` target's echo block (after line 31):

```makefile
	@echo "  make mc-label  UNLABELED_DATASET_PATH=datasets/CEFlows/CEFlows2_merged.csv MC_MAPPING_ARGS='--mapping 192.168.1.192,192.168.1.103,XMasAttack'"
```

- [ ] **Step 2: Verify the target runs**

Run:
```bash
make mc-label UNLABELED_DATASET_PATH=/tmp/claude-1001/-home-claude-dfair/3d069162-b78a-4238-91bc-37c15564913a/scratchpad/mc_label_smoke.csv MC_MAPPING_ARGS="--mapping 1.2.3.4,5.6.7.8,XMasAttack"
```

First create the smoke-test CSV:
```bash
printf 'src_ip,dst_ip\n1.2.3.4,5.6.7.8\n9.9.9.9,9.9.9.8\n' > /tmp/claude-1001/-home-claude-dfair/3d069162-b78a-4238-91bc-37c15564913a/scratchpad/mc_label_smoke.csv
```

Expected: prints `Multiclass-labeled dataset saved to: .../mc_label_smoke_mc_labeled.csv`, and that file exists with `MC_Label` column `['XMasAttack', 'Benign']`.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: add mc-label Makefile target for multiclass labeling script"
```

---

## Self-Review

**Spec coverage:** Issue #91 asks for "multiclass labeling (categorical attack-type assignment)" as a new script. Task 1-5 deliver `scripts/mc_labeling.py` with full CLI, validation, and CSV transform. Task 6 wires it into the same `make <target> PATH=...` convention as the existing `label`/`merge` targets. No sub-requirement of #91 is left uncovered.

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code.

**Type consistency:** `AttackMapping`, `MCLabelConfig`, `_parse_mapping`, `_validate_inputs`, `_add_multiclass_labels`, and `main` are used with identical signatures across all tasks that reference them.

**Known pre-existing issue, not fixed here:** The existing `label:` Makefile target (line 67-69) invokes `scripts/labeling.py --dataset_path "..."`, but `scripts/labeling.py`'s actual argparse only accepts positional `dataset_path src_ip dest_ip` — no `--dataset_path` flag exists, so `make label` is currently broken independent of this work. Also, `.PHONY` declares `bin-label` while the target is named `label`. Flagging for a separate fix; out of scope for #91.
