#!/usr/bin/env bash
# Phase-1 smoke: multiclass FIRCE on DFAIR, HTTP_Flood held out, novelty on.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$REPO/datasets/DFAIR/combined_data_with_okpVacc_modified.csv"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir "$WORK/CETrain"
cd "$WORK"

uv run --project "$REPO" --no-sync python - "$DATA" "$WORK" <<'PY'
import sys

import numpy as np
import pandas as pd

data, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(data)
rng = np.random.default_rng(42)

sessions = df.groupby('session_id')['Label'].first()
ids = sessions.index.to_numpy()
is_http = (sessions == 'HTTPAttack').to_numpy()
to_stream = is_http | (rng.random(len(ids)) < 0.5)
stream_ids = set(ids[to_stream])

in_stream = df['session_id'].isin(stream_ids)
train = df[~in_stream]
stream = df[in_stream]

cap = 20000
attacks = stream[stream['Label'] != 'Benign']
benign = stream[stream['Label'] == 'Benign']
benign = benign.sample(n=max(0, min(len(benign), cap - len(attacks))), random_state=42)
stream = pd.concat([attacks, benign]).sample(frac=1, random_state=42)

assert not set(train['session_id']) & set(stream['session_id'])
train.to_csv(f'{out}/CETrain/train.csv', index=False)
stream.to_csv(f'{out}/stream.csv', index=False)
print(f'train={len(train)} stream={len(stream)}')
print(stream['Label'].value_counts().to_string())
PY

set +e
uv run --project "$REPO" --no-sync firce CETrain/train.csv stream.csv \
  --modelType multi --modelVariant rf --monitorType ce --ceType ice \
  --labelColumn Label --heldOut HTTP_Flood --novelty --noveltyTau 1.0 --noveltySelectiveMode sampled \
  --pipeline simulation --useCircularLogger 2>&1 | tee run.log
status=${PIPESTATUS[0]}
set -e

if [ "$status" -ne 0 ]; then
  echo "phase1_smoke: firce exited $status" >&2
  exit "$status"
fi

if ! awk '/\[novelty\] chunk=/ { for (i = 1; i <= NF; i++) if ($i ~ /^flagged=/ && substr($i, 9) + 0 > 0) found = 1 } END { exit !found }' run.log; then
  echo 'phase1_smoke: no [novelty] chunk= line with flagged > 0' >&2
  exit 1
fi
echo 'phase1_smoke: OK'
