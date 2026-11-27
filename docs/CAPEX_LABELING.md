# CAPEX window labeling

`make capex-label` runs `scripts/capex_labeling.py`. It labels a CICFlowMeter flow CSV with `MC_Label` and
`MC_Partial` by joining CAPEX ground-truth attack windows on time overlap and target IP. Flows that match no
window are `Benign`.

## Inputs

- Flows: `data/flows/<device>_<session>_flow.csv`
- Ground truth: `data/logs/<device>_<session>_GT.csv`, with columns
  `session_id,device,target_ip,label,attempt,start_unix,end_unix`

## Join rule

A flow matches a window when its span `[timestamp, timestamp + flow_duration]` overlaps
`[start_unix, end_unix]` and the window's `target_ip` is the flow's source or destination IP. Otherwise the flow
is `Benign`.

- `MC_Partial` is true when the flow is not fully inside the winning window, or when it matched more than one window.
- When several windows match, the larger overlap wins; on a tie the earlier `start_unix` wins.

## Timezones

CICFlowMeter writes naive local time cut to whole seconds, so the zone the flows were captured in must be known.
`--flowTz` names that zone (an IANA zone name) and defaults to the host's local zone (DST-aware).

Flows whose timestamp falls in a repeated or skipped DST hour cannot be placed on the timeline. They are dropped
from the output, counted in the `Dropped (DST transition): N` report line, and a warning goes to stderr.

To avoid this, run the extraction in CAPEX with `TZ=UTC scripts/run_flows.sh` and label with `--flowTz UTC`
(`FLOW_TZ=UTC` for `make capex-label`).

## Rejected input

The script exits 1 with an error for a missing flow or ground-truth file, a missing column, blank `target_ip`,
`label`, `start_unix` or `end_unix` values, `end_unix < start_unix`, an unknown timezone, an excluded label (below), and any label outside the canonical class set.

## Excluded CAPEX labels

These labels are excluded from Direction 3 (see [#194](https://github.com/DFAIR-LAB-Augusta/XSecIoT/issues/194)) and rejected with a message saying so:

- `Config_Tamper_Setup_CGI`
- `Config_Tamper_TR064_SOAP`

The class registry in `src/firce/classes.py` is unchanged. Re-run the session without these attacks or remove their
rows from the ground-truth CSV.

## First-capture check

Compare the earliest attack-window flow with `start_unix` to confirm the flow timezone is right (tracked on #174).
