# #100 (D3 4/5): MITRE ATT&CK / Taxonomy Alignment Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map #99's LLM-generated candidate behavior descriptions (`suggested_label`/`summary`) onto real MITRE ATT&CK techniques, as heuristic/evidence-based suggestions - explicitly not ground truth - the fourth (optional) of 5 Direction 3 sub-issues.

**Architecture:** New module `src/firce/novelty/mitre_mapping.py` plus a curated real reference dataset `src/firce/novelty/data/mitre_technique_subset.json`. Rather than inventing a generic/arbitrary technique list, the reference dataset is scoped to the exact 14 MITRE ATT&CK techniques the user's own CAPEX repo (`~/dfair/repos/CAPEX`, the attack-generation tool that produces this project's training data) already targets in its attack library (`docs/superpowers/specs/2026-09-03-attack-library-expansion-design.md`, issue #66) - Active Scanning (T1595), Network Service Discovery (T1046), Brute Force (T1110), Default Accounts (T1078.001), Service Exhaustion Flood (T1499.002), Application or System Exploitation (T1499.004), Exploit Public-Facing Application (T1190), Application Layer Protocol (T1071), DNS [C2] (T1071.004), Exfiltration Over C2 Channel (T1041), Service Stop (T1489), Data Manipulation (T1565), Reflection Amplification (T1498.002), ARP Cache Poisoning (T1557.002). This directly aligns the taxonomy with what this system will actually see in its own training data, rather than a disconnected generic list. Technique id/name/description/tactics were fetched from the real, official MITRE ATT&CK STIX bundle (`github.com/mitre-attack/attack-stix-data`, `enterprise-attack.json`) and verified against the CAPEX design doc's technique IDs - not typed from memory, to avoid any risk of a hallucinated ID or description. Mapping itself is a lightweight, heuristic TF-IDF cosine-similarity match (`sklearn.feature_extraction.text.TfidfVectorizer`, already a dependency - no new ML dependency needed) between the LLM report's text and each technique's name+description, returning a ranked top-k list with similarity scores - explicitly presented as suggestions to evaluate for usefulness/stability, per the proposal's framing, never as a definitive classification.

**Tech Stack:** Python, scikit-learn (already a dependency), pytest.

## Global Constraints

- The MITRE technique reference data must be real, verified data (fetched from the official MITRE ATT&CK STIX source and cross-checked against CAPEX's design doc IDs) - never fabricated technique IDs/names/descriptions.
- No new dependencies - `TfidfVectorizer` is already available via the existing `scikit-learn` dependency.
- Output is framed as heuristic suggestions with a similarity score, never a single definitive answer - matches the issue's explicit "optional... evaluated for usefulness and stability rather than treated as definitive ground truth" framing.

---

### Task 1: Curated MITRE technique reference data + loader

**Files:**
- Create: `src/firce/novelty/data/mitre_technique_subset.json` (already staged - real data fetched from `github.com/mitre-attack/attack-stix-data/enterprise-attack.json`, filtered to the 14 technique IDs CAPEX's attack library targets, cross-verified against `~/dfair/repos/CAPEX/docs/superpowers/specs/2026-09-03-attack-library-expansion-design.md`)
- Create: `src/firce/novelty/mitre_mapping.py`
- Test: `tests/test_novelty_mitre_mapping.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `load_technique_reference() -> Dict[str, dict]` - used by Task 2.

- [ ] **Step 1: Verify the staged reference data**

The file `src/firce/novelty/data/mitre_technique_subset.json` should already exist (staged before this plan was written). Confirm it:
```bash
python3 -c "import json; d = json.load(open('src/firce/novelty/data/mitre_technique_subset.json')); print(len(d), 'techniques'); print(sorted(d.keys()))"
```
Expected: `14 techniques` and a list including `T1041, T1046, T1071, T1071.004, T1078.001, T1110, T1190, T1489, T1498.002, T1499.002, T1499.004, T1557.002, T1565, T1595`. Each entry has `technique_id`, `name`, `description`, `tactics` keys.

- [ ] **Step 2: Write the failing test**

Create `tests/test_novelty_mitre_mapping.py`:

```python
from firce.novelty.mitre_mapping import load_technique_reference


def test_load_technique_reference_returns_expected_technique_ids():
    reference = load_technique_reference()

    assert 'T1046' in reference
    assert reference['T1046']['name'] == 'Network Service Discovery'
    assert 'description' in reference['T1046']
    assert 'tactics' in reference['T1046']
    assert len(reference) == 14
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_novelty_mitre_mapping.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'firce.novelty.mitre_mapping'`

- [ ] **Step 4: Write the implementation**

Create `src/firce/novelty/mitre_mapping.py`:

```python
# firce.novelty.mitre_mapping
"""
MITRE ATT&CK / taxonomy alignment mapping for LLM-suggested behavior
descriptions (dissertation proposal Section 4.4.5).

Explicitly optional and heuristic/evidence-based, per the proposal - mapping
suggestions are "evaluated for usefulness and stability rather than treated
as definitive ground truth."

The reference technique set (data/mitre_technique_subset.json) is scoped to
the 14 MITRE ATT&CK techniques the CAPEX attack-generation tool
(~/dfair/repos/CAPEX, the source of this project's labeled training data)
already targets in its attack library - not a generic/disconnected list.
Technique id/name/description/tactics were fetched from the official MITRE
ATT&CK STIX bundle (github.com/mitre-attack/attack-stix-data) and
cross-verified against CAPEX's own design doc technique IDs.
"""

import json
import logging

from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).parent / 'data' / 'mitre_technique_subset.json'


def load_technique_reference() -> Dict[str, dict]:
    """
    Load the curated MITRE ATT&CK technique reference set.

    Returns:
        Dict mapping technique_id (e.g. 'T1046') to a dict with keys
        'technique_id', 'name', 'description', 'tactics' (list of tactic names).
    """
    with open(_DATA_PATH) as f:
        return json.load(f)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_novelty_mitre_mapping.py -v --no-cov`
Expected: PASS (1 test)

- [ ] **Step 6: Commit**

```bash
git add src/firce/novelty/data/mitre_technique_subset.json src/firce/novelty/mitre_mapping.py tests/test_novelty_mitre_mapping.py
git commit -m "feat: add curated MITRE ATT&CK technique reference (aligned with CAPEX) (#100)"
```

---

### Task 2: `suggest_mitre_techniques` - heuristic TF-IDF mapping

**Files:**
- Modify: `src/firce/novelty/mitre_mapping.py`
- Modify: `tests/test_novelty_mitre_mapping.py`

**Interfaces:**
- Consumes: `load_technique_reference` (Task 1), accepts #99's `generate_report` output shape directly (`{'summary': ..., 'suggested_label': ..., 'raw_output': ...}`).
- Produces: `suggest_mitre_techniques(report: dict, top_k: int = 3, reference: Optional[Dict[str, dict]] = None) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_novelty_mitre_mapping.py`:

```python
from firce.novelty.mitre_mapping import suggest_mitre_techniques


def test_suggest_mitre_techniques_ranks_scanning_report_toward_discovery_techniques():
    report = {
        'summary': 'A burst of probes against multiple network services was observed.',
        'suggested_label': 'possible scanning burst',
        'raw_output': 'Summary: A burst of probes against multiple network services was observed.\nSuggested label: possible scanning burst',
    }

    suggestions = suggest_mitre_techniques(report, top_k=3)

    assert len(suggestions) == 3
    top_ids = {s['technique_id'] for s in suggestions}
    assert 'T1595' in top_ids or 'T1046' in top_ids  # Active Scanning / Network Service Discovery
    for s in suggestions:
        assert set(s.keys()) == {'technique_id', 'name', 'score'}
        assert isinstance(s['score'], float)
    # Sorted descending by score.
    scores = [s['score'] for s in suggestions]
    assert scores == sorted(scores, reverse=True)


def test_suggest_mitre_techniques_ranks_flood_report_toward_impact_techniques():
    report = {
        'summary': 'A reflection amplification flood overwhelmed the target service.',
        'suggested_label': 'possible DDoS flood',
        'raw_output': '',
    }

    suggestions = suggest_mitre_techniques(report, top_k=3)

    top_ids = {s['technique_id'] for s in suggestions}
    assert 'T1498.002' in top_ids or 'T1499.002' in top_ids


def test_suggest_mitre_techniques_falls_back_to_raw_output_when_summary_and_label_are_none():
    report = {'summary': None, 'suggested_label': None, 'raw_output': 'network service scan probe discovery'}

    suggestions = suggest_mitre_techniques(report, top_k=1)

    assert len(suggestions) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_novelty_mitre_mapping.py -v --no-cov -k suggest_mitre_techniques`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `suggest_mitre_techniques`**

Add to `src/firce/novelty/mitre_mapping.py` (add imports at top: `from typing import List, Optional` and `from sklearn.feature_extraction.text import TfidfVectorizer`):

```python
def suggest_mitre_techniques(
    report: dict, top_k: int = 3, reference: Optional[Dict[str, dict]] = None
) -> List[dict]:
    """
    Heuristically rank MITRE ATT&CK techniques by textual similarity to an
    LLM-generated report (#99's generate_report output).

    This is a heuristic suggestion mechanism, not a classifier - per the
    proposal, results should be "evaluated for usefulness and stability
    rather than treated as definitive ground truth."

    Args:
        report: Output of firce.novelty.llm_reporting.generate_report
            ({'summary': str | None, 'suggested_label': str | None, 'raw_output': str}).
        top_k: Number of top-scoring techniques to return.
        reference: Technique reference dict (defaults to load_technique_reference()).

    Returns:
        List of up to top_k dicts ({'technique_id', 'name', 'score'}), sorted by
        descending similarity score. Empty summary/suggested_label fields fall
        back to raw_output, matching generate_report's graceful-degradation design.
    """
    if reference is None:
        reference = load_technique_reference()

    query_parts = [part for part in (report.get('summary'), report.get('suggested_label')) if part]
    query_text = ' '.join(query_parts) if query_parts else (report.get('raw_output') or '')

    technique_ids = list(reference.keys())
    corpus = [f"{reference[tid]['name']} {reference[tid]['description']}" for tid in technique_ids]

    vectorizer = TfidfVectorizer(stop_words='english')
    technique_vectors = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query_text])

    similarities = (technique_vectors @ query_vector.T).toarray().ravel()
    ranked_indices = similarities.argsort()[::-1][:top_k]

    return [
        {
            'technique_id': technique_ids[i],
            'name': reference[technique_ids[i]]['name'],
            'score': float(similarities[i]),
        }
        for i in ranked_indices
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_novelty_mitre_mapping.py -v --no-cov`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/firce/novelty/mitre_mapping.py tests/test_novelty_mitre_mapping.py
git commit -m "feat: add suggest_mitre_techniques heuristic TF-IDF mapping (#100)"
```

---

### Task 3: End-to-end integration with #99's real generate_report output, full suite, push, PR

**Files:**
- Modify: `tests/test_novelty_mitre_mapping.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_novelty_mitre_mapping.py`:

```python
def test_suggest_mitre_techniques_end_to_end_with_hand_built_report_shape():
    # Mirrors the exact dict shape firce.novelty.llm_reporting.generate_report
    # returns (verified in #99) without re-running a real LLM generation here -
    # that pipeline is already covered end-to-end in #99's own test suite.
    report = {
        'summary': 'Repeated failed login attempts against a default account were observed.',
        'suggested_label': 'possible credential brute force',
        'raw_output': (
            'Summary: Repeated failed login attempts against a default account were observed.\n'
            'Suggested label: possible credential brute force\n'
        ),
    }

    suggestions = suggest_mitre_techniques(report, top_k=2)

    top_ids = {s['technique_id'] for s in suggestions}
    assert 'T1110' in top_ids or 'T1078.001' in top_ids  # Brute Force / Default Accounts
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_novelty_mitre_mapping.py -v --no-cov`
Expected: PASS (all tests in the file).

- [ ] **Step 3: Commit**

```bash
git add tests/test_novelty_mitre_mapping.py
git commit -m "test: add end-to-end MITRE mapping sanity check (#100)"
```

- [ ] **Step 4: Run the full lean-CI suite and lint**

```bash
uv run pytest -q --no-cov
uv run ruff check src/firce/novelty/ tests/test_novelty_mitre_mapping.py
uv run ruff format --check src/firce/novelty/ tests/test_novelty_mitre_mapping.py
```
Expected: pytest exit code 0; ruff clean.

- [ ] **Step 5: Push and open the PR**

```bash
git push origin 100-mitre-mapping
gh pr create --base multiclass --head 100-mitre-mapping \
  --title "feat: MITRE ATT&CK taxonomy alignment mapping (#100, D3 4/5, optional)" \
  --body "$(cat <<'EOF'
## Summary
- New module `src/firce/novelty/mitre_mapping.py`: heuristic TF-IDF cosine-similarity mapping from #99's LLM-generated report text to candidate MITRE ATT&CK techniques, explicitly framed as suggestions to "evaluate for usefulness and stability," never as ground truth, per the proposal's own framing for this optional sub-issue.
- Reference dataset (`src/firce/novelty/data/mitre_technique_subset.json`) is scoped to the exact 14 techniques the user's own CAPEX repo (the tool that generates this project's labeled training data) already targets in its attack library, rather than a generic/disconnected technique list - directly aligns the taxonomy with what this system will actually see. Technique id/name/description/tactics were fetched from the official MITRE ATT&CK STIX bundle (`github.com/mitre-attack/attack-stix-data`) and cross-verified against CAPEX's own design doc technique IDs - not typed from memory.
- No new dependency - reuses `scikit-learn`'s `TfidfVectorizer`, already a project dependency.
- Fourth (optional) of 5 sub-issues under #96. Depends on #99 (merged).

## Test plan
- [x] Reference data loader covers all 14 expected technique IDs with real name/description/tactics.
- [x] TF-IDF mapping verified against 3 realistic report texts (scanning, flood, brute-force) - each correctly ranks the semantically-relevant technique(s) in its top-k, confirmed via direct execution before writing the assertions.
- [x] Graceful fallback to `raw_output` when `summary`/`suggested_label` are both `None`, matching #99's `generate_report`'s own degradation behavior.
- [x] `uv run pytest -q`: full suite passes (exit 0).
- [x] `ruff check`/`ruff format --check`: clean.
EOF
)"
```

## Self-Review

**Spec coverage:** Task 1 builds the real, verified reference dataset + loader. Task 2 implements the heuristic mapping. Task 3 validates against a realistic report shape and ships.

**Placeholder scan:** No TBD/TODO; every step has literal runnable code.

**Type consistency:** `suggest_mitre_techniques(report: dict, top_k: int = 3, reference: Optional[Dict[str, dict]] = None) -> List[dict]` consumes exactly the dict shape `generate_report` (#99) returns and the dict shape `load_technique_reference` (Task 1) returns - no shape mismatches introduced.

**Explicitly out of scope:** No live wiring into the runtime pipeline (that's #142, filed as a follow-up to #99, and would naturally extend to include this mapping too). No embedding-based/semantic-model similarity (TF-IDF keyword overlap is the deliberately lightweight heuristic choice, matching the issue's own "heuristic/evidence-based" framing and avoiding a new dependency) - a follow-up issue could explore this later if TF-IDF proves insufficient, but nothing in #100 commits to that. No expansion of the technique reference set beyond CAPEX's current 14 - if CAPEX's attack library grows (#66's roadmap), this reference set should be regenerated from the same source, but that's out of scope for this issue.
