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
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer

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
