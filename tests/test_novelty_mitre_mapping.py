from firce.novelty.mitre_mapping import load_technique_reference, suggest_mitre_techniques


def test_load_technique_reference_returns_expected_technique_ids():
    reference = load_technique_reference()

    assert 'T1046' in reference
    assert reference['T1046']['name'] == 'Network Service Discovery'
    assert 'description' in reference['T1046']
    assert 'tactics' in reference['T1046']
    assert len(reference) == 14


def test_suggest_mitre_techniques_ranks_scanning_report_toward_discovery_techniques():
    report = {
        'summary': 'A burst of probes against multiple network services was observed.',
        'suggested_label': 'possible scanning burst',
        'raw_output': (
            'Summary: A burst of probes against multiple network services was observed.\n'
            'Suggested label: possible scanning burst'
        ),
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
