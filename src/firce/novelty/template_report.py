# firce.novelty.template_report
"""
Deterministic template report baseline (Study C).

Produces the same structured report as the LLM path (summary, suggested_label,
raw_output) from the XAI explanation and novelty context, with no model loaded.
Output depends only on the inputs.
"""

_SUMMARY_MAX = 200
_LABEL_MAX = 80

# CICFlowMeter feature name -> (phrase when contribution is positive, phrase when negative).
FEATURE_PHRASES: dict[str, tuple[str, str]] = {
    'flow_duration': ('long flow duration', 'short flow duration'),
    'flow_byts_s': ('high byte rate', 'low byte rate'),
    'flow_pkts_s': ('high packet rate', 'low packet rate'),
    'fwd_pkts_s': ('high forward packet rate', 'low forward packet rate'),
    'bwd_pkts_s': ('high backward packet rate', 'low backward packet rate'),
    'tot_fwd_pkts': ('many forward packets', 'few forward packets'),
    'tot_bwd_pkts': ('many backward packets', 'few backward packets'),
    'totlen_fwd_pkts': ('large forward payload', 'small forward payload'),
    'totlen_bwd_pkts': ('large backward payload', 'small backward payload'),
    'fwd_pkt_len_max': ('large max forward packet', 'small max forward packet'),
    'fwd_pkt_len_mean': ('large mean forward packet', 'small mean forward packet'),
    'bwd_pkt_len_max': ('large max backward packet', 'small max backward packet'),
    'bwd_pkt_len_mean': ('large mean backward packet', 'small mean backward packet'),
    'pkt_len_max': ('large max packet length', 'small max packet length'),
    'pkt_len_mean': ('large mean packet length', 'small mean packet length'),
    'pkt_len_std': ('variable packet lengths', 'uniform packet lengths'),
    'pkt_size_avg': ('large average packet size', 'small average packet size'),
    'fwd_header_len': ('long forward headers', 'short forward headers'),
    'bwd_header_len': ('long backward headers', 'short backward headers'),
    'flow_iat_mean': ('long gaps between packets', 'short gaps between packets'),
    'flow_iat_max': ('long maximum packet gap', 'short maximum packet gap'),
    'flow_iat_std': ('irregular packet timing', 'regular packet timing'),
    'fwd_iat_mean': ('long forward packet gaps', 'short forward packet gaps'),
    'bwd_iat_mean': ('long backward packet gaps', 'short backward packet gaps'),
    'fin_flag_cnt': ('many FIN flags', 'few FIN flags'),
    'syn_flag_cnt': ('many SYN flags', 'few SYN flags'),
    'rst_flag_cnt': ('many RST flags', 'few RST flags'),
    'psh_flag_cnt': ('many PSH flags', 'few PSH flags'),
    'ack_flag_cnt': ('many ACK flags', 'few ACK flags'),
    'urg_flag_cnt': ('many URG flags', 'few URG flags'),
    'down_up_ratio': ('high download/upload ratio', 'low download/upload ratio'),
    'init_fwd_win_byts': ('large initial forward window', 'small initial forward window'),
    'init_bwd_win_byts': ('large initial backward window', 'small initial backward window'),
    'active_mean': ('long active periods', 'short active periods'),
    'idle_mean': ('long idle periods', 'short idle periods'),
    'src_port': ('high source port', 'low source port'),
    'dst_port': ('high destination port', 'low destination port'),
    'protocol': ('high protocol number', 'low protocol number'),
}


def _phrase(feature: str, contribution: float) -> str:
    positive = contribution >= 0
    if feature in FEATURE_PHRASES:
        return FEATURE_PHRASES[feature][0 if positive else 1]
    return f'{feature} high' if positive else f'{feature} low'


def generate_template_report(xai_result: dict, novelty_context: dict, top_k: int = 5) -> dict:
    """
    Build a deterministic structured report from XAI output and novelty context.

    Args:
        xai_result: Output of explain_with_shap/explain_with_lime
            ({'predicted_class', 'contributions': {feature: float}}).
        novelty_context: Novelty-signal scalars (max_softmax, tau, alpha).
        top_k: Number of top features (by |contribution|, ties by name) to describe.

    Returns:
        Dict with keys 'summary' (<= 200 chars), 'suggested_label' (<= 80 chars,
        never None) and 'raw_output' (str).
    """
    predicted_class = xai_result['predicted_class']
    contributions = xai_result['contributions']
    ranked = sorted(contributions.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:top_k]
    phrases = '; '.join(_phrase(name, value) for name, value in ranked)

    parts = [f'Flow nearest to {predicted_class}']
    if phrases:
        parts.append(f'driven by {phrases}')
    context = ', '.join(
        f'{key}={novelty_context[key]:.3f}' for key in ('max_softmax', 'tau', 'alpha') if key in novelty_context
    )
    if context:
        parts.append(f'({context})')
    summary = ' '.join(parts)[:_SUMMARY_MAX]
    suggested_label = f'unknown (nearest: {predicted_class})'[:_LABEL_MAX]
    raw_output = f'summary: {summary}\nsuggested_label: {suggested_label}'
    return {'summary': summary, 'suggested_label': suggested_label, 'raw_output': raw_output}


class TemplateReportBackend:
    """Marker backend selecting the template report path; loads no model."""

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        raise NotImplementedError('TemplateReportBackend produces reports via generate_template_report, not generate()')
