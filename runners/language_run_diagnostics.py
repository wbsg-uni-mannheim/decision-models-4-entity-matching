"""Paired language diagnostics from complete, request-verified run caches."""
import math

from stability_run_diagnostics import f1, index_cell

LANGUAGES = {'billiger-en-en-100un', 'billiger-de-en-100un', 'billiger-en-de-100un'}


def paired_values(model, reference, variant):
    base, other = index_cell(reference), index_cell(variant)
    if base.keys() != other.keys():
        raise ValueError('Language pair sets disagree')
    a = {c['pair_id']: c for c in reference['cases']}
    b = {c['pair_id']: c for c in variant['cases']}
    for key in base:
        if (base[key][0] != other[key][0] or a[key]['label'] != base[key][0]
                or b[key]['label'] != other[key][0] or a[key]['entity_ids'] != b[key]['entity_ids']):
            raise ValueError('Language identities or labels disagree')
        for side in ['left', 'right']:
            if a[key]['records'][side].keys() != b[key]['records'][side].keys():
                raise ValueError('Language field sets disagree')
    n = len(base)
    correct = [(base[k][1] == base[k][0], other[k][1] == other[k][0]) for k in base]
    values = {'f1_delta_vs_de_de': f1(other)-f1(base),
              'prediction_flip_rate': sum(base[k][1] != other[k][1] for k in base)/n,
              'both_correct_fraction': sum(x and y for x, y in correct)/n,
              'only_variant_correct_fraction': sum(not x and y for x, y in correct)/n,
              'only_de_de_correct_fraction': sum(x and not y for x, y in correct)/n,
              'both_wrong_fraction': sum(not x and not y for x, y in correct)/n}
    if model == 'luna':
        if reference['prob'] is not None or variant['prob'] is not None:
            raise ValueError('Generative Luna has no protocol probability score')
        absolute = signed = None
    else:
        indexed = []
        for cell in [reference, variant]:
            scores = cell['prob']
            if scores is None or len(scores) != n:
                raise ValueError('Missing or misaligned language probabilities')
            if any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in scores):
                raise ValueError('Invalid language probability')
            indexed.append({c['pair_id']: float(p) for c, p in zip(cell['cases'], scores)})
        shifts = [indexed[1][k]-indexed[0][k] for k in base]
        absolute = math.fsum(abs(x) for x in shifts)/n
        signed = math.fsum(shifts)/n
    values.update(mean_absolute_probability_shift=absolute, mean_signed_probability_shift=signed)
    return n, values


def export_language_diagnostics(run_id, model, benchmark, reference, variant):
    if benchmark not in LANGUAGES:
        raise ValueError('Only aligned billiger language variants support this contrast')
    n, values = paired_values(model, reference, variant)
    return [{'run_id': run_id, 'model': model, 'condition': 'variants/paired_language_vs_de_de',
             'benchmark': benchmark, 'metric': metric, 'value': value, 'n': n, 'split': 'test',
             'available': value is not None,
             'unavailable_reason': 'Generative Luna does not return probability scores' if value is None else None,
             'scope': 'Descriptive paired language contrast; translation not human-validated; no WDC full-split pairing or inferential significance'}
            for metric, value in values.items()]
