"""Read-only development diagnostics; repeat indices never become study run IDs."""
import math


def export_probability_diagnostics(run_id, model, benchmark, baseline, first, second):
    """Align three measurements by ID; missing model scores are never zero ranges."""
    repeat_diagnostics(baseline, first, second)
    cells = [baseline, first, second]
    probabilities = [data['prob'] for data in cells]
    unavailable = model == 'luna'
    if unavailable:
        if any(p is not None for p in probabilities):
            raise ValueError('Generative Luna has no protocol probability score')
        values = [None, None]
    else:
        indexed = []
        for data, scores in zip(cells, probabilities):
            if scores is None or len(scores) != len(data['cases']):
                raise ValueError('Missing or misaligned probability scores')
            if any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in scores):
                raise ValueError('Invalid probability score')
            indexed.append({case['pair_id']: float(p) for case, p in zip(data['cases'], scores)})
        ranges = [max(block[k] for block in indexed)-min(block[k] for block in indexed)
                  for k in indexed[1]]
        values = [math.fsum(ranges)/len(ranges), max(ranges)]
    names = ['mean_within_pair_probability_range', 'max_within_pair_probability_range']
    return [{'run_id': run_id, 'model': model, 'condition': 'stability/within_run_probability_range',
             'benchmark': benchmark, 'metric': metric, 'value': value, 'n': len(first['cases']),
             'split': 'development', 'available': not unavailable,
             'unavailable_reason': 'Generative Luna does not return probability scores' if unavailable else None,
             'scope': 'Descriptive range across three within-run measurements; not a three-study-run SD'}
            for metric, value in zip(names, values)]


def index_cell(data):
    cases, labels, predictions = data['cases'], data['y'], data['pred']
    if not cases or not len(cases) == len(labels) == len(predictions):
        raise ValueError('Empty or misaligned diagnostic cell')
    result = {}
    for case, label, prediction in zip(cases, labels, predictions):
        if case['pair_id'] in result or label not in (0, 1) or prediction not in (0, 1):
            raise ValueError('Duplicate ID or nonbinary diagnostic value')
        result[case['pair_id']] = (int(label), int(prediction))
    return result


def f1(index):
    tp = sum(y == 1 and p == 1 for y, p in index.values())
    fp = sum(y == 0 and p == 1 for y, p in index.values())
    fn = sum(y == 1 and p == 0 for y, p in index.values())
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0


def factorial_diagnostics(cells):
    names = ['I3_identity_json', 'I3_domain_json', 'X_P_identity_0', 'X_P_domain_0']
    if set(cells) != set(names):
        raise ValueError('Complete fixed factorial design required')
    indexed = {name: index_cell(cells[name]) for name in names}
    reference = indexed[names[0]]
    for other in indexed.values():
        if other.keys() != reference.keys() or any(other[k][0] != reference[k][0] for k in reference):
            raise ValueError('Factorial pair IDs or labels disagree')
    scores = {name: f1(value) for name, value in indexed.items()}
    return {'n': len(reference), 'f1_interaction':
            (scores['I3_domain_json']-scores['X_P_domain_0'])-
            (scores['I3_identity_json']-scores['X_P_identity_0'])}


def repeat_diagnostics(baseline, first, second):
    base, one, two = map(index_cell, [baseline, first, second])
    if one.keys() != two.keys() or not one.keys() <= base.keys():
        raise ValueError('Repeated pair sets disagree')
    if any(base[k][0] != one[k][0] or two[k][0] != one[k][0] for k in one):
        raise ValueError('Repeated labels disagree')
    n = len(one)
    return {'n': n,
            'all_three_prediction_agreement': sum(base[k][1] == one[k][1] == two[k][1] for k in one)/n,
            'baseline_vs_repeat1_flip_rate': sum(base[k][1] != one[k][1] for k in one)/n,
            'baseline_vs_repeat2_flip_rate': sum(base[k][1] != two[k][1] for k in one)/n,
            'repeat1_vs_repeat2_flip_rate': sum(one[k][1] != two[k][1] for k in one)/n}


def export_diagnostics(run_id, model, benchmark, factorial, baseline, first, second):
    rows = []
    for condition, values in [('stability/factorial_contrast', factorial_diagnostics(factorial)),
                              ('stability/within_run_repeatability', repeat_diagnostics(baseline, first, second))]:
        for metric, value in values.items():
            if metric != 'n':
                rows.append({'run_id': run_id, 'model': model, 'condition': condition,
                             'benchmark': benchmark, 'metric': metric, 'value': value,
                             'n': values['n'], 'split': 'development',
                             'scope': 'Descriptive within-run diagnostic; not test performance or three independent study runs'})
    return rows
