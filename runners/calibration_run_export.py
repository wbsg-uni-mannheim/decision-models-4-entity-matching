"""Read-only calibration export with full development reconstruction before test."""
import json
import math
import numpy as np

METRICS = ['f1', 'precision', 'recall', 'accuracy', 'brier', 'log_loss',
           'ece_10_equal_width', 'average_precision', 'error_detection_auroc']
REGIMES = ['within', 'leave_one_out', 'opposite_domain']


def read(path):
    return json.loads(path.read_text())


def same(expected, actual, path='selection'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(path+' fields differ')
        for k, v in expected.items():
            same(v, actual[k], path+'/'+k)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(path+' matrix differs')
        for i, (a, b) in enumerate(zip(expected, actual)):
            same(a, b, path+'/'+str(i))
    elif isinstance(expected, float):
        tolerance = 1e-6 if '/parameters/' in path else 1e-12
        if (not isinstance(actual, (int, float)) or not math.isfinite(actual)
                or not math.isclose(expected, actual, rel_tol=tolerance, abs_tol=tolerance)):
            raise ValueError(path+' numeric drift')
    elif expected != actual:
        raise ValueError(path+' differs')


def reconstruct(module, development, saved):
    """Preserve unavailable all-method entries and the frozen nested matrix order."""
    expected = []
    for model in module.DECISION_MODELS:
        for target in module.BENCHMARKS:
            for regime in REGIMES:
                sources = module.source_benchmarks(target, regime)
                for n in module.BUDGETS:
                    base = dict(model=model, benchmark=target, regime=regime,
                                labels_per_source=n, sources=sources,
                                label_budget_total=n*len(sources))
                    try:
                        fits, tunes = [], []
                        for b in sources:
                            cases, y, p = development[model, b]
                            i, j = module.partition(cases, n)
                            if set(i) & set(j) or len(i)+len(j) != n:
                                raise RuntimeError('Invalid development partition')
                            fits.append((y[i], p[i]))
                            tunes.append((y[j], p[j]))
                        fy, fp = (np.concatenate([x[k] for x in fits]) for k in [0, 1])
                        ty, tp = (np.concatenate([x[k] for x in tunes]) for k in [0, 1])
                    except ValueError as exc:
                        expected.append({**base, 'method': 'all', 'status': 'unavailable', 'reason': str(exc)})
                        continue
                    for method in module.METHODS:
                        meta = {**base, 'method': method}
                        try:
                            parameters = module.fit(method, fy, fp)
                            # Compare refit parameters with tolerance, then verify the
                            # threshold against the saved parameters actually evaluated.
                            candidate = saved[len(expected)] if len(expected) < len(saved) else {}
                            if candidate.get('status') == 'frozen':
                                same(parameters, candidate.get('parameters'), 'selection/parameters')
                                parameters = candidate['parameters']
                            threshold = module.threshold(ty, module.transform(parameters, tp))
                        except ValueError as exc:
                            if 'selection/parameters' in str(exc):
                                raise
                            expected.append({**meta, 'status': 'unavailable', 'reason': str(exc)})
                            continue
                        expected.append({**meta, 'status': 'frozen', 'parameters': parameters,
                                         'threshold': threshold, 'fit_n': len(fy), 'tune_n': len(ty),
                                         'fit_positive': int(fy.sum()), 'tune_positive': int(ty.sum())})
    same(expected, saved)


def export_calibration(workspace, module, analyze, cell, run_id, digest):
    marker = read(workspace/'artifacts/expanded_calibration_complete.json')
    saved = read(workspace/'configs/calibration_selection.json')
    if (marker.get('status') != 'complete_with_explicit_unavailable_fits'
            or marker.get('selection_sha256') != digest(saved)):
        raise ValueError('Calibration incomplete or selection changed')
    manifest = module.verify()
    if marker.get('new_decisions') != manifest['new_decisions']:
        raise ValueError('Calibration completion count changed')
    previous = analyze.payload
    rows = []
    try:
        analyze.payload = module.payload
        dev = {}
        for model in module.DECISION_MODELS:
            for b in module.BENCHMARKS:
                pool = read(workspace/'data'/b/'calibration_pool.json')
                blocks = [cell(model, 'B0', b, 'route')[0]]
                if manifest['benchmarks'][b]['extra_n']:
                    blocks.append(cell(model, 'B0', b, 'calibration_extra')[0])
                indexed = {}
                for block in blocks:
                    for case, probability in zip(block['cases'], block['prob']):
                        key = case['pair_id']
                        if key in indexed:
                            raise ValueError('Duplicate calibration source pair')
                        indexed[key] = (case, float(probability))
                probabilities = []
                for case in pool:
                    source, probability = indexed[case['pair_id']]
                    if source != case or not math.isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError('Calibration source alignment/probability mismatch')
                    probabilities.append(probability)
                dev[model, b] = (pool, np.array([c['label'] for c in pool]), np.array(probabilities))
        reconstruct(module, dev, saved)
        # Only after the whole development matrix was checked may test cells load.
        for entry in saved:
            methods = module.METHODS if entry['method'] == 'all' else [entry['method']]
            for method in methods:
                for rule in ['fixed_0.5', 'development_F1']:
                    meta = dict(run_id=run_id, model=entry['model'], benchmark=entry['benchmark'],
                                condition=f"calibration/{entry['regime']}/{method}/{entry['labels_per_source']}/{rule}",
                                split='test', labels_per_source=entry['labels_per_source'],
                                label_budget_total=entry['label_budget_total'], selected_variant='B0')
                    if entry['status'] == 'unavailable':
                        rows.extend({**meta, 'metric': k, 'value': None, 'unavailable': entry['reason']}
                                    for k in METRICS)
                        continue
                    data = cell(entry['model'], 'B0', entry['benchmark'], 'test')[0]
                    p = module.transform(entry['parameters'], data['prob'])
                    threshold = .5 if rule == 'fixed_0.5' else entry['threshold']
                    metrics = module.diagnostics(data['y'], p, p >= threshold)
                    rows.extend({**meta, 'metric': k, 'value': metrics[k], 'n': len(data['y'])}
                                for k in METRICS)
    finally:
        analyze.payload = previous
    return rows
