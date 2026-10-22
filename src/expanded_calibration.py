"""E4: frozen nested label budgets, independent fit/tune, cached B0 tests."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import analyze
import run
from prepare import case, read_rows, write_frozen
from protocol import BENCHMARKS, MODELS, digest, payload, canonical

PROJECT = Path(__file__).resolve().parents[1]
DECISION_MODELS = [m for m in MODELS if m != 'luna']
PUBLICATION = {'dblp-acm', 'dblp-scholar', 'semi-heter'}
BUDGETS = [25, 100, 500]
METHODS = ['raw', 'temperature', 'logistic']


def read(path):
    return json.loads(path.read_text())


def threshold(y, p):
    """Exact F1 optimum; ties prefer nearest .5, then larger threshold."""
    y, p = np.asarray(y), np.asarray(p)
    if len(set(y)) < 2:
        raise ValueError('Threshold selection requires both classes')
    candidates = np.unique(np.r_[p, .5, np.nextafter(1., 2.)])
    order = np.argsort(p, kind='stable')
    indices = np.searchsorted(p[order], candidates, side='left')
    below = np.r_[0, np.cumsum(y[order])][indices]
    tp = y.sum()-below
    fp, fn = len(y)-indices-tp, below
    denominator = 2*tp+fp+fn
    f1 = np.divide(2*tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    winner = min(range(len(candidates)), key=lambda i: (-f1[i], abs(candidates[i]-.5), -candidates[i]))
    return float(candidates[winner])


def fit(method, y, p):
    y, p = np.asarray(y), np.asarray(p)
    if method == 'raw':
        return {'method': method}
    if len(set(y)) < 2:
        raise ValueError('Calibration fit requires both classes')
    z = logit(np.clip(p, 1e-6, 1-1e-6))
    if method == 'temperature':
        def loss(log_temperature):
            scaled = z / np.exp(log_temperature)
            return float(np.mean(np.logaddexp(0, scaled)-y*scaled))
        result = minimize_scalar(loss, bounds=(-5., 5.), method='bounded', options={'xatol': 1e-8})
        if not result.success:
            raise ValueError('Temperature optimization failed')
        return {'method': method, 'temperature': float(np.exp(result.x))}
    model = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, tol=1e-9).fit(z[:, None], y)
    if model.n_iter_[0] >= 1000:
        raise ValueError('Logistic calibration did not converge')
    return {'method': method, 'coefficient': float(model.coef_[0, 0]), 'intercept': float(model.intercept_[0])}


def transform(parameters, p):
    p = np.asarray(p)
    if parameters['method'] == 'raw':
        return p
    z = logit(np.clip(p, 1e-6, 1-1e-6))
    if parameters['method'] == 'temperature':
        return expit(z / parameters['temperature'])
    return expit(parameters['coefficient']*z + parameters['intercept'])


def partition(rows, n):
    """Input order is frozen hash order; label-independent and nested budgets."""
    if len(rows) < n:
        raise ValueError(f'Only {len(rows)} eligible pairs; budget {n} unavailable')
    # Alternating assignment preserves nested fit and tune sets across budgets.
    return np.arange(n)[::2], np.arange(n)[1::2]


def source_benchmarks(target, regime):
    if regime == 'within':
        return [target]
    if regime == 'leave_one_out':
        return [b for b in BENCHMARKS if b != target]
    return [b for b in BENCHMARKS if (b in PUBLICATION) != (target in PUBLICATION)]


def prepare():
    core = read(PROJECT/'configs/manifest.json')
    manifest = {'registration': 'Post-registered after preliminary B0 test inspection; no test-based tuning',
                'budgets': BUDGETS, 'methods': METHODS, 'models': DECISION_MODELS,
                'split': 'Independent alternating fit/tune halves of nested hash-ordered development pools',
                'threshold_rule': 'max positive F1; ties nearest .5, then larger threshold',
                'probability_clip_for_fitting': 1e-6, 'temperature_log_bounds': [-5, 5],
                'logistic_C': 1., 'logistic_class_weight': None,
                'transfer_budget': 'n labels per source benchmark; report total, never call pooled total n',
                'regimes': ['within', 'leave_one_out', 'opposite_domain'], 'benchmarks': {},
                'test_overlap': 'Exclude exact input overlap with any core test set from all new pools',
                'test_oracle': 'Retrospective diagnostic only; never used for selection',
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    test_inputs = set()
    for b in BENCHMARKS:
        test_inputs.update(c['input_hash'] for c in read(PROJECT/'data'/b/'test.json'))
    for b in BENCHMARKS:
        directory = PROJECT/'data/source'/b
        for source, expected in core['benchmarks'][b]['source_sha256'].items():
            assert hashlib.sha256((directory/Path(source).name).read_bytes()).hexdigest() == expected
        valid = directory/('wdcproducts80cc20rnd000un_valid_large.json.gz' if b == 'wdc' else b+'-valid.csv')
        rows = read_rows(valid)
        if set(rows[0]) == {'pair_id'}:
            source = directory/(b+'-valid.json.gz')
            if not source.exists():
                source = directory/(b+'-train.json')
            indexed = {str(r['pair_id']): r for r in read_rows(source)}
            rows = [indexed[str(r['pair_id'])] for r in rows]
        excluded = set(test_inputs)
        for split in ['prompt', 'expanded_screen']:
            path = PROJECT/'data'/b/(split+'.json')
            if path.exists():
                excluded.update(c['input_hash'] for c in read(path))
        unique, ambiguous = {}, set()
        for raw in rows:
            c = case(raw, b)
            h = c['input_hash']
            if h in excluded:
                continue
            if h in unique and unique[h]['label'] != c['label']:
                ambiguous.add(h)
            unique.setdefault(h, c)
        eligible = [c for h, c in unique.items() if h not in ambiguous]
        route = read(PROJECT/'data'/b/'route.json')
        by_hash = {c['input_hash']: c for c in eligible}
        reused = [c for c in route if c['input_hash'] in by_hash]
        reused_hashes = {c['input_hash'] for c in reused}
        extra = sorted([c for c in eligible if c['input_hash'] not in reused_hashes],
                       key=lambda c: digest(['E4-pool-v1', b, c['pair_id']]))[:max(0, 500-len(reused))]
        pool = sorted(reused+extra, key=lambda c: digest(['E4-nested-v1', b, c['pair_id']]))
        assert len({c['input_hash'] for c in pool}) == len(pool)
        assert not {c['input_hash'] for c in pool} & excluded
        write_frozen(PROJECT/'data'/b/'calibration_pool.json', pool)
        write_frozen(PROJECT/'data'/b/'calibration_extra.json', extra)
        manifest['benchmarks'][b] = {'n': len(pool), 'positive': sum(c['label'] for c in pool),
                                     'pool_sha256': digest(pool), 'extra_n': len(extra),
                                     'extra_sha256': digest(extra), 'reused_route_n': len(reused),
                                     'source_sha256': core['benchmarks'][b]['source_sha256'],
                                     'budgets': {str(n): {'available': len(pool) >= n,
                                         'positive': sum(c['label'] for c in pool[:n]) if len(pool) >= n else None}
                                         for n in BUDGETS}}
    manifest['new_decisions'] = len(DECISION_MODELS)*sum(v['extra_n'] for v in manifest['benchmarks'].values())
    example = {'benchmark': 'wdc', 'records': {'left': {}, 'right': {}}}
    manifest['requests'] = {m: payload(example, m, 'B0') for m in DECISION_MODELS}
    write_frozen(PROJECT/'configs/calibration_manifest.json', manifest)
    print(canonical({'new_decisions': manifest['new_decisions'], 'pools': {b: v['n'] for b, v in manifest['benchmarks'].items()}}), flush=True)


def verify():
    manifest = read(PROJECT/'configs/calibration_manifest.json')
    assert manifest['budgets'] == BUDGETS and manifest['methods'] == METHODS
    assert manifest['models'] == DECISION_MODELS
    assert manifest['implementation_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sample = {'benchmark': 'wdc', 'records': {'left': {}, 'right': {}}}
    assert manifest['requests'] == {m: payload(sample, m, 'B0') for m in DECISION_MODELS}
    for b, info in manifest['benchmarks'].items():
        for split, key in [('calibration_pool', 'pool_sha256'), ('calibration_extra', 'extra_sha256')]:
            assert digest(read(PROJECT/'data'/b/(split+'.json'))) == info[key]
        core = read(PROJECT/'configs/manifest.json')
        assert digest(read(PROJECT/'data'/b/'route.json')) == core['benchmarks'][b]['splits']['route']['sha256']
    return manifest


def development(model, b):
    pool = read(PROJECT/'data'/b/'calibration_pool.json')
    blocks = [analyze.load(model, 'B0', b, 'route')]
    if read(PROJECT/'data'/b/'calibration_extra.json'):
        blocks.append(analyze.load(model, 'B0', b, 'calibration_extra'))
    probabilities = {c['pair_id']: float(p) for block in blocks for c, p in zip(block['cases'], block['prob'])}
    return pool, np.array([c['label'] for c in pool]), np.array([probabilities[c['pair_id']] for c in pool])


def selection():
    entries = []
    for model in DECISION_MODELS:
        dev = {b: development(model, b) for b in BENCHMARKS}
        for target in BENCHMARKS:
            for regime in ['within', 'leave_one_out', 'opposite_domain']:
                sources = source_benchmarks(target, regime)
                for n in BUDGETS:
                    base = {'model': model, 'benchmark': target, 'regime': regime,
                            'labels_per_source': n, 'sources': sources, 'label_budget_total': n*len(sources)}
                    try:
                        fits, tunes = [], []
                        for b in sources:
                            rows, y, p = dev[b]
                            i, j = partition(rows, n)
                            fits.append((y[i], p[i]))
                            tunes.append((y[j], p[j]))
                        fy, fp = (np.concatenate([x[k] for x in fits]) for k in [0, 1])
                        ty, tp = (np.concatenate([x[k] for x in tunes]) for k in [0, 1])
                    except ValueError as exc:
                        entries.append({**base, 'method': 'all', 'status': 'unavailable', 'reason': str(exc)})
                        continue
                    for method in METHODS:
                        try:
                            parameters = fit(method, fy, fp)
                            tuned = threshold(ty, transform(parameters, tp))
                            entries.append({**base, 'method': method, 'status': 'frozen', 'parameters': parameters,
                                            'threshold': tuned, 'fit_n': len(fy), 'tune_n': len(ty),
                                            'fit_positive': int(fy.sum()), 'tune_positive': int(ty.sum())})
                        except ValueError as exc:
                            entries.append({**base, 'method': method, 'status': 'unavailable', 'reason': str(exc)})
    # All development decisions are persisted before loading any test predictions/labels.
    write_frozen(PROJECT/'configs/calibration_selection.json', entries)
    return entries


def diagnostics(y, p, pred):
    cal = analyze.calibration(y, p)
    error = np.asarray(y) != pred
    uncertainty = np.where(pred, 1-p, p)
    return {**analyze.metric(y, pred), 'average_precision': float(average_precision_score(y, p)),
            **{k: cal[k] for k in ['brier', 'log_loss', 'ece_10_equal_width']},
            'error_detection_auroc': float(roc_auc_score(error, uncertainty)) if len(set(error)) == 2 else None}


def report(entries):
    out = PROJECT/'artifacts/results/expanded_calibration'
    out.mkdir(parents=True, exist_ok=True)
    rows, curves, intervals = [], [], []
    tests = {(m, b): analyze.load(m, 'B0', b, 'test') for m in DECISION_MODELS for b in BENCHMARKS}
    for entry in entries:
        if entry['status'] != 'frozen':
            continue
        data = tests[(entry['model'], entry['benchmark'])]
        p = transform(entry['parameters'], data['prob'])
        meta = {k: entry[k] for k in ['model', 'benchmark', 'regime', 'method', 'labels_per_source', 'label_budget_total']}
        for rule, t in [('fixed_0.5', .5), ('development_F1', entry['threshold'])]:
            pred = p >= t
            rows.append({**meta, 'rule': rule, 'threshold': t, **diagnostics(data['y'], p, pred)})
        curves.append({**meta, 'reliability': analyze.calibration(data['y'], p),
                       'risk_coverage_at_fixed_0.5': analyze.risk_curve(data['y'], p >= .5, p)})
        if entry['regime'] == 'within':
            intervals.append({**meta, **analyze.bootstrap_delta(data['cases'], data['y'], p >= entry['threshold'], data['pred'])})
    oracles = []
    for (m, b), d in tests.items():
        t = threshold(d['y'], d['prob'])
        oracles.append({'model': m, 'benchmark': b, 'regime': 'RETROSPECTIVE_TEST_ORACLE_NOT_DEPLOYABLE',
                        'threshold': t, **analyze.metric(d['y'], d['prob'] >= t)})
    for name, value in [('metrics', rows), ('curves', curves), ('intervals', intervals), ('retrospective_oracle', oracles),
                        ('unavailable', [e for e in entries if e['status'] != 'frozen'])]:
        (out/(name+'.json')).write_text(canonical(value)+'\n')
    analyze.csv_save(out/'metrics.csv', rows)
    print('E4 complete', len(rows), 'metric rows;', len(entries)-sum(e['status'] == 'frozen' for e in entries), 'unavailable fits', flush=True)


def infer(model, manifest):
    for b, info in manifest['benchmarks'].items():
        if info['extra_n']:
            run.run_cell(model, 'B0', b, 'calibration_extra', workers=8)


def cost_forecast(manifest):
    rows = []
    for m in DECISION_MODELS:
        for b, info in manifest['benchmarks'].items():
            d = analyze.load(m, 'B0', b, 'route')
            rows.append({'model': m, 'benchmark': b, 'new_decisions': info['extra_n'],
                         'estimated_known_usd': float(d['costs'].mean()*info['extra_n']),
                         'estimated_unknown_allowance_usd': .05*d['unknown_cost_attempts']/len(d['cases'])*info['extra_n']})
    budget = read(PROJECT/'artifacts/budget.json')
    forecast = {'basis': 'Mean B0 D_route observed cost; unknown attempts conservatively charged .05; estimate, not invoice or guarantee',
                'budget_at_start': budget, 'shared_cap_usd': 100,
                'estimated_new_block_usd': sum(r['estimated_known_usd']+r['estimated_unknown_allowance_usd'] for r in rows),
                'rows': rows}
    path = PROJECT/'artifacts/calibration_cost_forecast.json'
    if not path.exists():
        path.write_text(canonical(forecast)+'\n')
    print('E4 COST FORECAST', canonical(forecast), flush=True)


def cost_report(manifest):
    rows = []
    for m in DECISION_MODELS:
        for b, info in manifest['benchmarks'].items():
            for split, reused in [('route', True), ('calibration_extra', False)]:
                if split == 'calibration_extra' and not info['extra_n']:
                    continue
                d = analyze.load(m, 'B0', b, split)
                rows.append({'model': m, 'benchmark': b, 'split': split, 'reused_core_calls': reused,
                             'n': len(d['cases']), 'known_cost_usd_including_failed_attempts': float(d['costs'].sum()),
                             'unknown_cost_attempts': d['unknown_cost_attempts']})
    (PROJECT/'artifacts/results/expanded_calibration/costs.json').write_text(canonical({
        'note': 'Reused D_route calls are already charged to the core and must not be billed twice. Full source-cell cost shown, not a charge per fitted condition.',
        'rows': rows})+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'run', 'analyze'])
    args = ap.parse_args()
    if args.mode == 'prepare':
        prepare()
        return
    manifest = verify()
    if args.mode == 'run':
        progress = read(PROJECT/'artifacts/progress.json')
        assert progress['expanded_E2_E3_screen']['status'] == 'complete'
        cost_forecast(manifest)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda m: infer(m, manifest), DECISION_MODELS))
    entries = selection()
    report(entries)
    cost_report(manifest)
    write_frozen(PROJECT/'artifacts/expanded_calibration_complete.json',
                 {'status': 'complete_with_explicit_unavailable_fits', 'selection_sha256': digest(entries),
                  'new_decisions': manifest['new_decisions'], 'full_expanded_study_complete': False})


if __name__ == '__main__':
    main()
