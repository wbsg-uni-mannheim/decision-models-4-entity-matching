"""Read-only E5 reconstruction: all development fits before any test access."""
import json
import math
import numpy as np
from live_cascade_export import check_call

MAIN = ['f1', 'precision', 'recall', 'accuracy', 'routed_fraction',
        'known_cost_usd_per_1000', 'unknown_attempts_per_1000',
        'corrected_error_fraction', 'introduced_error_fraction']


def read(path):
    return json.loads(path.read_text())


def compare(a, b, where='selection', tolerance=1e-12):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError(where+' fields differ')
        for k, v in a.items():
            compare(v, b[k], where+'/'+k, tolerance)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError(where+' length differs')
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, where+'/'+str(i), tolerance)
    elif isinstance(a, float):
        if not isinstance(b, (int, float)) or not math.isfinite(b) or not math.isclose(a, b, abs_tol=tolerance, rel_tol=tolerance):
            raise ValueError(where+' numeric drift')
    elif a != b:
        raise ValueError(where+' differs: '+repr(a)+' vs '+repr(b))


def aligned(cells):
    ref = next(iter(cells.values()))
    for d in cells.values():
        if d['cases'] != ref['cases'] or not np.array_equal(d['y'], ref['y']):
            raise ValueError('Cascade source pair/record/label alignment differs')


def reconstruct(module, data, saved):
    expected = []
    for b in module.BENCHMARKS:
        train = {m: data[m, b, 'prompt'] for m in module.MODELS}
        dev = {m: data[m, b, 'route'] for m in module.MODELS}
        aligned(train); aligned(dev)
        for m in module.DECISIONS:
            for strategy in module.STRATEGIES:
                base = dict(model=m, benchmark=b, strategy=strategy, status='frozen',
                            peer=module.PEERS[m] if strategy == 'disagreement' else None)
                try:
                    if strategy == 'calibrated_risk':
                        base['calibration'] = module.calibrate('logistic', train[m]['y'], train[m]['prob'])
                    if strategy == 'learned_gain':
                        base['learned'] = module.learned_parameters(train[m], train['luna'])
                except ValueError as exc:
                    expected.append({**base, 'status': 'unavailable', 'reason': str(exc)})
                    continue
                candidate = saved[len(expected)] if len(expected) < len(saved) else {}
                for parameter in ['calibration', 'learned']:
                    if parameter in base:
                        compare(base[parameter], candidate.get(parameter), 'fit/'+parameter, 1e-6)
                        base[parameter] = candidate[parameter]  # evaluate exactly frozen parameters
                for fraction in module.FRACTIONS:
                    entry = {**base, 'fraction': fraction}
                    n = round(len(dev[m]['cases'])*fraction)
                    if strategy == 'asymmetric':
                        entry.update(module.asymmetric_selection(dev[m], dev['luna'], n))
                    else:
                        values = module.score(entry, dev[m]['cases'], dev[m]['prob'], dev[module.PEERS[m]]['prob'])
                        entry['cutoff'] = module.cutoff(dev[m]['cases'], values, n, values > 0 if strategy == 'disagreement' else None)
                    entry['actual_dev_routes'] = int(module.route(entry, dev[m]['cases'], dev[m]['prob'], dev[module.PEERS[m]]['prob']).sum())
                    entry['dev_n'] = len(dev[m]['cases'])
                    expected.append(entry)
    compare(expected, saved)


def evaluate(entry, cells, module, analyze):
    aligned(cells)
    d, luna = cells[entry['model']], cells['luna']
    peer = cells[entry['peer']] if entry['peer'] else None
    mask = module.route(entry, d['cases'], d['prob'], peer['prob'] if peer else None)
    overhead = d['costs']+(peer['costs'] if peer else 0.)
    unknown = d['unknown_cost_by_pair']+(peer['unknown_cost_by_pair'] if peer else 0)
    def cost(route):
        return float((overhead+np.where(route, luna['costs'], 0.)).sum()), int(unknown.sum()+luna['unknown_cost_by_pair'][route].sum())
    pred = np.where(mask, luna['pred'], d['pred'])
    known, missing = cost(mask)
    metrics = analyze.metric(d['y'], pred)
    corrected = int((mask & (d['pred'] != d['y']) & (luna['pred'] == d['y'])).sum())
    introduced = int((mask & (d['pred'] == d['y']) & (luna['pred'] != d['y'])).sum())
    random = []
    for seed in range(20):
        selected = np.zeros(len(mask), dtype=bool)
        selected[np.random.default_rng(seed).choice(len(mask), int(mask.sum()), replace=False)] = True
        k, u = cost(selected)
        random.append(dict(seed=seed, f1=analyze.metric(d['y'], np.where(selected, luna['pred'], d['pred']))['f1'], known_cost_usd=k, unknown_cost_attempts=u))
    oracle = module.oracle_mask(d['cases'], d['y'], d['pred'], luna['pred'], int(mask.sum()))
    return {**metrics, 'test_routes': int(mask.sum()), 'test_fraction': float(mask.mean()),
            'known_cost_usd': known, 'unknown_cost_attempts': missing,
            'corrected_errors': corrected, 'introduced_errors': introduced,
            'random_same_route_count': random}, {
                **analyze.metric(d['y'], np.where(oracle, luna['pred'], d['pred'])),
                'quota_at_most': int(mask.sum()), 'oracle_routes': int(oracle.sum())}


def normalized(run_id, entry, benchmark, result, oracle):
    condition = 'cascades/'+entry['strategy']+'/'+str(entry['fraction'])
    meta = dict(run_id=run_id, model=entry['model'], benchmark=benchmark, split='test', selected_variant='B0')
    if result is None:
        values = dict.fromkeys(MAIN)
        n = None
    else:
        n = result['n']
        values = {k: result[k] for k in MAIN[:4]}
        values.update(routed_fraction=result['test_fraction'], known_cost_usd_per_1000=1000*result['known_cost_usd']/n,
                      unknown_attempts_per_1000=1000*result['unknown_cost_attempts']/n,
                      corrected_error_fraction=result['corrected_errors']/n, introduced_error_fraction=result['introduced_errors']/n)
    rows = [{**meta, 'condition': condition, 'n': n, 'metric': k, 'value': v} for k, v in values.items()]
    for seed in range(20):
        random = None if result is None else result['random_same_route_count'][seed]
        values = {'f1': None if random is None else random['f1'],
                  'known_cost_usd_per_1000': None if random is None else 1000*random['known_cost_usd']/n,
                  'unknown_attempts_per_1000': None if random is None else 1000*random['unknown_cost_attempts']/n}
        rows.extend({**meta, 'condition': condition+'/random/'+str(seed), 'n': n, 'metric': k, 'value': v} for k, v in values.items())
    values = {'f1': None if oracle is None else oracle['f1'],
              'routed_fraction': None if oracle is None else oracle['oracle_routes']/n}
    rows.extend({**meta, 'condition': condition+'/retrospective_oracle', 'n': n, 'metric': k, 'value': v} for k, v in values.items())
    if result is None:
        for row in rows:
            row['unavailable'] = entry['reason']
    return rows


def export_cascades(workspace, module, analyze, protocol, cell, origin, run_id):
    marker = read(workspace/'artifacts/expanded_cascades_offline_complete.json')
    selected = read(workspace/'configs/expanded_cascade_selection.json')
    if (marker.get('status') != 'complete_with_explicit_unavailable_routers'
            or marker.get('selection_sha256') != protocol.digest(selected)):
        raise ValueError('Incomplete cascades or changed selection')
    module.verify()
    def load(m, b, split):
        d = cell(m, 'B0', b, split)[0]
        indexed = {c['pair_id']: c for c in d['cases']}
        for row in d['responses']:
            check_call(row, indexed[row['pair_id']], m, protocol, origin)
        failure = workspace/'runs/quality'/m/'B0'/b/(split+'.failures.jsonl')
        if failure.exists():
            for row in map(json.loads, failure.read_text().splitlines()):
                check_call(row, indexed[row['pair_id']], m, protocol, origin)
        return d
    development = {(m, b, split): load(m, b, split) for b in module.BENCHMARKS for m in module.MODELS for split in ['prompt', 'route']}
    reconstruct(module, development, selected)
    saved = read(workspace/'artifacts/results/expanded_cascades/metrics.json')
    oracles = read(workspace/'artifacts/results/expanded_cascades/test_oracles.json')
    key = lambda r: (r['model'], r['benchmark'], r['strategy'], r['target_fraction'])
    expected_keys = {(e['model'], b, e['strategy'], e['fraction']) for b in [*module.BENCHMARKS, 'billiger-de-50un']
                     for e in selected if e['status'] == 'frozen' and e['benchmark'] == ('billiger-de' if b == 'billiger-de-50un' else b)}
    if marker['metric_rows'] != len(expected_keys):
        raise ValueError('Cascade completion count differs')
    for records in [saved, oracles]:
        if len(records) != len(expected_keys) or {key(r) for r in records} != expected_keys:
            raise ValueError('Cascade archived matrix differs')
    saved, oracles = ({key(r): r for r in records} for records in [saved, oracles])
    output = []
    for b in [*module.BENCHMARKS, 'billiger-de-50un']:
        cells = {m: load(m, b, 'test') for m in module.MODELS}
        for entry in selected:
            if entry['benchmark'] != ('billiger-de' if b == 'billiger-de-50un' else b):
                continue
            if entry['status'] != 'frozen':
                for fraction in module.FRACTIONS:
                    output.extend(normalized(run_id, {**entry, 'fraction': fraction}, b, None, None))
                continue
            result, oracle = evaluate(entry, cells, module, analyze)
            k = entry['model'], b, entry['strategy'], entry['fraction']
            compare(result, {field: saved[k][field] for field in result}, 'metrics/'+str(k))
            compare(oracle, {field: oracles[k][field] for field in oracle}, 'oracle')
            output.extend(normalized(run_id, entry, b, result, oracle))
    return output
