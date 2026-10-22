"""E5: development-frozen routers, cached full-test evaluation and fresh live chains."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
import analyze
import run
from expanded_calibration import fit as calibrate, transform
from prepare import write_frozen
from protocol import BENCHMARKS, MODELS, canonical, digest, payload

PROJECT = Path(__file__).resolve().parents[1]
DECISIONS = [m for m in MODELS if m != 'luna']
PEERS = {m: DECISIONS[(i+1) % len(DECISIONS)] for i, m in enumerate(DECISIONS)}
STRATEGIES = ['symmetric', 'asymmetric', 'calibrated_risk', 'disagreement', 'learned_gain']
FRACTIONS = [.1, .25, .5]


def read(path):
    return json.loads(path.read_text())


def tie(c):
    return digest(['E5-route-tie-v1', c['pair_id']])


def cutoff(cases, scores, n, eligible=None):
    eligible = np.ones(len(cases), dtype=bool) if eligible is None else eligible
    ranked = sorted([(float(s), tie(c)) for c, s, ok in zip(cases, scores, eligible) if ok], reverse=True)
    n = min(n, len(ranked))
    return {'score': ranked[n-1][0], 'tie': ranked[n-1][1], 'selected_dev': n} if n else None


def accept(cases, scores, boundary):
    if boundary is None:
        return np.zeros(len(cases), dtype=bool)
    return np.array([(float(s), tie(c)) >= (boundary['score'], boundary['tie']) for c, s in zip(cases, scores)])


def features(cases, probability):
    rows = []
    for c, p in zip(cases, probability):
        values = [v for side in c['records'].values() for v in side.values()]
        missing = sum(v is None or v == '' for v in values)/max(1, len(values))
        rows.append([float(p), min(p, 1-p), float(p >= .5), np.log1p(len(canonical(c['records']))), missing])
    return np.asarray(rows)


def learned_parameters(data, fallback):
    gain = (data['pred'] != data['y']).astype(float)-(fallback['pred'] != data['y']).astype(float)
    if len(set(gain)) < 2:
        raise ValueError('No variation in development fallback gain; learned router not identifiable')
    x = features(data['cases'], data['prob'])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.
    model = Ridge(alpha=1.).fit((x-mean)/scale, gain)
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'coefficient': model.coef_.tolist(),
            'intercept': float(model.intercept_), 'target': 'decision error minus fallback error on D_prompt'}


def score(entry, cases, probability, peer_probability=None):
    p = np.asarray(probability)
    strategy = entry['strategy']
    if strategy in ['symmetric', 'asymmetric']:
        return np.minimum(p, 1-p)
    if strategy == 'calibrated_risk':
        calibrated = transform(entry['calibration'], p)
        # Acceptance still uses the original B0 class, not a recalibrated class.
        return np.where(p >= .5, 1-calibrated, calibrated)
    if strategy == 'disagreement':
        peer = np.asarray(peer_probability)
        return ((p >= .5) != (peer >= .5)).astype(float)*(1+np.abs(p-peer))
    parameters = entry['learned']
    x = (features(cases, p)-parameters['mean'])/parameters['scale']
    return x @ np.asarray(parameters['coefficient'])+parameters['intercept']


def route(entry, cases, probability, peer_probability=None):
    values = score(entry, cases, probability, peer_probability)
    if entry['strategy'] == 'asymmetric':
        prediction = np.asarray(probability) >= .5
        return ((~prediction) & accept(cases, values, entry['negative_cutoff'])) | (prediction & accept(cases, values, entry['positive_cutoff']))
    mask = accept(cases, values, entry['cutoff'])
    if entry['strategy'] == 'disagreement':
        mask &= (np.asarray(probability) >= .5) != (np.asarray(peer_probability) >= .5)
    return mask


def asymmetric_selection(data, fallback, n):
    p = data['prob']
    positive = p >= .5
    uncertain = np.minimum(p, 1-p)
    candidates = []
    for positive_n in range(max(0, n-int((~positive).sum())), min(n, int(positive.sum()))+1):
        entry = {'strategy': 'asymmetric',
                 'positive_cutoff': cutoff(data['cases'], uncertain, positive_n, positive),
                 'negative_cutoff': cutoff(data['cases'], uncertain, n-positive_n, ~positive)}
        mask = route(entry, data['cases'], p)
        value = analyze.metric(data['y'], np.where(mask, fallback['pred'], data['pred']))['f1']
        candidates.append((value, positive_n, entry))
    # Fixed tie rule favors the smaller positive-prediction routing allocation.
    value, positive_n, entry = min(candidates, key=lambda x: (-x[0], x[1]))
    return {**entry, 'selected_dev_f1': value, 'positive_dev_routes': positive_n, 'negative_dev_routes': n-positive_n}


def prepare():
    core = read(PROJECT/'configs/manifest.json')
    supplement = read(PROJECT/'configs/billiger50_manifest.json')
    manifest = {'registration': 'Post-registered extension; no test-result-based router selection',
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'calibration_implementation_sha256': hashlib.sha256((PROJECT/'src/expanded_calibration.py').read_bytes()).hexdigest(),
                'core_manifest_sha256': digest(core), 'supplement_manifest_sha256': digest(supplement),
                'strategies': STRATEGIES, 'target_fractions': FRACTIONS, 'models': DECISIONS,
                'fixed_disagreement_peers': PEERS, 'peer_rule': 'Next Decision Model in the frozen protocol order; no quality-based partner selection',
                'calibration_fit': 'Logistic C=1, original B0 D_prompt; thresholds on disjoint D_route',
                'learned_gain_fit': 'Ridge alpha=1, standardized [p, min(p,1-p), B0 class, log1p descriptive input length, missing-field fraction], trained on D_prompt fallback gain; thresholds on D_route',
                'asymmetric_fit': 'D_route allocation of the fixed quota across B0 positive/negative classes, maximizing cascade positive F1; ties fewer positive routes',
                'disagreement_quota': 'At most target fraction on D_route; only disagreeing pairs eligible, never fill with agreeing pairs',
                'selection_persistence': 'Freeze every strategy and fraction before loading test predictions',
                'oracle': 'Exact retrospective best positive F1 using at most the same number of fallback calls; labels never used by deployable routes',
                'random_control': '20 seeds with exactly the same test route count; same Decision and peer overhead',
                'intervals': 'Paired component and pair bootstrap at prespecified primary .25 quota; other quotas descriptive pending full statistics',
                'live': {'strategies': STRATEGIES, 'fraction': .25, 'cases_per_family': 20,
                         'families': list(BENCHMARKS), 'sample': 'First 20 D_route cases by hash [E5-live-v1, benchmark, pair_id]',
                         'concurrent_chains': 6, 'within_chain': 'Sequential fresh primary, optional peer, then routed Luna; no quality cache hits',
                         'failure_policy': 'Stop on a failed chain. Resume only complete-chain successes; rerun interrupted chains and retain all failed call charges.',
                         'maximum_calls_before_retries': 10560},
                'data': {}}
    for b in BENCHMARKS:
        manifest['data'][b] = {}
        for split in ['prompt', 'route', 'test']:
            cases = read(PROJECT/'data'/b/(split+'.json'))
            assert digest(cases) == core['benchmarks'][b]['splits'][split]['sha256']
            manifest['data'][b][split] = digest(cases)
        cases = read(PROJECT/'data'/b/'route.json')
        sample = sorted(cases, key=lambda c: digest(['E5-live-v1', b, c['pair_id']]))[:20]
        write_frozen(PROJECT/'data'/b/'cascade_live.json', sample)
        manifest['data'][b]['cascade_live'] = digest(sample)
    cases = read(PROJECT/'data/billiger-de-50un/test.json')
    assert digest(cases) == supplement['test_sha256']
    manifest['data']['billiger-de-50un'] = {'test': digest(cases)}
    write_frozen(PROJECT/'configs/expanded_cascade_manifest.json', manifest)
    print('E5 fixed strategy manifest frozen', flush=True)


def verify():
    manifest = read(PROJECT/'configs/expanded_cascade_manifest.json')
    assert manifest['implementation_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert manifest['calibration_implementation_sha256'] == hashlib.sha256((PROJECT/'src/expanded_calibration.py').read_bytes()).hexdigest()
    assert manifest['core_manifest_sha256'] == digest(read(PROJECT/'configs/manifest.json'))
    assert manifest['supplement_manifest_sha256'] == digest(read(PROJECT/'configs/billiger50_manifest.json'))
    for b, splits in manifest['data'].items():
        for split, expected in splits.items():
            assert digest(read(PROJECT/'data'/b/(split+'.json'))) == expected
    return manifest


def select():
    selected = []
    for b in BENCHMARKS:
        train = {m: analyze.load(m, 'B0', b, 'prompt') for m in MODELS}
        dev = {m: analyze.load(m, 'B0', b, 'route') for m in MODELS}
        for m in DECISIONS:
            for strategy in STRATEGIES:
                base = {'model': m, 'benchmark': b, 'strategy': strategy, 'status': 'frozen',
                        'peer': PEERS[m] if strategy == 'disagreement' else None}
                try:
                    if strategy == 'calibrated_risk':
                        base['calibration'] = calibrate('logistic', train[m]['y'], train[m]['prob'])
                    if strategy == 'learned_gain':
                        base['learned'] = learned_parameters(train[m], train['luna'])
                except ValueError as exc:
                    selected.append({**base, 'status': 'unavailable', 'reason': str(exc)})
                    continue
                for fraction in FRACTIONS:
                    n = round(len(dev[m]['cases'])*fraction)
                    entry = {**base, 'fraction': fraction}
                    if strategy == 'asymmetric':
                        entry.update(asymmetric_selection(dev[m], dev['luna'], n))
                    else:
                        values = score(entry, dev[m]['cases'], dev[m]['prob'], dev[PEERS[m]]['prob'])
                        eligible = values > 0 if strategy == 'disagreement' else None
                        entry['cutoff'] = cutoff(dev[m]['cases'], values, n, eligible)
                    entry['actual_dev_routes'] = int(route(entry, dev[m]['cases'], dev[m]['prob'], dev[PEERS[m]]['prob']).sum())
                    entry['dev_n'] = len(dev[m]['cases'])
                    selected.append(entry)
    write_frozen(PROJECT/'configs/expanded_cascade_selection.json', selected)
    return selected


def oracle_mask(cases, y, prediction, fallback, quota):
    """Optimal F1 over useful FN/FP corrections under an at-most-quota constraint."""
    eligible = (prediction != y) & (fallback == y)
    fn = sorted(np.flatnonzero(eligible & (y == 1)), key=lambda i: tie(cases[i]))
    fp = sorted(np.flatnonzero(eligible & (y == 0)), key=lambda i: tie(cases[i]))
    base = analyze.metric(y, prediction)
    candidates = []
    for nfn in range(min(quota, len(fn))+1):
        nfp = min(quota-nfn, len(fp))
        tp, falsep, falsen = base['tp']+nfn, base['fp']-nfp, base['fn']-nfn
        f1 = 2*tp/(2*tp+falsep+falsen) if 2*tp+falsep+falsen else 0.
        candidates.append((f1, nfn+nfp, nfn, nfp))
    _, _, nfn, nfp = max(candidates)
    mask = np.zeros(len(y), dtype=bool)
    mask[fn[:nfn]+fp[:nfp]] = True
    return mask


def report(selected):
    rows, controls, intervals, oracles, provenance = [], [], [], [], []
    for b in [*BENCHMARKS, 'billiger-de-50un']:
        cells = {m: analyze.load(m, 'B0', b, 'test') for m in MODELS}
        source = 'billiger-de' if b == 'billiger-de-50un' else b
        luna = cells['luna']
        for m, d in cells.items():
            provenance.append({'benchmark': b, 'model': m, 'source': 'existing B0 cache; no new paid calls',
                               'known_cost_usd': float(d['costs'].sum()), 'unknown_cost_attempts': d['unknown_cost_attempts']})
            controls.append({'benchmark': b, 'model': m, **analyze.metric(d['y'], d['pred'])})
        for entry in selected:
            if entry['benchmark'] != source or entry['status'] != 'frozen':
                continue
            m = entry['model']; d = cells[m]; peer = cells[entry['peer']] if entry['peer'] else None
            mask = route(entry, d['cases'], d['prob'], peer['prob'] if peer else None)
            pred = np.where(mask, luna['pred'], d['pred'])
            overhead = d['costs']+(peer['costs'] if peer else 0.)
            unknown = d['unknown_cost_by_pair']+(peer['unknown_cost_by_pair'] if peer else 0)
            random_results = []
            for seed in range(20):
                random_mask = np.zeros(len(mask), dtype=bool)
                random_mask[np.random.default_rng(seed).choice(len(mask), int(mask.sum()), replace=False)] = True
                random_results.append({'seed': seed, 'f1': analyze.metric(d['y'], np.where(random_mask, luna['pred'], d['pred']))['f1'],
                                       'known_cost_usd': float((overhead+np.where(random_mask, luna['costs'], 0.)).sum()),
                                       'unknown_cost_attempts': int(unknown.sum()+luna['unknown_cost_by_pair'][random_mask].sum())})
            meta = {'model': m, 'peer': entry['peer'], 'benchmark': b, 'source_development': source,
                    'strategy': entry['strategy'], 'target_fraction': entry['fraction']}
            missing_costs = int(unknown.sum()+luna['unknown_cost_by_pair'][mask].sum())
            rows.append({**meta, 'dev_routes': entry['actual_dev_routes'], 'dev_n': entry['dev_n'],
                         'test_routes': int(mask.sum()), 'test_fraction': float(mask.mean()),
                         **analyze.metric(d['y'], pred), 'known_cost_usd': float((overhead+np.where(mask, luna['costs'], 0.)).sum()),
                         'unknown_cost_attempts': missing_costs, 'luna_control_unknown_cost_attempts': luna['unknown_cost_attempts'],
                         'all_comparison_costs_known': missing_costs == 0 and luna['unknown_cost_attempts'] == 0,
                         'corrected_errors': int((mask & (d['pred'] != d['y']) & (luna['pred'] == d['y'])).sum()),
                         'introduced_errors': int((mask & (d['pred'] == d['y']) & (luna['pred'] != d['y'])).sum()),
                         'random_same_route_count': random_results})
            oracle = oracle_mask(d['cases'], d['y'], d['pred'], luna['pred'], int(mask.sum()))
            oracles.append({**meta, 'interpretation': 'RETROSPECTIVE TEST ORACLE; not deployable',
                            'quota_at_most': int(mask.sum()), 'oracle_routes': int(oracle.sum()),
                            **analyze.metric(d['y'], np.where(oracle, luna['pred'], d['pred']))})
            if entry['fraction'] == .25:
                intervals.append({**meta, 'vs_luna': analyze.bootstrap_delta(d['cases'], d['y'], pred, luna['pred']),
                                  'vs_decision_only': analyze.bootstrap_delta(d['cases'], d['y'], pred, d['pred'])})
    out = PROJECT/'artifacts/results/expanded_cascades'
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('metrics', rows), ('controls', controls), ('primary_intervals', intervals), ('test_oracles', oracles),
                        ('reused_test_cost_provenance', provenance), ('unavailable', [e for e in selected if e['status'] != 'frozen'])]:
        (out/(name+'.json')).write_text(canonical(value)+'\n')
    analyze.csv_save(out/'metrics.csv', [{k: v for k, v in r.items() if k != 'random_same_route_count'} for r in rows])
    write_frozen(PROJECT/'artifacts/expanded_cascades_offline_complete.json',
                 {'status': 'complete_with_explicit_unavailable_routers', 'selection_sha256': digest(selected),
                  'metric_rows': len(rows), 'new_api_calls': 0, 'live_latency_complete': False})
    print('E5 OFFLINE COMPLETE', len(rows), 'rows; live latency remains separate', flush=True)


def execute_chain(entry, case, credential, observe=None):
    calls = []
    def record(call):
        calls.append(call)
        if observe is not None:
            observe(call, len(calls)-1)
    start = time.perf_counter()
    primary = None; peer = None; routed = False; fallback = None
    try:
        primary = run.request(case, entry['model'], 'B0', credential); record(primary)
        if not primary.get('error') and entry['peer']:
            peer = run.request(case, entry['peer'], 'B0', credential); record(peer)
        if not any(c.get('error') for c in calls):
            routed = bool(route(entry, [case], [primary['probability_match']], [peer['probability_match']] if peer else None)[0])
            if routed:
                fallback = run.request(case, 'luna', 'B0', credential); record(fallback)
    except Exception as exc:
        # A ceiling refusal between stages must not discard already paid primary/peer calls.
        record({'error': type(exc).__name__, 'attempts': []})
    elapsed = time.perf_counter()-start
    failed = any(c.get('error') for c in calls)
    return {'pair_id': case['pair_id'], 'router_hash': digest(entry), 'input_hash': case['input_hash'],
            'primary_request_hash': digest(payload(case, entry['model'], 'B0')), 'calls': calls,
            'wall_seconds': elapsed, 'routed': routed, 'failed': failed,
            'prediction': None if failed else (fallback or primary)['prediction']}


def live_task(model, selected):
    credential = run.key()
    for entry in selected:
        if entry['model'] != model or entry['status'] != 'frozen' or entry['fraction'] != .25:
            continue
        b, strategy = entry['benchmark'], entry['strategy']
        cases = read(PROJECT/'data'/b/'cascade_live.json')
        directory = PROJECT/'runs/cascade_live'/model/strategy/b
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/'chains.jsonl'
        with (directory/'writer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            cached = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
            by_id = {c['pair_id']: c for c in cached}
            assert len(by_id) == len(cached)
            for c in cases:
                if c['pair_id'] in by_id:
                    prior = by_id[c['pair_id']]
                    assert prior['router_hash'] == digest(entry) and prior['input_hash'] == c['input_hash']
                    assert prior['primary_request_hash'] == digest(payload(c, model, 'B0')) and not prior['failed']
                    continue
                chain_id = str(uuid.uuid4())
                def observe(call, call_index):
                    # Persist each returned paid call before requesting the next stage.
                    with (directory/'call_journal.jsonl').open('a') as journal:
                        journal.write(canonical({'chain_id': chain_id, 'pair_id': c['pair_id'],
                                                 'router_hash': digest(entry), 'call_index': call_index, 'call': call})+'\n')
                result = {**execute_chain(entry, c, credential, observe), 'chain_id': chain_id}
                dest = directory/'failed_chains.jsonl' if result['failed'] else path
                with dest.open('a') as handle:
                    handle.write(canonical(result)+'\n')
                if result['failed']:
                    raise RuntimeError(f'Live cascade failed: {model}/{strategy}/{b}; prior paid calls retained')
            print('LIVE CELL COMPLETE', model, strategy, b, len(cases), flush=True)


def live_report(selected):
    rows = []
    for e in selected:
        if e['status'] != 'frozen' or e['fraction'] != .25:
            continue
        cases = read(PROJECT/'data'/e['benchmark']/'cascade_live.json')
        directory = PROJECT/'runs/cascade_live'/e['model']/e['strategy']/e['benchmark']
        successful = [json.loads(s) for s in (directory/'chains.jsonl').read_text().splitlines()]
        assert len(successful) == len(cases) and {s['pair_id'] for s in successful} == {c['pair_id'] for c in cases}
        indexed = {c['pair_id']: c for c in cases}
        for s in successful:
            c = indexed[s['pair_id']]
            assert s['router_hash'] == digest(e) and s['input_hash'] == c['input_hash'] and not s['failed']
            assert s['primary_request_hash'] == digest(payload(c, e['model'], 'B0'))
        failure = directory/'failed_chains.jsonl'
        failed = [json.loads(s) for s in failure.read_text().splitlines()] if failure.exists() else []
        journal = [json.loads(s) for s in (directory/'call_journal.jsonl').read_text().splitlines()]
        assert len({(j['chain_id'], j['call_index']) for j in journal}) == len(journal)
        finished_ids = {s['chain_id'] for s in successful+failed}
        orphaned_ids = {j['chain_id'] for j in journal}-finished_ids
        for chain in successful+failed:
            logged = sorted([j for j in journal if j['chain_id'] == chain['chain_id']], key=lambda j: j['call_index'])
            assert digest([j['call'] for j in logged]) == digest(chain['calls'])
        attempts = [a for j in journal for a in j['call']['attempts']]
        duration = [s['wall_seconds'] for s in successful]
        rows.append({'model': e['model'], 'benchmark': e['benchmark'], 'strategy': e['strategy'],
                     'n': len(successful), 'routed': sum(s['routed'] for s in successful),
                     'end_to_end_p50_seconds': float(np.quantile(duration, .5)),
                     'end_to_end_p95_seconds': float(np.quantile(duration, .95)),
                     'failed_chains': len(failed), 'failed_chain_wall_seconds': sum(s['wall_seconds'] for s in failed),
                     'interrupted_chains_with_journaled_calls': len(orphaned_ids),
                     'latency_scope': 'Successful complete fresh chains; interrupted-chain end-to-end latency unavailable',
                     'known_cost_usd_including_failures': sum(a['cost'] for a in attempts if a['cost'] is not None),
                     'unknown_cost_attempts': sum(a['cost'] is None for a in attempts), 'concurrent_chains_configured': 6})
    out = PROJECT/'artifacts/results/expanded_cascades'
    (out/'live_latency.json').write_text(canonical(rows)+'\n')
    analyze.csv_save(out/'live_latency.csv', rows)
    write_frozen(PROJECT/'artifacts/expanded_cascades_live_complete.json', {'status': 'complete', 'cells': len(rows)})


def live_forecast(selected):
    references = {(b, m): analyze.load(m, 'B0', b, 'prompt') for b in BENCHMARKS for m in MODELS}
    estimated, worst_case, calls = 0., 0., 0
    for e in selected:
        if e['status'] != 'frozen' or e['fraction'] != .25:
            continue
        def mean_cost(m):
            d = references[(e['benchmark'], m)]
            return (float(d['costs'].sum())+.05*d['unknown_cost_attempts'])/len(d['cases'])
        fixed = mean_cost(e['model'])+(mean_cost(e['peer']) if e['peer'] else 0.)
        rate = e['actual_dev_routes']/e['dev_n']
        estimated += 20*(fixed+rate*mean_cost('luna'))
        worst_case += 20*(fixed+mean_cost('luna'))
        calls += 20*(2+int(bool(e['peer'])))
    value = {'basis': 'D_prompt mean observed charges plus .05 allowance per unknown attempt, before live cache reuse',
             'estimated_usd_using_dev_route_rate': estimated, 'estimated_usd_if_every_pair_routes': worst_case,
             'maximum_new_calls_before_retries_and_cache': calls,
             'budget_snapshot': read(PROJECT/'artifacts/budget.json'), 'project_cap_usd': 100,
             'caveat': 'These are cost estimates, not guaranteed bounds; shared per-call budget guard remains active.'}
    path = PROJECT/'artifacts/expanded_cascade_live_cost_forecast.json'
    if not path.exists():
        path.write_text(canonical(value)+'\n')
    print('LIVE COST FORECAST', canonical(value), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'analyze', 'live'])
    args = ap.parse_args()
    if args.mode == 'prepare':
        prepare(); return
    verify()
    if args.mode == 'analyze':
        assert read(PROJECT/'artifacts/progress.json')['supplement_billiger50']['status'] == 'complete_automated'
        report(select())
        return
    assert read(PROJECT/'artifacts/expanded_cascades_offline_complete.json')['status'] == 'complete_with_explicit_unavailable_routers'
    selected = read(PROJECT/'configs/expanded_cascade_selection.json')
    live_forecast(selected)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda m: live_task(m, selected), DECISIONS))
    live_report(selected)


if __name__ == '__main__':
    main()
