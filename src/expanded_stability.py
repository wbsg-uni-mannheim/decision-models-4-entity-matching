"""Fixed E3 prompt x format design and two fresh repeated B0 runs."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import analyze
import run
from expanded_protocol import payload as expanded_payload
from prepare import write_frozen
from protocol import MODELS, canonical, digest, state

PROJECT = Path(__file__).resolve().parents[1]
SCREEN_BENCHMARKS = ['abt-buy', 'wdc', 'dblp-scholar']
INTERACTIONS = {'I3_identity_json': 'X_P_identity_0', 'I3_domain_json': 'X_P_domain_0'}


def read(path):
    return json.loads(path.read_text())


def payload(case, alias, variant='B0'):
    if variant not in INTERACTIONS:
        return expanded_payload(case, alias, variant)
    body = expanded_payload(case, alias, INTERACTIONS[variant])
    text = state(case, 'S1')
    if 'questions' in body:
        body['state'] = text
    else:
        body['messages'][1]['content'] = text
    return body


def prepare():
    screen = read(PROJECT/'configs/expanded_screen_manifest.json')
    manifest = {'registration': 'Post-registered fixed factorial contrasts; no test-based selection',
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'expanded_protocol_sha256': hashlib.sha256((PROJECT/'src/expanded_protocol.py').read_bytes()).hexdigest(),
                'screen_manifest_sha256': digest(screen), 'models': MODELS,
                'interaction_design': {'prompts': ['identity_0', 'domain_0'], 'formats': ['text', 'JSON'],
                                       'text_cells': 'Reuse unchanged broad-screen cells',
                                       'contrast': '(domain_JSON - domain_text) - (identity_JSON - identity_text), positive-class F1'},
                'repeats': {'n': 100, 'additional_runs': 2, 'baseline': 'Same cases from broad-screen B0',
                            'fresh_namespaces': ['e3_repeat_1', 'e3_repeat_2'],
                            'sampling': 'First 100 by hash of [E3-repeat-v1, benchmark, pair_id]; no label stratification'},
                'new_decisions_before_cache': len(MODELS)*len(SCREEN_BENCHMARKS)*(2*500+2*100),
                'benchmarks': {}, 'requests': {}}
    for b in SCREEN_BENCHMARKS:
        cases = read(PROJECT/'data'/b/'expanded_screen.json')
        assert digest(cases) == screen['benchmarks'][b]['sha256'] and len(cases) == 500
        repeat = sorted(cases, key=lambda c: digest(['E3-repeat-v1', b, c['pair_id']]))[:100]
        write_frozen(PROJECT/'data'/b/'expanded_repeat.json', repeat)
        manifest['benchmarks'][b] = {'repeat_sha256': digest(repeat), 'screen_sha256': digest(cases),
                                     'repeat_positive': sum(c['label'] for c in repeat)}
        sample = {'benchmark': b, 'records': {'left': {}, 'right': {}}}
        manifest['requests'][b] = {m: {v: payload(sample, m, v) for v in INTERACTIONS} for m in MODELS}
    write_frozen(PROJECT/'configs/expanded_stability_manifest.json', manifest)
    print('E3 FIXED DESIGN FROZEN', manifest['new_decisions_before_cache'], 'additional calls before cache', flush=True)


def verify():
    manifest = read(PROJECT/'configs/expanded_stability_manifest.json')
    assert manifest['implementation_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert manifest['expanded_protocol_sha256'] == hashlib.sha256((PROJECT/'src/expanded_protocol.py').read_bytes()).hexdigest()
    assert manifest['screen_manifest_sha256'] == digest(read(PROJECT/'configs/expanded_screen_manifest.json'))
    for b, info in manifest['benchmarks'].items():
        for split, key in [('expanded_repeat', 'repeat_sha256'), ('expanded_screen', 'screen_sha256')]:
            assert digest(read(PROJECT/'data'/b/(split+'.json'))) == info[key]
    return manifest


def repeat_summary(baseline, first, second):
    """Align via pair IDs, rejecting missing/duplicate cases rather than row-order assumptions."""
    def index(d):
        result = {c['pair_id']: i for i, c in enumerate(d['cases'])}
        assert len(result) == len(d['cases'])
        return result
    original, other = index(baseline), index(second)
    requested = index(first)
    assert requested.keys() == other.keys() and requested.keys() <= original.keys()
    ids = list(requested)
    bi, fi, si = ([lookup[k] for k in ids] for lookup in [original, requested, other])
    y = first['y'][fi]
    assert np.array_equal(y, baseline['y'][bi]) and np.array_equal(y, second['y'][si])
    predictions = np.array([baseline['pred'][bi], first['pred'][fi], second['pred'][si]])
    flipped = np.any(predictions != predictions[0], axis=0)
    result = {'n': len(ids), 'positive': int(y.sum()), 'all_three_prediction_agreement': float((~flipped).mean()),
              'baseline_vs_repeat1_flip_rate': float((predictions[0] != predictions[1]).mean()),
              'baseline_vs_repeat2_flip_rate': float((predictions[0] != predictions[2]).mean()),
              'repeat1_vs_repeat2_flip_rate': float((predictions[1] != predictions[2]).mean()),
              'run_metrics': [analyze.metric(y, p) for p in predictions],
              'changed_pair_ids': [pid for pid, changed in zip(ids, flipped) if changed]}
    if baseline['prob'] is not None:
        probabilities = np.array([baseline['prob'][bi], first['prob'][fi], second['prob'][si]])
        result['mean_within_pair_probability_range'] = float(np.ptp(probabilities, axis=0).mean())
        result['max_within_pair_probability_range'] = float(np.ptp(probabilities, axis=0).max())
    return result


def model_task(m):
    for b in SCREEN_BENCHMARKS:
        for v in INTERACTIONS:
            run.run_cell(m, v, b, 'expanded_screen', workers=8)
        for namespace in ['e3_repeat_1', 'e3_repeat_2']:
            run.run_cell(m, 'B0', b, 'expanded_repeat', workers=8, namespace=namespace)


def forecast():
    rows = []
    for m in MODELS:
        for b in SCREEN_BENCHMARKS:
            for v, n, reference in [('I3_identity_json', 500, 'X_P_identity_0'),
                                   ('I3_domain_json', 500, 'X_P_domain_0'), ('B0_repeats', 200, 'B0')]:
                d = analyze.load(m, reference, b, 'expanded_screen')
                rows.append({'model': m, 'benchmark': b, 'condition': v, 'calls_before_cache': n,
                             'estimated_usd_before_cache': (float(d['costs'].sum())+.05*d['unknown_cost_attempts'])/len(d['cases'])*n})
    value = {'basis': 'Corresponding screen condition means; errors with unknown cost get .05 allowance. JSON length can change actual costs. Conservative gross estimate before resume cache.',
             'budget_snapshot': read(PROJECT/'artifacts/budget.json'), 'project_cap_usd': 100,
             'estimated_usd_before_cache': sum(r['estimated_usd_before_cache'] for r in rows), 'rows': rows}
    path = PROJECT/'artifacts/expanded_stability_cost_forecast.json'
    if not path.exists():
        path.write_text(canonical(value)+'\n')
    print('E3 cost estimate before cache', value['estimated_usd_before_cache'], 'USD; hard cap unchanged', flush=True)


def report():
    contrasts, repeats, metrics, costs = [], [], [], []
    for m in MODELS:
        for b in SCREEN_BENCHMARKS:
            factorial = {}
            for v in [*INTERACTIONS, *INTERACTIONS.values()]:
                d = analyze.load(m, v, b, 'expanded_screen')
                value = analyze.metric(d['y'], d['pred'])
                factorial[v] = value['f1']
                metrics.append({'model': m, 'benchmark': b, 'condition': v, **value})
                costs.append({'model': m, 'benchmark': b, 'condition': v,
                              'already_charged_to_screen': v not in INTERACTIONS,
                              'known_cost_usd': float(d['costs'].sum()), 'unknown_cost_attempts': d['unknown_cost_attempts']})
            contrasts.append({'model': m, 'benchmark': b, 'split': 'expanded_screen_development',
                              'f1_interaction': (factorial['I3_domain_json']-factorial['X_P_domain_0'])-
                                                (factorial['I3_identity_json']-factorial['X_P_identity_0']),
                              'interpretation': 'Fixed descriptive factorial contrast, not a test-set performance claim or significance test'})
            baseline = analyze.load(m, 'B0', b, 'expanded_screen')
            fresh = [analyze.load(m, 'B0', b, 'expanded_repeat', namespace=n) for n in ['e3_repeat_1', 'e3_repeat_2']]
            repeats.append({'model': m, 'benchmark': b, **repeat_summary(baseline, *fresh)})
            for i, d in enumerate(fresh, 1):
                costs.append({'model': m, 'benchmark': b, 'condition': 'B0_repeat_'+str(i),
                              'already_charged_to_screen': False, 'known_cost_usd': float(d['costs'].sum()),
                              'unknown_cost_attempts': d['unknown_cost_attempts']})
    out = PROJECT/'artifacts/results/expanded_stability'
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('factorial_metrics', metrics), ('factorial_contrasts', contrasts), ('repeatability', repeats), ('costs', costs)]:
        (out/(name+'.json')).write_text(canonical(value)+'\n')
    analyze.csv_save(out/'factorial_metrics.csv', metrics)
    analyze.csv_save(out/'factorial_contrasts.csv', contrasts)
    write_frozen(PROJECT/'artifacts/expanded_stability_complete.json',
                 {'status': 'complete', 'factorial_contrasts': len(contrasts), 'repeatability_cells': len(repeats),
                  'full_expanded_study_complete': False, 'inferential_interaction_statistics': 'pending expanded statistics package'})
    print('E3 stability complete', len(contrasts), 'contrasts and', len(repeats), 'three-run repeatability cells', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'run', 'analyze'])
    args = ap.parse_args()
    if args.mode == 'prepare':
        prepare()
        return
    verify()
    run.payload = payload
    analyze.payload = payload
    if args.mode == 'run':
        assert read(PROJECT/'artifacts/progress.json')['expanded_E2_E3_screen']['status'] == 'complete'
        forecast()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
            list(pool.map(model_task, MODELS))
    report()


if __name__ == '__main__':
    main()
