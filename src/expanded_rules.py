"""Evaluate frozen rules from the user-authorized offline Astra CLI agent."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import analyze
import protocol
import expanded_protocol
import run
from prepare import write_frozen
from protocol import BENCHMARKS, MODELS, canonical, digest

PROJECT = Path(__file__).resolve().parents[1]
RULES = {}
MANUAL = "X_P_domain_0"
CONDITIONS = {"R_rules": "rules", MANUAL: "quality"}


def read(path):
    return json.loads(path.read_text())


def validate_rules(obj):
    assert obj['schema_version'] == 1 and obj['generator_model'] == 'gpt-6-astra'
    assert set(obj['families']) == set(BENCHMARKS), 'All eight training families required'
    for b, entry in obj['families'].items():
        assert set(entry) == {'rules'}, ('Unexpected rule fields', b)
        rules = entry['rules']
        assert isinstance(rules, list) and 1 <= len(rules) <= 8
        assert all(isinstance(r, str) and r.strip() and len(r) <= 240 for r in rules)
        assert len(set(rules)) == len(rules)
    return {b: entry['rules'] for b, entry in obj['families'].items()}


def payload(c, alias, variant='R_rules'):
    if variant in ['B0', MANUAL]:
        return expanded_protocol.payload(c, alias, variant)
    assert variant == 'R_rules'
    family = 'billiger-de' if c['benchmark'] == 'billiger-de-50un' else c['benchmark']
    body = protocol.payload(c, alias, 'B0')
    text = protocol.BASE + '\nGeneral rules derived from a separate labeled training sample:\n' + '\n'.join('- '+r for r in RULES[family])
    if alias == 'luna':
        body['messages'][0]['content'] = text
    else:
        body['questions']['match']['instructions'] = text
    return body


def prepare(agent_output):
    obj = read(agent_output/'rules.json')
    rules = validate_rules(obj)
    training = read(PROJECT/'configs/rule_agent_training_manifest.json')
    assert all(training['benchmarks'][b]['available'] for b in BENCHMARKS)
    # Parent validates actual CLI trace/artifacts first; an agent's self-validation is insufficient.
    receipt = read(PROJECT/'artifacts/astra_agent/acceptance.json')
    assert receipt['accepted'] is True
    raw_hash = hashlib.sha256((agent_output/'rules.json').read_bytes()).hexdigest()
    assert receipt['rules_sha256'] == raw_hash
    write_frozen(PROJECT/'configs/learned_rules.json', obj)
    spec = {'registration': 'User replaced one-shot Luna induction with iterative gpt-6-astra Codex CLI research agent before any rule generation or evaluation',
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol_sha256': hashlib.sha256((PROJECT/'src/protocol.py').read_bytes()).hexdigest(),
            'expanded_protocol_sha256': hashlib.sha256((PROJECT/'src/expanded_protocol.py').read_bytes()).hexdigest(),
            'training_manifest_sha256': digest(training), 'rules_sha256': digest(obj), 'acceptance_sha256': digest(receipt),
            'models': MODELS, 'generator': {'model': 'gpt-6-astra', 'interface': 'Codex CLI', 'monetary_cost': 'not established by token counts; separate usage provenance'},
            'selection': 'None. Rules fixed before D_prompt and full-test evaluation. No post-test rule revision.',
            'billiger50': 'Reuse billiger100-family rules; supplementary transfer, outside original macro',
            'new_generation_api_calls': 0, 'new_development_decisions': 11200, 'new_test_decisions': 242634,
            'manual_control': {'variant': MANUAL, 'choice': 'Fixed existing domain wording 0, not selected from scores',
                               'namespace': 'quality', 'cache': 'Reuse byte-identical E2 requests; fill ALL missing full-test cells',
                               'development_decisions_gross': 11200, 'test_decisions_gross': 242634},
            'comparisons': ['R_rules versus same-model B0', 'R_rules versus fixed same-model X_P_domain_0'],
            'stage_order': 'Complete all development cells before starting full test; no rule or control selection',
            'project_cap_usd': 100, 'agent_receipt': receipt,
            'human_review': 'No human labels or human rule validation claimed'}
    RULES.update(rules)
    spec['request_templates'] = {b: {m: {v: payload({'benchmark': b, 'records': {'left': {}, 'right': {}}}, m, v)
                                      for v in ['B0', *CONDITIONS]} for m in MODELS}
                                 for b in [*BENCHMARKS, 'billiger-de-50un']}
    spec['cells'] = [{"model": m, "variant": v, "benchmark": b, "split": split, "namespace": ns}
                     for m in MODELS for v, ns in CONDITIONS.items() for b in [*BENCHMARKS, 'billiger-de-50un']
                     for split in (['test'] if b == 'billiger-de-50un' else ['prompt', 'test'])]
    write_frozen(PROJECT/'configs/expanded_rules_manifest.json', spec)


def verify():
    spec = read(PROJECT/'configs/expanded_rules_manifest.json')
    assert spec['implementation_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert spec['protocol_sha256'] == hashlib.sha256((PROJECT/'src/protocol.py').read_bytes()).hexdigest()
    assert spec['expanded_protocol_sha256'] == hashlib.sha256((PROJECT/'src/expanded_protocol.py').read_bytes()).hexdigest()
    assert spec['models'] == MODELS
    training = read(PROJECT/'configs/rule_agent_training_manifest.json')
    assert digest(training) == spec['training_manifest_sha256']
    for path, expected in training['excluded_split_hashes'].items():
        assert digest(read(PROJECT/path)) == expected
    for b, info in training['benchmarks'].items():
        assert digest(read(PROJECT/'data'/b/'rules_train.json')) == info['training_sha256']
    obj = read(PROJECT/'configs/learned_rules.json')
    assert digest(obj) == spec['rules_sha256']
    assert digest(read(PROJECT/'artifacts/astra_agent/acceptance.json')) == spec['acceptance_sha256']
    RULES.update(validate_rules(obj))
    return spec


def quality_task(m, split):
    benchmarks = list(BENCHMARKS) + (['billiger-de-50un'] if split == 'test' else [])
    for b in benchmarks:
        for variant, namespace in CONDITIONS.items():
            run.run_cell(m, variant, b, split, workers=8, namespace=namespace)


def load_cell(m, variant, b, split):
    previous = analyze.payload
    try:
        analyze.payload = payload
        return analyze.load(m, variant, b, split, namespace=CONDITIONS.get(variant, 'quality'))
    finally:
        analyze.payload = previous


def report(spec):
    metrics, intervals, costs = [], [], []
    for m in MODELS:
        for b in [*BENCHMARKS, 'billiger-de-50un']:
            for split in ['test'] if b == 'billiger-de-50un' else ['prompt', 'test']:
                data = {v: load_cell(m, v, b, split) for v in ['B0', *CONDITIONS]}
                d = data['R_rules']
                for v, control in data.items():
                    assert [c['pair_id'] for c in d['cases']] == [c['pair_id'] for c in control['cases']]
                    assert np.array_equal(d['y'], control['y'])
                    metrics.append({'model': m, 'benchmark': b, 'split': split, 'variant': v,
                                    **analyze.metric(control['y'], control['pred'])})
                    costs.append({'model': m, 'benchmark': b, 'split': split, 'variant': v,
                                  'namespace': CONDITIONS.get(v, 'quality'),
                                  'known_cost_usd': float(control['costs'].sum()),
                                  'unknown_cost_attempts': control['unknown_cost_attempts']})
                if split == 'test':
                    for v in ['B0', MANUAL]:
                        intervals.append({'model': m, 'benchmark': b, 'variant': 'R_rules', 'control': v,
                                          **analyze.bootstrap_delta(d['cases'], d['y'], d['pred'], data[v]['pred'])})
    macro = []
    for m in MODELS:
        for v in ['B0', *CONDITIONS]:
            values = [r['f1'] for r in metrics if r['model'] == m and r['variant'] == v
                      and r['benchmark'] in BENCHMARKS and r['split'] == 'test']
            assert len(values) == len(BENCHMARKS)
            macro.append({'model': m, 'variant': v, 'benchmarks': len(values), 'macro_match_f1': float(np.mean(values))})
    out = PROJECT/'artifacts/results/expanded_rules';out.mkdir(parents=True, exist_ok=True)
    for name, value in [('metrics', metrics), ('intervals', intervals), ('costs', costs), ('macro', macro),
                        ('rules', read(PROJECT/'configs/learned_rules.json')), ('provenance', spec)]:
        (out/(name+'.json')).write_text(canonical(value)+'\n')
    analyze.csv_save(out/'metrics.csv', metrics)
    (out/'COST_README.txt').write_text(
        'Costs cover unique model/variant/benchmark/split cache cells, including errors and unknown attempts. '
        'B0 costs were charged in the core; manual X_P_domain_0 costs can also occur in E2 screening/confirmation reports. '
        'Do not add reused cache costs twice. The shared project ledger is authoritative for total accounted spend. '
        'Astra CLI usage is separate provenance, not an inferred dollar amount.\n')
    write_frozen(PROJECT/'artifacts/expanded_rules_complete.json', {'status': 'complete',
                 'test_cells_per_condition': len(MODELS)*(len(BENCHMARKS)+1),
                 'development_cells_per_condition': len(MODELS)*len(BENCHMARKS),
                 'evaluated_conditions': list(CONDITIONS), 'controls': ['B0', MANUAL],
                 'human_rule_quality_review': 'not performed', 'full_expanded_study_complete': False})


def forecast():
    rows = []
    for m in MODELS:
        for b in [*BENCHMARKS, 'billiger-de-50un']:
            family = 'billiger-de' if b == 'billiger-de-50un' else b
            reference = load_cell(m, 'B0', family, 'prompt')
            allowance = (float(reference['costs'].sum())+.05*reference['unknown_cost_attempts'])/len(reference['cases'])
            for split in ['test'] if b == 'billiger-de-50un' else ['prompt', 'test']:
                cases = read(PROJECT/'data'/b/(split+'.json'))
                for v, ns in CONDITIONS.items():
                    path = PROJECT/'runs'/ns/m/v/b/(split+'.jsonl')
                    records = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
                    cached = {r['pair_id']: r for r in records}
                    assert len(records) == len(cached), 'Duplicate cached response'
                    assert set(cached) <= {c['pair_id'] for c in cases}
                    for c in cases:
                        if c['pair_id'] in cached:
                            assert cached[c['pair_id']]['request_hash'] == digest(payload(c, m, v))
                    n = len(cases)-len(cached)
                    rows.append({'model': m, 'variant': v, 'benchmark': b, 'split': split, 'new_decisions': n,
                                 'reused_decisions': len(cached), 'estimated_quality_usd': n*allowance})
    value = {'rows': rows, 'quality_estimate_usd': sum(r['estimated_quality_usd'] for r in rows),
             'basis': 'B0 D_prompt means incl. errors/unknown allowance; changed rule/domain tokens may raise costs; excludes successful exact cache hits',
             'generation': 'Separate Codex CLI usage provenance; no zero-cost claim',
             'budget_snapshot': read(PROJECT/'artifacts/budget.json'), 'project_cap_usd': 100}
    path = PROJECT/'artifacts/rules_cost_forecast.json'
    if not path.exists():
        path.write_text(canonical(value)+'\n')
    print('RULES outstanding B0-based estimate USD', value['quality_estimate_usd'], '; changed tokens not included; hard cap unchanged', flush=True)


def main():
    ap = argparse.ArgumentParser();ap.add_argument('mode', choices=['prepare', 'run', 'analyze']);ap.add_argument('--agent-output', type=Path);args = ap.parse_args()
    if args.mode == 'prepare':
        assert args.agent_output is not None
        prepare(args.agent_output);return
    spec = verify()
    run.payload = payload
    if args.mode == 'run':
        forecast()
        for split in ['prompt', 'test']:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
                list(pool.map(lambda m: quality_task(m, split), MODELS))
    report(spec)


if __name__ == '__main__':
    main()
