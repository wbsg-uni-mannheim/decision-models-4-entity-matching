"""E6 existing source variants; no model, prompt, label or original macro changes."""
import argparse
import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import analyze
import run
from prepare import case, read_rows, write_frozen
from protocol import MODELS, canonical, digest, payload

PROJECT = Path(__file__).resolve().parents[1]
SOURCES = {
    'en': ('billiger-en/billiger-en-gs.json.gz', '71eef93151a23f2fa2753f33936f9494919b2ce45ffa02c67d926966dae3ca39'),
    'wdc-000un': ('wdc/wdcproducts80cc20rnd000un_gs.json.gz', 'da61541a1a53f427af40b1c3f72009ddf12b3a194a424695cf825ae97d1894c7'),
    'wdc-050un': ('wdc/wdcproducts80cc20rnd050un_gs.json.gz', '54608e9400cccb7cd46f923996323d92f72f6b205eaee5a99b542c32b23c2f72'),
}
LANGUAGES = {'billiger-en-en-100un': ('en', 'en'), 'billiger-de-en-100un': ('de', 'en'),
             'billiger-en-de-100un': ('en', 'de')}
VARIANTS = {**{b: 'billiger-de' for b in LANGUAGES}, 'wdc-000un': 'wdc', 'wdc-050un': 'wdc'}


def read(path):
    return json.loads(path.read_text())


def align(reference, other):
    """Full paired alignment requires identical identities and labels, not row order."""
    index = {c['pair_id']: c for c in other}
    assert len(index) == len(other), 'Duplicate pair ID'
    assert len({c['pair_id'] for c in reference}) == len(reference)
    assert {c['pair_id'] for c in reference} == index.keys(), 'Pair set mismatch'
    result = [index[c['pair_id']] for c in reference]
    for a, b in zip(reference, result):
        assert a['entity_ids'] == b['entity_ids'] and a['label'] == b['label'], 'Identity/label mismatch'
        for s in ['left', 'right']:
            assert a['records'][s].keys() == b['records'][s].keys(), 'Field mismatch'
    return result


def language_cases(de, en):
    en = align(de, en)
    for a, b in zip(de, en):
        for s in ['left', 'right']:
            for f in ['brand', 'price']:
                assert a['records'][s][f] == b['records'][s][f], 'Non-text source attribute changed'
    result = {}
    for name, languages in LANGUAGES.items():
        rows = []
        for a, b in zip(de, en):
            c = copy.deepcopy(a)
            c['benchmark'] = name
            for s, lang in zip(['left', 'right'], languages):
                c['records'][s] = copy.deepcopy((a if lang == 'de' else b)['records'][s])
            c['input_hash'] = digest(c['records'])
            rows.append(c)
        result[name] = rows
    return result


def overlap(rows, reference):
    index = {c['pair_id']: c for c in reference}
    common = [c for c in rows if c['pair_id'] in index]
    assert all(c['label'] == index[c['pair_id']]['label'] and c['entity_ids'] == index[c['pair_id']]['entity_ids'] for c in common)
    hashes = {c['input_hash'] for c in reference}
    return {'common_pair_ids': len(common), 'identical_descriptive_inputs': sum(c['input_hash'] in hashes for c in rows)}


def prepare(source):
    core = {b: read(PROJECT/'data'/b/'test.json') for b in ['billiger-de', 'wdc']}
    inputs, provenance = {}, {}
    for key, (relative, expected) in SOURCES.items():
        path = source/relative
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, ('Source changed', relative)
        dest = PROJECT/'data/source/e6'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert dest.read_bytes() == raw
        else:
            dest.write_bytes(raw)
        inputs[key] = [case(r, 'billiger-de' if key == 'en' else 'wdc') for r in read_rows(dest)]
        provenance[key] = {'source': str(path), 'copy': str(dest.relative_to(PROJECT)), 'sha256': expected}
    rows_by_variant = language_cases(core['billiger-de'], inputs['en'])
    for b in ['wdc-000un', 'wdc-050un']:
        rows_by_variant[b] = inputs[b]
        for c in rows_by_variant[b]:
            c['benchmark'] = b
    manifest = {'registration': 'Post-registered extension; fixed B0; no test-based selection',
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'protocol_sha256': hashlib.sha256((PROJECT/'src/protocol.py').read_bytes()).hexdigest(),
                'sources': provenance, 'models': MODELS, 'variant': 'B0', 'project_cap_usd': 100,
                'core_sha256': {b: digest(rows) for b, rows in core.items()}, 'conditions': {},
                'language_provenance': 'Existing EN source; mixed directions derived by aligned side substitution, not existing mixed files. All source values preserved. Source translation quality is not human-validated.',
                'language_inspection': {'sampling': '8 first pairs by digest([language-inspection-v1, pair_id]) inspected for English content; no outcome-based filtering',
                    'changed_fields': {f: {s: sum(a['records'][s][f] != b['records'][s][f] for a, b in zip(core['billiger-de'], align(core['billiger-de'], inputs['en']))) for s in ['left', 'right']} for f in ['brand', 'name', 'desc', 'price']}},
                'analysis': {'language': 'Paired F1 delta vs original DE-DE100, prediction flips, probability shifts; source translation noise is a confound',
                             'wdc': 'Descriptive full-split results only; not paired full-split intervals and not a causal unseen effect',
                             'macro': 'Separate supplements; do not change original eight-family macro or selection',
                             'missing': '20cc/50cc Gold JSON not found in documented source scan; no invented variants'},
                'new_decisions_before_cache': sum(len(v) for v in rows_by_variant.values())*len(MODELS)}
    for b, rows in rows_by_variant.items():
        family = VARIANTS[b]
        expected_n, expected_pos = (4447, 348) if family == 'billiger-de' else (4500, 500)
        assert len(rows) == expected_n and sum(c['label'] for c in rows) == expected_pos
        assert len({c['pair_id'] for c in rows}) == len(rows)
        write_frozen(PROJECT/'data'/b/'test.json', rows)
        dev_overlap = {}
        for split in ['prompt', 'route', 'expanded_screen', 'calibration_pool']:
            path = PROJECT/'data'/family/(split+'.json')
            if path.exists():
                dev_overlap[split] = overlap(rows, read(path))
        manifest['conditions'][b] = {'family': family, 'n': len(rows), 'positive': sum(c['label'] for c in rows),
                                     'sha256': digest(rows), 'core_overlap': overlap(rows, core[family]),
                                     'development_overlap': dev_overlap,
                                     'request_templates': {m: payload({'benchmark': b, 'records': {'left': {}, 'right': {}}}, m) for m in MODELS}}
    write_frozen(PROJECT/'configs/expanded_variant_manifest.json', manifest)
    print('E6 frozen', manifest['new_decisions_before_cache'], 'new B0 decisions before retries/cache', flush=True)


def verify():
    manifest = read(PROJECT/'configs/expanded_variant_manifest.json')
    assert manifest['implementation_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert manifest['protocol_sha256'] == hashlib.sha256((PROJECT/'src/protocol.py').read_bytes()).hexdigest()
    assert manifest['models'] == MODELS
    for b, info in manifest['conditions'].items():
        assert digest(read(PROJECT/'data'/b/'test.json')) == info['sha256']
    for b, expected in manifest['core_sha256'].items():
        assert digest(read(PROJECT/'data'/b/'test.json')) == expected
    for info in manifest['sources'].values():
        assert hashlib.sha256((PROJECT/info['copy']).read_bytes()).hexdigest() == info['sha256']
    return manifest


def paired_language(reference, other):
    ordered = align(reference['cases'], other['cases'])
    index = {c['pair_id']: i for i, c in enumerate(other['cases'])}
    ids = [index[c['pair_id']] for c in ordered]
    pred, y = other['pred'][ids], reference['y']
    assert np.array_equal(y, other['y'][ids])
    ref = reference['pred']
    result = {'prediction_flip_rate': float((pred != ref).mean()),
              'both_correct': int(((pred == y) & (ref == y)).sum()),
              'only_variant_correct': int(((pred == y) & (ref != y)).sum()),
              'only_de_correct': int(((pred != y) & (ref == y)).sum()),
              'both_wrong': int(((pred != y) & (ref != y)).sum()),
              **analyze.bootstrap_delta(reference['cases'], y, pred, ref)}
    if reference['prob'] is not None:
        result['mean_absolute_probability_shift'] = float(np.abs(other['prob'][ids]-reference['prob']).mean())
    return result


def report():
    manifest = verify()
    metrics, paired, costs = [], [], []
    for m in MODELS:
        base = {b: analyze.load(m, 'B0', b, 'test') for b in ['billiger-de', 'wdc']}
        for b, d in base.items():
            metrics.append({'model': m, 'benchmark': b, 'reused_core': True, **analyze.metric(d['y'], d['pred'])})
        for b, info in manifest['conditions'].items():
            d = analyze.load(m, 'B0', b, 'test')
            metrics.append({'model': m, 'benchmark': b, 'reused_core': False, **analyze.metric(d['y'], d['pred'])})
            costs.append({'model': m, 'benchmark': b, 'known_cost_usd': float(d['costs'].sum()), 'unknown_cost_attempts': d['unknown_cost_attempts']})
            if info['family'] == 'billiger-de':
                paired.append({'model': m, 'variant': b, 'control': 'billiger-de-100un', **paired_language(base['billiger-de'], d)})
    out = PROJECT/'artifacts/results/expanded_variants'
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('metrics', metrics), ('paired_language', paired), ('costs', costs), ('provenance', manifest)]:
        (out/(name+'.json')).write_text(canonical(value)+'\n')
    analyze.csv_save(out/'metrics.csv', metrics)
    write_frozen(PROJECT/'artifacts/expanded_variants_complete.json', {'status': 'complete', 'new_cells': len(costs),
        'reused_core_cells': 14, 'language_comparisons': len(paired), 'wdc_full_split_paired_tests': False,
        'full_expanded_study_complete': False})


def forecast():
    rows = []
    for m in MODELS:
        for b, family in VARIANTS.items():
            d = analyze.load(m, 'B0', family, 'test')
            cases = read(PROJECT/'data'/b/'test.json')
            rows.append({'model': m, 'benchmark': b, 'n': len(cases),
                         'estimated_usd': len(cases)*(float(d['costs'].sum())+.05*d['unknown_cost_attempts'])/len(d['cases'])})
    value = {'basis': 'Same-family B0 cost, including failed attempts and .05 allowance per unknown attempt; text length may differ; gross before cache',
             'budget_snapshot': read(PROJECT/'artifacts/budget.json'), 'rows': rows,
             'estimated_usd': sum(r['estimated_usd'] for r in rows), 'project_cap_usd': 100}
    path = PROJECT/'artifacts/expanded_variants_cost_forecast.json'
    if not path.exists():
        path.write_text(canonical(value)+'\n')
    print('E6 estimated gross USD', value['estimated_usd'], '; shared hard cap unchanged', flush=True)


def model_task(m):
    for b in VARIANTS:
        run.run_cell(m, 'B0', b, 'test', workers=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'run', 'analyze'])
    ap.add_argument('--source', type=Path)
    args = ap.parse_args()
    if args.mode == 'prepare':
        assert args.source is not None
        prepare(args.source)
        return
    verify()
    if args.mode == 'run':
        forecast()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
            list(pool.map(model_task, MODELS))
    report()


if __name__ == '__main__':
    main()
