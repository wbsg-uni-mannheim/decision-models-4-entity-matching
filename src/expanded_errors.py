"""Descriptive E6 errors and observed-graph consistency; no inferred gold labels."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import analyze
from prepare import write_frozen
from protocol import BENCHMARKS, MODELS, canonical, digest, state

PROJECT = Path(__file__).resolve().parents[1]
BENCHES = [*BENCHMARKS, 'billiger-de-50un']


def read(path):
    return json.loads(path.read_text())


def features(c):
    values = [v for record in c['records'].values() for v in record.values()]
    missing = sum(v is None or v == '' or v == [] for v in values)
    # Purely lexical numeric-set discrepancy, not a validated identity conflict.
    numbers = [{x for v in c['records'][s].values() if v is not None for x in re.findall(r'\d+(?:[.,]\d+)?', str(v))} for s in ['left', 'right']]
    return {'chars': len(state(c, 'B0')), 'missing_fields': missing,
            'numeric_set_discrepancy': bool(numbers[0] and numbers[1] and numbers[0] != numbers[1])}


def graph_diagnostics(cases, positive):
    """Graph closure can reveal contradictions only on already observed edges."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    partners = defaultdict(set)
    for c, p in zip(cases, positive):
        if p:
            l, r = ('L', c['entity_ids']['left']), ('R', c['entity_ids']['right'])
            parent[find(l)] = find(r)
            partners[l].add(r)
            partners[r].add(l)
    contradictions = [c['pair_id'] for c, p in zip(cases, positive) if not p and
                      find(('L', c['entity_ids']['left'])) == find(('R', c['entity_ids']['right']))]
    return {'positive_edges': int(sum(positive)), 'negative_edges_with_positive_path': len(contradictions),
            'contradiction_pair_ids': contradictions,
            'nodes_with_multiple_positive_partners': sum(len(v) > 1 for v in partners.values()),
            'max_positive_partners': max(map(len, partners.values()), default=0)}


def agreement(y, pred, luna):
    return {'both_correct': int(((pred == y) & (luna == y)).sum()),
            'only_model_correct': int(((pred == y) & (luna != y)).sum()),
            'only_luna_correct': int(((pred != y) & (luna == y)).sum()),
            'both_wrong': int(((pred != y) & (luna != y)).sum())}


def prepare():
    spec = {'registration': 'Post-registered descriptive diagnostics; no selection on test outcomes',
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'models': MODELS, 'benchmarks': {}, 'confidence_thresholds': [.9, .99],
            'strata': 'B0 character-length quartiles from D_prompt; any missing descriptive field; lexical numeric-set discrepancy (not semantic conflict)',
            'graph': 'Side-specific entity-ID graph; contradictions only for observed negative edges linked by positive paths. No inferred gold pairs. One-to-one is not assumed from observed degree.',
            'statistics': 'Descriptive subgroup analysis; no significance claims or prompt tuning; extended inferential statistics separate'}
    for b in BENCHES:
        family = 'billiger-de' if b == 'billiger-de-50un' else b
        dev = read(PROJECT/'data'/family/'prompt.json')
        cuts = np.quantile([features(c)['chars'] for c in dev], [.25, .5, .75]).tolist()
        spec['benchmarks'][b] = {'length_cuts': cuts, 'cut_source': family+'/prompt', 'source_sha256': digest(dev)}
    write_frozen(PROJECT/'configs/expanded_error_manifest.json', spec)


def verify():
    spec = read(PROJECT/'configs/expanded_error_manifest.json')
    assert spec['implementation_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert spec['models'] == MODELS
    for info in spec['benchmarks'].values():
        assert digest(read(PROJECT/'data'/(info['cut_source']+'.json'))) == info['source_sha256']
    return spec


def masks(cases, cuts):
    fs = [features(c) for c in cases]
    lengths = np.searchsorted(cuts, [f['chars'] for f in fs], side='left')
    missing = np.array([f['missing_fields'] > 0 for f in fs])
    numbers = np.array([f['numeric_set_discrepancy'] for f in fs])
    return {'all': np.ones(len(cases), dtype=bool), **{'length_bin_'+str(i+1): lengths == i for i in range(4)},
            'missing_any': missing, 'missing_none': ~missing,
            'numeric_discrepancy': numbers, 'no_numeric_discrepancy': ~numbers}


def report():
    spec = verify()
    strata, confident, graph, provenance = [], [], [], []
    for b, info in spec['benchmarks'].items():
        luna = analyze.load('luna', 'B0', b, 'test')
        cases, y = luna['cases'], luna['y']
        subsets = masks(cases, info['length_cuts'])
        gold = graph_diagnostics(cases, y)
        entity_text = defaultdict(set)
        for c in cases:
            for side in ['left', 'right']:
                entity_text[(side, c['entity_ids'][side])].add(digest(c['records'][side]))
        provenance.append({'benchmark': b, 'reference_graph': gold,
                           'entities_with_multiple_descriptive_records': sum(len(v) > 1 for v in entity_text.values()),
                           'one_to_one_validation': 'not established; multiple positive partners are descriptive, not automatically errors',
                           'reference_transitivity': 'contradictions observed' if gold['negative_edges_with_positive_path'] else 'no contradiction among observed edges; unobserved relations remain unknown'})
        for m in MODELS:
            d = luna if m == 'luna' else analyze.load(m, 'B0', b, 'test')
            assert [c['pair_id'] for c in cases] == [c['pair_id'] for c in d['cases']]
            assert np.array_equal(y, d['y'])
            for name, mask in subsets.items():
                strata.append({'benchmark': b, 'model': m, 'stratum': name, 'positive': int(y[mask].sum()),
                               **analyze.metric(y[mask], d['pred'][mask]), **agreement(y[mask], d['pred'][mask], luna['pred'][mask])})
            graph.append({'benchmark': b, 'model': m, **graph_diagnostics(cases, d['pred'])})
            for t in spec['confidence_thresholds']:
                if d['prob'] is None:
                    confident.append({'benchmark': b, 'model': m, 'threshold': t, 'available': False,
                                      'reason': 'Generative binary output supplies no probability'})
                    continue
                p = d['prob']
                accepted = np.maximum(p, 1-p) >= t
                errors = accepted & (d['pred'] != y)
                confident.append({'benchmark': b, 'model': m, 'threshold': t, 'available': True,
                                  'accepted': int(accepted.sum()), 'errors': int(errors.sum()),
                                  'error_rate': float(errors.sum()/accepted.sum()) if accepted.any() else None,
                                  'false_positives': int((errors & (y == 0)).sum()),
                                  'false_negatives': int((errors & (y == 1)).sum()),
                                  'pair_ids': [c['pair_id'] for c, e in zip(cases, errors) if e]})
    out = PROJECT/'artifacts/results/expanded_errors'
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('strata', strata), ('high_confidence', confident), ('graph', graph), ('reference_structure', provenance), ('protocol', spec)]:
        (out/(name+'.json')).write_text(canonical(value)+'\n')
    analyze.csv_save(out/'strata.csv', strata)
    write_frozen(PROJECT/'artifacts/expanded_errors_complete.json', {'status': 'complete', 'model_benchmark_cells': len(graph),
                 'new_api_calls': 0, 'human_audit': 'pending', 'full_expanded_study_complete': False})
    print('E6 offline error diagnostics complete', len(graph), 'cells', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'analyze'])
    args = ap.parse_args()
    prepare() if args.mode == 'prepare' else report()


if __name__ == '__main__':
    main()
