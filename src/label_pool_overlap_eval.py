"""Overlap of the Sol label sets and pools with the test sets, per benchmark. Read-only, no inference (Aaron, relayed 6 Nov).

Question: are the label pools of the benchmarks with an unseen dimension (WDC Products, billiger.de) disjoint from the test
entities? Sources compared with data/<b>/test.json: the 200 and 2,000 sets (all pairs and train split), the labelling run's
candidate pool (faiss_candidates.csv minus official test and validation pairs, as labelsets_sol_20261028.Data.pool) and its
record tables (canonical/left.csv, right.csv), and the official training set (train file minus the validation pair ids).
Entity = cluster_id of the official pair files (benchmarks/<b>/*.json*, id -> cluster per side); cluster_check reports whether
matches share a cluster and non-matches do not, per test set.
Per source: test pairs with a record in the source (same side as in the test pair; any side), test pairs with both records,
share of unique test records in the source, test pairs and matching test pairs with an entity in the source, with both
entities, share of unique test entities in the source, and test records with identical content to a source record (any
side, typed records of labelsets_sol_20261028.typed_records).

    .venv/bin/python src/label_pool_overlap_eval.py --out artifacts/results/label_pool_overlap_20261106.json
"""
import argparse
import csv
import glob
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'artifacts/operations')]
import labelsets_sol_20261028 as ls  # noqa: E402
import protocol  # noqa: E402

B = list(protocol.BENCHMARKS)
SETS = json.loads((ROOT/'configs/labeled_chapter_sets_20261028.json').read_text())['benchmarks']
TRAIN = {'wdc': 'wdcproducts80cc20rnd000un_train_large.json.gz', 'abt-buy': 'abt-buy-train.json'}
VALID = {'wdc': 'wdcproducts80cc20rnd000un_valid_large.json.gz'}


def rows_of(p):
    p = str(p)
    text = (gzip.open(p, 'rt') if p.endswith('.gz') else open(p)).read().strip()
    if text.startswith('{') and '\n{' not in text:
        d = json.loads(text)
        return [{k: v[i] for k, v in d.items()} for i in next(iter(d.values()))]
    return json.loads(text) if text.startswith('[') else [json.loads(x) for x in text.splitlines()]


def clusters(b):
    """{side: {record id: cluster id}} over all official pair files of the benchmark; conflicts counted."""
    out, conflicts = {'left': {}, 'right': {}}, 0
    for p in sorted(glob.glob(str(ls.LAB/'benchmarks'/b/'*.json*'))):
        n = os.path.basename(p)
        if 'batch' in n or 'manifest' in n:
            continue
        rows = rows_of(p)
        if not rows or 'cluster_id_left' not in rows[0]:
            continue
        for r in rows:
            for s in ('left', 'right'):
                i, c = str(r['id_'+s]), str(r['cluster_id_'+s])
                conflicts += out[s].setdefault(i, c) != c
    return out, conflicts


def official_train(b):
    d = ls.LAB/'benchmarks'/b
    path = d/TRAIN.get(b, '')
    if b not in TRAIN:
        path = d/f'{b}-train.json.gz'
    rows = rows_of(path)
    vp = d/VALID.get(b, f'{b}-valid.csv')
    if str(vp).endswith('.csv'):
        with open(vp) as f:
            valid = {r['pair_id'] for r in csv.DictReader(f)}
    else:
        valid = {str(r['pair_id']) for r in rows_of(vp)}
    return [(str(r['id_left']), str(r['id_right'])) for r in rows if str(r['pair_id']) not in valid], len(rows), len(valid)


def share(a, b):
    return a/b if b else None


def overlap(pairs, tests, cl, typed):
    L, R = {p[0] for p in pairs}, {p[1] for p in pairs}
    any_side = L | R
    ent = {cl['left'][i] for i in L if i in cl['left']} | {cl['right'][i] for i in R if i in cl['right']}
    content = {protocol.digest(typed[s][i]) for s, ids in (('left', L), ('right', R)) for i in ids if i in typed[s]}
    n = len(tests)
    pos = [c for c in tests if c['label'] == 1]
    rec_l, rec_r = {c['entity_ids']['left'] for c in tests}, {c['entity_ids']['right'] for c in tests}
    ents = {cl[s].get(c['entity_ids'][s]) for c in tests for s in ('left', 'right')} - {None}

    def has_ent(c, s):
        return cl[s].get(c['entity_ids'][s]) in ent

    return {'source_pairs': len(pairs), 'source_records': len(L)+len(R), 'source_entities': len(ent),
            'source_records_without_cluster': sum(i not in cl['left'] for i in L)+sum(i not in cl['right'] for i in R),
            'test_pairs_with_record_same_side': share(sum(c['entity_ids']['left'] in L or c['entity_ids']['right'] in R for c in tests), n),
            'test_pairs_with_record_any_side': share(sum(c['entity_ids']['left'] in any_side or c['entity_ids']['right'] in any_side for c in tests), n),
            'test_pairs_with_both_records': share(sum(c['entity_ids']['left'] in L and c['entity_ids']['right'] in R for c in tests), n),
            'unique_test_records_in_source': share(len(rec_l & L)+len(rec_r & R), len(rec_l)+len(rec_r)),
            'test_pairs_with_entity': share(sum(has_ent(c, 'left') or has_ent(c, 'right') for c in tests), n),
            'test_pairs_with_both_entities': share(sum(has_ent(c, 'left') and has_ent(c, 'right') for c in tests), n),
            'matching_test_pairs_with_entity': share(sum(has_ent(c, 'left') or has_ent(c, 'right') for c in pos), len(pos)),
            'unique_test_entities_in_source': share(len(ents & ent), len(ents)),
            'test_records_with_identical_content': share(
                sum(protocol.digest(c['records'][s]) in content for c in tests for s in ('left', 'right')), 2*n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'units': 'shares as fractions', 'benchmarks': {}}
    for b in B:
        tests = json.loads((ROOT/'data'/b/'test.json').read_text())
        cl, conflicts = clusters(b)
        typed = ls.typed_records(b)
        tc = [(c, cl['left'].get(c['entity_ids']['left']), cl['right'].get(c['entity_ids']['right'])) for c in tests]
        known = [x for x in tc if x[1] is not None and x[2] is not None]
        res = {'test_pairs': len(tests), 'cluster_check': {
            'test_pairs_with_cluster': share(len(known), len(tests)), 'id_conflicts': conflicts,
            'matches_same_cluster': share(sum(x[1] == x[2] for x in known if x[0]['label'] == 1), sum(x[0]['label'] == 1 for x in known)),
            'non_matches_different_cluster': share(sum(x[1] != x[2] for x in known if x[0]['label'] == 0), sum(x[0]['label'] == 0 for x in known))},
            'sources': {}}
        for n in ('200', '2000'):
            for part in ('all', 'train'):
                with gzip.open(ROOT/SETS[b][n]['files'][part]['path'], 'rt') as f:
                    pairs = [(str(r['id_left']), str(r['id_right'])) for r in map(json.loads, f)]
                res['sources'][f'set{n}_{part}'] = overlap(pairs, tests, cl, typed)
        run = ls.LAB/ls.RUNS[b]
        with open(run/'faiss_candidates.csv') as f:
            cand = {(r['id1'], r['id2']) for r in csv.DictReader(f)}
        with open(ROOT/'data'/b/'valid.json') as f:
            bad = {tuple(c['pair_id'].split('#', 1)) for c in json.load(f)} | {tuple(c['pair_id'].split('#', 1)) for c in tests}
        res['sources']['candidate_pool'] = overlap(sorted(cand-bad), tests, cl, typed)
        table = {}
        for s in ('left', 'right'):
            with open(run/'canonical'/f'{s}.csv') as f:
                table[s] = [r['id'] for r in csv.DictReader(f)]
        # the record tables have no pairs: every left and right record counts as a source record of its side
        m = max(len(table['left']), len(table['right']))
        res['sources']['pool_record_tables'] = overlap([(table['left'][i % len(table['left'])], table['right'][i % len(table['right'])]) for i in range(m)],
                                                       tests, cl, typed)
        pairs, n_file, n_valid = official_train(b)
        res['sources']['official_train'] = {**overlap(pairs, tests, cl, typed), 'train_file_rows': n_file, 'validation_pair_ids': n_valid}
        out['benchmarks'][b] = res
        print(b, res['cluster_check'], flush=True)
        for k, v in res['sources'].items():
            print(' ', k, v['source_pairs'], 'pair+rec', round(v['test_pairs_with_record_same_side'], 4), 'rec', round(v['unique_test_records_in_source'], 4),
                  'pair+ent', round(v['test_pairs_with_entity'], 4), 'match+ent', round(v['matching_test_pairs_with_entity'], 4),
                  'ent', round(v['unique_test_entities_in_source'] or 0, 4), 'content', round(v['test_records_with_identical_content'], 4), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
