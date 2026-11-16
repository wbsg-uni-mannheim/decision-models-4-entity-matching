"""B0 on the Sol-labelled sets of the labelled-data chapter, for threshold tuning without official validation labels.

Aaron (5 Nov, relayed): tune the decision thresholds (Section 3.3) on our own Sol labels of the 200 and the 2,000 set
instead of the official validation sets ("super lass das jetzt laufen ja gerne auf allen ... mach erstmal 200 und 2k").
Cases: data/<b>/solset.json, the union of set200_all and set2000_all of configs/labeled_chapter_sets_20261028.json
(16,449 pairs; 1,151 pairs lie in both sets), set200 first, then the new pairs of set2000, each in file order. A case has
the form of data/<b>/test.json (prepare.case): the typed records of the set row (the records of the Sol request, sorted
keys), pair_id <id_left>#<id_right>, label = Sol-B0 label, input_hash = digest(records). Membership per set is in
configs/solset_b0_cases_20261105.json. Requests: prompt12.payload with B0, the request of the B0 test runs.
Six decision models (Luna generative returns no probability), one run.
"""
import gzip
import hashlib
import json
from pathlib import Path

import prompt12
import protocol

PROJECT = Path(__file__).resolve().parents[1]
SPLIT = 'solset'
RUN_MODELS = [m for m in prompt12.RUN_MODELS if m != 'luna']
PROMPTS = ['B0']
SIZES = ['200', '2000']
SETS = 'configs/labeled_chapter_sets_20261028.json'
CASES = 'configs/solset_b0_cases_20261105.json'


def register_models():
    prompt12.register_models()


def run_variants(alias):
    return list(PROMPTS)


def benchmarks(variant):
    return list(protocol.BENCHMARKS)


def payload(case, alias, variant):
    assert variant in PROMPTS, variant
    return prompt12.payload(case, alias, variant)


def set_rows(root, b, n):
    cfg = json.loads((Path(root)/SETS).read_text())['benchmarks'][b][n]['files']['all']
    p = Path(root)/cfg['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == cfg['sha256'], (b, n)
    rows = [json.loads(line) for line in gzip.open(p, 'rt')]
    assert len(rows) == cfg['n'] and sum(r['label'] for r in rows) == cfg['pos'], (b, n)
    return rows


def case(row, b):
    fields = sorted(protocol.BENCHMARKS[b]['fields'])
    out = {'benchmark': b, 'pair_id': f"{row['id_left']}#{row['id_right']}", 'label': int(row['label']),
           'records': {s: {f: row[f+'_'+s] for f in fields} for s in ('left', 'right')},
           'entity_ids': {s: str(row['id_'+s]) for s in ('left', 'right')}}
    assert out['pair_id'] == row['pair_id'] and out['label'] in (0, 1), (b, row['pair_id'])
    out['input_hash'] = protocol.digest(out['records'])
    return out


def export_cases(root=PROJECT):
    """Write data/<b>/solset.json once and the membership config; independent overlap check against test and valid."""
    import prepare
    root = Path(root)
    report = {'description': __doc__, 'benchmarks': {}}
    for b in protocol.BENCHMARKS:
        cases, member = {}, {}
        for n in SIZES:
            member[n] = []
            for row in set_rows(root, b, n):
                c = case(row, b)
                if c['pair_id'] in cases:
                    assert cases[c['pair_id']] == c, (b, c['pair_id'])  # the same pair in both sets: identical case
                else:
                    cases[c['pair_id']] = c
                member[n].append(c['pair_id'])
            assert len(set(member[n])) == len(member[n]), (b, n)
        rows = list(cases.values())
        test = json.loads((root/'data'/b/'test.json').read_text())
        valid = json.loads((root/'data'/b/'valid.json').read_text())
        ids, hashes = set(cases), {c['input_hash'] for c in rows}
        ents = {(c['entity_ids']['left'], c['entity_ids']['right']) for c in rows}
        overlap = {}
        for name, other in (('test', test), ('valid', valid)):
            overlap[name] = {'pair_id': len(ids & {c['pair_id'] for c in other}),
                             'entity_pair': len(ents & {(c['entity_ids']['left'], c['entity_ids']['right']) for c in other}),
                             'input_hash': len(hashes & {c['input_hash'] for c in other}),
                             'n': len(other)}
            assert not overlap[name]['pair_id'] and not overlap[name]['entity_pair'], (b, name, overlap[name])
        for s in ('left', 'right'):  # key order and value types of the test records
            assert all(list(c['records'][s]) == list(test[0]['records'][s]) for c in rows), (b, s)
        prepare.write_frozen(root/'data'/b/(SPLIT+'.json'), rows)
        report['benchmarks'][b] = {
            'n': len(rows), 'positive': sum(c['label'] for c in rows), 'sha256': protocol.digest(rows),
            'sets': {n: {'n': len(member[n]), 'positive': sum(cases[i]['label'] for i in member[n]), 'pair_ids': member[n]}
                     for n in SIZES},
            'in_both_sets': len(set(member['200']) & set(member['2000'])), 'overlap': overlap}
    prepare.write_frozen(root/CASES, report)
    return report


def tuning_sets(root=PROJECT):
    """{size: {benchmark: [pair_id]}} from the membership config."""
    cfg = json.loads((Path(root)/CASES).read_text())['benchmarks']
    return {n: {b: cfg[b]['sets'][n]['pair_ids'] for b in cfg} for n in SIZES}


if __name__ == '__main__':
    r = export_cases()
    print(json.dumps({b: {k: v for k, v in x.items() if k != 'sets'} | {'sets': {n: {k: v for k, v in s.items() if k != 'pair_ids'}
                      for n, s in x['sets'].items()}} for b, x in r['benchmarks'].items()}, indent=1))
