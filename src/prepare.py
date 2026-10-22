"""Freeze the eight original test sets and disjoint development samples."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path
from protocol import BENCHMARKS, MODELS, VARIANTS, canonical, digest, payload, state

def read_rows(path):
    if path.suffix == '.csv':
        with path.open() as f:
            return list(csv.DictReader(f))
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return [json.loads(line) for line in f if line.strip()]
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    # Original pandas orient=columns JSON, preserve source row order.
    return [{k: v[i] for k, v in data.items()} for i in next(iter(data.values()))]

def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value

def case(row, b):
    fields = BENCHMARKS[b]['fields']
    row = dict(row)
    if b == 'abt-buy':
        # Official training configuration declares name/title aliases.
        for s in ['left', 'right']:
            if 'name_'+s not in row and 'title_'+s in row:
                row['name_'+s] = row['title_'+s]
    for s in ['left', 'right']:
        assert all(f + '_' + s in row for f in fields), (b, s, set(row))
    result = {'benchmark': b, 'pair_id': str(row['pair_id']), 'label': int(row['label']),
              'records': {s: {f: clean(row[f+'_'+s]) for f in fields} for s in ['left', 'right']},
              'entity_ids': {s: str(row['id_'+s]) for s in ['left', 'right']}}
    assert result['label'] in (0, 1)
    result['input_hash'] = digest(result['records'])
    return result

def choose(rows, n, salt):
    # Largest-remainder proportional stratification, then deterministic hash sampling.
    n = min(n, len(rows))
    groups = {y: [r for r in rows if r['label'] == y] for y in [0, 1]}
    npos = round(n * len(groups[1]) / len(rows))
    out = []
    for y, count in [(0, n-npos), (1, npos)]:
        out.extend(sorted(groups[y], key=lambda r: digest([salt, r['pair_id']]))[:count])
    return sorted(out, key=lambda r: digest([salt, 'order', r['pair_id']]))

def write_frozen(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical(value) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == data, 'Frozen file changed: ' + str(path)
    else:
        path.write_bytes(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--project', required=True)
    args = ap.parse_args()
    root, project = Path(args.source), Path(args.project)
    manifest = {'source_root': str(root), 'benchmarks': {}, 'models': MODELS,
                'split_seed': '20261013-v1', 'variants': VARIANTS, 'strict_threshold': .5}
    for b, spec in BENCHMARKS.items():
        directory = root / b
        test_path = directory / ('wdcproducts80cc20rnd100un_gs.json.gz' if b == 'wdc' else b+'-gs.json.gz')
        if b == 'billiger-de':
            # Cluster keeps both 50un and 100un; 100un matches the planned local paper split byte-for-byte.
            test_path = directory / 'billiger-de-gs-100un.json.gz'
        valid_path = directory / ('wdcproducts80cc20rnd000un_valid_large.json.gz' if b == 'wdc' else b+'-valid.csv')
        sources = [test_path, valid_path]
        tests = [case(r, b) for r in read_rows(test_path)]
        valid_rows = read_rows(valid_path)
        if set(valid_rows[0]) == {'pair_id'}:
            source = directory / (b+'-valid.json.gz')
            if not source.exists():
                source = directory / (b+'-train.json')
            sources.append(source)
            indexed = {str(r['pair_id']): r for r in read_rows(source)}
            valid_rows = [indexed[str(r['pair_id'])] for r in valid_rows]
        valid = [case(r, b) for r in valid_rows]
        assert len(tests) == spec['count'] and sum(r['label'] for r in tests) == spec['positive'], b
        assert len({r['pair_id'] for r in tests}) == len(tests), b
        assert len({r['pair_id'] for r in valid}) == len(valid), b
        test_ids = {r['pair_id'] for r in tests}
        assert not test_ids & {r['pair_id'] for r in valid}, ('valid/test pair overlap', b)
        test_inputs = {r['input_hash'] for r in tests}
        # Remove duplicated inputs from development only; keep all benchmark test rows.
        unique = {}
        ambiguous = set()
        for r in valid:
            if r['input_hash'] not in test_inputs:
                if r['input_hash'] in unique:
                    if r['label'] != unique[r['input_hash']]['label']:
                        ambiguous.add(r['input_hash'])
                unique.setdefault(r['input_hash'], r)
        eligible = [r for h,r in unique.items() if h not in ambiguous]
        n = min(200, len(eligible)//2)
        assert n >= 20, ('insufficient development pairs', b)
        d_prompt = choose(eligible, n, '20261013:prompt:'+b)
        selected = {r['pair_id'] for r in d_prompt}
        d_route = choose([r for r in eligible if r['pair_id'] not in selected], n, '20261013:route:'+b)
        splits = {'test': tests, 'prompt': d_prompt, 'route': d_route}
        for name, rows in splits.items():
            write_frozen(project/'data'/b/(name+'.json'), rows)
        source_hashes = {}
        for source in sources:
            data = source.read_bytes()
            source_hashes[str(source)] = hashlib.sha256(data).hexdigest()
            dest = project/'data'/'source'/b/source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                assert dest.read_bytes() == data
            else:
                shutil.copyfile(source, dest)
        report = {'source_sha256': source_hashes, 'fields': spec['fields'],
                  'splits': {k: {'n': len(v), 'positive': sum(r['label'] for r in v), 'sha256': digest(v)} for k,v in splits.items()},
                  'valid_original': len(valid), 'valid_unique_disjoint_inputs': len(eligible),
                  'valid_conflicting_input_groups_excluded': len(ambiguous),
                  'test_duplicate_inputs': len(tests)-len(test_inputs),
                  'max_input_chars': max(len(state(r, 'B0')) for r in tests+valid),
                  'entity_overlap_dev_test': {s: len({r['entity_ids'][s] for r in d_prompt+d_route} & {r['entity_ids'][s] for r in tests}) for s in ['left','right']}}
        manifest['benchmarks'][b] = report
        print(b, canonical(report), flush=True)
    # Freeze exact prompt strings and request templates before any test inference.
    sample = {'benchmark': 'wdc', 'records': {'left': {}, 'right': {}}}
    manifest['request_templates'] = {m: {v: payload(sample, m, v) for v in VARIANTS} for m in MODELS}
    write_frozen(project/'configs'/'manifest.json', manifest)
    print('FROZEN', sum(x['splits']['test']['n'] for x in manifest['benchmarks'].values()))

if __name__ == '__main__':
    main()
