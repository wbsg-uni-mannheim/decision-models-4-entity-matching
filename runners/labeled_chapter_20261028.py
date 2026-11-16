"""Labelled-data chapter, part 1: sets, frozen splits, test-record embedding plan, logistic regression.

Sets per benchmark from the Sol-B0 label sets (runs/labelsets_sol_b0_20261028): 200, 4000 and a
2000 set nested in the 4000 set (profile mechanism of the labelling repo: the first k positives and
2000-k negatives in labelling order, k at the 4000 set's positive share; the seed comes first and is
checked to be inside). One stratified 80/20 train/validation split per set: within each class the
pairs are ranked by sha256('labeled-chapter-split|<benchmark>|<pair_id>'), the first round(0.2 n)
go to validation. All files are frozen with sha256 in MANIFEST; every method uses these splits.

Logistic regression: exactly the features of the labelling paper's AL-ML committee
(active_learning_ml._build_feature_matrix: per configured field numeric closeness / exact match /
token Jaccard, plus the cosine of the 256-d text-embedding-3-small record embeddings). Records and
embeddings come from the labelling run's canonical tables; records missing there (most test
records, all of WDC, Semi-HETER and billiger-de) use the official typed record and need a new
embedding (embed-plan writes the requests, embed runs them; paid, not yet released). C and the
decision threshold are tuned on the validation split; test F1 needs the test embeddings.

Run with /path/to/envs/ditto-modern/bin/python.
"""
import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path('/path/to/decision-em')
LAB = Path('/path/to/Automatic-data-labeling')
LABELS = 'runs/labelsets_sol_b0_20261028'
OUT = 'runs/labeled_chapter_20261028'
MANIFEST = 'configs/labeled_chapter_sets_20261028.json'
SIZES = [200, 2000, 4000]
VALID_SHARE = .2
EMB_MODEL, EMB_DIM = 'text-embedding-3-small', 256
C_GRID = [.01, .1, 1., 10., 100.]
THRESHOLDS = [round(.05*i, 2) for i in range(1, 20)]
EMBED_APPROVAL = 'artifacts/operations/approval_labeled_chapter_embed_aaron_20261028.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_gz(p):
    with gzip.open(p, 'rt') as f:
        return [json.loads(x) for x in f]


def write_gz(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, 'wt') as f:
        f.writelines(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows)


def modules():
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'artifacts/operations'), str(LAB/'scripts/labeling')]
    import labelsets_sol_20261028 as ls
    return ls


# ------------------------------------------------------------------- sets

def set_rows(b, size):
    if size == 2000:
        rows = read_gz(ROOT/LABELS/b/f'{b}_set4000_train.json.gz')
        k = round(2000*sum(r['label'] for r in rows)/len(rows))
        pos = [r for r in rows if r['label']][:k]; neg = [r for r in rows if not r['label']][:2000-k]
        keep = {r['pair_id'] for r in pos+neg}
        return [r for r in rows if r['pair_id'] in keep]
    return read_gz(ROOT/LABELS/b/f'{b}_set{size}_train.json.gz')


def split(b, rows):
    train, valid = [], []
    for y in (0, 1):
        cls = sorted((r for r in rows if r['label'] == y),
                     key=lambda r: hashlib.sha256(f"labeled-chapter-split|{b}|{r['pair_id']}".encode()).hexdigest())
        n_valid = round(VALID_SHARE*len(cls))
        valid += cls[:n_valid]; train += cls[n_valid:]
    order = {r['pair_id']: i for i, r in enumerate(rows)}
    return sorted(train, key=lambda r: order[r['pair_id']]), sorted(valid, key=lambda r: order[r['pair_id']])


def make_sets(benchmarks):
    ls = modules()
    require(not (ROOT/MANIFEST).exists(), 'Manifest exists; sets and splits are write-once')
    spec = {'labels': LABELS, 'labels_report_sha256': {}, 'valid_share': VALID_SHARE, 'split_rule': __doc__.split('\n\n')[1],
            'script_sha256': sha(__file__), 'benchmarks': {}}
    for b in benchmarks:
        sel = json.loads((ROOT/LABELS/'selection'/f'{b}.json').read_text())
        seed = {f'{s[2]}#{s[3]}' for s in sel['seed']}
        info = {}
        for size in SIZES:
            rows = set_rows(b, size)
            ids = [r['pair_id'] for r in rows]
            require(len(rows) == size and len(set(ids)) == size, f'{b}/{size}: size or duplicates')
            if size == 2000:
                require(set(ids) <= {r['pair_id'] for r in read_gz(ROOT/LABELS/b/f'{b}_set4000_train.json.gz')}, 'not nested')
            train, valid = split(b, rows)
            files = {}
            for name, part in (('all', rows), ('train', train), ('valid', valid)):
                p = ROOT/OUT/'sets'/b/f'set{size}_{name}.json.gz'
                require(not p.exists(), 'exists: '+str(p))
                write_gz(p, part)
                files[name] = {'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'n': len(part), 'pos': sum(r['label'] for r in part)}
            info[str(size)] = {'files': files, 'seed_inside': seed <= set(ids), 'pos_share': round(files['all']['pos']/size, 4)}
        info['2000_in_4000'] = True
        spec['benchmarks'][b] = info
        print(b, json.dumps({s: [info[str(s)]['files']['train']['n'], info[str(s)]['files']['valid']['n'], info[str(s)]['pos_share'], info[str(s)]['seed_inside']]
                             for s in SIZES}), flush=True)
    (ROOT/MANIFEST).write_text(json.dumps(spec, indent=1, sort_keys=True)+'\n')


def manifest_rows(b, size, name):
    m = json.loads((ROOT/MANIFEST).read_text())['benchmarks'][b][str(size)]['files'][name]
    require(sha(ROOT/m['path']) == m['sha256'], 'Split drift: '+m['path'])
    return read_gz(ROOT/m['path'])


# -------------------------------------------------------- records, embeddings

class Records:
    """Records and embeddings keyed by internal ids: L:i / R:j (labelling run) or LT:<id> / RT:<id> (new)."""

    def __init__(self, b):
        import numpy as np
        import yaml
        ls = modules()
        alm, _, _ = ls.lab_modules()
        cfg = yaml.safe_load((LAB/'configs/labeling/benchmarks_active.yaml').read_text())['benchmarks'][b]
        run = LAB/ls.RUNS[b]
        self.b, self.alm, self.np = b, alm, np
        self.fields = list(cfg['fields'])
        self.typed = ls.typed_records(b)
        self.test_records = {'left': {}, 'right': {}}  # exactly the records of data/<b>/test.json
        for c in json.loads((ROOT/'data'/b/'test.json').read_text()):
            for side in ('left', 'right'):
                self.test_records[side][c['entity_ids'][side]] = c['records'][side]
        tables = {'left': alm._load_df(run/'canonical/left.csv', 'L', {}), 'right': alm._load_df(run/'canonical/right.csv', 'R', {})}
        emb_dir = LAB/cfg['embeddings']['dir']
        self.emb = {'left': np.load(emb_dir/cfg['embeddings']['left_file']).astype(np.float32),
                    'right': np.load(emb_dir/cfg['embeddings']['right_file']).astype(np.float32)}
        self.maps, self.idx, self.by_id = {}, {}, {}
        for s, df in tables.items():
            self.maps[s] = {r['__rid']: r for r in df.to_dict('records')}
            self.idx[s] = {r: i for i, r in enumerate(df['__rid'])}
            self.by_id[s] = {}
            for rid, i in zip(df['__rid'], df['id'].astype(str)):
                self.by_id[s].setdefault(i, rid)
        self.pair_rids = {}
        for r in map(json.loads, (ROOT/LABELS/b/'labels.jsonl').read_text().splitlines()):
            self.pair_rids[(r['id1'], r['id2'])] = (r['rid1'], r['rid2'])
        self.new = self.load_new()

    def official(self, side, i):
        t = self.test_records[side].get(i) or self.typed[side][i]
        rec = {f: t.get(f, t.get('name') if f == 'title' else None) for f in self.fields}
        return {**{f: '' if v is None else v for f, v in rec.items()}, 'id': i}

    def missing(self, pairs):
        """Record ids without a labelling-run row, per side."""
        out = {'left': set(), 'right': set()}
        for a, c in pairs:
            if (a, c) in self.pair_rids:
                continue
            if a not in self.by_id['left']: out['left'].add(a)
            if c not in self.by_id['right']: out['right'].add(c)
        return out

    def load_new(self):
        new = {}
        for s in ('left', 'right'):
            p = ROOT/OUT/'embeddings'/f'{self.b}_{s}.npz'
            if p.exists():
                z = self.np.load(p)
                new[s] = dict(zip(z['ids'].tolist(), z['vectors']))
        return new

    def keys(self, pairs):
        """Internal keys, maps, index and embedding arrays for _build_feature_matrix; None if an embedding is missing."""
        np = self.np
        maps, idx, emb = {s: dict(self.maps[s]) for s in self.maps}, {s: dict(self.idx[s]) for s in self.idx}, dict(self.emb)
        extra = {'left': [], 'right': []}
        out = []
        for a, c in pairs:
            if (a, c) in self.pair_rids:
                out.append(self.pair_rids[(a, c)]); continue
            k = []
            for s, i, tag in (('left', a, 'LT'), ('right', c, 'RT')):
                if i in self.by_id[s]:
                    k.append(self.by_id[s][i]); continue
                vec = self.new.get(s, {}).get(i)
                if vec is None:
                    return None
                key = f'{tag}:{i}'
                if key not in maps[s]:
                    maps[s][key] = self.official(s, i); idx[s][key] = len(emb[s])+len(extra[s]); extra[s].append(vec)
                k.append(key)
            out.append(tuple(k))
        for s in extra:
            if extra[s]:
                emb[s] = np.vstack([emb[s], np.array(extra[s], dtype=np.float32)])
        return out, maps, idx, emb

    def features(self, pairs):
        import pandas as pd
        got = self.keys(pairs)
        if got is None:
            return None
        keys, maps, idx, emb = got
        df = pd.DataFrame(keys, columns=['id1', 'id2'])
        return self.alm._build_feature_matrix(df, maps['left'], maps['right'], idx['left'], idx['right'],
                                              emb['left'], emb['right'], feature_fields=self.fields)


def test_pairs(b):
    cases = json.loads((ROOT/'data'/b/'test.json').read_text())
    return [(c['entity_ids']['left'], c['entity_ids']['right']) for c in cases], [c['label'] for c in cases]


def _value(v, keep_float):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15 and not keep_float:
        v = int(v)
    s = str(v).strip()
    return '' if s.lower() == 'nan' else s


TEXT_VARIANTS = {  # how the labelling run's embedding inputs were joined (differs per benchmark build)
    'skip_empty': lambda vals, kf: ' '.join(x for x in (_value(v, kf) for v in vals) if x),
    'keep_empty': lambda vals, kf: ' '.join(_value(v, kf) for v in vals),
}


def embed_spec(b, rec):
    """Field order (embedding manifest) and the text variant that reproduces most original inputs, per side."""
    import csv
    man = LAB/'benchmarks'/b/f'{b}-batch_embed_manifest.json'
    fields = json.loads(man.read_text()).get('embed_fields') if man.exists() else None
    spec = {}
    for s in ('left', 'right'):
        order = (fields.get(s) if isinstance(fields, dict) else fields) or rec.fields
        reqs = {}
        mp = LAB/'benchmarks'/b/f'{b}-batch_embed_{s}_map.csv'
        lp = LAB/'benchmarks'/b/f'{b}-batch_embed_{s}.jsonl'
        if mp.exists() and lp.exists():
            ids = {r['custom_id']: r['id'] for r in csv.DictReader(open(mp))}
            for line in open(lp):
                q = json.loads(line); reqs[ids[q['custom_id']]] = q['body']['input']
        best = ('skip_empty', False, -1)
        for name, fn in TEXT_VARIANTS.items():
            for kf in (False, True):
                hits = sum(fn([rec.official(s, i).get(f) for f in order], kf) == t
                           for i, t in reqs.items() if i in rec.typed[s] or i in rec.test_records[s])
                if hits > best[2]:
                    best = (name, kf, hits)
        spec[s] = {'order': order, 'variant': best[0], 'keep_float': best[1], 'identical': best[2], 'compared': len(reqs)}
    return spec


def embed_text(spec, rec, s, i):
    sp = spec[s]
    return TEXT_VARIANTS[sp['variant']]([rec.official(s, i).get(f) for f in sp['order']], sp['keep_float']).strip() or i


def embed_plan(benchmarks):
    """Requests for the records without embedding (test pairs), with the text format of the labelling run's inputs."""
    total = {'records': 0, 'chars': 0}
    for b in benchmarks:
        rec = Records(b)
        spec = embed_spec(b, rec)
        print(b, 'embedding text spec', json.dumps(spec), flush=True)
        pairs, _ = test_pairs(b)
        miss = rec.missing(pairs)
        d = ROOT/OUT/'embeddings'/'requests'; d.mkdir(parents=True, exist_ok=True)
        info = {}
        for s in ('left', 'right'):
            rows = []
            for i in sorted(miss[s]):
                t = embed_text(spec, rec, s, i)
                rows.append({'custom_id': f'{s}-{i}', 'method': 'POST', 'url': '/v1/embeddings',
                             'body': {'model': EMB_MODEL, 'dimensions': EMB_DIM, 'input': t}})
            (d/f'{b}_{s}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
            chars = sum(len(r['body']['input']) for r in rows)
            info[s] = [len(rows), chars]
            total['records'] += len(rows); total['chars'] += chars
        print(b, 'missing records (n, chars) L/R', info, flush=True)
    tokens = total['chars']/4
    print('total', total, f'~{tokens:,.0f} tokens, ~{tokens*0.02e-6:.4f} USD at 0.02 USD/1M (text-embedding-3-small)')


def embed(benchmarks):
    """Paid (text-embedding-3-small, ~0.02 USD): embeds the planned records; needs Aaron's approval file."""
    import numpy as np
    from openai import OpenAI
    a = json.loads((ROOT/EMBED_APPROVAL).read_text())
    require(a.get('approved_by') == 'Aaron' and a.get('scope') == 'labeled_chapter_embeddings' and a.get('quote'), 'Approval missing')
    sys.path.insert(0, str(ROOT/'artifacts/operations'))
    import astra_discovery_submit
    import reference_sol_direct_20261021 as ref
    budget = ref.Budget(ROOT, f"labeled-chapter-embed-{int(__import__('time').time())}", float(a['run_cap_usd']), a['quote'], tier='default')
    client = OpenAI(api_key=astra_discovery_submit.credential(ROOT), timeout=120, max_retries=4)
    try:
        for b in benchmarks:
            for side in ('left', 'right'):
                out = ROOT/OUT/'embeddings'/f'{b}_{side}.npz'
                reqs = [json.loads(x) for x in (ROOT/OUT/'embeddings'/'requests'/f'{b}_{side}.jsonl').read_text().splitlines()]
                if out.exists() or not reqs:
                    continue
                vecs = []
                dims = Records(b).emb[side].shape[1]  # wdc: 1536 (built without a dimensions parameter), others 256
                extra = {} if dims == 1536 else {'dimensions': dims}
                for i in range(0, len(reqs), 64):
                    chunk = reqs[i:i+64]
                    r = client.embeddings.create(model=EMB_MODEL, input=[q['body']['input'] for q in chunk], **extra)
                    budget.admit(); budget.settle(r.usage.total_tokens*0.02e-6)
                    vecs += [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
                np.savez(out, ids=np.array([q['custom_id'].split('-', 1)[1] for q in reqs]), vectors=np.array(vecs, dtype=np.float32))
                print(b, side, len(vecs), flush=True)
    finally:
        budget.ledger('reconcile')
        print('embedding cost USD', round(budget.spent, 4))


# ---------------------------------------------------------------- logistic regression

def f1(y, p):
    tp = sum(a and b for a, b in zip(y, p)); fp = sum(b and not a for a, b in zip(y, p)); fn = sum(a and not b for a, b in zip(y, p))
    return 2*tp/(2*tp+fp+fn) if tp else 0.


def lr(benchmarks, sizes):
    from sklearn.linear_model import LogisticRegression
    results = {}
    for b in benchmarks:
        rec = Records(b)
        pairs_test, y_test = test_pairs(b)
        X_test = rec.features(pairs_test)
        for size in sizes:
            data = {}
            for name in ('train', 'valid'):
                rows = manifest_rows(b, size, name)
                data[name] = (rec.features([(r['id_left'], r['id_right']) for r in rows]), [r['label'] for r in rows])
            best = None
            for C in C_GRID:
                m = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', C=C).fit(*data['train'])
                prob = m.predict_proba(data['valid'][0])[:, 1]
                for t in THRESHOLDS:
                    score = f1(data['valid'][1], [p >= t for p in prob])
                    if best is None or score > best[0]:
                        best = (score, C, t, m)
            score, C, t, m = best
            res = {'C': C, 'threshold': t, 'valid_f1': round(score, 4), 'n_train': len(data['train'][1]),
                   'n_valid': len(data['valid'][1]), 'features': rec.fields+['embedding_cosine']}
            if X_test is not None:
                res['test_f1'] = round(f1(y_test, [p >= t for p in m.predict_proba(X_test)[:, 1]]), 4)
            else:
                res['test_f1'] = None
                res['test_missing'] = {s: len(v) for s, v in rec.missing(pairs_test).items()}
            results.setdefault(b, {})[size] = res
            print(b, size, json.dumps(res), flush=True)
    p = ROOT/OUT/'lr'/'results.json'; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=1)+'\n')


def main():
    ls = modules()
    ap = argparse.ArgumentParser()
    ap.add_argument('--action', required=True, choices=['sets', 'embed-plan', 'embed', 'lr'])
    ap.add_argument('--benchmarks', nargs='+', default=list(ls.RUNS))
    ap.add_argument('--sizes', nargs='+', type=int, default=SIZES)
    args = ap.parse_args()
    if args.action == 'sets':
        make_sets(args.benchmarks)
    elif args.action == 'embed-plan':
        embed_plan(args.benchmarks)
    elif args.action == 'embed':
        embed(args.benchmarks)
    else:
        lr(args.benchmarks, args.sizes)


if __name__ == '__main__':
    main()
