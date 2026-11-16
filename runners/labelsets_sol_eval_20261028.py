"""Sol-B0 label sets vs official benchmark training labels (pairs in either orientation). Read-only."""
import gzip, json, sys
from pathlib import Path
sys.argv = ['x']; sys.path.insert(0, '/path/to/decision-em/artifacts/operations')
import labelsets_sol_20261028 as L
L.lab_modules()
import prepare
TRAIN = {'abt-buy': 'abt-buy-train.json', 'wdc': 'wdcproducts80cc20rnd000un_train_large.json.gz'}
out = {}
for b in L.RUNS:
    p = L.LAB/'benchmarks'/b/TRAIN.get(b, f'{b}-train.json.gz')
    text = gzip.open(p, 'rt').read() if p.suffix == '.gz' else None
    if text is not None and text.lstrip().startswith('{') and '\n{' not in text.strip():
        data = json.loads(text); rows = [{k: v[i] for k, v in data.items()} for i in next(iter(data.values()))]
    else:
        rows = prepare.read_rows(p)
    gold = {}
    for r in rows:
        a, c, y = str(r['id_left']), str(r['id_right']), int(r['label'])
        gold[(a, c)] = y; gold.setdefault((c, a), y)
    sel = json.load(open(L.ROOT/L.OUT/'selection'/f'{b}.json'))
    gpt = {(s[2], s[3]): s[4] for s in sel['seed']+sel['base']}
    res = {'train_pairs': len(rows), 'train_pos_share': round(sum(int(r['label']) for r in rows)/len(rows), 3)}
    for name in ('set200', 'set4000'):
        f = L.ROOT/L.OUT/b/f'{b}_{name}_train.json.gz'
        items = [json.loads(x) for x in gzip.open(f, 'rt')]
        ov = [(it['label'], gold[(it['id_left'], it['id_right'])], gpt.get((it['id_left'], it['id_right']))) for it in items
              if (it['id_left'], it['id_right']) in gold]
        def prf(i):
            pairs = [(x[i], x[1]) for x in ov if x[i] is not None]
            tp = sum(p and y for p, y in pairs); fp = sum(p and not y for p, y in pairs); fn = sum(not p and y for p, y in pairs)
            tn = len(pairs)-tp-fp-fn
            pr = tp/(tp+fp) if tp+fp else None; rc = tp/(tp+fn) if tp+fn else None
            return {'n': len(pairs), 'precision': pr and round(pr, 3), 'recall': rc and round(rc, 3),
                    'f1': round(2*pr*rc/(pr+rc), 3) if pr and rc else None,
                    'accuracy': round((tp+tn)/len(pairs), 3) if pairs else None, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
        res[name] = {'n': len(items), 'sol_pos_share': round(sum(it['label'] for it in items)/len(items), 3),
                     'overlap_with_train': len(ov), 'overlap_gold_pos_share': round(sum(x[1] for x in ov)/len(ov), 3) if ov else None,
                     'sol_vs_gold': prf(0), 'gpt_vs_gold': prf(2)}
    out[b] = res
    print(b, json.dumps(res), flush=True)
json.dump(out, open(L.ROOT/L.OUT/'eval_vs_benchmark_train.json', 'w'), indent=1)
