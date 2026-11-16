"""LR gold control: the labelled-chapter LR pipeline trained on the official benchmark training set. Read-only, free.

Same records, embeddings, features, LogisticRegression settings, C grid and threshold grid as labeled_chapter_20261028.lr,
and the same protocol as the Sol-set LR: the official training pairs (loader of labelsets_sol_eval_20261028.py) minus
pairs that are official validation or test pairs are split 80/20 within each class by labeled_chapter_20261028.split
(sha256 rank); train on the 80 %, C and threshold = maximum F1 on the 20 %, applied to test. Oracle variant: the same
model with the grid threshold that maximizes test F1. Side note (official_validation_variant): C and threshold tuned on
the official validation set (data/<b>/valid.json) instead, only where every validation pair has embeddings (WDC,
Billiger, Semi-HETER and Dn7 validation records have none or only part). No new embeddings, no API calls.
Writes lr/gold_control.json (write-once). Run with /path/to/envs/ditto-modern/bin/python.
"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labeled_chapter_20261028 as lc  # noqa: E402

NAMES = {'abt-buy': 'abt-buy-train.json', 'wdc': 'wdcproducts80cc20rnd000un_train_large.json.gz'}


def official_train(b):
    import prepare
    p = lc.LAB/'benchmarks'/b/NAMES.get(b, f'{b}-train.json.gz')
    text = gzip.open(p, 'rt').read() if p.suffix == '.gz' else None
    if text is not None and text.lstrip().startswith('{') and '\n{' not in text.strip():
        data = json.loads(text)
        return [{k: v[i] for k, v in data.items()} for i in next(iter(data.values()))]
    return prepare.read_rows(p)


def prf(y, p):
    tp = sum(a and b for a, b in zip(y, p)); fp = sum(b and not a for a, b in zip(y, p)); fn = sum(a and not b for a, b in zip(y, p))
    return {'precision': tp/(tp+fp) if tp+fp else 0., 'recall': tp/(tp+fn) if tp+fn else 0., 'f1': lc.f1(y, p)}


def tune(X, y, Xv, yv):
    from sklearn.linear_model import LogisticRegression
    best = None
    for C in lc.C_GRID:
        m = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', C=C).fit(X, y)
        prob = m.predict_proba(Xv)[:, 1]
        for t in lc.THRESHOLDS:
            score = lc.f1(yv, [p >= t for p in prob])
            if best is None or score > best[0]:
                best = (score, C, t, m)
    return best


def scored(best, X_test, y_test):
    score, C, t, m = best
    pt = m.predict_proba(X_test)[:, 1]
    oracle = max(((lc.f1(y_test, [p >= u for p in pt]), u) for u in lc.THRESHOLDS), key=lambda x: x[0])
    return {'C': C, 'threshold': t, 'tuning_f1': score, 'test': prf(y_test, [p >= t for p in pt]),
            'oracle_threshold': oracle[1], 'oracle_test': prf(y_test, [p >= oracle[1] for p in pt])}


def main():
    ls = lc.modules()
    ls.lab_modules()
    out_path = lc.ROOT/lc.OUT/'lr'/'gold_control.json'
    assert not out_path.exists(), out_path
    out = {'description': __doc__, 'benchmarks': {}}
    for b in ls.RUNS:
        rec = lc.Records(b)

        def ok(a, c):
            return (a, c) in rec.pair_rids or ((a in rec.by_id['left'] or a in rec.new.get('left', {}))
                                               and (c in rec.by_id['right'] or c in rec.new.get('right', {})))
        valid = json.loads((lc.ROOT/'data'/b/'valid.json').read_text())
        pairs_test, y_test = lc.test_pairs(b)
        X_test = rec.features(pairs_test)
        held = {(c['entity_ids']['left'], c['entity_ids']['right']) for c in valid} | set(pairs_test)
        rows = official_train(b)
        gold, seen = [], set()
        for r in rows:
            a, c = str(r['id_left']), str(r['id_right'])
            if (a, c) not in held and (a, c) not in seen:
                seen.add((a, c)); gold.append({'pair_id': f'{a}#{c}', 'pair': (a, c), 'label': int(r['label'])})
        assert all(ok(*g['pair']) for g in gold), b
        train, tune_split = lc.split(b, gold)
        X, y = rec.features([g['pair'] for g in train]), [g['label'] for g in train]
        Xv, yv = rec.features([g['pair'] for g in tune_split]), [g['label'] for g in tune_split]
        res = {'train_pairs_official': len(rows), 'pairs_after_holdout_removal': len(gold), 'train_pairs': len(train),
               'tuning_pairs': len(tune_split), 'pos_share': sum(g['label'] for g in gold)/len(gold),
               **scored(tune(X, y, Xv, yv), X_test, y_test)}
        va = [((c['entity_ids']['left'], c['entity_ids']['right']), c['label']) for c in valid]
        covered = sum(ok(*p) for p, _ in va)
        res['official_validation_variant'] = {'valid_pairs': len(va), 'valid_pairs_with_embeddings': covered}
        if covered == len(va):
            Xg, yg = rec.features([g['pair'] for g in gold]), [g['label'] for g in gold]
            Xo, yo = rec.features([p for p, _ in va]), [y for _, y in va]
            res['official_validation_variant'] |= {'trained_on': 'all pairs after holdout removal',
                                                   **scored(tune(Xg, yg, Xo, yo), X_test, y_test)}
        out['benchmarks'][b] = res
        print(b, json.dumps(res), flush=True)
    bs = out['benchmarks'].values()
    out['mean'] = {'test_f1': sum(r['test']['f1'] for r in bs)/len(bs), 'oracle_test_f1': sum(r['oracle_test']['f1'] for r in bs)/len(bs)}
    with out_path.open('x') as f:
        json.dump(out, f, indent=1)
    print(out['mean'])


if __name__ == '__main__':
    main()
