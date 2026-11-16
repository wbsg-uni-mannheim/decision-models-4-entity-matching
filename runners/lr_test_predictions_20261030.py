"""Per-pair LR test predictions for the paired bootstrap (labeled_chapter_20261028.lr stored only F1). Read-only, free.

Same records, embeddings, features, C grid, threshold grid and tie rule as labeled_chapter_20261028.lr; the chosen
C, threshold and the rounded test F1 must equal lr/results.json. Writes lr/test_predictions.json (write-once).
Run with /path/to/envs/ditto-modern/bin/python.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labeled_chapter_20261028 as lc  # noqa: E402


def main():
    from sklearn.linear_model import LogisticRegression
    ls = lc.modules()
    ref = json.loads((lc.ROOT/lc.OUT/'lr'/'results.json').read_text())
    out_path = lc.ROOT/lc.OUT/'lr'/'test_predictions.json'
    assert not out_path.exists(), out_path
    out = {}
    for b in ls.RUNS:
        rec = lc.Records(b)
        cases = json.loads((lc.ROOT/'data'/b/'test.json').read_text())
        pairs_test, y_test = lc.test_pairs(b)
        X_test = rec.features(pairs_test)
        for size in lc.SIZES:
            data = {}
            for name in ('train', 'valid'):
                rows = lc.manifest_rows(b, size, name)
                data[name] = (rec.features([(r['id_left'], r['id_right']) for r in rows]), [r['label'] for r in rows])
            best = None
            for C in lc.C_GRID:
                m = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', C=C).fit(*data['train'])
                prob = m.predict_proba(data['valid'][0])[:, 1]
                for t in lc.THRESHOLDS:
                    score = lc.f1(data['valid'][1], [p >= t for p in prob])
                    if best is None or score > best[0]:
                        best = (score, C, t, m)
            score, C, t, m = best
            pred = [int(p >= t) for p in m.predict_proba(X_test)[:, 1]]
            r = ref[b][str(size)]
            assert (C, t, round(lc.f1(y_test, pred), 4)) == (r['C'], r['threshold'], r['test_f1']), (b, size, C, t)
            out.setdefault(b, {})[str(size)] = {'C': C, 'threshold': t, 'test_f1': lc.f1(y_test, pred),
                                                'predictions': {c['pair_id']: p for c, p in zip(cases, pred)}}
            print(b, size, C, t, round(lc.f1(y_test, pred), 4), flush=True)
    with out_path.open('x') as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()
