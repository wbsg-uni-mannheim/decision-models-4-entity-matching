import json
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import expanded_calibration as cal
from similarity_baseline import tokens, scores
from prepare import case
from protocol import digest


class CalibrationTests(unittest.TestCase):
    def test_threshold_ties_and_extremes(self):
        self.assertEqual(cal.threshold([0, 1], [0., 1.]), .5)
        self.assertAlmostEqual(cal.threshold([0, 0, 1, 1], [.1, .7, .8, .9]), .8)
        with self.assertRaises(ValueError):
            cal.threshold([0, 0], [.1, .2])
        rng = np.random.default_rng(18)
        for _ in range(30):
            y = np.r_[0, 1, rng.integers(0, 2, 28)]
            p = np.round(rng.random(30), 1)
            candidates = sorted(set(p.tolist()+[.5, float(np.nextafter(1., 2.))]))
            expected = min(candidates, key=lambda t: (-cal.analyze.metric(y, p >= t)['f1'], abs(t-.5), -t))
            self.assertEqual(cal.threshold(y, p), expected)

    def test_nested_independent_halves(self):
        small = cal.partition(list(range(500)), 25)
        big = cal.partition(list(range(500)), 100)
        self.assertFalse(set(small[0]) & set(small[1]))
        self.assertTrue(set(small[0]) <= set(big[0]))
        self.assertTrue(set(small[1]) <= set(big[1]))
        with self.assertRaises(ValueError):
            cal.partition(list(range(200)), 500)

    def test_calibration_endpoint_numerics_and_missing_class(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([0., .9, .1, 1., .2, .8])
        for method in cal.METHODS:
            parameters = cal.fit(method, y, p)
            q = cal.transform(parameters, p)
            self.assertTrue(np.isfinite(q).all())
            self.assertTrue(((q >= 0) & (q <= 1)).all())
        for method in ['temperature', 'logistic']:
            with self.assertRaises(ValueError):
                cal.fit(method, [0, 0], [.1, .2])

    def test_similarity_uses_only_descriptive_values(self):
        self.assertEqual(tokens({'title': 'ＡＢＣ abc', 'empty': None}), {'abc'})
        c = {'label': 1, 'pair_id': 'gold-secret', 'records': {'left': {'name': 'A B'}, 'right': {'title': 'B C'}}}
        self.assertAlmostEqual(scores([c])[0], 1/3)
        self.assertEqual(scores([{'records': {'left': {}, 'right': {}}}])[0], 0.)

    def test_preparation_reuses_route_and_excludes_other_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'data/source/abt-buy/abt-buy-valid.csv'
            source.parent.mkdir(parents=True)
            raw = [{'pair_id': str(i), 'id_left': str(i), 'id_right': str(i), 'label': i % 2,
                    **{f+'_'+side: str(i)+' '+f for f in ['name', 'description', 'price'] for side in ['left', 'right']}}
                   for i in range(650)]
            with source.open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=raw[0].keys())
                writer.writeheader()
                writer.writerows(raw)
            rows = [case(r, 'abt-buy') for r in raw]
            path = root/'data/abt-buy'
            path.mkdir()
            splits = {'route': rows[:200], 'prompt': rows[200:210], 'expanded_screen': rows[210:220], 'test': rows[220:230]}
            for split, data in splits.items():
                (path/(split+'.json')).write_text(json.dumps(data))
            (root/'configs').mkdir()
            (root/'configs/manifest.json').write_text(json.dumps({'benchmarks': {'abt-buy': {
                'source_sha256': {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
                'splits': {'route': {'sha256': digest(splits['route'])}}}}}))
            with patch.object(cal, 'PROJECT', root), patch.object(cal, 'BENCHMARKS', {'abt-buy': {}}), \
                    patch.object(cal, 'DECISION_MODELS', ['jev']):
                cal.prepare()
                manifest = cal.verify()
                self.assertEqual(manifest['new_decisions'], 300)
                self.assertEqual(manifest['benchmarks']['abt-buy']['reused_route_n'], 200)
                pool = json.loads((path/'calibration_pool.json').read_text())
                forbidden = {r['input_hash'] for split in ['test', 'prompt', 'expanded_screen'] for r in splits[split]}
                self.assertFalse({r['input_hash'] for r in pool} & forbidden)
                cal.prepare()  # Exact-byte frozen preparation is safely repeatable.

    def test_selection_frozen_before_any_test_read_and_missing_budgets_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'pair_id': str(i), 'label': (i//2) % 2,
                     'entity_ids': {'left': str(i), 'right': str(i)}} for i in range(32)]
            y = np.array([r['label'] for r in rows])
            p = np.where(y, .8, .2)
            def dev(model, b):
                return rows, y, p
            def load(model, variant, b, split):
                self.assertEqual(split, 'test')
                self.assertTrue((root/'configs/calibration_selection.json').exists())
                return {'cases': rows, 'y': y, 'prob': p, 'pred': p >= .5}
            with patch.object(cal, 'PROJECT', root), patch.object(cal, 'DECISION_MODELS', ['jev']), \
                    patch.object(cal, 'BENCHMARKS', {'abt-buy': {}, 'dblp-acm': {}}), \
                    patch.object(cal, 'development', dev), patch.object(cal.analyze, 'load', load):
                entries = cal.selection()
                self.assertTrue(any(e['status'] == 'unavailable' and e['labels_per_source'] == 500 for e in entries))
                self.assertTrue(all(e['benchmark'] not in e['sources'] for e in entries if e['regime'] == 'leave_one_out'))
                # Runs full numerical report without making API calls.
                cal.report(entries)
                result = json.loads((root/'artifacts/results/expanded_calibration/metrics.json').read_text())
                self.assertEqual(len(result), 36)
                self.assertTrue(all(r['f1'] == 1. for r in result))
                self.assertEqual(cal.selection(), entries)


if __name__ == '__main__':
    unittest.main()
