import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import analyze
import billiger50
from protocol import MODELS, BENCHMARKS, canonical, digest, payload


class SupplementTests(unittest.TestCase):
    def test_same_information_and_prompt(self):
        records = {s: {f: 'example' for f in BENCHMARKS['billiger-de']['fields']}
                   for s in ['left', 'right']}
        old = {'benchmark': 'billiger-de', 'records': records}
        new = {**old, 'benchmark': billiger50.BENCHMARK}
        for model in MODELS:
            self.assertEqual(payload(old, model), payload(new, model))
        self.assertEqual(len(BENCHMARKS), 8)

    def test_offline_report_and_incomplete_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [{'benchmark': billiger50.BENCHMARK, 'pair_id': str(i),
                      'label': i % 2, 'records': {'left': {'name': str(i)}, 'right': {'name': str(i)}},
                      'entity_ids': {'left': str(i), 'right': str(i)}} for i in range(24)]
            folder = root/'data'/billiger50.BENCHMARK
            folder.mkdir(parents=True)
            (folder/'test.json').write_text(canonical(cases))
            for model in MODELS:
                path = root/'runs/quality'/model/'B0'/billiger50.BENCHMARK
                path.mkdir(parents=True)
                rows = [{'pair_id': c['pair_id'], 'request_hash': digest(payload(c, model)),
                         'prediction': c['label'], 'probability_match': .8 if c['label'] else .2,
                         'attempts': [{'cost': .001, 'seconds': .1}, {'cost': None, 'seconds': .1}]}
                        for c in cases]
                (path/'test.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows))
            original = analyze.bootstrap_delta
            with patch.object(billiger50, 'PROJECT', root), patch.object(analyze, 'PROJECT', root), \
                 patch.object(billiger50, 'verify'), patch.object(billiger50, 'bootstrap_delta',
                 side_effect=lambda *args: original(*args, repeats=20)):
                billiger50.analyze()
                output = root/'artifacts/results/billiger50'
                metrics = json.loads((output/'metrics.json').read_text())
                self.assertEqual(len(metrics), 7)
                self.assertTrue(all(m['f1'] == 1. and m['unknown_cost_attempts'] == 24 for m in metrics))
                self.assertIn('75.71', (output/'comparison.md').read_text())
                (path/'test.jsonl').write_text('')
                with self.assertRaises(AssertionError):
                    billiger50.analyze()


if __name__ == '__main__':
    unittest.main()
