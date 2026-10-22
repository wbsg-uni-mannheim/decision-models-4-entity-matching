import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from calibration_run_export import same, reconstruct, export_calibration, METRICS


def fixture(root):
    for path in ['artifacts', 'configs', 'data/b']:
        (root/path).mkdir(parents=True)
    cases = [{'pair_id': str(i), 'label': i % 2} for i in range(4)]
    (root/'data/b/calibration_pool.json').write_text(json.dumps(cases))
    saved = []
    for regime in ['within', 'leave_one_out', 'opposite_domain']:
        base = dict(model='m', benchmark='b', regime=regime, sources=['b'])
        saved += [{**base, 'labels_per_source': 4, 'label_budget_total': 4,
                   'method': 'raw', 'status': 'frozen', 'parameters': {'method': 'raw'},
                   'threshold': .5, 'fit_n': 2, 'tune_n': 2, 'fit_positive': 0, 'tune_positive': 2},
                  {**base, 'labels_per_source': 6, 'label_budget_total': 6,
                   'method': 'all', 'status': 'unavailable', 'reason': 'Insufficient pool'}]
    (root/'configs/calibration_selection.json').write_text(json.dumps(saved))
    (root/'artifacts/expanded_calibration_complete.json').write_text(json.dumps({
        'status': 'complete_with_explicit_unavailable_fits', 'selection_sha256': saved,
        'new_decisions': 0}))
    def partition(rows, n):
        if n > len(rows):
            raise ValueError('Insufficient pool')
        return np.arange(n)[::2], np.arange(n)[1::2]
    module = SimpleNamespace(DECISION_MODELS=['m'], BENCHMARKS=['b'], BUDGETS=[4, 6], METHODS=['raw'],
        source_benchmarks=lambda b, r: ['b'], partition=partition, fit=lambda *args: {'method': 'raw'},
        threshold=lambda *args: .5, transform=lambda pars, p: p, payload=object(),
        diagnostics=lambda *args: {k: .5 for k in METRICS},
        verify=Mock(return_value={'new_decisions': 0, 'benchmarks': {'b': {'extra_n': 0}}}))
    data = {'cases': cases, 'prob': np.array([.1, .9, .2, .8]), 'y': np.array([0, 1, 0, 1])}
    return module, data, saved


class CalibrationExportTests(unittest.TestCase):
    def test_complete_matrix_and_explicit_unavailable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module, data, _ = fixture(root)
            analyze = SimpleNamespace(payload='original')
            calls = []
            def cell(m, v, b, split):
                calls.append(split)
                self.assertIs(analyze.payload, module.payload)
                return data, 0
            rows = export_calibration(root, module, analyze, cell, 2, lambda x: x)
            self.assertEqual(len(rows), 108)
            self.assertEqual(sum(r['value'] is None for r in rows), 54)
            self.assertEqual(calls[0], 'route')
            self.assertTrue(all(r['run_id'] == 2 for r in rows))
            self.assertEqual(analyze.payload, 'original')

    def test_changed_development_stops_before_test_and_restores(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module, data, _ = fixture(root)
            module.threshold = lambda *args: .6
            cell = Mock(return_value=(data, 0))
            analyze = SimpleNamespace(payload='original')
            with self.assertRaisesRegex(ValueError, 'threshold'):
                export_calibration(root, module, analyze, cell, 1, lambda x: x)
            self.assertEqual([c.args[3] for c in cell.call_args_list], ['route'])
            self.assertEqual(analyze.payload, 'original')

    def test_partial_matrix_rejected(self):
        with TemporaryDirectory() as tmp:
            module, data, saved = fixture(Path(tmp))
            dev = {('m', 'b'): (data['cases'], data['y'], data['prob'])}
            with self.assertRaisesRegex(ValueError, 'matrix'):
                reconstruct(module, dev, saved[:-1])
            saved[1]['reason'] = 'fabricated'
            with self.assertRaisesRegex(ValueError, 'reason'):
                reconstruct(module, dev, saved)

    def test_completion_and_source_alignment(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module, data, _ = fixture(root)
            analyze = SimpleNamespace(payload='original')
            data['cases'][0]['label'] = 1
            with self.assertRaisesRegex(ValueError, 'alignment'):
                export_calibration(root, module, analyze, Mock(return_value=(data, 0)), 1, lambda x: x)
            (root/'artifacts/expanded_calibration_complete.json').write_text('{"status":"running"}')
            cell = Mock()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                export_calibration(root, module, analyze, cell, 1, lambda x: x)
            cell.assert_not_called()

    def test_parameter_tolerance_does_not_allow_threshold_drift(self):
        same({'parameters': {'temperature': 1.}}, {'parameters': {'temperature': 1.+1e-8}})
        with self.assertRaises(ValueError):
            same({'threshold': .5}, {'threshold': .50001})
        with self.assertRaises(ValueError):
            same({'parameters': {'temperature': 1.}}, {'parameters': {'temperature': float('nan')}})


if __name__ == '__main__':
    unittest.main()
