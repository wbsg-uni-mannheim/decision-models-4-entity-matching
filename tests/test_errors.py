import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import expanded_errors as e


def pair(pid, left, right):
    return {'pair_id': pid, 'entity_ids': {'left': left, 'right': right},
            'records': {'left': {'title': 'A 12', 'missing': None}, 'right': {'title': 'A 13', 'missing': ''}}}


class ErrorTests(unittest.TestCase):
    def test_transitivity_only_observed_edges(self):
        cases = [pair('ax', 'a', 'x'), pair('ay', 'a', 'y'), pair('bx', 'b', 'x'), pair('by', 'b', 'y')]
        d = e.graph_diagnostics(cases, [1, 1, 1, 0])
        self.assertEqual(d['contradiction_pair_ids'], ['by'])
        self.assertEqual(d['nodes_with_multiple_positive_partners'], 2)
        self.assertEqual(e.graph_diagnostics(cases[:3], [1, 1, 1])['negative_edges_with_positive_path'], 0)
        self.assertEqual(e.graph_diagnostics(cases, [0, 0, 0, 0])['negative_edges_with_positive_path'], 0)

    def test_side_namespaces_and_no_one_to_one_assumption(self):
        cases = [pair('a', 'id', 'id'), pair('b', 'id', 'different')]
        d = e.graph_diagnostics(cases, [0, 1])
        self.assertEqual(d['negative_edges_with_positive_path'], 0)

    def test_features_do_not_depend_on_labels(self):
        c = pair('a', 'a', 'b')
        fs = e.features(c)
        self.assertTrue(fs['numeric_set_discrepancy'])
        self.assertEqual(fs['missing_fields'], 2)
        c.update(label=1, prediction=0)
        self.assertEqual(e.features(c), fs)
        groups = e.masks([c], [fs['chars'], fs['chars'], fs['chars']])
        self.assertEqual(sum(groups['length_bin_'+str(i)][0] for i in range(1, 5)), 1)

    def test_complete_report_with_empty_strata_and_no_luna_probabilities(self):
        cases = [pair('ax', 'a', 'x'), pair('ay', 'a', 'y')]
        data = {'cases': cases, 'y': np.array([1, 0]), 'pred': np.array([1, 1]), 'prob': None}
        decision = {**data, 'pred': np.array([1, 0]), 'prob': np.array([.99, .1])}
        spec = {'benchmarks': {'sample': {'length_cuts': [1, 2, 3]}}, 'confidence_thresholds': [.9, .99]}
        with tempfile.TemporaryDirectory() as td, patch.object(e, 'PROJECT', Path(td)), patch.object(e, 'MODELS', ['luna', 'jev']), patch.object(e, 'verify', return_value=spec), patch.object(e.analyze, 'load', side_effect=lambda m, *a: data if m == 'luna' else decision):
            e.report()
            out = Path(td)/'artifacts/results/expanded_errors'
            rows = json.loads((out/'strata.json').read_text())
            self.assertEqual(len(rows), 18)
            self.assertTrue(any(r['n'] == 0 and r['accuracy'] is None for r in rows))
            conf = json.loads((out/'high_confidence.json').read_text())
            self.assertFalse(conf[0]['available'])
            self.assertEqual(conf[2]['errors'], 0)
            self.assertTrue((Path(td)/'artifacts/expanded_errors_complete.json').exists())

    def test_error_groups_partition(self):
        y, model, luna = np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), np.array([0, 1, 1, 0])
        counts = e.agreement(y, model, luna)
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(sum(counts.values()), 4)


if __name__ == '__main__':
    unittest.main()
