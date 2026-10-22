import copy
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import expanded_confirmation
from unittest.mock import patch
import expanded_rules as r
import protocol


class RulesTests(unittest.TestCase):
    def test_complete_family_rules_validation(self):
        obj = {'schema_version': 1, 'generator_model': 'gpt-6-astra', 'families': {b: {'rules': ['Compare identity evidence.']} for b in protocol.BENCHMARKS}}
        self.assertEqual(len(r.validate_rules(obj)), 8)
        for mutate in [lambda x: x['families'].pop('wdc'), lambda x: x.update(generator_model='luna'),
                       lambda x: x['families']['wdc'].update(rules=['x'*241]),
                       lambda x: x['families']['wdc'].update(rules=['same', 'same'])]:
            other = copy.deepcopy(obj);mutate(other)
            with self.assertRaises(AssertionError):
                r.validate_rules(other)

    def test_same_rules_all_models_and_billiger_transfer(self):
        case = {'benchmark': 'billiger-de-50un', 'records': {'left': {'name': 'A'}, 'right': {'name': 'B'}}, 'pair_id': 'SECRET', 'label': 1}
        with patch.object(r, 'RULES', {'billiger-de': ['Compare capacities.']}):
            for m in protocol.MODELS:
                original = protocol.payload(case, m)
                value = r.payload(case, m)
                if m == 'luna':
                    self.assertEqual(value['messages'][1], original['messages'][1])
                    self.assertIn('Compare capacities.', value['messages'][0]['content'])
                    self.assertEqual(value['response_format'], original['response_format'])
                else:
                    self.assertEqual(value['state'], original['state'])
                    self.assertIn('Compare capacities.', value['questions']['match']['instructions'])
                    self.assertEqual(value['questions']['match']['type'], 'noul')
                self.assertNotIn('SECRET', str(value))
                self.assertEqual(value['provider'], original['provider'])

    def test_manual_control_reuses_existing_confirmation_request(self):
        for b in [*protocol.BENCHMARKS, 'billiger-de-50un']:
            c = {'benchmark': b, 'records': {'left': {'title': 'a'}, 'right': {'title': 'b'}}}
            for m in protocol.MODELS:
                self.assertEqual(r.payload(c, m, r.MANUAL), expanded_confirmation.payload(c, m, r.MANUAL))
                self.assertEqual(r.payload(c, m, 'B0'), protocol.payload(c, m, 'B0'))

    def test_report_requires_all_controls_and_excludes_supplement_from_macro(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);(root/'configs').mkdir()
            (root/'configs/learned_rules.json').write_text('{}')
            y = np.array([0, 1, 0, 1])
            calls = []
            def load(m, v, b, split, namespace='quality'):
                calls.append((m, v, b, split, namespace))
                self.assertEqual(namespace, 'rules' if v == 'R_rules' else 'quality')
                pred = y if v == 'R_rules' and b != 'billiger-de-50un' else np.zeros(4, dtype=int)
                cases = [{'pair_id': str(i), 'entity_ids': {'left': str(i), 'right': str(i)}} for i in range(4)]
                return {'cases': cases, 'y': y, 'pred': pred, 'costs': np.ones(4), 'unknown_cost_attempts': 2}
            with patch.object(r, 'PROJECT', root), patch.object(r, 'MODELS', {'jev': {}, 'luna': {}}), \
                    patch.object(r, 'BENCHMARKS', {'billiger-de': {}, 'wdc': {}}), patch.object(r.analyze, 'load', load):
                r.report({'frozen': True})
                metrics = json.loads((root/'artifacts/results/expanded_rules/metrics.json').read_text())
                intervals = json.loads((root/'artifacts/results/expanded_rules/intervals.json').read_text())
                macro = json.loads((root/'artifacts/results/expanded_rules/macro.json').read_text())
                costs = json.loads((root/'artifacts/results/expanded_rules/costs.json').read_text())
                self.assertEqual(len(metrics), 30)
                self.assertEqual(len(intervals), 12)
                self.assertEqual({i['control'] for i in intervals}, {'B0', r.MANUAL})
                self.assertEqual({i['macro_match_f1'] for i in macro if i['variant'] == 'R_rules'}, {1.})
                self.assertTrue(all(c['known_cost_usd'] == 4 and c['unknown_cost_attempts'] == 2 for c in costs))
                self.assertEqual(len(calls), len(set(calls)))
            (root/'artifacts/expanded_rules_complete.json').unlink()
            def incomplete(*a, **kw):
                raise AssertionError('Incomplete manual cell')
            with patch.object(r, 'PROJECT', root), patch.object(r.analyze, 'load', incomplete):
                with self.assertRaisesRegex(AssertionError, 'Incomplete'):
                    r.report({})
                self.assertFalse((root/'artifacts/expanded_rules_complete.json').exists())

    def test_development_and_test_namespace_matrix(self):
        calls = []
        with patch.object(r, 'BENCHMARKS', {'wdc': {}, 'billiger-de': {}}), \
                patch.object(r.run, 'run_cell', lambda *a, **kw: calls.append((a, kw))):
            r.quality_task('jev', 'prompt')
            self.assertEqual(len(calls), 4)
            self.assertFalse(any(a[2] == 'billiger-de-50un' for a, kw in calls))
            r.quality_task('jev', 'test')
            self.assertEqual(len(calls), 10)
            self.assertTrue(all(kw['namespace'] == r.CONDITIONS[a[1]] for a, kw in calls))


if __name__ == '__main__':
    unittest.main()
