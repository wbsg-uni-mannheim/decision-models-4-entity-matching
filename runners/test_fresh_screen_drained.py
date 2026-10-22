import copy
import json
from pathlib import Path
import socket
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import expanded_screen
import expanded_protocol
import protocol
import fresh_screen_drained as component
import test_authorized_recovery_200 as budget_fixture
import test_fresh_supplement_drained as http_fixture


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.original = budget_fixture.AuthorizedBudget().setup_root(self.root, 100, 3)
        self.target = self.root/'replications/run02'
        (self.target/'artifacts').mkdir(parents=True)
        self.cases = {}
        for b in expanded_screen.SCREEN_BENCHMARKS:
            data = self.target/'data'/b; data.mkdir(parents=True)
            cases = [{'benchmark': b, 'pair_id': str(i), 'label': 1,
                      'records': {'left': {'title': str(i), 'brand': 'A', 'price': None},
                                  'right': {'title': str(i), 'brand': 'A', 'price': 1.0}}} for i in range(2)]
            for c in cases: c['input_hash'] = protocol.digest(c['records'])
            (data/'expanded_screen.json').write_text(json.dumps(cases))
            self.cases[b] = cases
        self.progress = self.target/'artifacts/progress.json'
        self.progress.write_text(json.dumps({'supplement_billiger50': {'status': 'complete_automated'}}))
        self.inference = types.SimpleNamespace(**{**vars(self.original), 'PROJECT': self.target,
                                                  'key': lambda: 'synthetic'})
        self.screen = types.SimpleNamespace(**{**vars(expanded_screen), 'PROJECT': self.target, 'verify': Mock()})
        self.journal = self.target/'artifacts/screen_attempt'
        self.bodies = []
        guard = patch.object(socket, 'socket', side_effect=AssertionError('Network forbidden'))
        guard.start(); self.addCleanup(guard.stop)

    def execute(self, run_id=2):
        return component.run_screen(self.root, self.target, run_id, self.inference,
                                    self.screen, expanded_protocol, protocol, self.journal)

    def http(self, request, **kwargs):
        return http_fixture.SupplementTests.http(self, request, **kwargs)

    def test_full_matrix_real_payload_parser_cache_and_cached_answers(self):
        isolated = types.SimpleNamespace(**{**vars(self.inference), 'payload': self.screen.payload,
            'account': component.shared_account(self.inference.account, self.root)})
        _, cell, _ = component.make_wrapper(self.root, 'core').authorized_build(isolated, 2)
        with patch('urllib.request.urlopen', side_effect=self.http):
            cell('jev', 'X_R_aligned', 'wdc', 'expanded_screen', workers=8)
        p = self.target/'runs/quality/jev/X_R_aligned/wdc/expanded_screen.jsonl'
        old = p.read_bytes(); progress = self.progress.read_bytes()
        self.bodies = []
        with patch('urllib.request.urlopen', side_effect=self.http) as http:
            result = self.execute()
        self.assertEqual(http.call_count, 1132)
        self.assertEqual(len(result['cells']), 567)
        self.assertTrue(result['inference_complete'])
        self.assertEqual(len(result['models']), 7)
        expected = [expanded_protocol.payload(c, m, v) for m in protocol.MODELS
                    for v in expanded_protocol.VARIANTS for b, cases in self.cases.items()
                    if (m, v, b) != ('jev', 'X_R_aligned', 'wdc') for c in cases]
        self.assertCountEqual(self.bodies, expected)
        self.assertEqual(p.read_bytes(), old)
        self.assertEqual(self.progress.read_bytes(), progress)
        self.assertFalse((self.target/'artifacts/results').exists())
        self.assertFalse(result['admission_granted'])
        self.assertFalse(result['completion_marker_written'])
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 101.134)
        self.assertAlmostEqual(budget['reserved'], 3)
        self.assertIs(self.original.payload, protocol.payload)
        self.assertEqual(self.original.account.__defaults__, (0, 100))
        # Every model retains original variant/benchmark order; each cell uses 8.
        for m in protocol.MODELS:
            for v in expanded_protocol.VARIANTS:
                for b in self.cases:
                    path = self.target/'runs/quality'/m/v/b/'expanded_screen.executions.jsonl'
                    self.assertEqual(json.loads(path.read_text())['workers'], 8)
        with self.assertRaises(ValueError): self.execute()

    def test_budget_rejections_drain_seven_models_without_paid_attempt(self):
        path = self.root/'artifacts/budget.json'
        value = json.loads(path.read_text()); value['spent'] = 196.99; path.write_text(json.dumps(value))
        before = path.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            result = self.execute()
        self.assertFalse(result['inference_complete'])
        self.assertTrue(all(m['status'] == 'failed' for m in result['models']))
        self.assertEqual(len(result['models']), 7)
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in result['cells']), 14)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']), 0)
        self.assertEqual(path.read_bytes(), before)
        paths = list((self.target/'runs').rglob('*.operations.jsonl'))
        self.assertEqual(len(paths), 7)
        rows = [json.loads(s) for s in paths[0].read_text().splitlines()]
        rows[0]['operational_stop'].update(budget_reservation_rejected=False,
            outcome_unknown=True, phase='future', exception_type='ValueError')
        rows[0]['error'] = 'unknown outcome'
        paths[0].write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'Unknown screening outcomes'):
            component.inventory(self.target, 2, self.screen, protocol)

    def test_parser_failure_preserves_sibling_and_drains_other_models(self):
        failed = expanded_protocol.payload(self.cases['abt-buy'][0], 'jev', 'B0')
        def http(request, **kwargs):
            if json.loads(request.data) == failed:
                return http_fixture.Response(json.dumps({'usage': {'cost': .001},
                    'model': protocol.MODELS['jev']['id'],
                    'provider': protocol.MODELS['jev']['provider'].split('/')[0]}).encode())
            return self.http(request, **kwargs)
        with patch('urllib.request.urlopen', side_effect=http): result = self.execute()
        self.assertFalse(result['inference_complete'])
        self.assertEqual(sum(m['status'] == 'failed' for m in result['models']), 1)
        self.assertEqual(sum(m['status'] == 'complete' for m in result['models']), 6)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']), 1)
        self.assertEqual(sum(c['responses'] for c in result['cells']), 973)
        self.assertAlmostEqual(sum(c['known_usd'] for c in result['cells']), .974)
        self.assertAlmostEqual(json.loads((self.root/'artifacts/budget.json').read_text())['spent'], 100.974)
        self.assertFalse((self.target/'artifacts/results').exists())
        with self.assertRaisesRegex(ValueError, 'Existing failures'):
            component.run_screen(self.root, self.target, 2, self.inference, self.screen,
                expanded_protocol, protocol, self.target/'artifacts/another_attempt')
        self.assertFalse((self.target/'artifacts/another_attempt').exists())

    def test_prerequisite_source_matrix_workspace_and_report_gates(self):
        self.progress.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'prerequisite'): self.execute()
        self.progress.write_text(json.dumps({'supplement_billiger50': {'status': 'complete_automated'}}))
        self.screen.PROJECT = self.root
        with self.assertRaisesRegex(ValueError, 'workspace'): self.execute()
        self.screen.PROJECT = self.target
        self.screen.__file__ = str(self.progress)
        with self.assertRaisesRegex(ValueError, 'source changed'): self.execute()
        self.screen.__file__ = expanded_screen.__file__
        self.screen.VARIANTS = ['B0']
        with self.assertRaisesRegex(ValueError, 'matrix'): self.execute()
        self.screen.VARIANTS = expanded_protocol.VARIANTS
        (self.target/'artifacts/results/expanded_screen').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'reports'): self.execute()
        self.assertFalse(self.journal.exists())

    def test_run_three_and_raw_input_request_variant_parser_identity(self):
        oldtarget = self.target
        self.target = self.root/'replications/run03'; oldtarget.rename(self.target)
        self.inference.PROJECT = self.target; self.screen.PROJECT = self.target
        self.journal = self.target/'artifacts/screen_attempt'
        with patch('urllib.request.urlopen', side_effect=self.http): result = self.execute(3)
        self.assertTrue(result['inference_complete'])
        self.assertFalse(oldtarget.exists())
        p = self.target/'runs/quality/jev/X_R_aligned/wdc/expanded_screen.jsonl'
        original = [json.loads(s) for s in p.read_text().splitlines()]
        self.assertTrue(all(r['study_run_id'] == 3 for r in original))
        for change in ({'study_run_id': 2}, {'request_hash': 'changed'}, {'prediction': 0},
                       {'input_hash': 'changed'}, {'variant': 'B0'}):
            rows = copy.deepcopy(original); rows[0].update(change)
            with self.assertRaises(ValueError):
                component.audit_cell(self.cases['wdc'], rows, [], protocol, expanded_protocol.payload,
                                     'jev', 'X_R_aligned', 'wdc', 3)
        component.audit_cell(self.cases['wdc'], original, [], protocol, expanded_protocol.payload,
                             'jev', 'X_R_aligned', 'wdc', 3)
        self.assertAlmostEqual(json.loads((self.root/'artifacts/budget.json').read_text())['reserved'], 3)


if __name__ == '__main__': unittest.main()
