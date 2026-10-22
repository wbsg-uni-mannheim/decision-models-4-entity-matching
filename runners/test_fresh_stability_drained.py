import copy
import json
from pathlib import Path
import socket
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import expanded_protocol
import expanded_stability
import protocol
import fresh_stability_drained as component
import test_authorized_recovery_200 as budget_fixture
import test_fresh_supplement_drained as http_fixture


class StabilityTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.original = budget_fixture.AuthorizedBudget().setup_root(self.root, 100, 3)
        self.target = self.root/'replications/run02'
        (self.target/'artifacts').mkdir(parents=True)
        self.cases = {}
        for b in component.BENCHMARKS:
            data = self.target/'data'/b; data.mkdir(parents=True)
            cases = [{'benchmark': b, 'pair_id': str(i), 'label': 1,
                      'records': {'left': {'title': str(i), 'price': None},
                                  'right': {'title': str(i), 'price': 1.0}}} for i in range(2)]
            for c in cases: c['input_hash'] = protocol.digest(c['records'])
            for split in ('expanded_screen', 'expanded_repeat'):
                (data/(split+'.json')).write_text(json.dumps(cases))
            self.cases[b] = cases
        self.progress = self.target/'artifacts/progress.json'
        self.progress.write_text(json.dumps({'expanded_E2_E3_screen': {'status': 'complete'}}))
        self.inference = types.SimpleNamespace(**{**vars(self.original), 'PROJECT': self.target,
                                                  'key': lambda: 'synthetic'})
        self.manifest = {'models': protocol.MODELS, 'new_decisions_before_cache': 25200,
            'benchmarks': {b: {} for b in component.BENCHMARKS},
            'repeats': {'n': 100, 'additional_runs': 2, 'fresh_namespaces': component.REPEATS},
            'requests': {b: {m: {v: expanded_stability.payload(
                {'benchmark': b, 'records': {'left': {}, 'right': {}}}, m, v)
                for v in component.INTERACTIONS} for m in protocol.MODELS} for b in component.BENCHMARKS}}
        self.stability = types.SimpleNamespace(**{**vars(expanded_stability), 'PROJECT': self.target,
                                                   'verify': Mock(return_value=self.manifest)})
        self.journal = self.target/'artifacts/stability_attempt'
        self.bodies = []
        guard = patch.object(socket, 'socket', side_effect=AssertionError('Network forbidden'))
        guard.start(); self.addCleanup(guard.stop)
        self.seed_controls()

    def http(self, request, **kwargs):
        return http_fixture.SupplementTests.http(self, request, **kwargs)

    def cell(self, run_id=2):
        isolated = types.SimpleNamespace(**{**vars(self.inference), 'payload': self.stability.payload,
            'account': component.shared_account(self.inference.account, self.root)})
        return component.make_wrapper(self.root, 'core').authorized_build(isolated, run_id)[1]

    def seed_controls(self):
        cell = self.cell()
        with patch('urllib.request.urlopen', side_effect=self.http):
            for ns, m, v, b, split in component.matrix(self.stability, controls=True):
                cell(m, v, b, split, workers=8, namespace=ns)
        self.bodies = []

    def execute(self, run_id=2):
        return component.run_stability(self.root, self.target, run_id, self.inference,
            self.stability, expanded_protocol, protocol, self.journal)

    def test_full_fixed_matrix_real_requests_parser_cache_repeat_separation(self):
        cell = self.cell()
        with patch('urllib.request.urlopen', side_effect=self.http):
            cell('jev', 'I3_identity_json', 'abt-buy', 'expanded_screen', workers=8)
            cell('jev', 'B0', 'abt-buy', 'expanded_repeat', workers=8, namespace='e3_repeat_1')
        before = component.control_hashes(self.target, self.stability)
        progress = self.progress.read_bytes()
        self.bodies = []
        with patch('urllib.request.urlopen', side_effect=self.http) as http: result = self.execute()
        self.assertEqual(http.call_count, 164)
        self.assertEqual(len(result['cells']), 84)
        self.assertEqual(len(result['controls']), 63)
        self.assertEqual(len(result['models']), 7)
        self.assertTrue(result['inference_complete'])
        expected = []
        for ns, m, v, b, split in component.matrix(self.stability):
            if (ns, m, v, b) not in [('quality', 'jev', 'I3_identity_json', 'abt-buy'),
                                    ('e3_repeat_1', 'jev', 'B0', 'abt-buy')]:
                expected.extend(self.stability.payload(c, m, v) for c in self.cases[b])
            path = self.target/'runs'/ns/m/v/b/(split+'.executions.jsonl')
            self.assertEqual(json.loads(path.read_text())['workers'], 8)
        self.assertCountEqual(expected, self.bodies)
        self.assertEqual(component.control_hashes(self.target, self.stability), before)
        self.assertEqual(self.progress.read_bytes(), progress)
        self.assertFalse((self.target/'artifacts/results').exists())
        self.assertFalse((self.target/'artifacts/expanded_stability_complete.json').exists())
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 100.294)  # 126 controls + 168 new responses
        self.assertAlmostEqual(budget['reserved'], 3)
        self.assertIs(self.original.payload, protocol.payload)
        self.assertEqual(self.original.account.__defaults__, (0, 100))
        with self.assertRaises(ValueError): self.execute()
        # A cached repeat must not be manufactured by copying the other namespace.
        first = self.target/'runs/e3_repeat_1/jev/B0/abt-buy/expanded_repeat.jsonl'
        second = self.target/'runs/e3_repeat_2/jev/B0/abt-buy/expanded_repeat.jsonl'
        second.write_bytes(first.read_bytes())
        with self.assertRaisesRegex(ValueError, 'Copied B0/repeat'):
            component.verify_repeat_separation(self.target, self.stability, protocol)

    def test_budget_rejections_drain_all_models_no_attempt_charge(self):
        path = self.root/'artifacts/budget.json'
        budget = json.loads(path.read_text()); budget['spent'] = 196.99; path.write_text(json.dumps(budget))
        before = path.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')): result = self.execute()
        self.assertFalse(result['inference_complete'])
        self.assertEqual(len(result['models']), 7)
        self.assertTrue(all(m['status'] == 'failed' for m in result['models']))
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in result['cells']), 14)
        self.assertEqual(sum(c['attempts'] for c in result['cells']), 0)
        self.assertEqual(path.read_bytes(), before)
        p = next((self.target/'runs').rglob('*.operations.jsonl'))
        rows = [json.loads(s) for s in p.read_text().splitlines()]
        rows[0]['operational_stop'].update(budget_reservation_rejected=False,
            outcome_unknown=True, phase='future', exception_type='ValueError')
        rows[0]['error'] = 'unknown outcome'
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'Unknown stability outcomes'):
            component.inventory(self.target, 2, self.stability, protocol)

    def test_parser_failure_keeps_sibling_and_other_model_results(self):
        failed = self.stability.payload(self.cases['abt-buy'][0], 'jev', 'I3_identity_json')
        def http(request, **kwargs):
            if json.loads(request.data) == failed:
                return http_fixture.Response(json.dumps({'usage': {'cost': .001},
                    'model': protocol.MODELS['jev']['id'],
                    'provider': protocol.MODELS['jev']['provider'].split('/')[0]}).encode())
            return self.http(request, **kwargs)
        with patch('urllib.request.urlopen', side_effect=http): result = self.execute()
        self.assertEqual(sum(m['status'] == 'failed' for m in result['models']), 1)
        self.assertEqual(sum(m['status'] == 'complete' for m in result['models']), 6)
        self.assertEqual(sum(c['responses'] for c in result['cells']), 145)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']), 1)
        self.assertAlmostEqual(sum(c['known_usd'] for c in result['cells']), .146)
        self.journal = self.target/'artifacts/another_attempt'
        with self.assertRaisesRegex(ValueError, 'Existing failures'): self.execute()
        self.assertFalse(self.journal.exists())

    def test_prerequisite_matrix_inputs_controls_and_source_guards(self):
        self.progress.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'prerequisite'): self.execute()
        self.progress.write_text(json.dumps({'expanded_E2_E3_screen': {'status': 'complete'}}))
        self.manifest['repeats']['additional_runs'] = 1
        with self.assertRaisesRegex(ValueError, 'manifest matrix'): self.execute()
        self.manifest['repeats']['additional_runs'] = 2
        self.manifest['requests']['wdc']['jev']['I3_identity_json'] = {}
        with self.assertRaisesRegex(ValueError, 'request templates'): self.execute()
        self.manifest['requests']['wdc']['jev']['I3_identity_json'] = expanded_stability.payload(
            {'benchmark': 'wdc', 'records': {'left': {}, 'right': {}}}, 'jev', 'I3_identity_json')
        p = self.target/'data/wdc/expanded_repeat.json'; original = p.read_bytes()
        cases = json.loads(original); cases[0]['label'] = 0; p.write_text(json.dumps(cases))
        with self.assertRaisesRegex(ValueError, 'alignment'): self.execute()
        p.write_bytes(original)
        self.stability.__file__ = str(p)
        with self.assertRaisesRegex(ValueError, 'source changed'): self.execute()
        self.stability.__file__ = expanded_stability.__file__
        p = self.target/'runs/quality/jev/B0/wdc/expanded_screen.jsonl'; p.unlink()
        with self.assertRaisesRegex(ValueError, 'controls incomplete'): self.execute()
        self.assertFalse(self.journal.exists())

    def test_run_three_identity_raw_parser_and_namespace_paths(self):
        old = self.target
        self.target = self.root/'replications/run03'; old.rename(self.target)
        self.inference.PROJECT = self.target; self.stability.PROJECT = self.target
        self.journal = self.target/'artifacts/stability_attempt'
        # Synthetic controls must themselves belong to run 3; no cross-run reuse.
        for p in (self.target/'runs').rglob('*.jsonl'):
            if p.name.endswith('.executions.jsonl'): continue
            rows = [json.loads(s) for s in p.read_text().splitlines()]
            for r in rows: r['study_run_id'] = 3
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with patch('urllib.request.urlopen', side_effect=self.http): result = self.execute(3)
        self.assertTrue(result['inference_complete'])
        p = self.target/'runs/e3_repeat_2/jev/B0/wdc/expanded_repeat.jsonl'
        original = [json.loads(s) for s in p.read_text().splitlines()]
        for change in ({'study_run_id': 2}, {'request_hash': 'bad'}, {'input_hash': 'bad'}, {'prediction': 0}):
            rows = copy.deepcopy(original); rows[0].update(change)
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError): component.inventory(self.target, 3, self.stability, protocol)
        p.write_text(''.join(json.dumps(r)+'\n' for r in original))
        self.assertTrue(all(r['study_run_id'] == 3 for r in original))


if __name__ == '__main__': unittest.main()
