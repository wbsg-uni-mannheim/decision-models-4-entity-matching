import copy
import io
import json
from pathlib import Path
import socket
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import billiger50
import protocol
import fresh_supplement_drained as component
import test_authorized_recovery_200 as budget_fixture


class Response(io.BytesIO):
    status = 200


class SupplementTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.original = budget_fixture.AuthorizedBudget().setup_root(self.root, 100, 3)
        self.target = self.root/'replications/run02'
        (self.target/'artifacts').mkdir(parents=True)
        data = self.target/'data'/component.BENCHMARK
        data.mkdir(parents=True)
        self.cases = [{'benchmark': component.BENCHMARK, 'pair_id': str(i), 'label': 1,
                       'records': {'left': {'title': str(i)}, 'right': {'title': str(i)}}} for i in range(2)]
        for c in self.cases: c['input_hash'] = protocol.digest(c['records'])
        (data/'test.json').write_text(json.dumps(self.cases))
        self.progress = self.target/'artifacts/progress.json'
        self.progress.write_text(json.dumps({'AP3_AP4_analysis': {'status': 'complete_automated'}}))
        self.inference = types.SimpleNamespace(**{**vars(self.original), 'PROJECT': self.target,
                                                  'key': lambda: 'synthetic'})
        # Real data manifests and terminal scheduler admission are NOT simulated
        # as evidence of a real start. Only this unit fixture's verify is mocked.
        self.supplement = types.SimpleNamespace(**{**vars(billiger50), 'PROJECT': self.target, 'verify': Mock()})
        self.journal = self.target/'artifacts/supplement_attempt'
        self.socket = patch.object(socket, 'socket', side_effect=AssertionError('Network forbidden'))
        self.socket.start(); self.addCleanup(self.socket.stop)

    def execute(self):
        return component.run_supplement(self.root, self.target, 2, self.inference,
                                        self.supplement, protocol, self.journal)

    def http(self, request, **kwargs):
        body = json.loads(request.data)
        model = next(m for m in protocol.MODELS if protocol.MODELS[m]['id'] == body['model'])
        raw = {'model': body['model'], 'provider': protocol.MODELS[model]['provider'].split('/')[0],
               'usage': {'cost': .001}}
        if model == 'luna':
            raw['choices'] = [{'finish_reason': 'stop', 'message': {'content': '{"match": true}'}}]
        else:
            raw['answers'] = {'match': {'type': 'noul', 'noul': .8}}
        self.bodies.append(body)
        return Response(json.dumps(raw).encode())

    def test_complete_real_request_parser_account_cache_and_cache_hit(self):
        self.bodies = []
        isolated = types.SimpleNamespace(**{**vars(self.inference),
            'account': component.shared_account(self.inference.account, self.root)})
        _, cell, _ = component.make_wrapper(self.root, 'core').authorized_build(isolated, 2)
        with patch('urllib.request.urlopen', side_effect=self.http):
            cell('jev', 'B0', component.BENCHMARK, 'test', workers=16)
        cached = self.target/'runs/quality/jev/B0'/component.BENCHMARK/'test.jsonl'
        old = cached.read_bytes(); progress = self.progress.read_bytes()
        self.bodies = []
        with patch('urllib.request.urlopen', side_effect=self.http) as http:
            result = self.execute()
        self.assertEqual(http.call_count, 12)
        self.assertEqual(cached.read_bytes(), old)
        self.assertEqual(result['run_id'], 2)
        self.assertTrue(result['inference_complete'])
        self.assertEqual(len(result['models']), 7)
        self.assertEqual(sum(c['responses'] for c in result['cells']), 14)
        expected = [protocol.payload(c, m, 'B0') for m in protocol.MODELS if m != 'jev' for c in self.cases]
        self.assertCountEqual(self.bodies, expected)
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 100.014)
        self.assertAlmostEqual(budget['reserved'], 3)
        self.assertEqual(self.progress.read_bytes(), progress)
        self.assertEqual(self.original.account.__defaults__, (0, 100))
        self.assertFalse(result['admission_granted'])
        self.assertFalse(result['completion_marker_written'])
        self.assertFalse((self.target/'artifacts/results').exists())
        with self.assertRaises(ValueError): self.execute()

    def test_budget_rejection_drains_all_models_without_http_or_failure_charges(self):
        path = self.root/'artifacts/budget.json'
        value = json.loads(path.read_text()); value['spent'] = 196.99; path.write_text(json.dumps(value))
        before = path.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No paid call')):
            result = self.execute()
        self.assertFalse(result['inference_complete'])
        self.assertEqual(len(result['models']), 7)
        self.assertTrue(all(m['status'] == 'failed' for m in result['models']))
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in result['cells']), 14)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']), 0)
        self.assertEqual(path.read_bytes(), before)

    def test_prerequisites_source_workspace_and_existing_reports_fail_closed(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            self.progress.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'prerequisite'): self.execute()
            self.progress.write_text(json.dumps({'AP3_AP4_analysis': {'status': 'complete_automated'}}))
            self.supplement.PROJECT = self.root
            with self.assertRaisesRegex(ValueError, 'workspace'): self.execute()
            self.supplement.PROJECT = self.target
            self.supplement.__file__ = str(self.progress)
            with self.assertRaisesRegex(ValueError, 'source changed'): self.execute()
            self.supplement.__file__ = billiger50.__file__
            (self.target/'artifacts/results/billiger50').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'reports'): self.execute()
        self.assertFalse(self.journal.exists())

    def test_run_three_has_separate_cache_and_shared_budget(self):
        original_target = self.target
        self.target = self.root/'replications/run03'
        original_target.rename(self.target)
        self.inference.PROJECT = self.target
        self.supplement.PROJECT = self.target
        self.journal = self.target/'artifacts/supplement_attempt'
        self.bodies = []
        with patch('urllib.request.urlopen', side_effect=self.http):
            result = component.run_supplement(self.root, self.target, 3, self.inference,
                                               self.supplement, protocol, self.journal)
        self.assertTrue(result['inference_complete'])
        self.assertFalse(original_target.exists())
        for p in (self.target/'runs').rglob('test.jsonl'):
            self.assertTrue(all(json.loads(s)['study_run_id'] == 3 for s in p.read_text().splitlines()))
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 100.014)
        self.assertAlmostEqual(budget['reserved'], 3)

    def test_raw_request_run_identity_and_input_tampering_rejected(self):
        self.bodies = []
        with patch('urllib.request.urlopen', side_effect=self.http): result = self.execute()
        self.assertTrue(result['inference_complete'])
        p = self.target/'runs/quality/jev/B0'/component.BENCHMARK/'test.jsonl'
        original = [json.loads(s) for s in p.read_text().splitlines()]
        for change in ({'study_run_id': 3}, {'request_hash': 'changed'}, {'prediction': 0}, {'input_hash': 'changed'}):
            rows = copy.deepcopy(original); rows[0].update(change)
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError): component.inventory(self.target, 2, protocol)
        p.write_text(''.join(json.dumps(r)+'\n' for r in original))


if __name__ == '__main__': unittest.main()
