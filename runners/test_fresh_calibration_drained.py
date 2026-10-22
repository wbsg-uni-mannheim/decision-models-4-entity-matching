import copy
import json
from pathlib import Path
import socket
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import expanded_calibration
import protocol
import fresh_calibration_drained as component
import test_authorized_recovery_200 as budget_fixture
import test_fresh_supplement_drained as http_fixture


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.original = budget_fixture.AuthorizedBudget().setup_root(self.root, 100, 3)
        self.target = self.root/'replications/run02'
        (self.target/'artifacts').mkdir(parents=True)
        (self.target/'configs').mkdir()
        self.manifest = {'models': list(expanded_calibration.DECISION_MODELS),
                         'benchmarks': {b: {'extra_n': 2} for b in protocol.BENCHMARKS},
                         'new_decisions': 96}
        self.cases, self.controls = {}, {}
        for b in protocol.BENCHMARKS:
            data = self.target/'data'/b; data.mkdir(parents=True)
            for split in ('route', 'calibration_extra'):
                cases = [{'benchmark': b, 'pair_id': split+str(i), 'label': i,
                          'records': {'left': {'title': str(i)}, 'right': {'title': str(i)}}} for i in range(2)]
                for c in cases: c['input_hash'] = protocol.digest(c['records'])
                (data/(split+'.json')).write_text(json.dumps(cases))
                self.cases[b, split] = cases
                if split == 'route':
                    for m in self.manifest['models']:
                        rows = []
                        for c in cases:
                            raw = {'model': protocol.MODELS[m]['id'],
                                   'provider': protocol.MODELS[m]['provider'].split('/')[0],
                                   'usage': {'cost': .001}, 'answers': {'match': {'type': 'noul', 'noul': .8}}}
                            rows.append({'benchmark': b, 'pair_id': c['pair_id'], 'input_hash': c['input_hash'],
                                'request_hash': protocol.digest(protocol.payload(c, m, 'B0')),
                                'model_alias': m, 'variant': 'B0', 'study_run_id': 2,
                                'prediction': 1, 'probability_match': .8,
                                'attempts': [{'status': 200, 'error': None, 'cost': .001,
                                              'seconds': 1, 'response': raw}]})
                        p = self.target/'runs/quality'/m/'B0'/b/'route.jsonl'; p.parent.mkdir(parents=True)
                        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                        self.controls[str(p.relative_to(self.target))] = p.read_bytes()
        self.progress = self.target/'artifacts/progress.json'
        self.progress.write_text(json.dumps({'expanded_E2_E3_screen': {'status': 'complete'}}))
        self.inference = types.SimpleNamespace(**{**vars(self.original), 'PROJECT': self.target,
                                                  'key': lambda: 'synthetic'})
        self.calibration = types.SimpleNamespace(**{**vars(expanded_calibration), 'PROJECT': self.target,
                                                    'verify': Mock(return_value=self.manifest)})
        self.journal = self.target/'artifacts/calibration_attempt'
        self.bodies = []
        ban = patch.object(socket, 'socket', side_effect=AssertionError('Network forbidden'))
        ban.start(); self.addCleanup(ban.stop)

    def execute(self, rid=2):
        return component.run_calibration_extra(self.root, self.target, rid, self.inference,
                                               self.calibration, protocol, self.journal)

    def http(self, request, **kwargs):
        return http_fixture.SupplementTests.http(self, request, **kwargs)

    def test_full_extra_matrix_real_functions_cache_hits_route_unchanged(self):
        isolated = types.SimpleNamespace(**{**vars(self.inference),
            'account': component.shared_account(self.inference.account, self.root)})
        _, cell, _ = component.make_wrapper(self.root, 'core').authorized_build(isolated, 2)
        with patch('urllib.request.urlopen', side_effect=self.http):
            cell('jev', 'B0', 'abt-buy', 'calibration_extra', workers=8)
        cached = self.target/'runs/quality/jev/B0/abt-buy/calibration_extra.jsonl'
        old = cached.read_bytes(); progress = self.progress.read_bytes(); self.bodies = []
        with patch('urllib.request.urlopen', side_effect=self.http) as http:
            result = self.execute()
        self.assertEqual(http.call_count, 94)
        self.assertTrue(result['inference_complete'])
        self.assertEqual(len(result['models']), 6)
        self.assertEqual(sum(c['responses'] for c in result['cells']), 96)
        self.assertEqual(len(result['reused_route_cells']), 48)
        self.assertCountEqual(self.bodies, [protocol.payload(c, m, 'B0')
            for m in self.manifest['models'] for b in self.manifest['benchmarks']
            if (m, b) != ('jev', 'abt-buy') for c in self.cases[b, 'calibration_extra']])
        self.assertEqual(cached.read_bytes(), old)
        for rel, raw in self.controls.items(): self.assertEqual((self.target/rel).read_bytes(), raw)
        self.assertEqual(self.progress.read_bytes(), progress)
        self.assertFalse((self.target/'configs/calibration_selection.json').exists())
        self.assertFalse((self.target/'artifacts/results').exists())
        self.assertFalse(result['admission_granted'])
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 100.096); self.assertAlmostEqual(budget['reserved'], 3)
        self.assertEqual(self.original.account.__defaults__, (0, 100))
        with self.assertRaises(ValueError): self.execute()

    def test_budget_stop_drains_six_models_without_http_or_charges(self):
        p = self.root/'artifacts/budget.json'
        value = json.loads(p.read_text()); value['spent'] = 196.99; p.write_text(json.dumps(value))
        old = p.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            result = self.execute()
        self.assertFalse(result['inference_complete'])
        self.assertEqual(len(result['models']), 6)
        self.assertTrue(all(m['status'] == 'failed' for m in result['models']))
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in result['cells']), 12)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']), 0)
        self.assertEqual(p.read_bytes(), old)

    def test_parser_failure_preserves_sibling_and_other_models(self):
        def http(req, **kwargs):
            response = self.http(req, **kwargs)
            body = json.loads(req.data)
            if body['model'] == protocol.MODELS['jev']['id'] and '"0"' in body['state']:
                raw = json.loads(response.getvalue()); raw['answers']['match']['noul'] = 1.5
                return http_fixture.Response(json.dumps(raw).encode())
            return response
        with patch('urllib.request.urlopen', side_effect=http): result = self.execute()
        self.assertFalse(result['inference_complete'])
        self.assertEqual(sum(m['status'] == 'complete' for m in result['models']), 5)
        self.assertEqual(sum(c['responses'] for c in result['cells']), 81)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']), 1)
        self.assertAlmostEqual(sum(c['known_usd'] for c in result['cells']), .082)
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 100.082); self.assertAlmostEqual(budget['reserved'], 3)

    def test_source_prerequisite_matrix_and_existing_selection_rejected(self):
        self.progress.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'prerequisite'): self.execute()
        self.progress.write_text(json.dumps({'expanded_E2_E3_screen': {'status': 'complete'}}))
        route = self.target/next(iter(self.controls)); original = route.read_bytes(); route.write_text('')
        with self.assertRaisesRegex(ValueError, 'route prerequisite'): self.execute()
        route.write_bytes(original)
        self.calibration.PROJECT = self.root
        with self.assertRaisesRegex(ValueError, 'workspace'): self.execute()
        self.calibration.PROJECT = self.target; self.calibration.__file__ = str(self.progress)
        with self.assertRaisesRegex(ValueError, 'source changed'): self.execute()
        self.calibration.__file__ = expanded_calibration.__file__
        self.manifest['new_decisions'] = 97
        with self.assertRaisesRegex(ValueError, 'matrix'): self.execute()
        self.manifest['new_decisions'] = 96
        (self.target/'configs/calibration_selection.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'Existing calibration'): self.execute()
        self.assertFalse(self.journal.exists())

    def test_run_three_and_raw_tampering(self):
        original = self.target; self.target = self.root/'replications/run03'; original.rename(self.target)
        self.inference.PROJECT = self.calibration.PROJECT = self.target
        self.journal = self.target/'artifacts/calibration_attempt'
        for rel in self.controls:
            p = self.target/rel
            rows = [json.loads(x) for x in p.read_text().splitlines()]
            for row in rows: row['study_run_id'] = 3
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with patch('urllib.request.urlopen', side_effect=self.http): result = self.execute(3)
        self.assertTrue(result['inference_complete']); self.assertFalse(original.exists())
        p = self.target/'runs/quality/jev/B0/abt-buy/calibration_extra.jsonl'
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        self.assertTrue(all(r['study_run_id'] == 3 for r in rows))
        for change in ({'study_run_id': 2}, {'request_hash': 'wrong'}, {'prediction': 0}, {'input_hash': 'wrong'}):
            edited = copy.deepcopy(rows); edited[0].update(change)
            p.write_text(''.join(json.dumps(r)+'\n' for r in edited))
            with self.assertRaises(ValueError): component.inventory(self.target, 3, protocol, self.manifest)

    def test_unknown_operational_outcome_blocks_without_creating_attempt(self):
        c = self.cases['abt-buy', 'calibration_extra'][0]
        row = {'benchmark': 'abt-buy', 'pair_id': c['pair_id'], 'input_hash': c['input_hash'],
               'request_hash': protocol.digest(protocol.payload(c, 'jev', 'B0')),
               'model_alias': 'jev', 'variant': 'B0', 'study_run_id': 2,
               'completed_at': 'synthetic', 'error': 'Synthetic unknown future failure',
               'operational_stop': {'phase': 'future', 'exception_type': 'RuntimeError',
                   'previous_attempts_preserved': 0, 'budget_reservation_rejected': False,
                   'outcome_unknown': True}}
        p = self.target/'runs/quality/jev/B0/abt-buy/calibration_extra.operations.jsonl'
        p.write_text(json.dumps(row)+'\n')
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            with self.assertRaisesRegex(ValueError, 'Unknown calibration outcomes'): self.execute()
        self.assertFalse(self.journal.exists())


if __name__ == '__main__': unittest.main()
