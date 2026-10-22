import json
from pathlib import Path
import socket
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import analyze
import expanded_confirmation as ec
import expanded_protocol as ep
import protocol
import fresh_confirmation_development_drained as component
import test_authorized_recovery_200 as budget_fixture
import test_fresh_supplement_drained as http_fixture


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.original = budget_fixture.AuthorizedBudget().setup_root(self.root, 100, 3)
        self.target = self.root/'replications/run02'
        (self.target/'artifacts').mkdir(parents=True)
        (self.target/'configs').mkdir()
        self.cases = {}
        for b in protocol.BENCHMARKS:
            directory = self.target/'data'/b; directory.mkdir(parents=True)
            cases = [{'benchmark': b, 'pair_id': str(i), 'label': 1,
                      'records': {'left': {'title': str(i)}, 'right': {'title': str(i)}}} for i in range(2)]
            for c in cases: c['input_hash'] = protocol.digest(c['records'])
            (directory/'prompt.json').write_text(json.dumps(cases)); self.cases[b] = cases
        self.progress = self.target/'artifacts/progress.json'
        self.progress.write_text(json.dumps({'expanded_E2_E3_screen': {'status': 'complete'}}))
        self.inference = types.SimpleNamespace(**{**vars(self.original), 'PROJECT': self.target, 'key': lambda: 'fake'})
        self.confirmation = types.SimpleNamespace(**{**vars(ec), 'PROJECT': self.target, 'verify': Mock()})
        self.analysis = types.SimpleNamespace(**{**vars(analyze), 'PROJECT': self.target})
        self.journal = self.target/'artifacts/confirmation_development_attempt'
        self.bodies = []
        guard = patch.object(socket, 'socket', side_effect=AssertionError('No network'))
        guard.start(); self.addCleanup(guard.stop)
        isolated = types.SimpleNamespace(**{**vars(self.inference), 'payload': ec.payload,
            'account': component.shared_account(self.inference.account, self.root)})
        _, self.cell, _ = component.make_wrapper(self.root, 'core').authorized_build(isolated, 2)
        with patch('urllib.request.urlopen', side_effect=self.http):
            for m in protocol.MODELS:
                for b in protocol.BENCHMARKS: self.cell(m, 'B0', b, 'prompt', workers=8)
        self.bodies = []

    def http(self, request, **kwargs):
        return http_fixture.SupplementTests.http(self, request, **kwargs)

    def execute(self, run_id=2):
        return component.run_development(self.root, self.target, run_id, self.inference,
            self.confirmation, ep, self.analysis, protocol, self.journal)

    def test_complete_development_real_http_parser_account_selection_no_test_files(self):
        controls = {p: p.read_bytes() for p in (self.target/'runs').rglob('prompt.jsonl')}
        with patch('urllib.request.urlopen', side_effect=self.http) as http:
            result = self.execute()
        self.assertEqual(http.call_count, 2240)
        self.assertEqual(len(result['cells']), 1176)
        self.assertEqual(sum(c['responses'] for c in result['cells']), 2352)
        self.assertTrue(result['development_complete'])
        chosen = result['selection_candidate']
        self.assertEqual(len(chosen['selected']), 378)
        self.assertEqual(len(chosen['test_cells']), 126)
        self.assertTrue(all(e['variant'] in ('B0', 'X_P_conservative_0') for e in chosen['selected']))
        self.assertEqual(len(chosen['development_metrics']), 1176)
        expected = [ec.payload(c, m, v) for m, v, b in component.matrix(ec)
                    if v != 'B0' for c in self.cases[b]]
        self.assertCountEqual(self.bodies, expected)
        self.assertTrue(all(p.read_bytes() == raw for p, raw in controls.items()))
        self.assertFalse(list((self.target/'configs').iterdir()))
        self.assertFalse(list((self.target/'runs').rglob('test*')))
        self.assertFalse((self.target/'artifacts/results').exists())
        self.assertFalse(result['test_calls_started'])
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 102.352)
        self.assertAlmostEqual(budget['reserved'], 3)
        with self.assertRaisesRegex(ValueError, 'Exclusive'): self.execute()

    def test_budget_drain_has_no_selection_test_or_failure_charges(self):
        p = self.root/'artifacts/budget.json';d=json.loads(p.read_text());d['spent']=196.99;p.write_text(json.dumps(d))
        before = p.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            result = self.execute()
        self.assertEqual(p.read_bytes(), before)
        self.assertFalse(result['development_complete'])
        self.assertIsNone(result['selection_candidate'])
        self.assertEqual(len(result['models']), 7)
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in result['cells']), 14)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']), 0)
        self.assertFalse(list((self.target/'runs').rglob('test*')))
        with self.assertRaisesRegex(ValueError, 'Existing expanded failures'):
            self.journal = self.target/'artifacts/other'; self.execute()

    def test_parser_failure_drains_sibling_and_other_models_blocks_selection(self):
        failed = ec.payload(self.cases['abt-buy'][0], 'jev', ec.PROMPT_VARIANTS[0])
        def http(req, **kwargs):
            if json.loads(req.data) == failed:
                return http_fixture.Response(json.dumps({'usage': {'cost': .001},
                    'model': protocol.MODELS['jev']['id'],
                    'provider': protocol.MODELS['jev']['provider'].split('/')[0]}).encode())
            return self.http(req, **kwargs)
        with patch('urllib.request.urlopen', side_effect=http): result = self.execute()
        self.assertEqual(sum(m['status'] == 'complete' for m in result['models']), 6)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']), 1)
        self.assertEqual(sum(c['responses'] for c in result['cells']), 2033)
        self.assertIsNone(result['selection_candidate'])
        self.assertAlmostEqual(sum(c['known_usd'] for c in result['cells']), 2.034)
        self.assertFalse(list((self.target/'runs').rglob('test*')))

    def test_incomplete_or_foreign_core_and_wrong_prerequisites_rejected(self):
        p = self.target/'runs/quality/jev/B0/abt-buy/prompt.jsonl';old=p.read_bytes()
        p.write_text('')
        with self.assertRaisesRegex(ValueError, 'Core B0'): self.execute()
        p.write_bytes(old);rows=[json.loads(s) for s in p.read_text().splitlines()];rows[0]['study_run_id']=3
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'study run'): self.execute()
        p.write_bytes(old);self.progress.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Screen prerequisite'): self.execute()
        self.assertFalse(self.journal.exists())

    def test_run03_identity_and_cached_failure_provenance(self):
        old = self.target
        target = self.root/'replications/run03'
        old.rename(target); self.target=target
        self.progress=target/'artifacts/progress.json'
        self.journal=target/'artifacts/confirmation_development_attempt'
        for obj in (self.inference, self.confirmation, self.analysis): obj.PROJECT=target
        # Adapt only synthetic fixture records; real run caches are never copied.
        for p in (target/'runs').rglob('prompt.jsonl'):
            rows=[json.loads(s) for s in p.read_text().splitlines()]
            for row in rows: row['study_run_id']=3
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        p=self.root/'artifacts/budget.json';d=json.loads(p.read_text());d['spent']=196.99;p.write_text(json.dumps(d))
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            result=self.execute(3)
        self.assertEqual(result['run_id'],3)
        for p in (target/'runs').rglob('*.operations.jsonl'):
            for line in p.read_text().splitlines(): self.assertEqual(json.loads(line)['study_run_id'],3)
        self.assertIsNone(result['selection_candidate'])
        with self.assertRaisesRegex(ValueError, 'Incomplete development'):
            component.selection_in_memory(target,3,self.confirmation,self.analysis,protocol)

    def test_source_matrix_identity_existing_selection_guards(self):
        with self.assertRaisesRegex(ValueError, 'identity'): self.execute(3)
        self.confirmation.SERIALIZATIONS=['B0']
        with self.assertRaisesRegex(ValueError, 'matrix'): self.execute()
        self.confirmation.SERIALIZATIONS=ec.SERIALIZATIONS
        self.confirmation.__file__=str(self.progress)
        with self.assertRaisesRegex(ValueError, 'source changed'): self.execute()
        self.confirmation.__file__=ec.__file__
        (self.target/'configs/expanded_selected_variants.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'Existing selection'): self.execute()
        self.assertFalse(self.journal.exists())


if __name__ == '__main__': unittest.main()
