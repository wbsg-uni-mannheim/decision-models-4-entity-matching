import json
import types
import unittest
from unittest.mock import patch

import analyze
import expanded_confirmation as ec
import expanded_protocol as ep
import protocol
import fresh_confirmation_selection as freeze
import fresh_confirmation_test_drained as component
from test_fresh_confirmation_development_drained import DevelopmentTests
from test_fresh_supplement_drained import Response


class SelectedTestTests(unittest.TestCase):
    http = DevelopmentTests.http
    execute = DevelopmentTests.execute

    def setUp(self):
        DevelopmentTests.setUp(self)
        with patch('urllib.request.urlopen', side_effect=self.http):
            self.assertTrue(self.execute()['development_complete'])
        self.guard = patch.object(freeze, 'verify'); self.guard.start(); self.addCleanup(self.guard.stop)
        freeze.freeze_development(*self.args(), self.journal/'models.jsonl')
        for b in [*protocol.BENCHMARKS, 'billiger-de-50un']:
            cases = json.loads(json.dumps(self.cases.get(b, self.cases['billiger-de'])))
            for c in cases:
                c['benchmark'] = b
                c['entity_ids'] = {'left': 'l'+c['pair_id'], 'right': 'r'+c['pair_id']}
            folder = self.target/'data'/b; folder.mkdir(exist_ok=True)
            (folder/'test.json').write_text(json.dumps(cases))
            with patch('urllib.request.urlopen', side_effect=self.http):
                for m in protocol.MODELS: self.cell(m, 'B0', b, 'test', workers=8)
        self.bodies = []

    def args(self):
        return self.root, self.target, 2, self.confirmation, ep, self.analysis, protocol

    def run_tests(self):
        return component.run_selected_tests(self.root, self.target, 2, self.inference,
            self.confirmation, ep, self.analysis, protocol)

    def test_full_real_pipeline_frozen_report_equivalence_and_immutable_development(self):
        protected = {p: p.read_bytes() for p in (self.target/'runs').rglob('prompt*') if p.is_file()}
        protected.update({self.target/p: (self.target/p).read_bytes() for p in (*freeze.OUTPUTS, freeze.RECEIPT)})
        with patch('urllib.request.urlopen', side_effect=self.http) as http:
            result = self.run_tests()
        self.assertTrue(result['test_complete']); self.assertEqual(http.call_count, 126)
        self.assertEqual(len(result['cells']), 126)
        self.assertEqual(sum(c['responses'] for c in result['cells']), 252)
        self.assertEqual(sum(m['status'] == 'complete' for m in result['models']), 7)
        self.assertTrue(all(p.read_bytes() == raw for p, raw in protected.items()))
        budget = self.root/'artifacts/budget.json'; old_budget = budget.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No report HTTP')):
            files = component.finish_report(*self.args())
        self.assertEqual(len(files), 7)
        self.assertEqual(budget.read_bytes(), old_budget)
        self.assertAlmostEqual(json.loads(old_budget)['reserved'], 3)
        self.assertEqual(len(json.loads((self.target/component.REPORT_DIR/'metrics.json').read_text())),378)
        self.assertEqual(len(json.loads((self.target/component.REPORT_DIR/'macro.json').read_text())),42)
        self.assertTrue(all(p.read_bytes() == raw for p, raw in protected.items()))
        # Independent invocation of the actual frozen report; compare all bytes.
        reference = self.root/'reference'; (reference/component.REPORT_DIR).mkdir(parents=True)
        loader = types.FunctionType(analyze.load.__code__,
            {**analyze.load.__globals__, 'PROJECT': self.target, 'payload': ec.payload}, argdefs=analyze.load.__defaults__)
        report = types.FunctionType(ec.report.__code__,
            {**ec.report.__globals__, 'PROJECT': reference,
             'analyze': types.SimpleNamespace(**{**vars(analyze), 'load': loader})})
        report(json.loads((self.target/freeze.OUTPUTS[0]).read_text()))
        for name in [component.MARKER, *(component.REPORT_DIR+'/'+n for n in component.REPORT_NAMES)]:
            self.assertEqual((self.target/name).read_bytes(), (reference/name).read_bytes())
        with self.assertRaisesRegex(ValueError, 'Existing or unexpected report'): self.run_tests()

    def test_budget_drains_all_models_no_failure_charges_report_or_repeat(self):
        p = self.root/'artifacts/budget.json'; d=json.loads(p.read_text()); d['spent']=196.99; p.write_text(json.dumps(d))
        old=p.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            result=self.run_tests()
        self.assertFalse(result['test_complete'])
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in result['cells']),14)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']),0)
        self.assertEqual(len(result['models']),7); self.assertEqual(p.read_bytes(),old)
        with self.assertRaisesRegex(ValueError,'All test model completions'): component.finish_report(*self.args())
        with self.assertRaisesRegex(ValueError,'Already attempted'): self.run_tests()
        self.assertFalse((self.target/component.MARKER).exists())

    def test_parser_failure_siblings_drain_and_no_report(self):
        cases=json.loads((self.target/'data/abt-buy/test.json').read_text())
        failed=ec.payload(cases[0],'jev','X_P_conservative_0')
        def http(request,**kwargs):
            if json.loads(request.data)==failed:
                return Response(json.dumps({'usage':{'cost':.001}, 'model':protocol.MODELS['jev']['id'],
                    'provider':protocol.MODELS['jev']['provider'].split('/')[0]}).encode())
            return self.http(request,**kwargs)
        with patch('urllib.request.urlopen',side_effect=http): result=self.run_tests()
        self.assertFalse(result['test_complete'])
        self.assertEqual(sum(m['status']=='complete' for m in result['models']),6)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']),1)
        self.assertEqual(sum(c['responses'] for c in result['cells']),235)
        self.assertAlmostEqual(sum(c['known_usd'] for c in result['cells']),.236)
        with self.assertRaisesRegex(ValueError,'All test model completions'): component.finish_report(*self.args())
        self.assertFalse((self.target/component.MARKER).exists())

    def test_provenance_raw_controls_and_partial_report_fail_closed(self):
        p=self.target/freeze.RECEIPT; old=p.read_bytes();p.unlink()
        with self.assertRaisesRegex(ValueError,'No committed'): self.run_tests()
        p.write_bytes(old)
        p=self.target/'runs/quality/jev/B0/abt-buy/test.jsonl';old=p.read_bytes();p.write_text('')
        with self.assertRaisesRegex(ValueError,'Complete same-run B0'): self.run_tests()
        rows=[json.loads(s) for s in old.splitlines()];rows[0]['study_run_id']=3
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'study run'): self.run_tests()
        p.write_bytes(old)
        self.assertFalse((self.target/component.ATTEMPT).exists())
        with patch('urllib.request.urlopen',side_effect=self.http): self.run_tests()
        p=self.target/component.ATTEMPT/'selection.json';old=p.read_bytes();d=json.loads(old);d['run_id']=3;p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'provenance mismatch'): component.finish_report(*self.args())
        p.write_bytes(old)
        original=component.durable_write;calls=[]
        def interrupted(path,raw):
            calls.append(path)
            if len(calls)==2: raise OSError('Synthetic report failure')
            original(path,raw)
        with patch.object(component,'durable_write',side_effect=interrupted):
            with self.assertRaisesRegex(OSError,'report failure'): component.finish_report(*self.args())
        self.assertTrue((self.target/component.REPORT_DIR/'metrics.json').exists())
        self.assertFalse((self.target/component.MARKER).exists())
        with self.assertRaisesRegex(ValueError,'Existing or unexpected report'): component.finish_report(*self.args())


if __name__ == '__main__': unittest.main()
