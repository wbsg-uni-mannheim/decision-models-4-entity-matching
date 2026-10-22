import json
import types
import unittest
from unittest.mock import patch

import analyze
import expanded_schema as es
import protocol
from authorized_recovery_200 import make_wrapper
from repeated_runs import shared_account
import fresh_schema_selection as freeze
import fresh_schema_test_drained as component
import test_fresh_schema_development_drained as fixture
import test_fresh_supplement_drained as b0_fixture
from test_schema_drained import Response


class SchemaTestTests(unittest.TestCase):
    identify = fixture.FreshSchemaTests.identify
    http = fixture.FreshSchemaTests.http
    execute = fixture.FreshSchemaTests.execute

    def setUp(self):
        fixture.FreshSchemaTests.setUp(self)

    def prepare(self, rid=2):
        self.rid = rid
        if rid == 3:
            target = self.root/'replications/run03'; self.target.rename(target); self.target = target
            self.schema.PROJECT = target; self.inference.PROJECT = target
            self.journal = target/'artifacts/schema_development_attempt'
        with patch('urllib.request.urlopen', side_effect=self.http) as http:
            self.assertTrue(self.execute(rid)['development_complete'])
        self.assertEqual(http.call_count, 700)
        guard = patch.object(freeze, 'verify'); guard.start(); self.addCleanup(guard.stop)
        freeze.freeze_development(*self.args(), self.journal/'models.jsonl')
        isolated = types.SimpleNamespace(**{**vars(self.inference),
            'account': shared_account(self.inference.account, self.root)})
        _, cell, _ = make_wrapper(self.root, 'core').authorized_build(isolated, rid)
        for b in [*es.BENCHMARKS, 'billiger-de-50un']:
            cases = json.loads(json.dumps(self.cases.get(b, self.cases['billiger-de'])))
            for c in cases:
                c['benchmark'] = b
                c['entity_ids'] = dict(left='l'+c['pair_id'], right='r'+c['pair_id'])
            directory = self.target/'data'/b; directory.mkdir(exist_ok=True)
            (directory/'test.json').write_text(json.dumps(cases))
            with patch('urllib.request.urlopen', side_effect=lambda *a, **kw:b0_fixture.SupplementTests.http(self,*a,**kw)):
                for m in es.MODELS: cell(m, 'B0', b, 'test', workers=8)
        self.bodies = []

    def args(self): return self.root, self.target, self.rid, self.schema, analyze

    def run_tests(self):
        return component.run_selected_tests(self.root, self.target, self.rid, self.inference, self.schema, analyze)

    def test_full_pipeline_real_frozen_report_bytes_and_protected_sources(self):
        self.prepare(3)
        protected = {p:p.read_bytes() for p in self.target.rglob('*') if p.is_file()}
        with patch('urllib.request.urlopen', side_effect=self.http) as calls:
            result = self.run_tests()
        self.assertEqual(calls.call_count,252); self.assertTrue(result['test_complete'])
        self.assertEqual(len(result['cells']),126); self.assertEqual(len(result['models']),7)
        self.assertEqual(sum(c['responses'] for c in result['cells']),252)
        self.assertTrue(all(p.read_bytes()==raw for p,raw in protected.items()))
        prior_budget = self.budget.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No report HTTP')):
            files = component.finish_report(*self.args())
        self.assertEqual(len(files),8); self.assertEqual(self.budget.read_bytes(),prior_budget)
        self.assertAlmostEqual(json.loads(prior_budget)['reserved'],3)
        self.assertAlmostEqual(json.loads(prior_budget)['spent'],101.078)
        self.assertTrue(all(p.read_bytes()==raw for p,raw in protected.items()))
        out = self.target/component.REPORT_DIR
        for n, count in [('metrics',126),('development',280),('macro',14),('order_effects_development',112),('costs',406),('intervals',63)]:
            self.assertEqual(len(json.loads((out/(n+'.json')).read_text())),count)
        metrics = json.loads((out/'metrics.json').read_text())
        self.assertTrue(all(r['coverage']==0 and r['accepted_risk'] is None for r in metrics if r['condition']=='C_review'))
        reference = self.root/'reference'
        for n in (*freeze.OUTPUTS,'artifacts/schema_pilot_status.json'):
            p=reference/n; p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((self.target/n).read_bytes())
        loader = types.FunctionType(es.load.__code__, {**es.load.__globals__,'PROJECT':self.target}, argdefs=es.load.__defaults__)
        baseline = types.FunctionType(analyze.load.__code__,{**analyze.load.__globals__,'PROJECT':self.target},argdefs=analyze.load.__defaults__)
        report = types.FunctionType(es.report.__code__,{**es.report.__globals__,'PROJECT':reference,'load':loader,
            'analyze':types.SimpleNamespace(**{**vars(analyze),'load':baseline})})
        report()
        for name in files: self.assertEqual((self.target/name).read_bytes(),(reference/name).read_bytes())
        with self.assertRaisesRegex(ValueError,'Existing report'):self.run_tests()

    def test_budget_stop_all_models_zero_http_no_failure_charge(self):
        self.prepare()
        d=json.loads(self.budget.read_text());d['spent']=196.99;self.budget.write_text(json.dumps(d));old=self.budget.read_bytes()
        with patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')): result=self.run_tests()
        self.assertFalse(result['test_complete']);self.assertEqual(len(result['models']),7)
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in result['cells']),14)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']),0)
        self.assertEqual(self.budget.read_bytes(),old)
        with self.assertRaisesRegex(ValueError,'All test model completions'):component.finish_report(*self.args())
        with self.assertRaisesRegex(ValueError,'Already attempted'):self.run_tests()
        self.assertFalse((self.target/component.MARKER).exists())

    def test_parser_failure_drains_sibling_and_other_models(self):
        self.prepare()
        case=json.loads((self.target/'data/abt-buy/test.json').read_text())[0]
        failed=es.wire(es.payload(case,'jev','C_match')).encode()
        def http(req,**kwargs):
            if req.data==failed:return Response(json.dumps({'usage':{'cost':.001}}).encode())
            return self.http(req,**kwargs)
        with patch('urllib.request.urlopen',side_effect=http):result=self.run_tests()
        self.assertFalse(result['test_complete'])
        self.assertEqual(sum(m['status']=='complete' for m in result['models']),6)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']),1)
        self.assertEqual(sum(c['responses'] for c in result['cells']),217)
        self.assertAlmostEqual(sum(c['known_usd'] for c in result['cells']),.218)
        with self.assertRaisesRegex(ValueError,'All test model completions'):component.finish_report(*self.args())
        self.assertFalse((self.target/component.MARKER).exists())

    def test_commit_controls_attempt_and_partial_report_guards(self):
        self.prepare()
        p=self.target/freeze.RECEIPT;old=p.read_bytes();p.unlink()
        with self.assertRaisesRegex(ValueError,'No committed'):self.run_tests()
        p.write_bytes(old)
        p=self.target/'runs/quality/jev/B0/abt-buy/test.jsonl';old=p.read_bytes();p.write_text('')
        with self.assertRaisesRegex(ValueError,'Complete same-run B0'):self.run_tests()
        rows=[json.loads(s) for s in old.splitlines()];rows[0]['study_run_id']=3
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'study run'):self.run_tests()
        p.write_bytes(old)
        p=self.target/'runs/schema/jev/C_match/abt-buy/test.executions.jsonl';p.write_text('')
        with self.assertRaisesRegex(ValueError,'Existing schema test'):self.run_tests()
        p.unlink();self.assertFalse((self.target/component.ATTEMPT).exists())
        with patch('urllib.request.urlopen',side_effect=self.http):self.run_tests()
        p=self.target/component.ATTEMPT/'selection.json';old=p.read_bytes();d=json.loads(old);d['run_id']=3;p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'provenance mismatch'):component.finish_report(*self.args())
        p.write_bytes(old)
        original=freeze.durable_write
        def interrupted(path,raw):
            if path==self.target/component.REPORT_DIR/'development.json':raise OSError('Synthetic publication failure')
            original(path,raw)
        with patch.object(freeze,'durable_write',side_effect=interrupted):
            with self.assertRaisesRegex(OSError,'publication failure'):component.finish_report(*self.args())
        self.assertTrue((self.target/component.REPORT_DIR/'metrics.json').exists())
        self.assertFalse((self.target/component.MARKER).exists())
        with self.assertRaisesRegex(ValueError,'Existing report'):component.finish_report(*self.args())

    def test_test_raw_runid_wire_and_operation_drift_block_report(self):
        self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.http):self.run_tests()
        p=self.target/'runs/schema/jev/C_match/abt-buy/test.jsonl';old=p.read_bytes()
        for field,value in [('study_run_id',3),('request_hash','wrong')]:
            rows=[json.loads(s) for s in old.splitlines()];rows[0][field]=value
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):component.finish_report(*self.args())
        p.write_text(old.decode().splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError,'Complete selected'):component.finish_report(*self.args())
        p.write_bytes(old)
        p=self.target/'runs/schema/jev/C_match/abt-buy/test.operations.jsonl';p.write_text('{}\n')
        with self.assertRaises((ValueError, KeyError)):component.finish_report(*self.args())
        p.unlink()
        p=self.target/'runs/schema/jev/C_match/abt-buy/prompt.jsonl';old=p.read_bytes();p.write_text('')
        with self.assertRaisesRegex(ValueError,'provenance changed'):component.finish_report(*self.args())
        p.write_bytes(old)
        self.assertFalse((self.target/component.REPORT_ATTEMPT).exists())


if __name__=='__main__':unittest.main()
