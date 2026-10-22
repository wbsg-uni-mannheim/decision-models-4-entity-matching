import copy
import json
from pathlib import Path
import socket
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import analyze
import expanded_schema as es
import protocol
import fresh_schema_development_drained as component
import test_authorized_recovery_200 as budget_fixture
from test_schema_drained import Response, raw_response


class FreshSchemaTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        original = budget_fixture.AuthorizedBudget().setup_root(self.root, 100, 3)
        self.target = self.root/'replications/run02'
        (self.target/'artifacts').mkdir(parents=True)
        (self.target/'configs').mkdir()
        self.cases = {}
        for b in es.BENCHMARKS:
            d = self.target/'data'/b; d.mkdir(parents=True)
            cases = [dict(benchmark=b, pair_id=b+'#'+str(i), label=1,
                records={'left': {'z': b+'Ä'+str(i), 'a': 'B'}, 'right': {'z': 'A', 'a': 'B'}}) for i in range(2)]
            for c in cases: c['input_hash'] = protocol.digest(c['records'])
            (d/'prompt.json').write_text(json.dumps(cases)); self.cases[b] = cases
            if b in component.PILOT_FAMILIES:
                (d/'schema_pilot.json').write_text(json.dumps(cases))
        self.progress = self.target/'artifacts/progress.json'
        self.progress.write_text(json.dumps({'AP3_AP4_analysis': {'status': 'complete_automated'}}))
        self.inference = types.SimpleNamespace(**{**vars(original), 'PROJECT': self.target, 'key': lambda: 'fake'})
        self.schema = types.SimpleNamespace(**{**vars(es), 'PROJECT': self.target,
            'verify': Mock(return_value={'pilot': {'families': component.PILOT_FAMILIES, 'pairs_per_family': 2}})})
        self.journal = self.target/'artifacts/schema_development_attempt'
        self.bodies, self.calls = [], []
        self.budget = self.root/'artifacts/budget.json'
        sock = patch.object(socket, 'socket', side_effect=AssertionError('No network'))
        sock.start(); self.addCleanup(sock.stop)

    def identify(self, body):
        model = next(m for m, spec in es.MODELS.items() if spec['id'] == body['model'])
        labels = (list(body['questions']['match']['criteria']) if model != 'luna' else
                  body['response_format']['json_schema']['schema']['properties']['match']['enum'])
        variant = next(v for v in es.CONDITIONS if es.options(v) == labels)
        return model, variant

    def http(self, req, **kwargs):
        body = json.loads(req.data); self.bodies.append(body)
        self.assertEqual(req.data, es.wire(body).encode())
        m, v = self.identify(body)
        return Response(json.dumps(raw_response(m, v)).encode())

    def execute(self, run_id=2, after_cell=None):
        build = component.build_schema
        def tracked(*args):
            request, cell, proof = build(*args)
            def call(m, v, b, split, **kwargs):
                self.calls.append((m, v, b, split, kwargs))
                value = cell(m, v, b, split, **kwargs)
                if after_cell: after_cell(m, v, b, split)
                return value
            return request, call, proof
        with patch.object(component, 'build_schema', side_effect=tracked):
            return component.run_development(self.root, self.target, run_id, self.inference,
                                              self.schema, analyze, self.journal)

    def test_full_real_request_parser_account_pilot_order_and_frozen_selection(self):
        original_globals = (es.CONTEXT, es.run.request, es.run.payload, es.run.parse_response)
        with patch('urllib.request.urlopen', side_effect=self.http) as http:
            result = self.execute()
        self.assertEqual(http.call_count, 700)
        self.assertEqual(len(result['pilot_cells']), 70)
        self.assertEqual(sum(c['responses'] for c in result['pilot_cells']), 140)
        self.assertEqual(len(result['cells']), 280)
        self.assertEqual(sum(c['responses'] for c in result['cells']), 560)
        self.assertTrue(result['development_complete'])
        self.assertEqual(len(result['models']), 7)
        self.assertEqual(self.calls[:70], [(m,v,b,'schema_pilot',dict(workers=2,namespace='schema_pilot'))
                                         for m,v,b in component.matrix(es, True)])
        for m in es.MODELS:
            self.assertEqual([c for c in self.calls[70:] if c[0] == m],
                [(a,v,b,'prompt',dict(workers=8,namespace='schema')) for a,v,b in component.matrix(es) if a == m])
        expected = [es.payload(c,m,v) for pilot in (True, False)
                    for m,v,b in component.matrix(es,pilot) for c in self.cases[b]]
        self.assertCountEqual(self.bodies, expected)
        self.assertFalse(list((self.target/'configs').iterdir()))
        self.assertFalse(list((self.target/'runs').rglob('test*')))
        self.assertEqual(original_globals, (es.CONTEXT, es.run.request, es.run.payload, es.run.parse_response))
        budget = json.loads(self.budget.read_text())
        self.assertAlmostEqual(budget['spent'],100.7); self.assertAlmostEqual(budget['reserved'],3)
        # Independently execute the original frozen select, only capturing its writes.
        captured = {}
        loader = types.FunctionType(es.load.__code__, {**es.load.__globals__, 'PROJECT': self.target}, argdefs=es.load.__defaults__)
        select = types.FunctionType(es.select.__code__, {**es.select.__globals__, 'PROJECT': self.target,
            'load': loader, 'write_frozen': lambda p, v: captured.update({p.name: v})})
        select()
        candidate = result['selection_candidate']
        self.assertEqual(candidate['selected'],captured['expanded_schema_selection.json'])
        self.assertEqual(candidate['test_cells'],[list(c) for c in captured['expanded_schema_test_cells.json']])
        self.assertEqual(len(candidate['test_cells']),126)
        self.assertTrue(all(v['global']=='C_match' for v in candidate['selected'].values()))
        # Every Review response remains abstention, never a third correct class.
        p=self.target/'runs/schema/jev/C_review/abt-buy/prompt.jsonl'
        self.assertTrue(all(json.loads(s)['prediction']==-1 for s in p.read_text().splitlines()))
        p=self.target/'runs/schema/jev/C_match/abt-buy/prompt.jsonl'; raw=p.read_bytes()
        rows=[json.loads(s) for s in raw.splitlines()]
        for field,value in [('study_run_id',3),('request_hash',protocol.digest(es.payload(self.cases['abt-buy'][0],'jev','C_match')))]:
            bad=copy.deepcopy(rows);bad[0][field]=value;p.write_text(''.join(json.dumps(r)+'\n' for r in bad))
            with self.assertRaises(ValueError):component.selection_in_memory(self.target,2,self.schema,analyze)
            p.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'Exclusive'):self.execute()

    def test_pilot_failure_drains_two_requests_and_blocks_development(self):
        wanted=es.payload(self.cases['abt-buy'][0],'jev','C_match')
        def http(req, **kwargs):
            if req.data==es.wire(wanted).encode():
                raw=raw_response('jev','C_match')
                raw['answers']['match']['probabilities']={'Match':.8,'No Match':.19}
                return Response(json.dumps(raw).encode())
            return self.http(req,**kwargs)
        with patch('urllib.request.urlopen',side_effect=http) as mocked:result=self.execute()
        self.assertEqual(mocked.call_count,2)
        self.assertFalse(result['pilot_complete']); self.assertEqual(result['models'],[])
        self.assertEqual(sum(c['responses'] for c in result['pilot_cells']),1)
        self.assertEqual(sum(c['failure_rows'] for c in result['pilot_cells']),1)
        self.assertAlmostEqual(sum(c['known_usd'] for c in result['pilot_cells']),.002)
        self.assertFalse((self.target/'runs/schema').exists())
        self.assertIsNone(result['selection_candidate'])
        self.journal=self.target/'artifacts/another_attempt'
        with self.assertRaisesRegex(ValueError,'Existing schema stage'):self.execute()

    def test_pilot_budget_stop_zero_http_and_no_failure_charge(self):
        d=json.loads(self.budget.read_text());d['spent']=196.99;self.budget.write_text(json.dumps(d));before=self.budget.read_bytes()
        with patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):result=self.execute()
        self.assertEqual(self.budget.read_bytes(),before)
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in result['pilot_cells']),2)
        self.assertEqual(sum(c['failure_rows'] for c in result['pilot_cells']),0)
        self.assertFalse(result['development_complete']);self.assertEqual(result['models'],[])
        self.assertFalse((self.target/'runs/schema').exists())

    def test_development_budget_stop_all_seven_models_and_run03_identity(self):
        target=self.root/'replications/run03';self.target.rename(target);self.target=target
        self.schema.PROJECT=target;self.inference.PROJECT=target
        self.journal=target/'artifacts/schema_development_attempt'
        last=component.matrix(es,True)[-1]
        def after(m,v,b,split):
            if split=='schema_pilot' and (m,v,b)==last:
                d=json.loads(self.budget.read_text());d['spent']=196.99;self.budget.write_text(json.dumps(d))
        with patch('urllib.request.urlopen',side_effect=self.http) as http:result=self.execute(3,after)
        self.assertEqual(http.call_count,140)
        self.assertTrue(result['pilot_complete']);self.assertFalse(result['development_complete'])
        self.assertEqual(len(result['models']),7)
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in result['cells']),14)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']),0)
        self.assertIsNone(result['selection_candidate'])
        for p in (target/'runs').rglob('*.operations.jsonl'):
            self.assertTrue(all(json.loads(s)['study_run_id']==3 for s in p.read_text().splitlines()))
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'],3)

    def test_development_parser_failure_drains_other_models_no_selection(self):
        wanted=es.payload(self.cases['abt-buy'][0],'jev','C_match'); seen=0
        def http(req,**kwargs):
            nonlocal seen
            if req.data==es.wire(wanted).encode():
                seen+=1
                if seen==2:
                    return Response(json.dumps({'usage':{'cost':.001}}).encode())
            return self.http(req,**kwargs)
        with patch('urllib.request.urlopen',side_effect=http):result=self.execute()
        self.assertTrue(result['pilot_complete'])
        self.assertEqual(sum(m['status']=='complete' for m in result['models']),6)
        self.assertEqual(sum(c['responses'] for c in result['cells']),481)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']),1)
        self.assertAlmostEqual(sum(c['known_usd'] for c in result['cells']),.482)
        self.assertIsNone(result['selection_candidate'])
        with self.assertRaisesRegex(ValueError,'Incomplete'):component.selection_in_memory(self.target,2,self.schema,analyze)

    def test_workspace_source_prerequisites_and_existing_journals_rejected(self):
        with self.assertRaisesRegex(ValueError,'identity'):self.execute(3)
        self.schema.__file__=str(self.progress)
        with self.assertRaisesRegex(ValueError,'source changed'):self.execute()
        self.schema.__file__=es.__file__; self.schema.CONDITIONS=['C_match']
        with self.assertRaisesRegex(ValueError,'matrix changed'):self.execute()
        self.schema.CONDITIONS=es.CONDITIONS;self.progress.write_text('{}')
        with self.assertRaisesRegex(ValueError,'prerequisite'):self.execute()
        self.progress.write_text(json.dumps({'AP3_AP4_analysis':{'status':'complete_automated'}}))
        d=self.target/'runs/schema';d.mkdir(parents=True);(d/'foreign.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'Unexpected schema'):self.execute()
        self.assertFalse(self.journal.exists())


if __name__ == '__main__':unittest.main()
