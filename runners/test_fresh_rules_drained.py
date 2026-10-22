import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import protocol
import fresh_rules_drained as component
from test_rules_drained import RulesDrainTests, Response
from test_authorized_recovery_200 import AuthorizedBudget


class FreshRulesTests(unittest.TestCase):
    def setUp(self):
        self.fixture = RulesDrainTests()
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.original = AuthorizedBudget().setup_root(self.root, 100, 3)
        self.target = self.root/'replications/run02'; self.target.parent.mkdir()
        self.fixture.root.rename(self.target)
        self.rid = 2
        self.rules = types.SimpleNamespace(**{**vars(self.fixture.rules), 'PROJECT': self.target})
        self.inference = types.SimpleNamespace(**{**vars(self.original), 'PROJECT': self.target, 'key': lambda: 'fake'})
        self.journal = self.target/'artifacts/rules_attempt'
        for p in (self.target/'runs').rglob('*.jsonl'):
            records = [json.loads(s) for s in p.read_text().splitlines()]
            for r in records: r['study_run_id'] = self.rid
            p.write_text(''.join(json.dumps(r)+'\n' for r in records))
        self.write('artifacts/progress.json', {k: {'status': 'complete_automated'} for k in ('AP3_AP4_analysis','supplement_billiger50')})
        for s in ('confirmation','schema'): self.write(f'artifacts/expanded_{s}_complete.json', {'status': 'complete'})
        self.bodies = []

    def write(self, rel, value):
        p = self.target/rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value)+'\n')

    def http(self, req, **kw):
        body = json.loads(req.data); self.bodies.append(body)
        model = next(m for m in protocol.MODELS if protocol.MODELS[m]['id'] == body['model'])
        if 'test A' in req.data.decode():
            frozen, _, cells, _ = component.prepare(self.target, self.rules, protocol)
            component.audit(self.target, self.rid, frozen, protocol, [c for c in cells if c['split'] == 'prompt'], True)
        return Response(json.dumps(self.fixture.raw(model)).encode())

    def execute(self):
        return component.run_rules(self.root,self.target,self.rid,self.inference,self.rules,protocol,self.journal)

    def test_full_matrix_real_payload_parser_account_cache_and_dev_before_test(self):
        prior = {p: p.read_bytes() for p in (self.target/'runs').rglob('*.jsonl')}
        original_drain = component.drain_model_tasks
        calls = []
        def drain(task, models, path, canonical):
            def counted(m): calls.append((path.name,m)); return task(m)
            return original_drain(counted,models,path,canonical)
        with patch('urllib.request.urlopen', side_effect=self.http), patch.object(component,'drain_model_tasks',drain):
            result = self.execute()
        self.assertTrue(result['inference_complete']); self.assertEqual(len(self.bodies),236)
        self.assertEqual(len(calls),14); self.assertEqual(len(set(calls)),14)
        self.assertCountEqual(self.bodies,[self.fixture.payload(self.fixture.cases[c['benchmark'],c['split']],c['model'],c['variant'])
            for c in self.fixture.cells if c not in self.fixture.cached])
        self.assertTrue(all(p.read_bytes()==raw for p,raw in prior.items()))
        self.assertEqual(sum(len(p['cells']) for p in result['phases']),238)
        budget=json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'],100.236);self.assertAlmostEqual(budget['reserved'],3)
        self.assertEqual(self.original.account.__defaults__,(0,100))
        self.assertFalse(result['report_written']);self.assertFalse((self.target/'artifacts/results').exists())
        self.assertEqual((__import__('run').payload,__import__('analyze').payload,__import__('expanded_rules').RULES),self.fixture.original)
        with self.assertRaisesRegex(ValueError,'Exclusive'):self.execute()

    def test_budget_drains_seven_models_without_test_or_http(self):
        p=self.root/'artifacts/budget.json'; d=json.loads(p.read_text());d['spent']=196.99;p.write_text(json.dumps(d));old=p.read_bytes()
        with patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')): result=self.execute()
        self.assertFalse(result['inference_complete']);self.assertEqual(len(result['phases']),1)
        phase=result['phases'][0];self.assertEqual(len(phase['models']),7)
        self.assertTrue(all(m['status']=='failed' for m in phase['models']))
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in phase['cells']),7)
        self.assertEqual(sum(c['failure_rows'] for c in phase['cells']),0);self.assertEqual(p.read_bytes(),old)
        self.assertFalse((self.journal/'test_models.jsonl').exists())

    def test_paid_parser_failure_drains_all_other_models_before_stopping(self):
        def bad(req,**kw):
            model=next(m for m in protocol.MODELS if protocol.MODELS[m]['id']==json.loads(req.data)['model'])
            raw=self.fixture.raw(model)
            if model=='jev':raw['answers']['match']['noul']=2
            return Response(json.dumps(raw).encode())
        with patch('urllib.request.urlopen',side_effect=bad):result=self.execute()
        phase=result['phases'][0]; self.assertFalse(result['inference_complete'])
        self.assertEqual(sum(m['status']=='complete' for m in phase['models']),6)
        self.assertEqual(sum(c['failure_rows'] for c in phase['cells']),1)
        self.assertEqual(sum(c['responses'] for c in phase['cells']),97) # 96 fresh + 1 preexisting manual
        self.assertAlmostEqual(sum(c['known_usd'] for c in phase['cells']),.098)
        self.assertFalse((self.journal/'test_models.jsonl').exists())

    def test_prerequisites_empty_prior_rule_journal_and_source_drift(self):
        p=self.target/'artifacts/expanded_schema_complete.json';old=p.read_bytes();p.unlink()
        with self.assertRaisesRegex(ValueError,'prerequisite'):self.execute()
        p.write_bytes(old)
        path=self.target/'runs/rules/jev/R_rules/abt-buy/prompt.jsonl';path.parent.mkdir(parents=True);path.touch()
        with self.assertRaisesRegex(ValueError,'absent rule'):self.execute()
        path.unlink()
        self.rules.__file__=str(p)
        with self.assertRaisesRegex(ValueError,'source'):self.execute()
        self.assertFalse(self.journal.exists())

    def test_manual_and_b0_foreign_run_request_raw_unknown_rejected_before_attempt(self):
        p=self.target/'runs/quality/jev/X_P_domain_0/abt-buy/prompt.jsonl';old=p.read_bytes();r=json.loads(old)
        for key,value in [('study_run_id',1),('request_hash','bad'),('input_hash','bad'),('prediction',0)]:
            changed=copy.deepcopy(r);changed[key]=value;p.write_text(json.dumps(changed)+'\n')
            with self.assertRaises(ValueError):self.execute()
        p.write_bytes(old)
        op={k:v for k,v in r.items() if k not in ('attempts','prediction','probability_match')}
        op.update(error='Operational future exception: IncompleteRead',operational_stop=dict(previous_attempts_preserved=0,
            budget_reservation_rejected=False,outcome_unknown=True,phase='future',exception_type='IncompleteRead'))
        q=p.with_name('prompt.operations.jsonl');q.write_text(json.dumps(op)+'\n')
        with self.assertRaisesRegex(ValueError,'Unknown'):self.execute()
        q.unlink();b=self.target/'runs/quality/jev/B0/abt-buy/prompt.jsonl';b.write_text('')
        with self.assertRaisesRegex(ValueError,'Incomplete'):self.execute()
        self.assertFalse(self.journal.exists())

    def test_run3_actual_new_ids_and_b0_control_drift(self):
        new=self.root/'replications/run03';self.target.rename(new);self.target=new;self.rid=3
        self.inference.PROJECT=new;self.rules.PROJECT=new;self.journal=new/'artifacts/rules_attempt'
        # Relabel only prior synthetic fixture controls, never real run caches.
        for p in (new/'runs').rglob('*.jsonl'):
            records=[json.loads(s) for s in p.read_text().splitlines()]
            for r in records:r['study_run_id']=3
            p.write_text(''.join(json.dumps(r)+'\n' for r in records))
        with patch('urllib.request.urlopen',side_effect=self.http):result=self.execute()
        self.assertTrue(result['inference_complete'])
        self.assertTrue(all(r['study_run_id']==3 for p in (new/'runs/rules').rglob('*.jsonl') if not any(s in p.name for s in ('executions','operations','failures')) for r in component.rows(p)))
        frozen,_,cells,_=component.prepare(new,self.rules,protocol)
        p=new/'runs/rules/jev/R_rules/abt-buy/prompt.jsonl';r=json.loads(p.read_text());r['study_run_id']=2;p.write_text(json.dumps(r)+'\n')
        with self.assertRaisesRegex(ValueError,'Foreign'):component.audit(new,3,frozen,protocol,cells)


if __name__=='__main__':unittest.main()
