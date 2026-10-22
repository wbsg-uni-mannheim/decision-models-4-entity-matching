import copy
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import analyze
import expanded_schema as es
import protocol
import fresh_schema_selection as freeze
from test_fresh_schema_development_drained import FreshSchemaTests
from test_schema_drained import Response, raw_response


class SchemaSelectionTests(unittest.TestCase):
    setUp = FreshSchemaTests.setUp
    identify = FreshSchemaTests.identify
    http = FreshSchemaTests.http
    execute = FreshSchemaTests.execute

    def args(self): return self.root, self.target, self.rid, self.schema, analyze

    def complete(self, rid=2):
        self.rid=rid
        if rid==3:
            target=self.root/'replications/run03';self.target.rename(target);self.target=target
            self.schema.PROJECT=target;self.inference.PROJECT=target
            self.journal=target/'artifacts/schema_development_attempt'
        with patch('urllib.request.urlopen',side_effect=self.http) as http:
            result=self.execute(rid)
        self.assertEqual(http.call_count,700)
        self.assertTrue(result['development_complete'])

    def test_full_raw_replay_frozen_selector_byte_equivalence_and_post_test_independence(self):
        self.complete()
        before={p:p.read_bytes() for p in self.target.rglob('*') if p.is_file()}
        budget=self.budget.read_bytes()
        reference=self.root/'reference';reference.mkdir()
        loader=types.FunctionType(es.load.__code__,{**es.load.__globals__,'PROJECT':self.target},argdefs=es.load.__defaults__)
        selector=types.FunctionType(es.select.__code__,{**es.select.__globals__,'PROJECT':reference,'load':loader})
        selector()
        with patch.object(freeze,'verify'),patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            receipt=freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
            selected=freeze.verify_frozen(*self.args())
            self.assertEqual(freeze.counts(selected),dict(pilot_cells=70,development_cells=280,
                pilot_responses=140,development_responses=560,selected_models=7,unique_test_cells=126))
            for n in freeze.OUTPUTS:self.assertEqual((self.target/n).read_bytes(),(reference/n).read_bytes())
            self.assertFalse(receipt['admission_granted'])
            with self.assertRaisesRegex(ValueError,'Existing freeze'):
                freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
            self.assertTrue(all(p.read_bytes()==raw for p,raw in before.items()))
            self.assertEqual(self.budget.read_bytes(),budget)
            self.assertFalse(list((self.target/'runs').rglob('test*')))
            # This consumer permits only selected test scopes without ever reading
            # test contents. Their validation belongs to the later test component.
            m,v,b=selected['test_cells'][0]
            test=self.target/'runs/schema'/m/v/b/'test.jsonl'
            test.write_text('Not read by development selection\n')
            self.assertEqual(freeze.verify_frozen(*self.args()),selected)
            test.unlink()
            extra=self.target/'runs/schema/jev/C_same/abt-buy/test.jsonl'
            extra.write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'Unexpected schema'):freeze.verify_frozen(*self.args())
            extra.unlink()
            for n,mutate,pattern in [
                (freeze.OUTPUTS[0],lambda d:{},'differs'),
                (freeze.RECEIPT,lambda d:{**d,'run_id':3},'identity'),
                (freeze.RECEIPT,lambda d:{**d,'admission_granted':True},'identity'),
                (freeze.RECEIPT,lambda d:{**d,'development_cells':279},'differs')]:
                p=self.target/n;raw=p.read_bytes();p.write_text(json.dumps(mutate(json.loads(raw))))
                with self.assertRaisesRegex(ValueError,pattern):freeze.verify_frozen(*self.args())
                p.write_bytes(raw)
            op=self.target/'runs/schema/jev/C_match/abt-buy/prompt.operations.jsonl';op.write_text('')
            with self.assertRaisesRegex(ValueError,'provenance changed'):freeze.verify_frozen(*self.args())
            op.unlink()
        self.assertFalse((self.target/'artifacts/expanded_schema_complete.json').exists())
        self.assertEqual(self.budget.read_bytes(),budget)

    def test_full_run03_pilot_status_model_completions_raw_runid_and_source_guards(self):
        self.complete(3)
        with patch.object(freeze,'verify'),patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            for p,mutation,pattern in [
                (self.journal/'pilot.jsonl',lambda rows:[rows[0],rows[2],rows[1],*rows[3:]],'ordered pilot'),
                (self.journal/'models.jsonl',lambda rows:[rows[0],{**rows[1],'status':'failed'},*rows[2:]],'model completions'),
                (self.target/'runs/schema/jev/C_match/abt-buy/prompt.jsonl',lambda rows:[{**rows[0],'study_run_id':2},*rows[1:]],'run'),
                (self.target/'runs/schema/jev/C_match/abt-buy/prompt.jsonl',lambda rows:rows[:1],'Incomplete')]:
                old=p.read_bytes();rows=[json.loads(s) for s in old.splitlines()]
                p.write_text(''.join(json.dumps(r)+'\n' for r in mutation(copy.deepcopy(rows))))
                with self.assertRaises(ValueError) as ctx:
                    freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
                self.assertIn(pattern.lower(),str(ctx.exception).lower())
                self.assertFalse((self.target/freeze.ATTEMPT).exists());p.write_bytes(old)
            p=self.target/'artifacts/schema_pilot_status.json';old=p.read_bytes();d=json.loads(old)
            d['cells'][0]['known_cost_usd']=42;p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'Pilot status'):freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
            p.write_bytes(old)
            self.schema.__file__=str(p)
            with self.assertRaisesRegex(ValueError,'source changed'):freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
            self.schema.__file__=es.__file__
            # Even a matching status/journal lie is checked against raw pilot costs.
            d['cells'][0]['known_cost_usd']=42;p.write_text(json.dumps(d))
            journal=self.journal/'pilot.jsonl';jr=journal.read_bytes();events=[json.loads(s) for s in jr.splitlines()]
            events[1]['known_cost_usd']=42;journal.write_text(''.join(json.dumps(e)+'\n' for e in events))
            with self.assertRaisesRegex(ValueError,'raw replay'):freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
            p.write_bytes(old);journal.write_bytes(jr)
            receipt=freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
            self.assertEqual(receipt['run_id'],3);freeze.verify_frozen(*self.args())

    def test_nontrivial_global_local_choices_and_billiger50_transfer_match_frozen(self):
        self.rid=2
        bodies={es.wire(es.payload(c,m,v)).encode():(m,v,b)
                for m in es.MODELS for v in es.CONDITIONS for b,cases in self.cases.items() for c in cases}
        choices={m:es.BINARY[i%4] for i,m in enumerate(es.MODELS)}
        alternatives={m:es.BINARY[(i+1)%4] for i,m in enumerate(es.MODELS)}
        def http(req,**kwargs):
            m,v,b=bodies[req.data]
            chosen=alternatives[m] if b in ('abt-buy','billiger-de') else choices[m]
            return Response(json.dumps(raw_response(m,v,positive=v==chosen)).encode())
        with patch('urllib.request.urlopen',side_effect=http) as calls:result=self.execute()
        self.assertTrue(result['development_complete']);self.assertEqual(calls.call_count,700)
        reference=self.root/'reference';reference.mkdir()
        loader=types.FunctionType(es.load.__code__,{**es.load.__globals__,'PROJECT':self.target},argdefs=es.load.__defaults__)
        selector=types.FunctionType(es.select.__code__,{**es.select.__globals__,'PROJECT':reference,'load':loader})
        selector()
        with patch.object(freeze,'verify'),patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            receipt=freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
            candidate=freeze.verify_frozen(*self.args())
        self.assertEqual(receipt['unique_test_cells'],147)
        for m in es.MODELS:
            self.assertEqual(candidate['selected'][m]['global'],choices[m])
            for b in ('abt-buy','billiger-de','billiger-de-50un'):
                self.assertEqual(candidate['selected'][m]['per_benchmark'][b],alternatives[m])
        for n in freeze.OUTPUTS:self.assertEqual((self.target/n).read_bytes(),(reference/n).read_bytes())

    def test_partial_write_cannot_be_consumed_or_blindly_repeated(self):
        self.complete()
        original=freeze.durable_write;calls=[]
        def interrupted(path,raw):
            calls.append(path)
            if len(calls)==2:raise OSError('Synthetic storage failure')
            original(path,raw)
        with patch.object(freeze,'verify'),patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            with patch.object(freeze,'durable_write',side_effect=interrupted):
                with self.assertRaisesRegex(OSError,'storage failure'):
                    freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
            self.assertTrue((self.target/freeze.OUTPUTS[0]).exists())
            self.assertFalse((self.target/freeze.RECEIPT).exists())
            with self.assertRaisesRegex(ValueError,'No committed'):freeze.verify_frozen(*self.args())
            with self.assertRaisesRegex(ValueError,'Existing freeze'):
                freeze.freeze_development(*self.args(),self.journal/'models.jsonl')

    def test_failed_development_and_forged_complete_ends_still_require_all_raw(self):
        self.rid=2
        last=('luna','C_review','dblp-scholar')
        def after(m,v,b,split):
            if split=='schema_pilot' and (m,v,b)==last:
                d=json.loads(self.budget.read_text());d['spent']=196.99;self.budget.write_text(json.dumps(d))
        with patch('urllib.request.urlopen',side_effect=self.http) as http:result=self.execute(2,after)
        self.assertEqual(http.call_count,140);self.assertFalse(result['development_complete'])
        with patch.object(freeze,'verify'):
            with self.assertRaisesRegex(ValueError,'model completions'):
                freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
            journal=self.journal/'models.jsonl';events=[json.loads(s) for s in journal.read_text().splitlines()]
            for e in events[1:]:e['status']='complete'
            journal.write_text(''.join(json.dumps(e)+'\n' for e in events))
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
        self.assertFalse((self.target/freeze.ATTEMPT).exists())
        self.assertFalse(any((self.target/p).exists() for p in freeze.OUTPUTS))


if __name__=='__main__':unittest.main()
