import copy
import http.client
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import grouped_inference
import schema_grouped_successor_20261016 as helper
from leased_budget import Credit
from operational_journal import inspect_operations
from resume_grouped_core3_fourth_20261016 import persist_prior
import test_schema_drained as fixture

ROOT=Path(__file__).resolve().parents[2]


class SuccessorTests(unittest.TestCase):
    setUp=fixture.SchemaDrainTests.setUp
    write_source=fixture.SchemaDrainTests.write_source
    http=fixture.SchemaDrainTests.http

    def prepare(self):
        self.prior=helper.capture_prior(self.root,self.schema,fixture.analyze,inspect_operations)
        self.attempt=self.root/helper.ATTEMPT; self.attempt.mkdir(parents=True)
        self.prior_path=self.attempt/'prior.json';persist_prior(self.prior_path,self.prior)
        self.prior_sha=helper.sha(self.prior_path)
        self.credit=Credit(self.root,'fast-schema-fixture',dict(prior_sha256=self.prior_sha,
            stage='schema',run_id=1,journal_dir=str(self.attempt/'models')))
        self.execute=helper.private_component(self.root,self.credit,self.prior_path,self.prior_sha,inspect_operations)

    def run_models(self,http=None):
        with patch('urllib.request.urlopen',side_effect=http or self.http) as network:
            result=self.execute(self.schema,fixture.analyze,self.inference)
        return result,network.call_count

    def close(self):
        return helper.reconcile(self.root,self.schema,fixture.protocol,self.prior_path,self.prior_sha,self.credit,inspect_operations)

    def test_whole_selected_matrix_and_real_partial_cache(self):
        # One actual prior mock HTTP, unlike the constructed pilot/dev records.
        m,v,b=self.cells[0]
        _,cell,_=fixture.component.build_schema(self.schema,self.inference)
        with patch('urllib.request.urlopen',side_effect=self.http) as http:
            cell(m,v,b,'test',workers=8,namespace='schema')
        self.assertEqual(http.call_count,1)
        before={p:p.read_bytes() for p in (self.root/'runs').rglob('*.jsonl')}
        original=(fixture.run.payload,fixture.run.account,fixture.component.build,fixture.es.CONTEXT)
        self.prepare(); result,calls=self.run_models(); closed=self.close()
        self.assertEqual(calls,125);self.assertEqual(closed['attempts'],125)
        self.assertAlmostEqual(closed['charged'],.125);self.assertTrue(closed['complete'])
        self.assertEqual(closed['prior_responses_excluded'],1)
        self.assertTrue(all(m['status']=='complete' for m in result['models']))
        self.assertTrue(all(p.read_bytes().startswith(v) for p,v in before.items()))
        self.assertEqual(original,(fixture.run.payload,fixture.run.account,fixture.component.build,fixture.es.CONTEXT))
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'],3.)
        self.assertFalse((self.root/'artifacts/expanded_schema_complete.json').exists())
        with self.assertRaises(ValueError): self.execute(self.schema,fixture.analyze,self.inference)

    def test_real_512_checkpoint_boundary_in_grouped_schema_transport(self):
        for b,case in self.cases.items():
            rows=[{**case,'pair_id':case['pair_id']+'#'+str(i)} for i in range(5)]
            (self.root/'data'/b/'test.json').write_text(json.dumps(rows))
        self.prepare();result,calls=self.run_models();closed=self.close()
        self.assertEqual(calls,630);self.assertEqual(closed['attempts'],630)
        self.assertAlmostEqual(closed['charged'],.630)
        self.assertEqual(self.credit.global_transactions,3)
        self.assertTrue(closed['complete'])
        self.assertEqual(len(result['models']),7)
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'],3.)

    def test_unknown_incomplete_read_keeps_credit_even_outstanding_zero(self):
        self.prepare()
        def http(req,**kwargs):
            if json.loads(req.data)['model']==fixture.protocol.MODELS['jev']['id']:
                raise http_client_error()
            return self.http(req,**kwargs)
        result,_=self.run_models(http)
        self.assertEqual(self.credit.outstanding,0);self.assertGreater(self.credit.unknown,0)
        self.assertEqual(sum(x['status']=='complete' for x in result['models']),6)
        before=self.budget.read_bytes()
        with self.assertRaisesRegex(ValueError,'Unknown'):self.close()
        self.assertEqual(before,self.budget.read_bytes());self.assertFalse(self.credit.closed)

    def test_parser_failure_is_paid_and_exactly_reconciled_without_report(self):
        self.prepare()
        def http(req,**kwargs):
            if json.loads(req.data)['model']==fixture.protocol.MODELS['jev']['id']:
                raw=fixture.raw_response('jev','C_review')
                raw['answers']['match']['probabilities']={'Match':.07,'No Match':.81,'Review':.11}
                return fixture.Response(json.dumps(raw).encode())
            return self.http(req,**kwargs)
        result,calls=self.run_models(http);closed=self.close()
        self.assertFalse(closed['complete']);self.assertEqual(closed['attempts'],calls)
        self.assertAlmostEqual(closed['charged'],calls*.001)
        self.assertEqual(sum(x['status']=='complete' for x in result['models']),6)
        self.assertFalse(closed['report_written'])

    def test_readable_storage_failure_retains_credit(self):
        self.prepare();original=grouped_inference.write_group
        def fail(path,*args,**kwargs):
            original(path,*args,**kwargs)
            if 'runs/schema/jev/' in str(path):raise OSError('unconfirmed fsync')
        with patch.object(grouped_inference,'write_group',side_effect=fail):
            self.run_models()
        self.assertTrue(self.credit.journal_io_failed)
        before=self.budget.read_bytes()
        with self.assertRaisesRegex(ValueError,'storage'):self.close()
        self.assertEqual(before,self.budget.read_bytes())

    def test_missing_model_end_and_raw_attempt_block_close(self):
        self.prepare();self.run_models()
        path=self.attempt/'models/models.jsonl';old=path.read_bytes()
        path.write_bytes(b'\n'.join(old.splitlines()[:-1])+b'\n')
        with self.assertRaisesRegex(ValueError,'model ends'):self.close()
        path.write_bytes(old)
        m,v,b=self.cells[0];raw=self.root/'runs/schema'/m/v/b/'test.jsonl';raw.write_text('')
        with self.assertRaisesRegex(ValueError,'reconcile'):self.close()
        self.assertFalse(self.credit.closed)

    def test_request_and_strict_new_run_id_guards(self):
        self.prepare();self.run_models()
        m,v,b=self.cells[0];p=self.root/'runs/schema'/m/v/b/'test.jsonl';row=json.loads(p.read_text())
        altered=copy.deepcopy(row);altered['request_hash']='bad';p.write_text(json.dumps(altered)+'\n')
        with self.assertRaises((ValueError,AssertionError)):self.close()
        altered=copy.deepcopy(row);del altered['study_run_id'];p.write_text(json.dumps(altered)+'\n')
        with self.assertRaisesRegex(ValueError,'strict Run1'):self.close()
        p.write_text(json.dumps(row)+'\n');self.assertTrue(self.close()['complete'])

    def test_prior_source_credit_and_repeat_guards_before_http(self):
        self.prepare();path=self.root/'data/abt-buy/test.json';old=path.read_bytes();path.write_text('[]')
        with self.assertRaisesRegex(ValueError,'source changed'):self.run_models()
        path.write_bytes(old)
        ledger=json.loads(self.budget.read_text());ledger['inference_credits'][self.credit.identity]['provenance']['run_id']=2
        self.budget.write_text(json.dumps(ledger))
        with self.assertRaisesRegex(ValueError,'Credit must bind'):self.run_models()
        self.assertEqual(self.credit.attempts,0)

    def test_receipt_contract_entire_history_matrix_and_credit_flags(self):
        d=helper.read(ROOT/helper.DIAGNOSIS);p=helper.read(ROOT/helper.PROOF)
        self.assertEqual(helper.receipt_contract(d,p)['missing'],221223)
        for field,value in [('credit_exactly_reconciled',False),('unresolved_raw_attempts',1),('independent_cells',583)]:
            changed=copy.deepcopy(p);changed[field]=value
            with self.assertRaises(ValueError):helper.receipt_contract(d,changed)
        changed=copy.deepcopy(d);changed['stage_history']=changed['stage_history'][:-1]
        with self.assertRaisesRegex(ValueError,'history'):helper.receipt_contract(changed,p)
        changed=copy.deepcopy(p);changed['remaining_cost_plan']['cells'][0]['missing']+=1
        with self.assertRaisesRegex(ValueError,'matrix'):helper.receipt_contract(d,changed)


def http_client_error():
    return http.client.IncompleteRead(b'partial')


if __name__=='__main__':unittest.main()
