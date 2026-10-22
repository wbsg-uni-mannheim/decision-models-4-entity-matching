import http.client
import json
import unittest
from unittest.mock import patch

import analyze
import expanded_rules
import protocol
import fresh_rules_grouped as grouped
import test_fresh_rules_drained as fixture
from leased_budget import Credit


class GroupedTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.FreshRulesTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root, self.target, self.rid = self.f.root, self.f.target, 2
        self.rules, self.inference = self.f.rules, self.f.inference
        self.journal = self.target/'artifacts/grouped_fixture/inference'
        self.journal.parent.mkdir()
        self.prior_path = self.journal.parent/'prior.json'
        # Entity IDs are report metadata only. Update synthetic training/input
        # manifest digests after extending this fixture, never any real input.
        training_path = self.target/'configs/rule_agent_training_manifest.json'
        training = json.loads(training_path.read_text())
        for (b, split), c in self.f.fixture.cases.items():
            c['entity_ids'] = dict(left='left-'+c['pair_id'], right='right-'+c['pair_id'])
            p = self.target/'data'/b/(split+'.json'); p.write_text(json.dumps([c]))
            training['excluded_split_hashes'][str(p.relative_to(self.target))] = protocol.digest([c])
        training_path.write_text(json.dumps(training))
        p = self.target/'configs/expanded_rules_manifest.json'; spec = json.loads(p.read_text())
        spec['training_manifest_sha256'] = protocol.digest(training); p.write_text(json.dumps(spec))
        self.budget = self.root/'artifacts/budget.json'

    def prepare(self):
        prior = grouped.capture_prior(self.root,self.target,self.rid,self.rules,protocol)
        grouped.files.durable_write(self.prior_path,(json.dumps(prior)+'\n').encode())
        self.prior_sha = grouped.files.sha(self.prior_path)
        self.credit = Credit(self.root,'fixture-rules',{'prior_sha256':self.prior_sha,'run_id':self.rid})
        self.runner = grouped.private_component(self.credit)
        return prior

    def run_rules(self):
        return self.runner(self.root,self.target,self.rid,self.inference,self.rules,protocol,self.journal)

    def reconcile(self):
        return grouped.reconcile(self.root,self.target,self.rid,self.prior_path,self.prior_sha,
            self.journal,self.credit,self.rules,protocol)

    def report(self):
        with patch.object(grouped,'verify',return_value={'fixture':True}):
            return grouped.finish_report(self.root,self.target,self.rid,self.rules,protocol,analyze,
                self.journal,self.journal.parent/'report_staging')

    def test_complete_real_drain_grouped_credit_and_byte_identical_frozen_report(self):
        prior=self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.f.http) as calls: result=self.run_rules()
        self.assertTrue(result['inference_complete']);self.assertEqual(calls.call_count,236)
        self.assertFalse(self.credit.closed);self.assertEqual(self.credit.attempts,236)
        self.assertEqual(self.credit.global_transactions,1)
        proof=self.reconcile();self.assertEqual(proof['attempts'],236)
        self.assertEqual(proof['prior_responses_excluded'],2);self.assertAlmostEqual(proof['charged'],.236)
        self.assertEqual(self.credit.global_transactions,2);self.assertTrue(self.credit.closed)
        d=json.loads(self.budget.read_text());self.assertAlmostEqual(d['spent'],100.236);self.assertAlmostEqual(d['reserved'],3)
        self.assertEqual(prior['control_sha256_or_absent'],grouped.component.hashes(self.target,
            [dict(c,variant='B0',namespace='quality') for c in prior['cells'] if c['variant']=='R_rules']))
        reports=self.report();self.assertEqual(len(reports),9)
        expected={'metrics':357,'intervals':126,'costs':357,'macro':21}
        for name,n in expected.items():self.assertEqual(len(json.loads((self.target/grouped.REPORT_DIR/(name+'.json')).read_text())),n)
        # Independently invoke the frozen report, retaining its own marker writer.
        independent=self.root/'independent';(independent/'configs').mkdir(parents=True)
        (independent/'configs/learned_rules.json').write_bytes((self.target/'configs/learned_rules.json').read_bytes())
        # The completed marker intentionally blocks prepare; use the immutable
        # fixture payload and manifest for this independent reporting invocation.
        spec=json.loads((self.target/'configs/expanded_rules_manifest.json').read_text())
        load=grouped.bind(analyze.load,PROJECT=self.target,payload=self.f.fixture.payload)
        def load_cell(m,v,b,s):return load(m,v,b,s,namespace=expanded_rules.CONDITIONS.get(v,'quality'))
        grouped.bind(expanded_rules.report,PROJECT=independent,load_cell=load_cell)(spec)
        for rel in reports:self.assertEqual((self.target/rel).read_bytes(),(independent/rel).read_bytes())
        with self.assertRaisesRegex(ValueError,'Existing report'):self.report()
        self.assertEqual((__import__('run').payload,analyze.payload,expanded_rules.RULES),self.f.fixture.original)

    def test_unknown_transport_outcome_keeps_credit_at_zero_outstanding(self):
        self.prepare();once=True
        def broken(req,**kw):
            nonlocal once
            if once:once=False;raise http.client.IncompleteRead(b'partial')
            return self.f.http(req,**kw)
        with patch('urllib.request.urlopen',side_effect=broken):
            with self.assertRaisesRegex(ValueError,'Unknown'):self.run_rules()
        self.assertEqual(self.credit.outstanding,0);self.assertEqual(self.credit.unknown,1)
        with self.assertRaisesRegex(ValueError,'Unknown'):self.reconcile()
        self.assertFalse(self.credit.closed);self.assertFalse((self.target/grouped.MARKER).exists())
        self.assertEqual(json.loads(self.budget.read_text())['reserved'],11)

    def test_readable_storage_failure_keeps_credit(self):
        import grouped_inference
        self.prepare();original=grouped_inference.write_group;once=True
        def broken(path,lines):
            nonlocal once
            original(path,lines)
            if once:once=False;raise OSError('unconfirmed fsync')
        with patch.object(grouped_inference,'write_group',side_effect=broken),patch('urllib.request.urlopen',side_effect=self.f.http):
            self.run_rules()
        with self.assertRaisesRegex(ValueError,'durability'):self.reconcile()
        self.assertFalse(self.credit.closed)

    def test_parser_failure_drains_development_and_closes_exact_delta_only(self):
        self.prepare()
        def broken(req,**kw):
            model=next(m for m in protocol.MODELS if protocol.MODELS[m]['id']==json.loads(req.data)['model'])
            raw=self.f.fixture.raw(model)
            if model=='jev':raw['answers']['match']['noul']=2
            return fixture.Response(json.dumps(raw).encode())
        with patch('urllib.request.urlopen',side_effect=broken):result=self.run_rules()
        self.assertFalse(result['inference_complete']);self.assertEqual(len(result['phases']),1)
        self.assertEqual(sum(x['status']=='complete' for x in result['phases'][0]['models']),6)
        self.assertFalse((self.journal/'test_models.jsonl').exists())
        result=self.reconcile();self.assertEqual(result['attempts'],97);self.assertAlmostEqual(result['charged'],.097)
        self.assertTrue(self.credit.closed)
        with self.assertRaisesRegex(ValueError,'development'):self.report()

    def test_missing_raw_attempt_or_model_end_holds_credit(self):
        self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.f.http):self.run_rules()
        p=self.target/'runs/rules/jev/R_rules/abt-buy/prompt.jsonl';raw=p.read_bytes();p.unlink()
        with self.assertRaisesRegex(ValueError,'incremental'):self.reconcile()
        p.write_bytes(raw)
        p=self.journal/'test_models.jsonl';raw=p.read_bytes();p.write_bytes(raw[:raw.rfind(b'\n',0,-1)+1])
        with self.assertRaisesRegex(ValueError,'model ends'):self.reconcile()
        self.assertFalse(self.credit.closed)

    def test_prior_prefix_identity_raw_and_receipt_guards(self):
        self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.f.http):self.run_rules()
        p=self.target/'runs/quality/jev/X_P_domain_0/abt-buy/prompt.jsonl';raw=p.read_bytes()
        row=json.loads(raw);row['completed_at']='different-but-valid';p.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError,'prefix'):self.reconcile()
        p.write_bytes(raw)
        for key,value in [('study_run_id',3),('request_hash','wrong')]:
            row=json.loads(raw);row[key]=value;p.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):self.reconcile()
        p.write_bytes(raw)
        self.prior_path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'receipt'):self.reconcile()
        self.assertFalse(self.credit.closed)

    def test_partial_publication_no_marker_or_blind_repeat(self):
        self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.f.http):self.run_rules()
        self.reconcile();original=grouped.files.durable_write
        def broken(path,data):
            if path==self.target/grouped.REPORT_DIR/'costs.json':raise OSError('publication failed')
            return original(path,data)
        with patch.object(grouped.files,'durable_write',side_effect=broken):
            with self.assertRaisesRegex(OSError,'publication'):self.report()
        self.assertTrue((self.target/grouped.REPORT_DIR/'metrics.json').exists())
        self.assertFalse((self.target/grouped.MARKER).exists())
        with self.assertRaisesRegex(ValueError,'Existing report'):self.report()

    def test_real_prior_read_only_controls_required_before_funding(self):
        p=self.target/'runs/quality/jev/B0/abt-buy/prompt.jsonl';old=p.read_bytes();p.write_text('')
        before=self.budget.read_bytes()
        with self.assertRaisesRegex(ValueError,'Incomplete'):self.prepare()
        self.assertEqual(self.budget.read_bytes(),before);self.assertFalse(self.prior_path.exists())
        p.write_bytes(old);p=self.target/'runs/rules/jev/R_rules/abt-buy/prompt.jsonl';p.parent.mkdir(parents=True);p.touch()
        with self.assertRaisesRegex(ValueError,'absent rule'):self.prepare()
        self.assertFalse(self.prior_path.exists())

    def test_run3_identity(self):
        new=self.root/'replications/run03';self.target.rename(new);self.target=new;self.rid=3
        self.f.target=new;self.f.rid=3;self.rules.PROJECT=new;self.inference.PROJECT=new
        self.journal=new/'artifacts/grouped_fixture/inference';self.prior_path=self.journal.parent/'prior.json'
        for p in (new/'runs').rglob('*.jsonl'):
            records=grouped.component.rows(p)
            for r in records:r['study_run_id']=3
            p.write_text(''.join(json.dumps(r)+'\n' for r in records))
        self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.f.http):self.run_rules()
        self.assertEqual(self.reconcile()['attempts'],236)
        self.assertTrue(all(r['study_run_id']==3 for p in (new/'runs/rules').rglob('*.jsonl')
            if not any(s in p.name for s in ('executions','operations','failures')) for r in grouped.component.rows(p)))

    def test_new_manual_tail_excludes_exact_old_prefix(self):
        import copy
        path=self.target/'data/abt-buy/test.json'; cases=json.loads(path.read_text())
        second=copy.deepcopy(cases[0]);second['pair_id']+='-second'
        second['records']['left']['name']='test C';second['input_hash']=protocol.digest(second['records'])
        second['entity_ids']={'left':'extra-left','right':'extra-right'}
        cases.append(second);path.write_text(json.dumps(cases))
        training_path=self.target/'configs/rule_agent_training_manifest.json'
        training=json.loads(training_path.read_text());training['excluded_split_hashes']['data/abt-buy/test.json']=protocol.digest(cases)
        training_path.write_text(json.dumps(training))
        path=self.target/'configs/expanded_rules_manifest.json';spec=json.loads(path.read_text())
        spec['training_manifest_sha256']=protocol.digest(training);path.write_text(json.dumps(spec))
        for model in protocol.MODELS:
            path=self.target/f'runs/quality/{model}/B0/abt-buy/test.jsonl'
            row=json.loads(path.read_text());row.update(pair_id=second['pair_id'],input_hash=second['input_hash'],
                request_hash=protocol.digest(self.f.fixture.payload(second,model,'B0')))
            with path.open('a') as out:out.write(json.dumps(row)+'\n')
        path=self.target/'runs/quality/jev/X_P_domain_0/abt-buy/test.jsonl';prior=path.read_bytes()
        self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.f.http) as http:self.run_rules()
        self.assertEqual(http.call_count,250);self.assertTrue(path.read_bytes().startswith(prior))
        self.assertEqual(len(grouped.component.rows(path)),2)
        proof=self.reconcile();self.assertEqual(proof['attempts'],250)
        self.assertAlmostEqual(proof['charged'],.250);self.assertEqual(proof['prior_responses_excluded'],2)

    def test_readable_model_journal_failure_is_not_durable_proof(self):
        self.prepare();original=grouped.component.drain_model_tasks
        def broken(*args):
            original(*args)
            raise OSError('model fsync confirmation absent')
        with patch.object(grouped.component,'drain_model_tasks',side_effect=broken),patch('urllib.request.urlopen',side_effect=self.f.http):
            self.runner=grouped.private_component(self.credit)
            with self.assertRaisesRegex(OSError,'model fsync'):self.run_rules()
        with self.assertRaisesRegex(ValueError,'durability'):self.reconcile()
        self.assertFalse(self.credit.closed)

    def test_budget_exhaustion_drains_without_test_or_foreign_hold_release(self):
        d=json.loads(self.budget.read_text());d['spent']=196.95;self.budget.write_text(json.dumps(d))
        self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.f.http) as calls:result=self.run_rules()
        self.assertFalse(result['inference_complete']);self.assertEqual(calls.call_count,1)
        self.assertEqual(len(result['phases'][0]['models']),7);self.assertFalse((self.journal/'test_models.jsonl').exists())
        result=self.reconcile();self.assertEqual(result['attempts'],1);self.assertAlmostEqual(result['charged'],.001)
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'],3)

    def test_retained_retry_unknown_is_reconciled_from_raw_not_operation_exemption(self):
        import io
        import urllib.error
        self.prepare();once=True
        def retry(req,**kw):
            nonlocal once
            if once:
                once=False
                raise urllib.error.HTTPError(req.full_url,503,'fixture overload',{},io.BytesIO(b'{"error":"overloaded"}'))
            return self.f.http(req,**kw)
        with patch('urllib.request.urlopen',side_effect=retry),patch('time.sleep'):
            self.assertTrue(self.run_rules()['inference_complete'])
        proof=self.reconcile();self.assertEqual(proof['attempts'],237);self.assertEqual(proof['unknown'],1)
        self.assertAlmostEqual(proof['charged'],.286);self.assertTrue(self.credit.closed)
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'],3)


if __name__=='__main__':unittest.main()
