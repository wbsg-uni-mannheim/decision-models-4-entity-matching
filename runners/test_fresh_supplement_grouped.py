import copy
import http.client
import json
import unittest
from unittest.mock import patch

import protocol
import fresh_supplement_grouped as grouped
import test_fresh_supplement_drained as fixture
from leased_budget import Credit


class GroupedSupplementTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.SupplementTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root,self.target,self.rid=self.f.root,self.f.target,2
        self.f.bodies=[]
        (self.target/'configs').mkdir()
        (self.target/'configs/billiger50_manifest.json').write_text('{"synthetic":true}')
        self.journal=self.target/'artifacts/grouped_fixture/inference'
        self.journal.parent.mkdir()
        self.prior_path=self.journal.parent/'prior.json'
        self.budget=self.root/'artifacts/budget.json'

    def prepare(self):
        prior=grouped.capture_prior(self.root,self.target,self.rid,self.f.supplement,protocol,self.journal)
        grouped.files.durable_write(self.prior_path,(json.dumps(prior)+'\n').encode())
        self.prior_sha=grouped.files.sha(self.prior_path)
        self.credit=Credit(self.root,'supplement-fixture',dict(prior_sha256=self.prior_sha,
            run_id=self.rid,stage='supplement',journal_dir=str(self.journal.relative_to(self.target))))
        self.runner=grouped.private_component(self.credit,self.prior_path,self.prior_sha)
        return prior

    def execute(self):
        return self.runner(self.root,self.target,self.rid,self.f.inference,self.f.supplement,protocol,self.journal)

    def reconcile(self):
        return grouped.reconcile(self.root,self.target,self.rid,self.prior_path,self.prior_sha,self.journal,
            self.credit,self.f.supplement,protocol)

    def seed_one_prior(self):
        data=self.target/'data'/grouped.component.BENCHMARK/'test.json'; old=data.read_bytes()
        data.write_text(json.dumps(self.f.cases[:1]))
        isolated=__import__('types').SimpleNamespace(**{**vars(self.f.inference),
            'account':grouped.component.shared_account(self.f.inference.account,self.root)})
        _,cell,_=grouped.component.make_wrapper(self.root,'core').authorized_build(isolated,self.rid)
        with patch('urllib.request.urlopen',side_effect=self.f.http):
            cell('jev','B0',grouped.component.BENCHMARK,'test',workers=16)
        data.write_bytes(old)
        path=self.target/'runs/quality/jev/B0'/grouped.component.BENCHMARK/'test.jsonl'
        return path,path.read_bytes()

    def test_partial_prior_full_real_pipeline_exact_delta_and_private_bindings(self):
        path,old=self.seed_one_prior(); progress=self.f.progress.read_bytes()
        original=(grouped.component.run_supplement,grouped.component.make_wrapper,self.f.original.account)
        prior=self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.f.http) as http:result=self.execute()
        self.assertEqual(http.call_count,13);self.assertTrue(result['inference_complete'])
        self.assertEqual(self.credit.global_transactions,1);self.assertEqual(self.credit.attempts,13)
        self.assertTrue(path.read_bytes().startswith(old));self.assertEqual(len(path.read_text().splitlines()),2)
        proof=self.reconcile();self.assertEqual(proof['attempts'],13);self.assertAlmostEqual(proof['charged'],.013)
        self.assertEqual(proof['prior_responses_excluded'],1);self.assertEqual(prior['valid_prior_responses'],1)
        self.assertTrue(self.credit.closed);self.assertEqual(self.credit.global_transactions,2)
        budget=json.loads(self.budget.read_text());self.assertAlmostEqual(budget['spent'],100.014)
        self.assertAlmostEqual(budget['reserved'],3);self.assertEqual(progress,self.f.progress.read_bytes())
        self.assertEqual(original,(grouped.component.run_supplement,grouped.component.make_wrapper,self.f.original.account))
        self.assertFalse((self.target/'artifacts/results').exists())
        with self.assertRaises(ValueError):self.execute()

    def test_run3_full_pipeline_and_512_checkpoint(self):
        data=self.target/'data'/grouped.component.BENCHMARK/'test.json'
        cases=[]
        for i in range(85):
            c=copy.deepcopy(self.f.cases[0]);c['pair_id']=str(i)
            c['records']['left']['title']=str(i);c['input_hash']=protocol.digest(c['records']);cases.append(c)
        data.write_text(json.dumps(cases))
        new=self.root/'replications/run03';self.target.rename(new);self.target=new;self.rid=3
        self.f.target=new;self.f.inference.PROJECT=new;self.f.supplement.PROJECT=new
        self.journal=new/'artifacts/grouped_fixture/inference';self.prior_path=self.journal.parent/'prior.json'
        self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.f.http) as http:result=self.execute()
        self.assertTrue(result['inference_complete']);self.assertEqual(http.call_count,595)
        self.assertGreaterEqual(self.credit.global_transactions,2)
        proof=self.reconcile();self.assertEqual(proof['attempts'],595);self.assertAlmostEqual(proof['charged'],.595)
        self.assertTrue(self.credit.closed)
        self.assertTrue(all(json.loads(s)['study_run_id']==3 for p in (new/'runs').rglob('test.jsonl') for s in p.read_text().splitlines()))
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'],3)

    def test_unknown_incomplete_read_zero_outstanding_keeps_hold(self):
        self.prepare();once=True
        def broken(req,**kw):
            nonlocal once
            if once:once=False;raise http.client.IncompleteRead(b'partial')
            return self.f.http(req,**kw)
        with patch('urllib.request.urlopen',side_effect=broken):
            with self.assertRaisesRegex(ValueError,'Unknown'):self.execute()
        self.assertEqual(self.credit.outstanding,0);self.assertEqual(self.credit.unknown,1)
        with self.assertRaisesRegex(ValueError,'Unknown'):self.reconcile()
        self.assertFalse(self.credit.closed);self.assertEqual(json.loads(self.budget.read_text())['reserved'],11)

    def test_readable_group_storage_failure_keeps_hold(self):
        import grouped_inference
        self.prepare();original=grouped_inference.write_group;once=True
        def broken(path,lines):
            nonlocal once
            original(path,lines)
            if once:once=False;raise OSError('unconfirmed fsync')
        with patch.object(grouped_inference,'write_group',side_effect=broken),patch('urllib.request.urlopen',side_effect=self.f.http):
            self.execute()
        with self.assertRaisesRegex(ValueError,'durability'):self.reconcile()
        self.assertFalse(self.credit.closed)

    def test_model_journal_fsync_failure_readable_bytes_keeps_hold(self):
        self.prepare();original=grouped.component.drain_model_tasks
        def broken(*args,**kwargs):
            original(*args,**kwargs);raise OSError('model journal durability')
        with patch.object(grouped.component,'drain_model_tasks',side_effect=broken):
            # Bind after patch: private globals are snapshots, not mutations.
            self.runner=grouped.private_component(self.credit,self.prior_path,self.prior_sha)
            with patch('urllib.request.urlopen',side_effect=self.f.http):
                with self.assertRaises(OSError):self.execute()
        with self.assertRaisesRegex(ValueError,'durability'):self.reconcile()
        self.assertFalse(self.credit.closed)

    def test_missing_raw_or_model_end_does_not_close(self):
        self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.f.http):self.execute()
        path=self.target/'runs/quality/jev/B0'/grouped.component.BENCHMARK/'test.jsonl';raw=path.read_bytes();path.unlink()
        with self.assertRaisesRegex(ValueError,'incremental'):self.reconcile()
        path.write_bytes(raw);path=self.journal/'models.jsonl';raw=path.read_bytes()
        path.write_bytes(raw[:raw.rfind(b'\n',0,-1)+1])
        with self.assertRaisesRegex(ValueError,'model ends'):self.reconcile()
        self.assertFalse(self.credit.closed)

    def test_parser_failure_siblings_drained_and_exact_cost_close(self):
        self.prepare()
        def broken(req,**kw):
            response=self.f.http(req,**kw);raw=json.loads(response.getvalue())
            if json.loads(req.data)['model']==protocol.MODELS['jev']['id']:
                raw['answers']['match']['noul']=2
            return fixture.Response(json.dumps(raw).encode())
        with patch('urllib.request.urlopen',side_effect=broken):result=self.execute()
        self.assertFalse(result['inference_complete']);self.assertEqual(sum(m['status']=='complete' for m in result['models']),6)
        proof=self.reconcile();self.assertEqual(proof['attempts'],14);self.assertAlmostEqual(proof['charged'],.014)
        self.assertFalse(proof['inference_complete']);self.assertTrue(self.credit.closed)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']),2)

    def test_budget_stop_is_not_a_paid_raw_attempt(self):
        value=json.loads(self.budget.read_text());value['spent']=196.95;self.budget.write_text(json.dumps(value))
        self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.f.http) as http:result=self.execute()
        self.assertFalse(result['inference_complete']);self.assertEqual(len(result['models']),7)
        self.assertGreater(sum(c['operations']['budget_rejections'] for c in result['cells']),0)
        proof=self.reconcile();self.assertEqual(proof['attempts'],http.call_count)
        self.assertGreater(http.call_count,0);self.assertLess(http.call_count,14)
        self.assertAlmostEqual(proof['charged'],.001*http.call_count);self.assertTrue(self.credit.closed)
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'],3)

    def test_receipt_prefix_raw_source_and_credit_identity_guards(self):
        path,old=self.seed_one_prior();self.prepare()
        with patch('urllib.request.urlopen',side_effect=self.f.http):self.execute()
        raw=path.read_bytes();rows=[json.loads(s) for s in raw.splitlines()]
        for change in ({'completed_at':'changed'},{'study_run_id':3},{'request_hash':'wrong'},{'prediction':0}):
            altered=copy.deepcopy(rows);altered[0].update(change);path.write_text(''.join(json.dumps(r)+'\n' for r in altered))
            with self.assertRaises(ValueError):self.reconcile()
        path.write_bytes(raw)
        value=json.loads(self.budget.read_text());original=copy.deepcopy(value)
        value['inference_credits'][self.credit.identity]['provenance']['journal_dir']='artifacts/foreign'
        self.budget.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'Credit does not bind'):self.reconcile()
        self.budget.write_text(json.dumps(original))
        p=self.target/'configs/billiger50_manifest.json';source=p.read_bytes();p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'sources'):self.reconcile()
        p.write_bytes(source);self.prior_path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'receipt'):self.reconcile()
        self.assertFalse(self.credit.closed)

    def test_retained_retry_unknown_is_charged_exactly_once(self):
        import io
        import urllib.error
        self.prepare();once=True
        def retry(req,**kw):
            nonlocal once
            if once:
                once=False
                raise urllib.error.HTTPError(req.full_url,503,'fixture',{},io.BytesIO(b'{"error":"overloaded"}'))
            return self.f.http(req,**kw)
        with patch('urllib.request.urlopen',side_effect=retry),patch('time.sleep'):
            self.assertTrue(self.execute()['inference_complete'])
        proof=self.reconcile();self.assertEqual(proof['attempts'],15);self.assertEqual(proof['unknown'],1)
        self.assertAlmostEqual(proof['charged'],.064);self.assertTrue(self.credit.closed)
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'],3)

    def test_prior_drift_and_prerequisites_block_before_http(self):
        before=self.budget.read_bytes();self.f.progress.write_text('{}')
        with self.assertRaisesRegex(ValueError,'prerequisite'):self.prepare()
        self.assertEqual(before,self.budget.read_bytes());self.assertFalse(self.prior_path.exists())
        self.f.progress.write_text('{"AP3_AP4_analysis":{"status":"complete_automated"}}');self.prepare()
        p=self.target/'runs/quality/jev/B0'/grouped.component.BENCHMARK/'test.jsonl'
        p.parent.mkdir(parents=True);p.touch()
        with patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            with self.assertRaisesRegex(ValueError,'Prior cache changed'):self.execute()
        self.assertEqual(self.credit.attempts,0);self.assertFalse(self.journal.exists())
        self.assertFalse(self.credit.closed)


if __name__=='__main__':unittest.main()
