import copy
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import analyze
import orchestrate
import protocol
import run_fresh_supplement_grouped as wrapper
import test_run_fresh_supplement_drained as fixture

ROOT = Path(__file__).resolve().parents[2]


class GroupedWrapperTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.WrapperTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        for name in ('root','target','history','progress','inference','supplement','admission','replica','policy'):
            setattr(self,name,getattr(self.f,name))
        for rel in wrapper.REQUIRED:
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
        (self.target/'configs').mkdir(exist_ok=True)
        (self.target/'configs/billiger50_manifest.json').write_text('{"synthetic":true}')
        reports={}
        for rel in wrapper.common.CORE_OUTPUTS:
            p=self.target/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic core report\n')
            reports[rel]=wrapper.sha(p)
        rows=[];prereq=[]
        for stage,job in (('smoke','40'),('core','42')):
            p=self.root/(stage+'_admission.json')
            p.write_text(json.dumps(dict(admitted=True,stage=stage,run_id=2,slurm_job_id=job)))
            pair=[dict(stage=stage,status=status,run_id=2,slurm_job_id=job,admission_sha256=wrapper.sha(p))
                  for status in ('started','complete')]
            if stage=='core':pair[-1]['report_sha256']=reports
            rows.extend(pair)
            prereq.append(dict(stage=stage,slurm_job_id=job,admission=p.name,admission_sha256=wrapper.sha(p),
                stage_end_sha256=protocol.digest(pair[-1]),report_sha256=reports if stage=='core' else {}))
        self.history.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.spec=copy.deepcopy(self.f.spec)
        self.spec.update(accounting_transport='bounded_process_credit',journal_transport='group_commit_v1',
            credit_chunk_usd=8,checkpoint_reservations=512,history_adapter='canonical_success_only_v1',
            prerequisites=prereq,core_stage_journal_sha256=wrapper.sha(self.history),
            source_sha256={r:wrapper.sha(self.root/r) for r in wrapper.REQUIRED})
        self.save()
        self.bad_account=None;self.join=self.spec['reviewed_dependencies']['99'];self.other='independent writer'
        self.plan=dict(full_stage_scenario=dict(known_proxy_usd=.014,historical_unknown_scenario_usd=0),
                       quote_or_hard_bound=False,future_cache_discount=False)

    def save(self):self.admission.write_text(json.dumps(self.spec))

    def scheduler(self,args,**kwargs):
        if args[0]=='sacct':
            job=args[args.index('-j')+1]
            result=''.join(f'{job}{s}|COMPLETED|0:0|now\n' for s in ('','.batch','.extern'))
            return self.bad_account(result) if self.bad_account else result
        if args[:3]==['scontrol','show','job']:
            return 'Dependency='+('(null)' if args[3]=='43' else self.join)
        if args[:3]==['scontrol','write','batch_script']:return self.other
        raise AssertionError(args)

    @contextmanager
    def gates(self):
        with patch.object(wrapper.old,'planning',return_value=self.plan), \
                patch.object(wrapper.common,'make_wrapper') as policy, \
                patch.object(wrapper.common.subprocess,'check_output',side_effect=self.scheduler), \
                patch.object(wrapper,'verify',return_value=self.replica):
            policy.return_value.review.return_value=self.policy
            yield

    def execute(self):
        return wrapper.execute_bound(self.root,self.target,2,self.admission,self.spec,'43',self.replica,
            self.inference,self.supplement,protocol,analyze,orchestrate)

    def ready(self):
        return wrapper.verify_ready(self.root,self.target,2,self.spec,'43',self.replica,self.supplement,protocol)

    def credit(self):return wrapper.read(self.root/'artifacts/budget.json')['inference_credits']['fresh-supplement-2-43']

    def assert_locks(self):
        for p in (self.root/'artifacts/operations/repetition_writer.lock',self.target/'artifacts/parallel_writer.lock'):
            with p.open('a') as f:
                with self.assertRaises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_full_real_pipeline_prior_before_funding_locks_and_frozen_report_bytes(self):
        old=self.history.read_bytes();original=wrapper.finish_report;calls=[]
        def report(*args):self.assert_locks();return original(*args)
        def http(req,**kw):
            self.assert_locks();prior=self.target/wrapper.ATTEMPT/'prior.json'
            row=self.credit();self.assertEqual(row['provenance']['prior_sha256'],wrapper.sha(prior))
            self.assertEqual(row['provenance']['journal_dir'],wrapper.ATTEMPT+'/inference')
            calls.append(req);return self.f.http(req,**kw)
        with self.gates(),patch('urllib.request.urlopen',side_effect=http),patch.object(wrapper,'finish_report',side_effect=report):
            self.assertEqual(self.execute(),0)
        self.assertEqual(len(calls),14);self.assertTrue(self.history.read_bytes().startswith(old))
        self.assertEqual([r['status'] for r in wrapper.old.rows(self.history)[-2:]],['started','complete'])
        self.assertEqual(self.credit()['status'],'closed');self.assertEqual(self.credit()['settled'],14)
        self.assertAlmostEqual(self.credit()['charged'],.014)
        self.assertAlmostEqual(wrapper.read(self.root/'artifacts/budget.json')['reserved'],3)
        evidence=wrapper.read(self.target/wrapper.ATTEMPT/'credit.json')
        self.assertEqual(evidence['credit']['global_transactions'],2)
        self.assertEqual(wrapper.read(self.progress)['supplement_billiger50']['status'],'complete_automated')
        independent=self.root/'independent'
        wrapper.bind(self.supplement.analyze,PROJECT=independent,verify=self.supplement.verify,
            load=wrapper.bind(analyze.load,PROJECT=self.target))()
        for name in wrapper.REPORT_NAMES:
            self.assertEqual((self.target/wrapper.REPORT_DIR/name).read_bytes(),(independent/wrapper.REPORT_DIR/name).read_bytes())
        self.assertEqual(len(wrapper.read(self.target/wrapper.REPORT_DIR/'metrics.json')),7)
        self.assertEqual(len(wrapper.read(self.target/wrapper.REPORT_DIR/'paired_intervals.json')),6)
        self.assertEqual(wrapper.read(self.target/wrapper.MARKER)['status'],'complete')
        with self.gates(),self.assertRaises(ValueError):self.execute()

    def test_existing_own_cache_prefix_excluded_from_incremental_credit(self):
        import types
        data=self.target/'data'/wrapper.grouped.component.BENCHMARK/'test.json';raw=data.read_bytes()
        data.write_text(json.dumps(self.f.cases[:1]))
        isolated=types.SimpleNamespace(**{**vars(self.inference),
            'account':wrapper.grouped.component.shared_account(self.inference.account,self.root)})
        _,cell,_=wrapper.grouped.component.make_wrapper(self.root,'core').authorized_build(isolated,2)
        with patch('urllib.request.urlopen',side_effect=self.f.http):
            cell('jev','B0',wrapper.grouped.component.BENCHMARK,'test',workers=16)
        data.write_bytes(raw)
        p=self.target/'runs/quality/jev/B0'/wrapper.grouped.component.BENCHMARK/'test.jsonl';old=p.read_bytes()
        self.spec['cache_sha256_or_absent']=wrapper.old.cache_inventory(self.target,protocol);self.save()
        with self.gates(),patch('urllib.request.urlopen',side_effect=self.f.http) as http:self.execute()
        self.assertEqual(http.call_count,13);self.assertTrue(p.read_bytes().startswith(old))
        self.assertEqual(self.credit()['settled'],13);self.assertAlmostEqual(self.credit()['charged'],.013)
        proof=wrapper.read(self.target/wrapper.ATTEMPT/'credit.json')
        self.assertEqual(proof['raw']['prior_responses_excluded'],1)

    def test_prior_file_or_directory_fsync_failure_prevents_funding(self):
        before=(self.root/'artifacts/budget.json').read_bytes()
        original=wrapper.files.durable_write
        def broken(path,raw):
            original(path,raw)
            if path.name=='prior.json':raise OSError('unconfirmed prior fsync')
        with self.gates(),patch.object(wrapper.files,'durable_write',side_effect=broken),patch.object(wrapper,'Credit') as fund:
            with self.assertRaises(OSError):self.execute()
            fund.assert_not_called()
        self.assertEqual(before,(self.root/'artifacts/budget.json').read_bytes())

    def test_budget_is_reread_before_funding(self):
        original=wrapper.planning;calls=0
        def planning(*args):
            nonlocal calls
            calls+=1
            if calls==2:
                p=self.root/'artifacts/budget.json';b=wrapper.read(p);b['spent']=195;p.write_text(json.dumps(b))
            return original(*args)
        with self.gates(),patch.object(wrapper,'planning',side_effect=planning),patch.object(wrapper,'Credit') as fund:
            with self.assertRaisesRegex(ValueError,'headroom'):self.execute()
            fund.assert_not_called()
        self.assertFalse((self.target/wrapper.ATTEMPT/'inference').exists())

    def test_all_steps_including_smoke_and_all_joins(self):
        cases=[lambda r:r.replace('.batch|COMPLETED','.batch|RUNNING'),
               lambda r:r.replace('.batch|COMPLETED|0:0','.batch|FAILED|1:0'),
               lambda r:r.split('40.extern')[0] if r.startswith('40|') else r]
        with self.gates():
            for change in cases:
                self.bad_account=change
                with self.assertRaises(ValueError):self.ready()
            self.bad_account=None
            for join in ('afterany:1','afterok:1?afterok:2','afterok:999'):
                self.join=join
                with self.assertRaises(ValueError):self.ready()
            self.join=self.spec['reviewed_dependencies']['99'];self.other='runner --run-id 2'
            with self.assertRaisesRegex(ValueError,'Another writer'):self.ready()

    def test_entire_history_admission_report_cache_source_and_unadmitted_guards(self):
        original=self.history.read_bytes()
        with self.gates():
            self.history.write_bytes(original+original[-100:]);self.spec['core_stage_journal_sha256']=wrapper.sha(self.history)
            with self.assertRaises((ValueError,json.JSONDecodeError)):self.ready()
            self.history.write_bytes(original);self.spec['core_stage_journal_sha256']=wrapper.sha(self.history)
            p=self.root/'core_admission.json';raw=p.read_bytes();p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'admission provenance'):self.ready()
            p.write_bytes(raw)
            p=self.target/next(iter(wrapper.common.CORE_OUTPUTS));raw=p.read_bytes();p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'reports changed'):self.ready()
            p.write_bytes(raw)
            self.spec['cache_sha256_or_absent']={}
            with self.assertRaisesRegex(ValueError,'cache changed'):self.ready()
            self.spec['cache_sha256_or_absent']=wrapper.old.cache_inventory(self.target,protocol)
            p=self.root/wrapper.SELF;raw=p.read_bytes();p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'source drift'):self.ready()
            p.write_bytes(raw);self.spec['admitted']=False
            with self.assertRaisesRegex(ValueError,'admission'):self.execute()
        self.assertFalse((self.target/wrapper.ATTEMPT).exists())

    def test_canonical_history_guard_rejects_even_completed_recovery_chain(self):
        rows=wrapper.old.rows(self.history);failed=copy.deepcopy(rows[2:])
        failed[1]['status']='failed'
        for r in failed:r['slurm_job_id']='41'
        rows[2:2]=failed
        self.history.write_text(''.join(json.dumps(r)+'\n' for r in rows));self.spec['core_stage_journal_sha256']=wrapper.sha(self.history)
        with self.gates(),self.assertRaisesRegex(ValueError,'entire core history'):self.ready()

    def test_unknown_incomplete_read_keeps_credit_with_zero_outstanding(self):
        once=True
        import http.client as client
        def broken(req,**kw):
            nonlocal once
            if once:once=False;raise client.IncompleteRead(b'partial')
            return self.f.http(req,**kw)
        with self.gates(),patch('urllib.request.urlopen',side_effect=broken),self.assertRaisesRegex(ValueError,'Unknown'):
            self.execute()
        self.assertEqual(self.credit()['status'],'active')
        retained=wrapper.old.rows(self.history)[-2]
        self.assertEqual(retained['status'],'credit_retained');self.assertEqual(retained['credit']['outstanding'],0)
        self.assertFalse((self.target/wrapper.MARKER).exists())

    def test_readable_storage_failure_keeps_credit(self):
        import grouped_inference
        original=grouped_inference.write_group;once=True
        def broken(*args):
            nonlocal once
            original(*args)
            if once:once=False;raise OSError('unconfirmed journal fsync')
        with self.gates(),patch('urllib.request.urlopen',side_effect=self.f.http), \
                patch.object(grouped_inference,'write_group',side_effect=broken),self.assertRaisesRegex(ValueError,'durability'):
            self.execute()
        self.assertEqual(self.credit()['status'],'active');self.assertFalse((self.target/wrapper.MARKER).exists())

    def test_missing_model_end_retains_credit(self):
        original=wrapper.grouped.private_component
        def factory(*args):
            run=original(*args)
            def broken(*a):
                result=run(*a);p=a[-1]/'models.jsonl';raw=p.read_bytes()
                p.write_bytes(raw[:raw.rfind(b'\n',0,-1)+1]);return result
            return broken
        with self.gates(),patch('urllib.request.urlopen',side_effect=self.f.http), \
                patch.object(wrapper.grouped,'private_component',side_effect=factory),self.assertRaisesRegex(ValueError,'model ends'):
            self.execute()
        self.assertEqual(self.credit()['status'],'active')

    def test_parser_failure_closes_only_exact_durable_cost_without_report(self):
        def broken(req,**kw):
            response=self.f.http(req,**kw);raw=json.loads(response.getvalue())
            if json.loads(req.data)['model']==protocol.MODELS['jev']['id']:raw['answers']['match']['noul']=2
            return __import__('test_fresh_supplement_drained').Response(json.dumps(raw).encode())
        with self.gates(),patch('urllib.request.urlopen',side_effect=broken),self.assertRaisesRegex(RuntimeError,'drained'):
            self.execute()
        self.assertEqual(self.credit()['status'],'closed');self.assertEqual(self.credit()['settled'],14)
        self.assertAlmostEqual(self.credit()['charged'],.014)
        ends=wrapper.old.rows(self.target/wrapper.ATTEMPT/'inference/models.jsonl')[1:-1]
        self.assertEqual(sum(r['status']=='complete' for r in ends),6)
        self.assertFalse((self.target/wrapper.MARKER).exists())

    def test_partial_report_publication_blocks_marker_and_retry(self):
        original=wrapper.files.durable_write
        def broken(path,raw):
            original(path,raw)
            if path==self.target/wrapper.REPORT_DIR/'metrics.json':raise OSError('report fsync')
        with self.gates(),patch('urllib.request.urlopen',side_effect=self.f.http), \
                patch.object(wrapper.files,'durable_write',side_effect=broken),self.assertRaises(OSError):self.execute()
        self.assertEqual(self.credit()['status'],'closed')
        self.assertTrue((self.target/wrapper.REPORT_DIR/'metrics.json').exists())
        self.assertFalse((self.target/wrapper.MARKER).exists())
        with self.gates(),self.assertRaises(ValueError):self.execute()


if __name__=='__main__':unittest.main()
