import copy
import fcntl
import http.client
import json
from pathlib import Path
import shutil
import types
import unittest
from unittest.mock import patch

import analyze
import expanded_schema as es
import protocol
from authorized_recovery_200 import make_wrapper
from repeated_runs import shared_account
import fresh_schema_selection as freeze
import test_fresh_schema_development_drained as fixture
import test_fresh_supplement_drained as b0_fixture
import run_fresh_schema_drained as wrapper

ROOT=Path(__file__).resolve().parents[2]


class WrapperTests(unittest.TestCase):
    identify=fixture.FreshSchemaTests.identify
    http=fixture.FreshSchemaTests.http

    def setUp(self):
        fixture.FreshSchemaTests.setUp(self)
        guard=patch.object(freeze,'verify');guard.start();self.addCleanup(guard.stop)
        for rel in wrapper.REQUIRED:
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
        isolated=types.SimpleNamespace(**{**vars(self.inference),'account':shared_account(self.inference.account,self.root)})
        _,cell,_=make_wrapper(self.root,'core').authorized_build(isolated,2)
        for b in [*es.BENCHMARKS,'billiger-de-50un']:
            cases=copy.deepcopy(self.cases.get(b,self.cases['billiger-de']))
            for c in cases:
                c['benchmark']=b;c['entity_ids']=dict(left='l'+c['pair_id'],right='r'+c['pair_id'])
            p=self.target/'data'/b;p.mkdir(exist_ok=True);(p/'test.json').write_text(json.dumps(cases))
            with patch('urllib.request.urlopen',side_effect=lambda *a,**kw:b0_fixture.SupplementTests.http(self,*a,**kw)):
                for m in es.MODELS:cell(m,'B0',b,'test',workers=8)
        self.bodies=[]
        self.progress.write_text(json.dumps({k:{'status':v} for k,v in dict(smoke='complete',AP1_baseline='complete',
            AP2_development='complete',AP2_confirmation='complete',AP1_timing='complete',
            AP3_AP4_analysis='complete_automated',supplement_billiger50='complete_automated').items()}))
        history=[dict(stage=stage,status=status,run_id=2,slurm_job_id=job)
            for stage,job in [('smoke','40'),('core','42'),('supplement','44')]
            for status in ('started','complete')]
        reports={}
        for stage,job,index in [('core','42',3),('supplement','44',5)]:
            p=self.root/(stage+'_admission.json');p.write_text(json.dumps(dict(admitted=True,stage=stage,run_id=2,slurm_job_id=job)))
            paths=wrapper.common.CORE_OUTPUTS if stage=='core' else {'artifacts/results/billiger50/metrics.json'}
            for rel in paths:
                q=self.target/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_text('fixture prerequisite report')
            reports[stage]={rel:wrapper.sha(self.target/rel) for rel in paths}
            history[index].update(admission_sha256=wrapper.sha(p),report_sha256=reports[stage])
        self.history=self.target/'artifacts/stage_execution.jsonl'
        self.history.write_text(''.join(json.dumps(r)+'\n' for r in history))
        self.spec=dict(admitted=True,stage='schema',run_id=2,slurm_job_id='43',replica=str(self.target),
            budget_cap_usd=200,budget_authority_sha256=wrapper.APPROVAL_SHA,max_invocations_per_model_per_phase=1,
            historical_unknown_outcomes_disclosed=True,two_phase_budget_gate=True,
            accounting_transport='bounded_process_credit',journal_transport='group_commit_v1',credit_chunk_usd=8,
            checkpoint_reservations=512,source_sha256={p:wrapper.sha(self.root/p) for p in wrapper.REQUIRED},
            runner_sha256='fixture',core_job_id='42',supplement_job_id='44',
            core_admission='core_admission.json',supplement_admission='supplement_admission.json',
            core_admission_sha256=history[3]['admission_sha256'],supplement_admission_sha256=history[5]['admission_sha256'],
            core_stage_end_sha256=protocol.digest(history[3]),supplement_stage_end_sha256=protocol.digest(history[5]),
            core_report_sha256=reports['core'],supplement_report_sha256=reports['supplement'],
            prerequisite_stage_journal_sha256=wrapper.sha(self.history),progress_sha256=wrapper.sha(self.progress),
            cache_sha256_or_absent=wrapper.cache_inventory(self.target,self.schema),parallel_policy='fixture_policy.json',
            parallel_policy_sha256='policy',reviewed_dependencies={'43':'(null)','99':'afterok:366694(unfulfilled)'})
        self.admission=self.root/'admission.json';self.admission.write_text(json.dumps(self.spec))
        self.replica={'runner_sha256':'fixture'}
        self.policy={'policy_sha256':'policy','reviewed_jobs':[{'id':'43','self':True},{'id':'99','self':False}]}
        self.accounts={j:''.join(f'{j}{s}|COMPLETED|0:0|now\n' for s in ('','.batch','.extern')) for j in ('42','44')}
        self.dependency=self.spec['reviewed_dependencies']['99'];self.other_script='independent root writer'
        self.plan={'known_proxy_usd':.7,'historical_unknown_scenario_usd':0}

    def scheduler(self,args,**kwargs):
        if args[0]=='sacct':return self.accounts[args[3]]
        if args[:3]==['scontrol','show','job']:
            return 'Dependency='+('(null)' if args[3]=='43' else self.dependency)
        if args[:3]==['scontrol','write','batch_script']:return self.other_script
        raise AssertionError(args)

    def contexts(self):
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch.object(wrapper,'planning',return_value=self.plan))
        stack.enter_context(patch.object(wrapper.common,'make_wrapper')).return_value.review.return_value=self.policy
        stack.enter_context(patch.object(wrapper.common.subprocess,'check_output',side_effect=self.scheduler))
        stack.enter_context(patch.object(wrapper,'verify',return_value=self.replica))
        return stack

    def ready(self,started_sha=None):
        return wrapper.verify_ready(self.root,self.target,2,self.spec,'43',self.replica,self.schema,started_sha)

    def execute(self):
        return wrapper.execute_bound(self.root,self.target,2,self.admission,self.spec,'43',self.replica,
            self.inference,self.schema,analyze)

    def held(self):
        for p in (self.root/'artifacts/operations/repetition_writer.lock',self.target/'artifacts/parallel_writer.lock'):
            with p.open('a') as f:
                with self.assertRaises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_full_two_phase_real_components_grouped_credit_and_report(self):
        old=self.history.read_bytes();b0={p:p.read_bytes() for p in (self.target/'runs/quality').rglob('*.jsonl')}
        original=wrapper.tests.finish_report
        def report(*args):
            self.held()
            budget=wrapper.read(self.budget)
            self.assertTrue(all(v['status']=='closed' for v in budget['inference_credits'].values()))
            gate=wrapper.read(self.target/wrapper.ATTEMPT/'selected_test_gate.json')
            self.assertEqual(gate['selected_test_costs']['missing_test_responses'],252)
            return original(*args)
        with self.contexts(),patch.object(wrapper,'verify_ready',wraps=wrapper.verify_ready) as ready, \
                patch.object(wrapper.tests,'finish_report',side_effect=report), \
                patch('urllib.request.urlopen',side_effect=self.http) as calls:
            self.assertEqual(self.execute(),0)
        self.assertEqual(calls.call_count,952);self.assertEqual(ready.call_count,2)
        self.assertTrue(self.history.read_bytes().startswith(old));self.assertTrue(all(p.read_bytes()==b for p,b in b0.items()))
        self.assertEqual([r['status'] for r in wrapper.rows(self.history)[-2:]],['started','complete'])
        self.assertEqual(len(wrapper.rows(self.history)[-1]['report_sha256']),8)
        self.assertEqual(len(wrapper.read(self.target/wrapper.tests.REPORT_DIR/'metrics.json')),126)
        budget=wrapper.read(self.budget);self.assertAlmostEqual(budget['spent'],101.078);self.assertAlmostEqual(budget['reserved'],3)
        credits=budget['inference_credits'];self.assertEqual(len(credits),2)
        self.assertEqual(credits['fresh-schema-2-43-development']['attempts'],700)
        self.assertEqual(credits['fresh-schema-2-43-test']['attempts'],252)
        self.assertTrue(all(v['status']=='closed' for v in credits.values()))
        with self.assertRaisesRegex(ValueError,'Prior schema'):self.execute()

    def test_post_freeze_budget_stop_preserves_selection_and_no_test_credit(self):
        original=wrapper.freeze.freeze_development
        def stopped(*args):
            result=original(*args);self.held()
            b=wrapper.read(self.budget);b['spent']=196.99;self.budget.write_text(json.dumps(b));return result
        with self.contexts(),patch.object(wrapper.freeze,'freeze_development',side_effect=stopped), \
                patch('urllib.request.urlopen',side_effect=self.http) as calls:
            with self.assertRaisesRegex(ValueError,'selected-test scenario'):self.execute()
        self.assertEqual(calls.call_count,700)
        self.assertTrue((self.target/freeze.RECEIPT).exists());self.assertFalse((self.target/wrapper.tests.ATTEMPT).exists())
        self.assertEqual(wrapper.rows(self.history)[-1]['phase'],'selected_test_gate')
        budget=wrapper.read(self.budget);self.assertAlmostEqual(budget['reserved'],3)
        self.assertEqual(len(budget['inference_credits']),1)
        self.assertEqual(next(iter(budget['inference_credits'].values()))['status'],'closed')
        with self.assertRaisesRegex(ValueError,'Prior schema'):self.execute()

    def test_development_budget_reread_before_credit_and_http(self):
        b=wrapper.read(self.budget);b['spent']=196.99;self.budget.write_text(json.dumps(b))
        with self.contexts(),patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            with self.assertRaisesRegex(ValueError,'selected-test scenario'):self.execute()
        self.assertFalse((self.target/freeze.RECEIPT).exists());self.assertNotIn('inference_credits',wrapper.read(self.budget))

    def test_unknown_transfer_keeps_credit_and_blocks_freeze(self):
        count=0
        def broken(req,**kw):
            nonlocal count
            count+=1
            if count==1:raise http.client.IncompleteRead(b'partial')
            return self.http(req,**kw)
        with self.contexts(),patch('urllib.request.urlopen',side_effect=broken):
            with self.assertRaisesRegex(ValueError,'Unknown schema outcomes'):self.execute()
        b=wrapper.read(self.budget);credit=next(iter(b['inference_credits'].values()))
        self.assertEqual(credit['status'],'active');self.assertAlmostEqual(b['reserved'],11)
        self.assertFalse((self.target/freeze.RECEIPT).exists())
        events=wrapper.rows(self.history)
        retained=next(r for r in events if r['status']=='credit_retained')
        self.assertEqual(retained['credit']['outstanding'],0)
        self.assertEqual(retained['credit']['settled'],2)
        self.assertEqual(retained['credit']['unknown'],1)
        self.assertTrue(list((self.target/'runs/schema_pilot').rglob('*.operations.jsonl')))

    def test_readable_journal_storage_failure_still_keeps_credit(self):
        import grouped_inference
        original=grouped_inference.write_group
        once=True
        def broken(path,lines):
            nonlocal once
            original(path,lines)
            if once:
                once=False;raise OSError('fsync outcome unknown')
        with self.contexts(),patch.object(grouped_inference,'write_group',side_effect=broken), \
                patch('urllib.request.urlopen',side_effect=self.http):
            with self.assertRaisesRegex(ValueError,'Journal durability'):self.execute()
        b=wrapper.read(self.budget);self.assertEqual(next(iter(b['inference_credits'].values()))['status'],'active')
        self.assertAlmostEqual(b['reserved'],11);self.assertFalse((self.target/freeze.RECEIPT).exists())

    def test_gates_all_steps_joins_scope_source_cache_and_prior_history(self):
        with self.contexts():
            self.ready()
            for job in ('42','44'):
                old=self.accounts[job];self.accounts[job]=old.replace('.extern|COMPLETED','.extern|RUNNING')
                with self.assertRaises(ValueError):self.ready()
                self.accounts[job]=old
            old=self.dependency;self.dependency='afterany:366694(unfulfilled)'
            with self.assertRaisesRegex(ValueError,'conjunctive'):self.ready()
            self.dependency=old
            self.other_script=str(self.target)
            with self.assertRaisesRegex(ValueError,'Another writer'):self.ready()
            self.other_script='independent'
            p=self.target/'runs/quality/jev/B0/abt-buy/test.jsonl';raw=p.read_bytes();p.write_text('')
            with self.assertRaisesRegex(ValueError,'B0 test controls'):self.ready()
            p.write_bytes(raw)
            p=self.root/wrapper.SELF;raw=p.read_bytes();p.write_text('drift')
            with self.assertRaisesRegex(ValueError,'source drift'):self.ready()
            p.write_bytes(raw)
            self.spec['previous_completed_stages']=[{'stage':'screen'}]
            with self.assertRaisesRegex(ValueError,'Intervening stages'):self.ready()
            self.spec.pop('previous_completed_stages')
            history=wrapper.rows(self.history);history.insert(3,dict(stage='core',status='failed',run_id=2,slurm_job_id='41',recovery='group_commit_v1'))
            self.history.write_text(''.join(json.dumps(r)+'\n' for r in history));self.spec['prerequisite_stage_journal_sha256']=wrapper.sha(self.history)
            with self.assertRaisesRegex(ValueError,'Core prerequisite'):self.ready()

    def test_locks_and_unadmitted_entry_no_attempt(self):
        lock=self.target/'artifacts/parallel_writer.lock'
        with lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):self.execute()
        self.spec['admitted']=False
        with self.assertRaisesRegex(ValueError,'reviewed two-phase'):self.execute()
        self.assertFalse((self.target/wrapper.ATTEMPT).exists())


if __name__=='__main__':unittest.main()
