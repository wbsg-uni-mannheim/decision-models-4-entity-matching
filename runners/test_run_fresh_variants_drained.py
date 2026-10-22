from contextlib import ExitStack
import fcntl
import http.client
import json
from pathlib import Path
import shutil
import types
import unittest
from unittest.mock import patch

import analyze
import protocol
import expanded_variants
import test_fresh_variants_drained as fixture
import run_fresh_variants_drained as wrapper

ROOT=Path(__file__).resolve().parents[2]


class WrapperTests(unittest.TestCase):
    http=fixture.VariantsTests.http
    seed_controls=fixture.VariantsTests.seed_controls

    def setUp(self):
        fixture.VariantsTests.setUp(self)
        for rel in wrapper.REQUIRED:
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
        for b,cases in self.cases.items():
            for c in cases:c['entity_ids']=dict(left='l'+c['pair_id'],right='r'+c['pair_id'])
            (self.target/'data'/b/'test.json').write_text(json.dumps(cases))
        self.progress.write_text(json.dumps({k:{'status':v} for k,v in dict(smoke='complete',AP1_baseline='complete',
            AP2_development='complete',AP2_confirmation='complete',AP1_timing='complete',AP3_AP4_analysis='complete_automated').items()}))
        history=[dict(stage=stage,status=status,run_id=2,slurm_job_id=job)
            for stage,job in [('smoke','40'),('core','42')] for status in ('started','complete')]
        admission=self.root/'core_admission.json';admission.write_text(json.dumps(dict(admitted=True,stage='core',run_id=2,slurm_job_id='42')))
        for rel in wrapper.common.CORE_OUTPUTS:
            p=self.target/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture prerequisite report')
        reports={p:wrapper.sha(self.target/p) for p in wrapper.common.CORE_OUTPUTS}
        history[-1].update(admission_sha256=wrapper.sha(admission),report_sha256=reports)
        self.history=self.target/'artifacts/stage_execution.jsonl';self.history.write_text(''.join(json.dumps(r)+'\n' for r in history))
        self.spec=dict(admitted=True,stage='variants',run_id=2,slurm_job_id='43',replica=str(self.target),
            budget_cap_usd=200,budget_authority_sha256=wrapper.APPROVAL_SHA,max_invocations_per_model=1,
            historical_unknown_outcomes_disclosed=True,accounting_transport='bounded_process_credit',
            journal_transport='group_commit_v1',credit_chunk_usd=8,checkpoint_reservations=512,
            source_sha256={p:wrapper.sha(self.root/p) for p in wrapper.REQUIRED},runner_sha256='fixture',
            core_job_id='42',core_admission='core_admission.json',core_admission_sha256=wrapper.sha(admission),
            core_stage_end_sha256=protocol.digest(history[-1]),core_report_sha256=reports,
            prerequisite_stage_journal_sha256=wrapper.sha(self.history),progress_sha256=wrapper.sha(self.progress),
            cache_sha256_or_absent=wrapper.snapshot(self.target,protocol),parallel_policy='fixture_policy.json',
            parallel_policy_sha256='policy',reviewed_dependencies={'43':'(null)','99':'afterok:366694(unfulfilled)'})
        self.admission=self.root/'admission.json';self.admission.write_text(json.dumps(self.spec))
        self.replica={'runner_sha256':'fixture'}
        self.policy={'policy_sha256':'policy','reviewed_jobs':[{'id':'43','self':True},{'id':'99','self':False}]}
        self.account=''.join(f'42{s}|COMPLETED|0:0|now\n' for s in ('','.batch','.extern'))
        self.dependency=self.spec['reviewed_dependencies']['99'];self.other_script='independent root writer'
        self.plan={'known_proxy_usd':.07,'historical_unknown_scenario_usd':0}
        self.budget=self.root/'artifacts/budget.json'

    def scheduler(self,args,**kwargs):
        if args[0]=='sacct':return self.account
        if args[:3]==['scontrol','show','job']:return 'Dependency='+('(null)' if args[3]=='43' else self.dependency)
        if args[:3]==['scontrol','write','batch_script']:return self.other_script
        raise AssertionError(args)

    def contexts(self):
        stack=ExitStack();stack.enter_context(patch.object(wrapper,'planning',return_value=self.plan))
        stack.enter_context(patch.object(wrapper.common,'make_wrapper')).return_value.review.return_value=self.policy
        stack.enter_context(patch.object(wrapper.common.subprocess,'check_output',side_effect=self.scheduler))
        stack.enter_context(patch.object(wrapper,'verify',return_value=self.replica))
        return stack

    def ready(self):return wrapper.verify_ready(self.root,self.target,2,self.spec,'43',self.replica,self.variants,protocol)
    def execute(self):return wrapper.execute_bound(self.root,self.target,2,self.admission,self.spec,'43',self.replica,self.inference,self.variants,protocol,analyze)

    def held(self):
        for p in (self.root/'artifacts/operations/repetition_writer.lock',self.target/'artifacts/parallel_writer.lock'):
            with p.open('a') as f:
                with self.assertRaises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_full_grouped_credit_raw_report_and_marker(self):
        old=self.history.read_bytes();controls=wrapper.component.control_hashes(self.target,protocol)
        report=wrapper.finish_report
        def checked(*args):
            self.held();self.assertEqual(next(iter(wrapper.read(self.budget)['inference_credits'].values()))['status'],'closed')
            return report(*args)
        with self.contexts(),patch.object(wrapper,'finish_report',side_effect=checked),patch('urllib.request.urlopen',side_effect=self.http) as calls:
            self.assertEqual(self.execute(),0)
        self.assertEqual(calls.call_count,70);self.assertTrue(self.history.read_bytes().startswith(old))
        self.assertEqual(wrapper.component.control_hashes(self.target,protocol),controls)
        self.assertCountEqual(self.bodies,[protocol.payload(c,m,'B0') for m in protocol.MODELS for b in wrapper.component.VARIANTS for c in self.cases[b]])
        self.assertEqual([r['status'] for r in wrapper.rows(self.history)[-2:]],['started','complete'])
        self.assertEqual(len(wrapper.rows(self.history)[-1]['report_sha256']),6)
        budget=wrapper.read(self.budget);self.assertAlmostEqual(budget['spent'],100.098);self.assertAlmostEqual(budget['reserved'],3)
        credit=next(iter(budget['inference_credits'].values()));self.assertEqual(credit['attempts'],70);self.assertEqual(credit['status'],'closed')
        self.assertEqual(len(wrapper.read(self.target/wrapper.REPORT_DIR/'metrics.json')),49)
        self.assertEqual(len(wrapper.read(self.target/wrapper.REPORT_DIR/'paired_language.json')),21)
        # Execute frozen report independently with same raw loader into a separate output.
        independent=self.root/'independent';independent.mkdir()
        bound=types.SimpleNamespace(**{**vars(analyze),'load':wrapper.bind(analyze.load,PROJECT=self.target)})
        frozen=wrapper.bind(expanded_variants.report,PROJECT=independent,verify=self.variants.verify,analyze=bound)
        frozen()
        for p in [wrapper.MARKER,*(wrapper.REPORT_DIR+'/'+n for n in wrapper.REPORT_NAMES)]:
            self.assertEqual((self.target/p).read_bytes(),(independent/p).read_bytes())
        with self.assertRaisesRegex(ValueError,'Prior attempt'):self.execute()

    def test_budget_reread_before_credit_or_http(self):
        d=wrapper.read(self.budget);d['spent']=196.99;self.budget.write_text(json.dumps(d))
        with self.contexts(),patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            with self.assertRaisesRegex(ValueError,'scenario'):self.execute()
        self.assertNotIn('inference_credits',wrapper.read(self.budget));self.assertFalse((self.target/wrapper.MARKER).exists())

    def test_unknown_operation_keeps_credit_even_outstanding_zero(self):
        count=0
        def broken(req,**kw):
            nonlocal count
            count+=1
            if count==1:raise http.client.IncompleteRead(b'partial')
            return self.http(req,**kw)
        with self.contexts(),patch('urllib.request.urlopen',side_effect=broken):
            with self.assertRaisesRegex(ValueError,'Unknown variant'):self.execute()
        d=wrapper.read(self.budget);self.assertEqual(next(iter(d['inference_credits'].values()))['status'],'active')
        self.assertAlmostEqual(d['reserved'],11)
        retained=next(r for r in wrapper.rows(self.history) if r['status']=='credit_retained')
        self.assertEqual(retained['credit']['outstanding'],0);self.assertEqual(retained['credit']['unknown'],1)
        self.assertFalse((self.target/wrapper.MARKER).exists())

    def test_readable_storage_failure_keeps_credit(self):
        import grouped_inference
        original=grouped_inference.write_group;once=True
        def broken(path,lines):
            nonlocal once
            original(path,lines)
            if once:once=False;raise OSError('fsync outcome unknown')
        with self.contexts(),patch.object(grouped_inference,'write_group',side_effect=broken),patch('urllib.request.urlopen',side_effect=self.http):
            with self.assertRaisesRegex(ValueError,'durability'):self.execute()
        self.assertEqual(next(iter(wrapper.read(self.budget)['inference_credits'].values()))['status'],'active')
        self.assertFalse((self.target/wrapper.MARKER).exists())

    def test_gates_core_steps_provenance_joins_overlap_source_cache_history(self):
        with self.contexts():
            self.ready();old=self.account;self.account=old.replace('.extern|COMPLETED','.extern|RUNNING')
            with self.assertRaises(ValueError):self.ready()
            self.account=old;old=self.dependency;self.dependency='afterany:366694'
            with self.assertRaisesRegex(ValueError,'conjunctive'):self.ready()
            self.dependency=old;self.other_script=str(self.target)
            with self.assertRaisesRegex(ValueError,'Another writer'):self.ready()
            self.other_script='independent'
            for rel in [wrapper.SELF,'core_admission.json']:
                p=self.root/rel;old=p.read_bytes();p.write_text('drift')
                with self.assertRaises(ValueError):self.ready()
                p.write_bytes(old)
            p=self.target/'runs/quality/jev/B0/wdc/test.jsonl';old=p.read_bytes();p.write_text('')
            with self.assertRaisesRegex(ValueError,'controls incomplete'):self.ready()
            p.write_bytes(old)
            p=self.target/'data/wdc-050un/test.json';old=p.read_bytes();p.write_text('[]')
            with self.assertRaisesRegex(ValueError,'Inputs or cache'):self.ready()
            p.write_bytes(old)
            self.spec['previous_completed_stages']=[{'stage':'screen'}]
            with self.assertRaisesRegex(ValueError,'Intervening'):self.ready()
            del self.spec['previous_completed_stages']
            hist=wrapper.rows(self.history);hist.insert(3,dict(stage='core',status='failed',run_id=2,slurm_job_id='41'))
            self.history.write_text(''.join(json.dumps(r)+'\n' for r in hist));self.spec['prerequisite_stage_journal_sha256']=wrapper.sha(self.history)
            with self.assertRaisesRegex(ValueError,'Core prerequisite'):self.ready()

    def test_partial_report_publication_no_marker_or_blind_repeat(self):
        original=wrapper.files.durable_write
        def broken(path,data):
            if path==self.target/wrapper.REPORT_DIR/'costs.json':raise OSError('publication failure')
            return original(path,data)
        with self.contexts(),patch.object(wrapper.files,'durable_write',side_effect=broken),patch('urllib.request.urlopen',side_effect=self.http):
            with self.assertRaisesRegex(OSError,'publication'):self.execute()
        self.assertTrue((self.target/wrapper.REPORT_DIR/'metrics.json').exists());self.assertFalse((self.target/wrapper.MARKER).exists())
        with self.assertRaisesRegex(ValueError,'Prior attempt'):self.execute()

    def test_paid_parser_failure_closes_only_exact_raw_credit(self):
        once=True
        def bad(req,**kw):
            nonlocal once
            response=self.http(req,**kw)
            if once and json.loads(req.data)['model']==protocol.MODELS['jev']['id']:
                once=False;value=json.loads(response.getvalue());value['answers']['match']['noul']=2
                return fixture.Response(json.dumps(value).encode())
            return response
        with self.contexts(),patch('urllib.request.urlopen',side_effect=bad):
            with self.assertRaisesRegex(RuntimeError,'model drain'):self.execute()
        d=wrapper.read(self.budget);credit=next(iter(d['inference_credits'].values()))
        self.assertEqual(credit['status'],'closed');self.assertEqual(credit['attempts'],62)
        self.assertAlmostEqual(d['reserved'],3);self.assertAlmostEqual(d['spent'],100.09)
        self.assertFalse((self.target/wrapper.MARKER).exists())

    def test_all_model_ends_required_before_report(self):
        original=wrapper.finish_report
        def broken(*args):
            p=self.target/wrapper.ATTEMPT/'inference/models.jsonl'
            events=wrapper.rows(p);events[1]['status']='failed'
            p.write_text(''.join(json.dumps(r)+'\n' for r in events))
            return original(*args)
        with self.contexts(),patch.object(wrapper,'finish_report',side_effect=broken),patch('urllib.request.urlopen',side_effect=self.http):
            with self.assertRaisesRegex(ValueError,'model ends'):self.execute()
        self.assertFalse((self.target/wrapper.MARKER).exists())

    def test_incomplete_model_journal_retains_credit_despite_full_raw(self):
        original=wrapper.component.drain_model_tasks
        def broken(*args):
            original(*args)
            journal=args[2];events=wrapper.rows(journal)
            journal.write_text(''.join(json.dumps(r)+'\n' for r in events[:-1]))
            raise OSError('model journal durability unknown')
        with self.contexts(),patch.object(wrapper.component,'drain_model_tasks',side_effect=broken),patch('urllib.request.urlopen',side_effect=self.http):
            with self.assertRaisesRegex(OSError,'model journal'):self.execute()
        self.assertEqual(next(iter(wrapper.read(self.budget)['inference_credits'].values()))['status'],'active')
        self.assertFalse((self.target/wrapper.MARKER).exists())

    def test_model_end_guard_and_both_locks_unadmitted(self):
        for path in (self.root/'artifacts/operations/repetition_writer.lock',self.target/'artifacts/parallel_writer.lock'):
            with path.open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):self.execute()
        self.spec['admitted']=False
        with self.assertRaisesRegex(ValueError,'reviewed'):self.execute()
        self.assertFalse((self.target/wrapper.ATTEMPT).exists())


if __name__=='__main__':unittest.main()
