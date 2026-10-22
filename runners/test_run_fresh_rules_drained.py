from contextlib import ExitStack
import fcntl
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import types
import unittest
from unittest.mock import patch

import analyze
import expanded_rules
import protocol
import test_fresh_rules_grouped as fixture
import run_fresh_rules_drained as wrapper

ROOT = Path(__file__).resolve().parents[2]


class RulesWrapperTests(unittest.TestCase):
    def setUp(self):
        self.g = fixture.GroupedTests(); self.g.setUp(); self.addCleanup(self.g.doCleanups)
        self.root,self.target = self.g.root,self.g.target
        self.rules,self.inference = self.g.rules,self.g.inference
        self.budget = self.g.budget
        for rel in wrapper.REQUIRED:
            p = self.root/rel; p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/rel,p)
        progress = dict(smoke='complete',AP1_baseline='complete',AP2_development='complete',
            AP2_confirmation='complete',AP1_timing='complete',AP3_AP4_analysis='complete_automated',
            supplement_billiger50='complete_automated',expanded_E2_E3_screen='complete')
        self.progress = self.target/'artifacts/progress.json'
        self.progress.write_text(json.dumps({k:{'status':v} for k,v in progress.items()}))
        self.history = self.target/'artifacts/stage_execution.jsonl'; self.history.touch()
        prerequisites = []
        for i,stage in enumerate(('smoke','core','supplement','screen','confirmation','schema')):
            job = str(40+i); before = wrapper.sha(self.history)
            admission = self.root/(stage+'_admission.json')
            admission.write_text(json.dumps(dict(admitted=True,stage=stage,run_id=2,slurm_job_id=job,
                prerequisite_stage_journal_sha256=before)))
            if stage in ('supplement','screen'):
                folder = self.target/'artifacts/results'/('billiger50' if stage=='supplement' else 'expanded_screen')
                folder.mkdir(parents=True); (folder/'fixture.json').write_text('{}')
            else:
                for rel in wrapper.report_paths(self.target,stage):
                    p = self.target/rel; p.parent.mkdir(parents=True,exist_ok=True)
                    if not p.exists():p.write_text('fixture prerequisite report')
            reports = {p:wrapper.sha(self.target/p) for p in wrapper.report_paths(self.target,stage)}
            pair = [dict(stage=stage,status=status,run_id=2,slurm_job_id=job,
                admission_sha256=wrapper.sha(admission),prior_stage_journal_sha256=before)
                for status in ('started','complete')]
            if stage!='smoke':pair[-1]['report_sha256']=reports
            with self.history.open('a') as f:f.write(''.join(json.dumps(r)+'\n' for r in pair))
            prerequisites.append(dict(stage=stage,slurm_job_id=job,admission=admission.name,
                admission_sha256=wrapper.sha(admission),stage_end_sha256=protocol.digest(pair[-1]),report_sha256=reports))
        self.spec = dict(admitted=True,stage='rules',run_id=2,slurm_job_id='50',replica=str(self.target),
            budget_cap_usd=200,budget_authority_sha256=wrapper.APPROVAL_SHA,
            max_invocations_per_model_per_phase=1,historical_unknown_outcomes_disclosed=True,
            accounting_transport='bounded_process_credit',journal_transport='group_commit_v1',
            credit_chunk_usd=8,checkpoint_reservations=512,history_adapter='canonical_success_only_v1',
            source_sha256={p:wrapper.sha(self.root/p) for p in wrapper.REQUIRED},runner_sha256='fixture',
            prerequisites=prerequisites,prerequisite_stage_journal_sha256=wrapper.sha(self.history),
            progress_sha256=wrapper.sha(self.progress),cache_inventory=wrapper.snapshot(self.target,self.rules,protocol),
            parallel_policy='fixture_policy.json',parallel_policy_sha256='policy',
            reviewed_dependencies={'50':'(null)','99':'afterok:366694(unfulfilled)'})
        self.admission=self.root/'admission.json'; self.save_spec()
        self.replica={'runner_sha256':'fixture'}
        self.policy={'policy_sha256':'policy','reviewed_jobs':[{'id':'50','self':True},{'id':'99','self':False}]}
        self.dependency=self.spec['reviewed_dependencies']['99'];self.script='independent writer'
        self.bad_step=False
        self.sources=[dict(namespace='quality',model=m,variant=v,benchmark=b,split='prompt',responses=1,
            known_proxy_usd=.001,historical_unknown_scenario_usd=0)
            for m in protocol.MODELS for v in ('B0','X_P_domain_0') for b in protocol.BENCHMARKS]

    def save_spec(self):self.admission.write_text(json.dumps(self.spec))

    def scheduler(self,args,**kwargs):
        if args[0]=='sacct':
            job=args[3]
            return ''.join(f'{job}{s}|'+('RUNNING|0:0|Unknown' if s=='.extern' and self.bad_step else 'COMPLETED|0:0|now')+'\n' for s in ('','.batch','.extern'))
        if args[:3]==['scontrol','show','job']:
            return 'Dependency='+('(null)' if args[3]=='50' else self.dependency)
        if args[:3]==['scontrol','write','batch_script']:return self.script
        raise AssertionError(args)

    def contexts(self):
        stack=ExitStack()
        stack.enter_context(patch.object(wrapper,'cost_sources',return_value=self.sources))
        stack.enter_context(patch.object(wrapper.common,'make_wrapper')).return_value.review.return_value=self.policy
        stack.enter_context(patch.object(wrapper.common.subprocess,'check_output',side_effect=self.scheduler))
        stack.enter_context(patch.object(wrapper,'verify',return_value=self.replica))
        stack.enter_context(patch.object(wrapper.grouped,'verify',return_value=self.replica))
        return stack

    def ready(self):return wrapper.verify_ready(self.root,self.target,2,self.spec,'50',self.replica,self.rules,protocol)
    def execute(self):return wrapper.execute_bound(self.root,self.target,2,self.admission,self.spec,'50',self.replica,self.inference,self.rules,protocol,analyze)

    def held(self):
        for p in (self.root/'artifacts/operations/repetition_writer.lock',self.target/'artifacts/parallel_writer.lock'):
            with p.open('a') as f:
                with self.assertRaises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_full_wrapper_actual_cache_discount_credit_locks_and_frozen_report(self):
        old=self.history.read_bytes(); finish=wrapper.grouped.finish_report; credit_init=wrapper.Credit
        def funded(*args,**kw):
            self.held();prior=self.target/wrapper.ATTEMPT/'prior.json'
            self.assertTrue(prior.exists());self.assertEqual(args[2]['prior_sha256'],wrapper.sha(prior))
            self.assertEqual(wrapper.read(prior)['valid_prior_responses'],2)
            return credit_init(*args,**kw)
        def checked(*args):
            self.held();self.assertEqual(next(iter(wrapper.read(self.budget)['inference_credits'].values()))['status'],'closed')
            return finish(*args)
        with self.contexts(),patch.object(wrapper,'Credit',side_effect=funded),patch.object(wrapper.grouped,'finish_report',side_effect=checked),patch('urllib.request.urlopen',side_effect=self.g.f.http) as calls:
            self.assertEqual(self.execute(),0)
        self.assertEqual(calls.call_count,236);self.assertTrue(self.history.read_bytes().startswith(old))
        plan=wrapper.read(self.target/wrapper.ATTEMPT/'cost.json')
        self.assertEqual(plan['missing'],236);self.assertEqual(plan['responses'],2);self.assertAlmostEqual(plan['known_proxy_usd'],.236)
        receipt=wrapper.read(self.target/wrapper.ATTEMPT/'credit.json')
        self.assertEqual(receipt['raw']['prior_responses_excluded'],2);self.assertEqual(receipt['raw']['attempts'],236)
        self.assertEqual(receipt['credit']['global_transactions'],2)
        d=wrapper.read(self.budget);self.assertAlmostEqual(d['spent'],100.236);self.assertAlmostEqual(d['reserved'],3)
        end=wrapper.rows(self.history)[-1];self.assertEqual(end['status'],'complete');self.assertEqual(len(end['report_sha256']),9)
        independent=self.root/'independent';(independent/'configs').mkdir(parents=True)
        (independent/'configs/learned_rules.json').write_bytes((self.target/'configs/learned_rules.json').read_bytes())
        load=wrapper.bind(analyze.load,PROJECT=self.target,payload=self.g.f.fixture.payload)
        def load_cell(m,v,b,s):return load(m,v,b,s,namespace=expanded_rules.CONDITIONS.get(v,'quality'))
        wrapper.bind(expanded_rules.report,PROJECT=independent,load_cell=load_cell)(wrapper.read(self.target/'configs/expanded_rules_manifest.json'))
        for p in end['report_sha256']:self.assertEqual((self.target/p).read_bytes(),(independent/p).read_bytes())
        with self.assertRaisesRegex(ValueError,'Prior attempt'):self.execute()

    def test_budget_rechecked_after_receipt_before_funding(self):
        original=wrapper.files.durable_write
        def change(path,data):
            result=original(path,data)
            if path.name=='cost.json':
                d=wrapper.read(self.budget);d['spent']=196.99;self.budget.write_text(json.dumps(d))
            return result
        with self.contexts(),patch.object(wrapper.files,'durable_write',side_effect=change),patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            with self.assertRaisesRegex(ValueError,'scenario'):self.execute()
        self.assertNotIn('inference_credits',wrapper.read(self.budget))

    def test_prior_receipt_failure_never_funds_or_starts(self):
        original=wrapper.files.durable_write
        def broken(path,data):
            original(path,data)
            if path.name=='prior.json':raise OSError('unconfirmed prior fsync')
        old=self.history.read_bytes()
        with self.contexts(),patch.object(wrapper.files,'durable_write',side_effect=broken),patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            with self.assertRaises(OSError):self.execute()
        self.assertEqual(self.history.read_bytes(),old);self.assertNotIn('inference_credits',wrapper.read(self.budget))
        with self.assertRaisesRegex(ValueError,'Prior attempt'):self.execute()

    def test_unknown_outcome_keeps_credit_even_zero_outstanding(self):
        once=True
        def broken(req,**kw):
            nonlocal once
            if once:once=False;raise http.client.IncompleteRead(b'partial')
            return self.g.f.http(req,**kw)
        with self.contexts(),patch('urllib.request.urlopen',side_effect=broken):
            with self.assertRaisesRegex(ValueError,'Unknown'):self.execute()
        d=wrapper.read(self.budget);self.assertEqual(next(iter(d['inference_credits'].values()))['status'],'active')
        retained=next(r for r in wrapper.rows(self.history) if r['status']=='credit_retained')
        self.assertEqual(retained['credit']['outstanding'],0);self.assertEqual(retained['credit']['unknown'],1)
        self.assertAlmostEqual(d['reserved'],11);self.assertFalse((self.target/wrapper.grouped.MARKER).exists())

    def test_readable_storage_failure_retains_credit(self):
        import grouped_inference
        original=grouped_inference.write_group;once=True
        def broken(path,lines):
            nonlocal once
            original(path,lines)
            if once:once=False;raise OSError('unconfirmed fsync')
        with self.contexts(),patch.object(grouped_inference,'write_group',side_effect=broken),patch('urllib.request.urlopen',side_effect=self.g.f.http):
            with self.assertRaisesRegex(ValueError,'durability'):self.execute()
        self.assertEqual(next(iter(wrapper.read(self.budget)['inference_credits'].values()))['status'],'active')

    def test_partial_publication_blocks_repeat_and_marker(self):
        original=wrapper.files.durable_write
        def broken(path,data):
            if path==self.target/wrapper.grouped.REPORT_DIR/'costs.json':raise OSError('publication failure')
            return original(path,data)
        with self.contexts(),patch.object(wrapper.files,'durable_write',side_effect=broken),patch('urllib.request.urlopen',side_effect=self.g.f.http):
            with self.assertRaisesRegex(OSError,'publication'):self.execute()
        self.assertTrue((self.target/wrapper.grouped.REPORT_DIR/'metrics.json').exists())
        self.assertFalse((self.target/wrapper.grouped.MARKER).exists())
        with self.assertRaisesRegex(ValueError,'Prior attempt'):self.execute()

    def test_all_steps_joins_overlap_sources_history_reports_cache(self):
        with self.contexts():
            self.ready();self.bad_step=True
            with self.assertRaises(ValueError):self.ready()
            self.bad_step=False;self.dependency='afterany:366694'
            with self.assertRaisesRegex(ValueError,'conjunctive'):self.ready()
            self.dependency=self.spec['reviewed_dependencies']['99'];self.script=str(self.target)
            with self.assertRaisesRegex(ValueError,'Another writer'):self.ready()
            self.script='independent'
            for p in [self.root/wrapper.SELF,self.root/'schema_admission.json',self.target/'artifacts/results/expanded_schema/metrics.json']:
                old=p.read_bytes();p.write_text('drift')
                with self.assertRaises(ValueError):self.ready()
                p.write_bytes(old)
            p=self.target/'runs/quality/jev/B0/wdc/test.jsonl';old=p.read_bytes();p.write_text('')
            with self.assertRaisesRegex(ValueError,'cache'):self.ready()
            self.spec['cache_inventory']=wrapper.snapshot(self.target,self.rules,protocol)
            with self.assertRaisesRegex(ValueError,'Incomplete'):self.ready()
            p.write_bytes(old);self.spec['cache_inventory']=wrapper.snapshot(self.target,self.rules,protocol)
            original=self.history.read_bytes();events=wrapper.rows(self.history)
            events.insert(3,dict(stage='core',status='failed',run_id=2,slurm_job_id='41'))
            self.history.write_text(''.join(json.dumps(r)+'\n' for r in events))
            self.spec['prerequisite_stage_journal_sha256']=wrapper.sha(self.history)
            with self.assertRaisesRegex(ValueError,'entire prerequisite'):self.ready()
            self.history.write_bytes(original);self.spec['prerequisite_stage_journal_sha256']=wrapper.sha(self.history)
            self.spec['prerequisites'][2]['stage']='stability'
            with self.assertRaisesRegex(ValueError,'entire prerequisite'):self.ready()

    def test_manual_cache_current_raw_validation_not_hypothetical_discount(self):
        with self.contexts():
            plan=wrapper.planning(self.root,self.target,2,self.rules,protocol)
            self.assertEqual(plan['responses'],2)
            p=self.target/'runs/quality/jev/X_P_domain_0/abt-buy/prompt.jsonl'
            # Locate actual fixture manual hit, then corrupt only its request.
            p=next(p for p in (self.target/'runs/quality').glob('*/X_P_domain_0/*/prompt.jsonl') if p.stat().st_size)
            records=wrapper.rows(p);records[0]['request_hash']='wrong'
            p.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(ValueError):wrapper.planning(self.root,self.target,2,self.rules,protocol)

    def test_cost_source_pin_gate(self):
        # Exact synthetic full source layout/proof tests the unmocked gate;
        # economic source correctness was independently checked in prior prep.
        cost=dict(quote_or_hard_bound=False,sources=self.sources,journal_sha256_or_absent={},protected_sha256={})
        for m in protocol.MODELS:
            for v in ('B0','X_P_domain_0'):
                for b in protocol.BENCHMARKS:
                    for suffix in wrapper.component.SUFFIXES:
                        rel=f'runs/quality/{m}/{v}/{b}/prompt{suffix}'
                        cost['journal_sha256_or_absent'][rel]=None
        for b in protocol.BENCHMARKS:
            p=self.root/f'data/{b}/prompt.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('[]')
            cost['protected_sha256'][str(p.relative_to(self.root))]=wrapper.sha(p)
        p=self.root/wrapper.COST;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(cost))
        proof=self.root/wrapper.PROOF;proof.write_text(json.dumps(dict(scenario_sha256=wrapper.sha(p))))
        with patch.object(wrapper,'COST_SHA',wrapper.sha(p)),patch.object(wrapper,'PROOF_SHA',wrapper.sha(proof)):
            self.assertEqual(wrapper.cost_sources(self.root,protocol),self.sources)
            journal=self.root/'runs/quality/jev/B0/abt-buy/prompt.jsonl';journal.parent.mkdir(parents=True);journal.touch()
            with self.assertRaisesRegex(ValueError,'journal changed'):wrapper.cost_sources(self.root,protocol)

    def test_paid_parser_failure_drains_before_exact_credit_close(self):
        def bad(req,**kw):
            model=next(m for m in protocol.MODELS if protocol.MODELS[m]['id']==json.loads(req.data)['model'])
            raw=self.g.f.fixture.raw(model)
            if model=='jev':raw['answers']['match']['noul']=2
            return fixture.fixture.Response(json.dumps(raw).encode())
        with self.contexts(),patch('urllib.request.urlopen',side_effect=bad):
            with self.assertRaisesRegex(RuntimeError,'phase model drain'):self.execute()
        credit=next(iter(wrapper.read(self.budget)['inference_credits'].values()))
        self.assertEqual(credit['status'],'closed');self.assertEqual(credit['attempts'],97)
        self.assertAlmostEqual(credit['charged'],.097)
        self.assertFalse((self.target/wrapper.ATTEMPT/'inference/test_models.jsonl').exists())

    def test_missing_model_end_keeps_credit(self):
        original=wrapper.private_component
        def builder(*args):
            runner=original(*args)
            def changed(*inner):
                result=runner(*inner)
                path=inner[-1]/'test_models.jsonl'
                events=wrapper.rows(path);events.pop(1)
                path.write_text(''.join(json.dumps(r)+'\n' for r in events))
                return result
            return changed
        with self.contexts(),patch.object(wrapper,'private_component',side_effect=builder),patch('urllib.request.urlopen',side_effect=self.g.f.http):
            with self.assertRaisesRegex(ValueError,'model ends'):self.execute()
        credit=next(iter(wrapper.read(self.budget)['inference_credits'].values()))
        self.assertEqual(credit['status'],'active');self.assertEqual(credit['outstanding_at_checkpoint'],0)
        self.assertFalse((self.target/wrapper.grouped.MARKER).exists())

    def test_foreign_lock_excludes_before_attempt(self):
        p=self.target/'artifacts/parallel_writer.lock'
        with p.open('a') as f,self.contexts():
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):self.execute()
        self.assertFalse((self.target/wrapper.ATTEMPT).exists())


if __name__=='__main__':unittest.main()
