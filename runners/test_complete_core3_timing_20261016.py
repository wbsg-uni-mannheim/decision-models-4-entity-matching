"""Offline full successor execution, publication faults and completion guards.

Scheduler/admission/history authority is synthetic. Inference, account, raw
parser, frozen analysis/report and independent numerical verifier execute.
"""
import fcntl
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import protocol
import analyze
import core3_final_validation_20261016 as final
import complete_core3_timing_20261016 as complete
import test_core3_timing_wrapper_20261016 as fixture_module


def fixture():
    f = fixture_module.TimingWrapperTests('test_full_frozen_analysis_in_private_workspace');f.setUp()
    for b,cases in f.cases.items():
        for split in ('route','test'):
            (f.target/'data'/b/(split+'.json')).write_text(json.dumps(cases))
        for m in protocol.MODELS:
            for v,splits in [('B0',('prompt','route','test'))]+(
                    [(v,('prompt','test') if v == 'P1' else ('prompt',)) for v in ('P1','P2','S1','O1')]
                    if m in protocol.ABLATION_MODELS else []):
                for split in splits:
                    p = f.target/'runs/quality'/m/v/b/(split+'.jsonl');p.parent.mkdir(parents=True,exist_ok=True)
                    rr = [{**f.record(c,m,i+1),'variant':v,'request_hash':protocol.digest(protocol.payload(c,m,v))} for i,c in enumerate(cases)]
                    p.write_text(''.join(json.dumps(r)+'\n' for r in rr))
    selected = {m:dict(variant='P1',development_macro_match_f1={v:2/3 for v in ('P1','P2','S1')}) for m in protocol.ABLATION_MODELS}
    (f.target/'configs/selected_variants.json').write_text(json.dumps(selected))
    (f.target/'configs/manifest.json').write_bytes((ROOT/'configs/manifest.json').read_bytes())
    (f.target/'artifacts/budget.json').symlink_to(f.root/'artifacts/budget.json')
    progress = {k:dict(status=v) for k,v in dict(smoke='complete',AP1_baseline='complete',AP2_development='complete',AP2_confirmation='complete',AP1_timing='needs_attention').items()}
    (f.target/'artifacts/progress.json').write_text(json.dumps(progress))
    stop = f.root/'artifacts/verification/grouped_stop_reconciled_20261015.json';stop.parent.mkdir();stop.write_text('{"resolved_reserve_events":{}}')
    f.d['controlled_stop_proof_sha256'] = final.sha(stop)
    f.d['cells'] = []
    for p in sorted((f.target/'runs/quality').rglob('*.jsonl')):
        m,v,b = p.parts[-4:-1]
        f.d['cells'].append(dict(namespace='quality',model=m,variant=v,benchmark=b,split=p.stem,responses=20,attempts=20,
                                unknown_attempts=0,failure_rows=0,known_usd=.02))
    f.d['protected_sha256_or_absent'] = {str(p.relative_to(f.target)):final.sha(p) for p in f.target.rglob('*') if p.is_file() and p.name != 'budget.json'}
    f.spec['sources'] = dict.fromkeys(complete.REQUIRED,'fixture')
    f.admission.write_text(json.dumps(f.spec))
    plan_path = f.root/final.PLAN;plan_path.parent.mkdir();plan_path.write_text(json.dumps(f.saved))
    f.old = {p:p.read_bytes() for p in f.target.rglob('*') if p.is_file()}
    return f


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture();self.addCleanup(self.f.doCleanups)
        self.gates = 0

    def assert_locks(self):
        f = self.f
        for p in (f.root/'artifacts/operations/repetition_writer.lock',f.target/'artifacts/parallel_writer.lock'):
            with p.open('a') as stream:
                with self.assertRaises(BlockingIOError):fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def gate(self,*args):
        self.assert_locks();self.gates += 1
        return dict(fixture_scheduler=True)

    def independent(self,root,admission_path,proto):
        self.assert_locks();f = self.f
        attempt = f.target/final.ATTEMPT
        events = final.rows(attempt/'execution.jsonl')
        self.assertEqual([e['status'] for e in events],['started','measured_pending_audit'])
        self.assertEqual(final.rows(f.target/'artifacts/stage_execution.jsonl')[1:],events)
        proof = final.verify_evidence(f.target,proto,f.saved,f.d,attempt)
        proof.update(status='independent_raw_execution_cost_numeric_report_verified',run_id=3,core_complete=False,
                     admission_sha256=final.sha(admission_path),prior_sha256=final.sha(attempt/'prior.json'),plan_sha256=final.PLAN_SHA)
        return proof

    def execute(self,gate=None,independent=None,use_runtime=False):
        f = self.f
        modules = (f.inference,None,protocol,analyze)
        if use_runtime:
            import shutil
            for name in ('protocol','run','orchestrate','analyze'):
                p = f.target/'src'/(name+'.py');p.parent.mkdir(exist_ok=True)
                shutil.copyfile(ROOT/'src'/(name+'.py'),p)
            with patch.object(complete.executor,'verify'):
                modules = complete.runtime(f.root)
            modules[0].key = lambda:'mock'
        with patch.dict(os.environ,{'SLURM_JOB_ID':'999'}),patch.object(complete,'current_gate',side_effect=gate or self.gate), \
            patch.object(complete.executor,'preflight',return_value=(f.target,f.saved,f.d,{'fixture':True})), \
            patch.object(final,'verify',side_effect=independent or self.independent),patch('urllib.request.urlopen',side_effect=f.http):
            return complete.execute(f.root,f.admission,*modules)

    def test_full_execution_independent_audit_exclusive_commit_and_repeat_guard(self):
        marker = self.execute(use_runtime=True);f = self.f
        self.assertEqual(self.gates,2);self.assertEqual(len(f.calls),420)
        self.assertEqual(marker,complete.require_committed(f.target))
        self.assertFalse(marker['full_expanded_study_complete']);self.assertFalse(marker['private_legacy_draft_published'])
        self.assertEqual(final.read(f.root/'artifacts/budget.json')['inference_credits'],{'foreign':{'status':'active','funded':3}})
        self.assertAlmostEqual(final.read(f.root/'artifacts/budget.json')['spent'],120.42)
        for p,content in f.old.items():
            if p.name == 'stage_execution.jsonl':self.assertTrue(p.read_bytes().startswith(content))
            elif p.name not in ('progress.json','budget.json'):self.assertEqual(p.read_bytes(),content)
        self.assertFalse((f.target/'manuscript/draft.md').exists())
        self.assertEqual(final.read(f.target/'artifacts/progress.json')['AP3_AP4_analysis']['status'],'complete_automated')
        with self.assertRaisesRegex(ValueError,'Never repeat'):self.execute()
        self.assertEqual(len(f.calls),420)

    def test_no_completion_on_independent_failure(self):
        with self.assertRaisesRegex(ValueError,'independent failure'):
            self.execute(independent=lambda *a:(_ for _ in ()).throw(ValueError('independent failure')))
        self.assertFalse((self.f.target/complete.RESULTS).exists());self.assertFalse((self.f.target/complete.MARKER).exists())
        self.assertEqual(final.rows(self.f.target/'artifacts/stage_execution.jsonl')[-1]['status'],'failed')
        self.assertEqual(final.read(self.f.target/'artifacts/progress.json')['AP1_timing']['status'],'needs_attention')

    def test_final_scheduler_gate_failure_preserves_private_reports(self):
        def gate(*args):
            result = self.gate(*args)
            if self.gates == 2:raise ValueError('new writer')
            return result
        with self.assertRaisesRegex(ValueError,'new writer'):self.execute(gate=gate)
        self.assertTrue((self.f.target/final.ATTEMPT/'staged_result.json').exists())
        self.assertFalse((self.f.target/complete.RESULTS).exists())

    def test_partial_publication_and_marker_fsync_failure_never_commit_or_retry(self):
        original = complete.durable_bytes
        def failed(path,content):
            self.assert_locks()
            original(path,content)
            if path.name == 'main_metrics.json':raise OSError('readable unconfirmed report')
        with patch.object(complete,'durable_bytes',side_effect=failed):
            with self.assertRaisesRegex(OSError,'unconfirmed report'):self.execute()
        self.assertTrue((self.f.target/complete.RESULTS/'main_metrics.json').exists())
        self.assertFalse((self.f.target/complete.MARKER).exists())
        with self.assertRaisesRegex(ValueError,'Never repeat'):self.execute()

    def test_marker_written_but_final_event_failure_is_not_committed(self):
        original = complete.executor.durable_append
        def append(path,row,canonical):
            if path.name == 'final_execution.jsonl' and row['status'] == 'complete':raise OSError('final event storage')
            return original(path,row,canonical)
        with patch.object(complete.executor,'durable_append',side_effect=append):
            with self.assertRaisesRegex(OSError,'final event storage'):self.execute()
        self.assertTrue((self.f.target/complete.MARKER).exists())
        with self.assertRaisesRegex(ValueError,'Incomplete final stage'):complete.require_committed(self.f.target)
        with self.assertRaisesRegex(ValueError,'Never repeat'):self.execute()

    def test_current_gate_all_steps_all_joins_overlap_and_budget(self):
        f = self.f;f.spec['policy'] = 'policy.json'
        (f.root/'configs/expanded_execution.json').write_text(json.dumps(dict(e2_confirmation_job=366694,e2_schema_job=366695)))
        review = dict(policy_sha256='fixture',reviewed_jobs=[dict(id='999',self=True,scope='core3')])
        state = dict(active_step=False,bad_join=False,unheld=False)
        def command(args,**kwargs):
            if args[0] == 'sacct':
                job = args[3]
                batch = 'RUNNING' if state['active_step'] and job == '365894' else 'FAILED'
                return f'{job}|FAILED|1:0|end|\n{job}.batch|{batch}|1:0|end|\n{job}.extern|COMPLETED|0:0|end|\n'
            job = args[3];dep = '366694:366695'+(':365300' if job == '365355' else '')
            if state['bad_join']:dep = '366694'
            return f'JobState=PENDING StartTime=Unknown Priority={1 if state["unheld"] else 0} Dependency=afterok:{dep}'
        with patch.object(complete.executor.authority,'authority'),patch.object(complete.executor,'verify'), \
            patch.object(final,'sha',return_value='fixture'),patch.object(complete.executor.authority,'private_module',return_value=types.SimpleNamespace(review=lambda *a:review)), \
            patch.object(complete.subprocess,'check_output',side_effect=command):
            checked = complete.current_gate(f.root,f.target,f.spec,'999')
            self.assertEqual(len(checked['historical_steps']),10)
            for key in state:
                state[key] = True
                with self.assertRaises((ValueError,RuntimeError,AssertionError)):complete.current_gate(f.root,f.target,f.spec,'999')
                state[key] = False
            review['reviewed_jobs'].append(dict(id='9999',scope='core3'))
            with self.assertRaisesRegex(ValueError,'overlapping'):complete.current_gate(f.root,f.target,f.spec,'999')
            review['reviewed_jobs'].pop()
            budget = final.read(f.root/'artifacts/budget.json');budget['spent'] = 199
            (f.root/'artifacts/budget.json').write_text(json.dumps(budget))
            with self.assertRaisesRegex(ValueError,'ceiling'):complete.current_gate(f.root,f.target,f.spec,'999')

    def test_cli_rejects_unadmitted_before_runtime_and_private_runtime_preserves_modules(self):
        self.f.admission.write_text('{"admitted":false}')
        with patch.object(complete,'runtime',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError,'Unadmitted final timing CLI'):
                complete.main(['--root',str(self.f.root),'--admission',str(self.f.admission)])
        import shutil
        for name in ('protocol','run','orchestrate','analyze'):
            p = self.f.target/'src'/(name+'.py');p.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT/'src'/(name+'.py'),p)
        import run
        import orchestrate
        originals = (run.PROJECT,orchestrate.PROJECT,analyze.PROJECT)
        with patch.object(complete.executor,'verify'):
            runtime = complete.runtime(self.f.root)
        self.assertEqual([m.PROJECT for m in runtime[:2]],[self.f.target,self.f.target])
        self.assertEqual(runtime[0].account.__globals__['PROJECT'],self.f.target)
        self.assertEqual(runtime[1].select.__globals__['PROJECT'],self.f.target)
        self.assertIs(runtime[3].save.__globals__,vars(runtime[3]))
        self.assertIs(runtime[3].load.__globals__,vars(runtime[3]))
        self.assertEqual((run.PROJECT,orchestrate.PROJECT,analyze.PROJECT),originals)
        self.assertEqual(self.f.calls,[])

    def test_unadmitted_rejected_before_locks_or_attempt(self):
        self.f.admission.write_text('{"admitted":false}')
        with self.assertRaisesRegex(ValueError,'Unadmitted'):
            complete.execute(self.f.root,self.f.admission,self.f.inference,None,protocol,analyze)
        self.assertFalse((self.f.target/final.ATTEMPT).exists())
        self.assertFalse((self.f.target/'artifacts/parallel_writer.lock').exists())
        self.assertEqual(self.f.calls,[])


if __name__ == '__main__':unittest.main()
