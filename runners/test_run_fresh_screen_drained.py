import copy
import fcntl
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import unittest

import analyze
import orchestrate
import protocol
import expanded_protocol
import run_fresh_screen_drained as wrapper
import test_fresh_screen_drained as fixture

ROOT = Path(__file__).resolve().parents[2]


class WrapperTests(unittest.TestCase):
    http = fixture.ScreeningTests.http

    def setUp(self):
        fixture.ScreeningTests.setUp(self)
        for rel in wrapper.REQUIRED:
            p = self.root/rel
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/rel, p)
        self.progress.write_text(json.dumps({k: {'status': v} for k, v in {
            'smoke': 'complete', 'AP1_baseline': 'complete', 'AP2_development': 'complete',
            'AP2_confirmation': 'complete', 'AP1_timing': 'complete', 'AP3_AP4_analysis': 'complete_automated', 'supplement_billiger50': 'complete_automated'}.items()}))
        self.history = self.target/'artifacts/stage_execution.jsonl'
        self.history.write_text(''.join(json.dumps({'stage': st, 'status': status, 'run_id': 2, 'slurm_job_id': job})+'\n'
                                       for st, status, job in [('smoke','started','40'),('smoke','complete','40'),
                                                               ('core','started','42'),('core','complete','42'),
                                                               ('supplement','started','44'),('supplement','complete','44')]))
        report = self.target/'artifacts/results/billiger50/metrics.json'
        report.parent.mkdir(parents=True);report.write_text('fixture prerequisite report')
        self.reports = {str(report.relative_to(self.target)): wrapper.sha(report)}
        history = wrapper.rows(self.history)
        history[-1].update(report_sha256=self.reports, admission_sha256='supplement-admission')
        self.history.write_text(''.join(json.dumps(r)+'\n' for r in history))
        self.spec = {'admitted': True, 'stage': 'screen', 'run_id': 2, 'slurm_job_id': '43',
            'replica': str(self.target), 'budget_cap_usd': 200, 'budget_authority_sha256': wrapper.APPROVAL_SHA,
            'max_invocations': 1, 'historical_unknown_outcomes_disclosed': True,
            'source_sha256': {r: wrapper.sha(self.root/r) for r in wrapper.REQUIRED},
            'runner_sha256': 'fixture', 'core_job_id': '42', 'supplement_job_id': '44', 'supplement_admission_sha256': 'supplement-admission',
            'supplement_report_sha256': self.reports, 'prerequisite_stage_journal_sha256': wrapper.sha(self.history),
            'progress_sha256': wrapper.sha(self.progress), 'cache_sha256_or_absent': wrapper.cache_inventory(self.target, self.screen),
            'parallel_policy': 'fixture_policy.json', 'parallel_policy_sha256': 'policy',
            'reviewed_dependencies': {'43': '(null)', '99': 'afterok:366115(unfulfilled)'}}
        self.admission = self.root/'admission.json'; self.admission.write_text(json.dumps(self.spec))
        self.replica = {'runner_sha256': 'fixture'}
        self.checked = {'policy': {'policy_sha256': 'policy'}}
        self.policy = {'policy_sha256': 'policy', 'reviewed_jobs': [{'id':'43','self':True}, {'id':'99','self':False}]}
        self.bodies = []
        self.accounting = '42|COMPLETED|0:0|now\n42.batch|COMPLETED|0:0|now\n42.extern|COMPLETED|0:0|now\n'
        self.supplement_accounting = self.accounting.replace('42', '44')
        self.dependency = 'afterok:366115(unfulfilled)'
        self.other_script = 'independent Run1 writer'

    def execute(self):
        return wrapper.execute_bound(self.root, self.target, 2, self.admission, self.spec, '43', self.replica,
                                     self.inference, self.screen, expanded_protocol, protocol, analyze, orchestrate)

    def scheduler(self, args, **kwargs):
        if args[0] == 'sacct': return self.accounting if args[3] == '42' else self.supplement_accounting
        if args[:3] == ['scontrol','show','job']:
            return 'Dependency='+('(null)' if args[3]=='43' else self.dependency)
        if args[:3] == ['scontrol','write','batch_script']: return self.other_script
        raise AssertionError(args)

    def ready(self):
        with patch.object(wrapper,'planning',return_value={'fixture':True}), \
                patch.object(wrapper.supplement_component,'inventory',return_value=[{'complete':True}]), \
                patch.object(wrapper,'make_wrapper') as factory, patch.object(wrapper.subprocess,'check_output',side_effect=self.scheduler):
            factory.return_value.review.return_value = self.policy
            return wrapper.verify_ready(self.root,self.target,2,self.spec,'43',self.replica,protocol,self.screen)

    def test_real_component_report_marker_and_locks_preserve_old_history(self):
        old = self.history.read_bytes(); original_report = wrapper.finish_report
        def report(*args):
            for path in (self.root/'artifacts/operations/repetition_writer.lock', self.target/'artifacts/parallel_writer.lock'):
                with path.open('a') as f:
                    with self.assertRaises(BlockingIOError): fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return original_report(*args)
        with patch.object(wrapper,'verify_ready',return_value=self.checked), patch.object(wrapper,'verify',return_value=self.replica), \
                patch.object(wrapper,'finish_report',side_effect=report), patch('urllib.request.urlopen',side_effect=self.http) as http:
            self.assertEqual(self.execute(),0)
        self.assertEqual(http.call_count,1134)
        self.assertTrue(self.history.read_bytes().startswith(old))
        new = wrapper.rows(self.history)[-2:]
        self.assertEqual([r['status'] for r in new],['started','complete'])
        self.assertEqual(new[0]['admission_sha256'],wrapper.sha(self.admission))
        self.assertEqual(len(new[1]['report_sha256']),2)
        metrics = wrapper.read(self.target/'artifacts/results/expanded_screen/development_metrics.json')
        self.assertEqual(len(metrics),567);self.assertTrue(all(r['n']==2 and r['f1']==1 for r in metrics))
        self.assertEqual(wrapper.read(self.progress)[wrapper.component.STAGE]['status'],'complete')
        budget = wrapper.read(self.root/'artifacts/budget.json')
        self.assertAlmostEqual(budget['spent'],101.134);self.assertAlmostEqual(budget['reserved'],3)
        with self.assertRaises(ValueError): self.execute()

    def test_real_budget_stop_no_report_or_complete_and_all_model_ends(self):
        p=self.root/'artifacts/budget.json';b=wrapper.read(p);b['spent']=196.99;p.write_text(json.dumps(b));before=p.read_bytes()
        with patch.object(wrapper,'verify_ready',return_value=self.checked), patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            with self.assertRaisesRegex(RuntimeError,'drained'):self.execute()
        self.assertEqual([r['status'] for r in wrapper.rows(self.history)[-2:]],['started','failed'])
        self.assertFalse((self.target/'artifacts/results/expanded_screen').exists())
        self.assertEqual(wrapper.read(self.progress)[wrapper.component.STAGE]['status'],'needs_attention')
        self.assertEqual(p.read_bytes(),before)
        models=wrapper.rows(self.target/wrapper.ATTEMPT/'inference/models.jsonl')
        self.assertEqual(sum(r['event']=='model_end' for r in models),7)

    def test_ready_actual_code_with_synthetic_scheduler_and_cost(self):
        result=self.ready()
        self.assertEqual(result['dependencies'],self.spec['reviewed_dependencies'])
        self.assertEqual(len(result['audited_cells']),567)

    def test_pending_failed_or_missing_core_step_denied(self):
        for value in (self.accounting.replace('42.batch|COMPLETED','42.batch|RUNNING'),
                      self.accounting.replace('42.batch|COMPLETED|0:0','42.batch|FAILED|1:0'),
                      self.accounting.split('42.extern')[0]):
            self.accounting=value
            with self.assertRaises(ValueError):self.ready()

    def test_supplement_steps_history_and_reports_required(self):
        original = self.supplement_accounting
        for value in (original.replace('44.batch|COMPLETED', '44.batch|RUNNING'),
                      original.replace('44.batch|COMPLETED|0:0', '44.batch|FAILED|1:0'),
                      original.split('44.extern')[0]):
            self.supplement_accounting = value
            with self.assertRaises(ValueError): self.ready()
        self.supplement_accounting = original
        old = self.history.read_text()
        self.history.write_text('\n'.join(old.splitlines()[:-2])+'\n')
        with self.assertRaisesRegex(ValueError, 'prerequisites'): self.ready()
        self.history.write_text(old)
        (self.target/'artifacts/results/billiger50/metrics.json').write_text('drift')
        with self.assertRaisesRegex(ValueError, 'reports changed'): self.ready()

    def test_all_joins_and_other_replica_writer_checked(self):
        for dep in ('afterany:366115(unfulfilled)','afterok:1?afterok:2','afterok:999(unfulfilled)'):
            self.dependency=dep
            with self.assertRaises(ValueError):self.ready()
        self.dependency=self.spec['reviewed_dependencies']['99'];self.other_script='python runner --run-id 2'
        with self.assertRaisesRegex(ValueError,'Another'):self.ready()

    def test_history_cache_and_source_drift_denied(self):
        before=self.history.read_text()
        self.history.write_text(before.replace('"complete"','"failed"'))
        with self.assertRaises(ValueError):self.ready()
        self.history.write_text(before)
        self.spec['cache_sha256_or_absent']={}
        with self.assertRaisesRegex(ValueError,'cache changed'):self.ready()
        self.spec['cache_sha256_or_absent']=wrapper.cache_inventory(self.target,self.screen)
        (self.root/wrapper.SELF).write_text('drift')
        with self.assertRaisesRegex(ValueError,'source drift'):self.ready()

    def test_no_admission_means_no_attempt_or_http(self):
        self.spec['admitted']=False
        with patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            with self.assertRaisesRegex(ValueError,'admission'):self.execute()
        self.assertFalse((self.target/wrapper.ATTEMPT).exists())

    def test_full_cost_scenario_budget_and_source_gates(self):
        for rel in (wrapper.COST,wrapper.COST_PROOF):
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
        expected=wrapper.read(self.root/wrapper.COST)['source_journal_sha256_or_absent']
        scope=wrapper.cache_inventory(self.root,self.screen)
        source={p:h for p,h in expected.items() if p in scope}
        with patch.object(wrapper,'cache_inventory',return_value=source):
            good=wrapper.planning(self.root,{'spent':100,'reserved':3},self.screen)
            self.assertFalse(good['future_cache_discount'])
            with self.assertRaisesRegex(ValueError,'headroom'):wrapper.planning(self.root,{'spent':195,'reserved':3},self.screen)
            source=copy.deepcopy(source);source[next(iter(source))]='drift'
        with patch.object(wrapper,'cache_inventory',return_value=source):
            with self.assertRaisesRegex(ValueError,'source journal'):wrapper.planning(self.root,{'spent':100,'reserved':3},self.screen)


if __name__ == '__main__':unittest.main()
