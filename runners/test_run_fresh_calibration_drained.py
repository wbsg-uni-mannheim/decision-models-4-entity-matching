import copy
import fcntl
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import unittest

import analyze
import types
import expanded_screen
import protocol
import run_fresh_calibration_drained as wrapper
import test_fresh_calibration_drained as fixture

ROOT = Path(__file__).resolve().parents[2]


class WrapperTests(unittest.TestCase):
    http = fixture.CalibrationTests.http

    def setUp(self):
        fixture.CalibrationTests.setUp(self)
        for rel in wrapper.REQUIRED:
            p = self.root/rel
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/rel, p)
        self.progress.write_text(json.dumps({k: {'status': v} for k, v in {
            'smoke': 'complete', 'AP1_baseline': 'complete', 'AP2_development': 'complete',
            'AP2_confirmation': 'complete', 'AP1_timing': 'complete', 'AP3_AP4_analysis': 'complete_automated', 'supplement_billiger50': 'complete_automated', 'expanded_E2_E3_screen': 'complete'}.items()}))
        self.history = self.target/'artifacts/stage_execution.jsonl'
        self.history.write_text(''.join(json.dumps({'stage': st, 'status': status, 'run_id': 2, 'slurm_job_id': job})+'\n'
                                       for st, status, job in [('smoke','started','40'),('smoke','complete','40'),
                                                               ('core','started','42'),('core','complete','42'),
                                                               ('supplement','started','44'),('supplement','complete','44'),('screen','started','45'),('screen','complete','45')]))
        report = self.target/'artifacts/results/billiger50/metrics.json'
        report.parent.mkdir(parents=True);report.write_text('fixture prerequisite report')
        self.reports = {str(report.relative_to(self.target)): wrapper.sha(report)}
        history = wrapper.rows(self.history)
        history[-3].update(report_sha256=self.reports, admission_sha256='supplement-admission')
        screen_report = self.target/'artifacts/results/expanded_screen/development_metrics.json'
        screen_report.parent.mkdir(parents=True); screen_report.write_text('fixture screen report')
        self.screen_reports = {str(screen_report.relative_to(self.target)): wrapper.sha(screen_report)}
        history[-1].update(report_sha256=self.screen_reports, admission_sha256='screen-admission')
        self.history.write_text(''.join(json.dumps(r)+'\n' for r in history))
        self.spec = {'admitted': True, 'stage': 'calibration', 'run_id': 2, 'slurm_job_id': '43',
            'replica': str(self.target), 'budget_cap_usd': 200, 'budget_authority_sha256': wrapper.APPROVAL_SHA,
            'max_invocations': 1, 'historical_unknown_outcomes_disclosed': True,
            'source_sha256': {r: wrapper.sha(self.root/r) for r in wrapper.REQUIRED},
            'runner_sha256': 'fixture', 'core_job_id': '42', 'supplement_job_id': '44', 'supplement_admission_sha256': 'supplement-admission',
            'supplement_report_sha256': self.reports, 'screen_job_id': '45',
            'screen_admission_sha256': 'screen-admission', 'screen_report_sha256': self.screen_reports, 'prerequisite_stage_journal_sha256': wrapper.sha(self.history),
            'progress_sha256': wrapper.sha(self.progress), 'cache_sha256_or_absent': wrapper.cache_inventory(self.target, self.calibration),
            'parallel_policy': 'fixture_policy.json', 'parallel_policy_sha256': 'policy',
            'reviewed_dependencies': {'43': '(null)', '99': 'afterok:366115(unfulfilled)'}}
        self.screen = types.SimpleNamespace(**{**vars(expanded_screen), 'PROJECT': self.target, 'verify': lambda: {}})
        self.admission = self.root/'admission.json'; self.admission.write_text(json.dumps(self.spec))
        self.replica = {'runner_sha256': 'fixture'}
        self.checked = {'policy': {'policy_sha256': 'policy'}}
        self.policy = {'policy_sha256': 'policy', 'reviewed_jobs': [{'id':'43','self':True}, {'id':'99','self':False}]}
        self.bodies = []
        self.accounting = '42|COMPLETED|0:0|now\n42.batch|COMPLETED|0:0|now\n42.extern|COMPLETED|0:0|now\n'
        self.supplement_accounting = self.accounting.replace('42', '44')
        self.screen_accounting = self.accounting.replace('42', '45')
        self.dependency = 'afterok:366115(unfulfilled)'
        self.other_script = 'independent Run1 writer'

        # 26 development pairs permit real frozen n=25 fits, while n=100/500
        # remain explicitly unavailable. Both alternating fit/tune halves have both classes.
        for b in protocol.BENCHMARKS:
            data = self.target/'data'/b
            route = []
            for i in range(24):
                c = {'benchmark': b, 'pair_id': 'route'+str(i), 'label': (i//2)%2,
                     'records': {'left': {'title': str(i)}, 'right': {'title': str(i)}}}
                c['input_hash'] = protocol.digest(c['records']); route.append(c)
            (data/'route.json').write_text(json.dumps(route))
            (data/'calibration_pool.json').write_text(json.dumps(route+self.cases[b,'calibration_extra']))
            tests = copy.deepcopy(self.cases[b,'calibration_extra'])
            for i,c in enumerate(tests):
                c['pair_id'] = 'test'+str(i); c['entity_ids'] = {'left': 'l'+str(i), 'right': 'r'+str(i)}
            (data/'test.json').write_text(json.dumps(tests))
            for m in self.manifest['models']:
                template = wrapper.rows(self.target/'runs/quality'/m/'B0'/b/'route.jsonl')[0]
                for split,cases in [('route',route),('test',tests)]:
                    records=[]
                    for c in cases:
                        r=copy.deepcopy(template)
                        r.update(pair_id=c['pair_id'], input_hash=c['input_hash'], request_hash=protocol.digest(protocol.payload(c,m,'B0')))
                        records.append(r)
                    (self.target/'runs/quality'/m/'B0'/b/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
        self.spec['cache_sha256_or_absent'] = wrapper.cache_inventory(self.target,self.calibration)
        self.admission.write_text(json.dumps(self.spec))

    def execute(self):
        return wrapper.execute_bound(self.root, self.target, 2, self.admission, self.spec, '43', self.replica,
                                     self.inference, self.calibration, self.screen, protocol, analyze)

    def scheduler(self, args, **kwargs):
        if args[0] == 'sacct': return {'42':self.accounting,'44':self.supplement_accounting,'45':self.screen_accounting}[args[3]]
        if args[:3] == ['scontrol','show','job']:
            return 'Dependency='+('(null)' if args[3]=='43' else self.dependency)
        if args[:3] == ['scontrol','write','batch_script']: return self.other_script
        raise AssertionError(args)

    def ready(self):
        with patch.object(wrapper,'planning',return_value={'fixture':True}), \
                patch.object(wrapper.supplement_component,'inventory',return_value=[{'complete':True}]), \
                patch.object(wrapper.screen_component,'inventory',return_value=[{'complete':True}]), \
                patch.object(wrapper,'make_wrapper') as factory, patch.object(wrapper.subprocess,'check_output',side_effect=self.scheduler):
            factory.return_value.review.return_value = self.policy
            return wrapper.verify_ready(self.root,self.target,2,self.spec,'43',self.replica,protocol,self.calibration,self.screen)

    def test_real_component_report_marker_and_locks_preserve_old_history(self):
        old = self.history.read_bytes(); original_report = wrapper.finish_report
        reused=wrapper.cache_inventory(self.target,self.calibration,('route','test'))
        progress=self.progress.read_bytes()
        original_audit=wrapper.audit_test_controls
        def audit(*args):
            self.assertTrue((self.target/'configs/calibration_selection.json').exists())
            return original_audit(*args)
        def report(*args):
            for path in (self.root/'artifacts/operations/repetition_writer.lock', self.target/'artifacts/parallel_writer.lock'):
                with path.open('a') as f:
                    with self.assertRaises(BlockingIOError): fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return original_report(*args)
        with patch.object(wrapper,'verify_ready',return_value=self.checked), patch.object(wrapper,'verify',return_value=self.replica), \
                patch.object(wrapper,'finish_report',side_effect=report), \
                patch.object(wrapper,'audit_test_controls',side_effect=audit), patch('urllib.request.urlopen',side_effect=self.http) as http:
            self.assertEqual(self.execute(),0)
        self.assertEqual(http.call_count,96)
        self.assertEqual(wrapper.cache_inventory(self.target,self.calibration,('route','test')),reused)
        self.assertEqual(self.progress.read_bytes(),progress)
        self.assertFalse((self.target/'artifacts/calibration_cost_forecast.json').exists())
        self.assertTrue(self.history.read_bytes().startswith(old))
        new = wrapper.rows(self.history)[-2:]
        self.assertEqual([r['status'] for r in new],['started','complete'])
        self.assertEqual(new[0]['admission_sha256'],wrapper.sha(self.admission))
        self.assertEqual(len(new[1]['report_sha256']),9)
        self.assertTrue(new[1]['selection_before_test'])
        metrics = wrapper.read(self.target/'artifacts/results/expanded_calibration/metrics.json')
        self.assertEqual(len(metrics),864); self.assertTrue(all(r['n']==2 for r in metrics))
        entries=wrapper.read(self.target/'configs/calibration_selection.json')
        self.assertEqual(sum(e['status']=='frozen' for e in entries),432)
        self.assertEqual(sum(e['status']=='unavailable' for e in entries),288)
        marker=wrapper.read(self.target/'artifacts/expanded_calibration_complete.json')
        self.assertEqual(marker['selection_sha256'],protocol.digest(entries))
        budget = wrapper.read(self.root/'artifacts/budget.json')
        self.assertAlmostEqual(budget['spent'],100.096);self.assertAlmostEqual(budget['reserved'],3)
        with self.assertRaises(ValueError): self.execute()

    def test_real_budget_stop_no_report_or_complete_and_all_model_ends(self):
        p=self.root/'artifacts/budget.json';b=wrapper.read(p);b['spent']=196.99;p.write_text(json.dumps(b));before=p.read_bytes()
        with patch.object(wrapper,'verify_ready',return_value=self.checked), patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            with self.assertRaisesRegex(RuntimeError,'drained'):self.execute()
        self.assertEqual([r['status'] for r in wrapper.rows(self.history)[-2:]],['started','failed'])
        self.assertFalse((self.target/'artifacts/results/expanded_calibration').exists())
        self.assertFalse((self.target/'configs/calibration_selection.json').exists())
        self.assertEqual(p.read_bytes(),before)
        models=wrapper.rows(self.target/wrapper.ATTEMPT/'inference/models.jsonl')
        self.assertEqual(sum(r['event']=='model_end' for r in models),6)

    def test_ready_actual_code_with_synthetic_scheduler_and_cost(self):
        result=self.ready()
        self.assertEqual(result['dependencies'],self.spec['reviewed_dependencies'])
        self.assertEqual(len(result['audited_cells']),48)

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
        self.spec['cache_sha256_or_absent']=wrapper.cache_inventory(self.target,self.calibration)
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
        scope=wrapper.cache_inventory(self.root,self.calibration,('calibration_extra',))
        source={p:h for p,h in expected.items() if p in scope}
        with patch.object(wrapper,'cache_inventory',return_value=source):
            good=wrapper.planning(self.root,{'spent':100,'reserved':3},self.calibration)
            self.assertFalse(good['future_cache_discount'])
            with self.assertRaisesRegex(ValueError,'headroom'):wrapper.planning(self.root,{'spent':196.5,'reserved':3},self.calibration)
            source=copy.deepcopy(source);source[next(iter(source))]='drift'
        with patch.object(wrapper,'cache_inventory',return_value=source):
            with self.assertRaisesRegex(ValueError,'source journal'):wrapper.planning(self.root,{'spent':100,'reserved':3},self.calibration)

    def test_screen_terminal_report_and_raw_gates(self):
        original=self.screen_accounting
        for value in (original.replace('45.batch|COMPLETED','45.batch|RUNNING'),
                      original.replace('45.batch|COMPLETED|0:0','45.batch|FAILED|1:0'),
                      original.split('45.extern')[0]):
            self.screen_accounting=value
            with self.assertRaises(ValueError): self.ready()
        self.screen_accounting=original
        self.spec['screen_admission_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'Screen report/admission'): self.ready()

    def test_raw_test_tampering_blocks_completion_after_selection(self):
        p=self.target/'runs/quality/jev/B0/abt-buy/test.jsonl'
        records=wrapper.rows(p); records[0]['study_run_id']=3
        p.write_text(''.join(json.dumps(r)+'\n' for r in records))
        with patch.object(wrapper,'verify_ready',return_value=self.checked), patch.object(wrapper,'verify',return_value=self.replica), \
                patch('urllib.request.urlopen',side_effect=self.http):
            with self.assertRaises(ValueError): self.execute()
        self.assertTrue((self.target/'configs/calibration_selection.json').exists())
        self.assertFalse((self.target/'artifacts/expanded_calibration_complete.json').exists())
        self.assertEqual(wrapper.rows(self.history)[-1]['status'],'failed')


if __name__ == '__main__':unittest.main()
