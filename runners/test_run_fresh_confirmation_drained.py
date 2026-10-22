import copy
import fcntl
import json
from pathlib import Path
import shutil
import types
import unittest
from unittest.mock import patch

import expanded_protocol as ep
import expanded_screen
import protocol
import run_fresh_confirmation_drained as wrapper
from test_fresh_confirmation_development_drained import DevelopmentTests

ROOT = Path(__file__).resolve().parents[2]


class WrapperTests(unittest.TestCase):
    http = DevelopmentTests.http

    def setUp(self):
        DevelopmentTests.setUp(self)
        self.guard = patch.object(wrapper.freeze, 'verify'); self.guard.start(); self.addCleanup(self.guard.stop)
        for rel in wrapper.REQUIRED:
            p = self.root/rel; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/rel, p)
        with patch('urllib.request.urlopen', side_effect=self.http):
            for m in wrapper.CORE_S1:
                for b in protocol.BENCHMARKS: self.cell(m, 'S1', b, 'prompt', workers=8)
            for b in [*protocol.BENCHMARKS, 'billiger-de-50un']:
                cases = copy.deepcopy(self.cases.get(b, self.cases['billiger-de']))
                for c in cases:
                    c['benchmark'] = b
                    c['entity_ids'] = {'left': 'l'+c['pair_id'], 'right': 'r'+c['pair_id']}
                folder = self.target/'data'/b; folder.mkdir(exist_ok=True)
                (folder/'test.json').write_text(json.dumps(cases))
                for m in protocol.MODELS: self.cell(m, 'B0', b, 'test', workers=8)
        self.progress.write_text(json.dumps({k: {'status': v} for k, v in {
            'smoke': 'complete', 'AP1_baseline': 'complete', 'AP2_development': 'complete',
            'AP2_confirmation': 'complete', 'AP1_timing': 'complete', 'AP3_AP4_analysis': 'complete_automated',
            'supplement_billiger50': 'complete_automated', 'expanded_E2_E3_screen': 'complete'}.items()}))
        core_admission = self.root/'core_admission.json'
        core_admission.write_text(json.dumps({'admitted': True, 'stage': 'core', 'run_id': 2, 'slurm_job_id': '42'}))
        for rel in wrapper.CORE_OUTPUTS:
            p = self.target/rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('fixture core report')
        history = [{'stage': stage, 'status': status, 'run_id': 2, 'slurm_job_id': job}
                   for stage, job in [('smoke','40'),('core','42'),('supplement','44'),('screen','45')]
                   for status in ('started','complete')]
        history[3]['admission_sha256'] = wrapper.sha(core_admission)
        reports = {}
        for stage, folder, index in [('supplement','billiger50',5),('screen','expanded_screen',7)]:
            p = self.target/'artifacts/results'/folder/'metrics.json'; p.parent.mkdir(parents=True); p.write_text('fixture prerequisite report')
            reports[stage] = {str(p.relative_to(self.target)): wrapper.sha(p)}
            history[index].update(report_sha256=reports[stage], admission_sha256=stage+'-admission')
        self.history = self.target/'artifacts/stage_execution.jsonl'
        self.history.write_text(''.join(json.dumps(r)+'\n' for r in history))
        self.spec = {'admitted': True, 'stage': 'confirmation', 'run_id': 2, 'slurm_job_id': '43',
            'replica': str(self.target), 'budget_cap_usd': 200, 'budget_authority_sha256': wrapper.APPROVAL_SHA,
            'max_invocations_per_model_per_phase': 1, 'historical_unknown_outcomes_disclosed': True,
            'two_phase_budget_gate': True, 'source_sha256': {r: wrapper.sha(self.root/r) for r in wrapper.REQUIRED},
            'runner_sha256': 'fixture', 'core_job_id': '42', 'supplement_job_id': '44', 'screen_job_id': '45',
            'core_admission': 'core_admission.json', 'core_admission_sha256': wrapper.sha(core_admission),
            'core_stage_end_sha256': protocol.digest(history[3]),
            'core_report_sha256': {p: wrapper.sha(self.target/p) for p in wrapper.CORE_OUTPUTS},
            'supplement_admission_sha256': 'supplement-admission', 'screen_admission_sha256': 'screen-admission',
            'supplement_report_sha256': reports['supplement'], 'screen_report_sha256': reports['screen'],
            'prerequisite_stage_journal_sha256': wrapper.sha(self.history), 'progress_sha256': wrapper.sha(self.progress),
            'cache_sha256_or_absent': wrapper.cache_inventory(self.target, self.confirmation, protocol),
            'parallel_policy': 'fixture_policy.json', 'parallel_policy_sha256': 'policy',
            'reviewed_dependencies': {'43': '(null)', '99': 'afterok:366115(unfulfilled)'}}
        self.screen = types.SimpleNamespace(**{**vars(expanded_screen), 'PROJECT': self.target, 'verify': lambda: {}})
        self.admission = self.root/'admission.json'; self.admission.write_text(json.dumps(self.spec))
        self.replica = {'runner_sha256': 'fixture'}
        self.checked = {'policy': {'policy_sha256': 'policy'}}
        self.policy = {'policy_sha256': 'policy', 'reviewed_jobs': [{'id': '43', 'self': True}, {'id': '99', 'self': False}]}
        self.accounts = {j: ''.join(f'{j}{suffix}|COMPLETED|0:0|now\n' for suffix in ('','.batch','.extern'))
                         for j in ('42','44','45','46','47')}
        self.dependency = self.spec['reviewed_dependencies']['99']; self.other_script = 'independent Run1 writer'
        self.bodies = []

    def execute(self):
        return wrapper.execute_bound(self.root, self.target, 2, self.admission, self.spec, '43', self.replica,
            self.inference, self.confirmation, ep, self.screen, protocol, self.analysis)

    def scheduler(self, args, **kwargs):
        if args[0] == 'sacct': return self.accounts[args[3]]
        if args[:3] == ['scontrol','show','job']:
            return 'Dependency='+('(null)' if args[3] == '43' else self.dependency)
        if args[:3] == ['scontrol','write','batch_script']: return self.other_script
        raise AssertionError(args)

    def ready(self, started_sha=None):
        with patch.object(wrapper, 'planning', return_value={'fixture': True}), \
                patch.object(wrapper.supplement_component, 'inventory', return_value=[{'complete': True}]), \
                patch.object(wrapper.screen_component, 'inventory', return_value=[{'complete': True}]), \
                patch.object(wrapper, 'make_wrapper') as factory, \
                patch.object(wrapper.subprocess, 'check_output', side_effect=self.scheduler):
            factory.return_value.review.return_value = self.policy
            return wrapper.verify_ready(self.root, self.target, 2, self.spec, '43', self.replica,
                protocol, self.confirmation, self.screen, started_sha)

    def held(self):
        for path in (self.root/'artifacts/operations/repetition_writer.lock', self.target/'artifacts/parallel_writer.lock'):
            with path.open('a') as f:
                with self.assertRaises(BlockingIOError): fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_full_actual_components_two_phase_receipt_reports_and_locks(self):
        old = self.history.read_bytes(); progress = self.progress.read_bytes()
        b0 = {p: p.read_bytes() for p in (self.target/'runs/quality').glob('*/B0/*/*.jsonl') if 'executions' not in p.name}
        original_report = wrapper.tests.finish_report
        def report(*args):
            self.held()
            receipt = wrapper.read(self.target/wrapper.ATTEMPT/'selected_test_gate.json')
            self.assertEqual(receipt['selected_test_costs']['development_cells_raw_verified'],1176)
            self.assertEqual(receipt['selected_test_costs']['missing_test_responses'],126)
            return original_report(*args)
        with patch.object(wrapper, 'verify_ready', wraps=wrapper.verify_ready) as ready, \
                patch.object(wrapper, 'planning', return_value={'fixture': True}), \
                patch.object(wrapper.supplement_component, 'inventory', return_value=[{'complete': True}]), \
                patch.object(wrapper.screen_component, 'inventory', return_value=[{'complete': True}]), \
                patch.object(wrapper, 'make_wrapper') as factory, \
                patch.object(wrapper.subprocess, 'check_output', side_effect=self.scheduler), \
                patch.object(wrapper, 'verify', return_value=self.replica), \
                patch.object(wrapper.tests, 'finish_report', side_effect=report), \
                patch('urllib.request.urlopen', side_effect=self.http) as http:
            factory.return_value.review.return_value = self.policy
            self.assertEqual(self.execute(),0)
        self.assertEqual(http.call_count,2302)  # 2176 new dev, 126 selected tests.
        self.assertEqual(ready.call_count,2)
        self.assertIsNotNone(ready.call_args.args[-1])
        self.assertEqual(self.progress.read_bytes(),progress)
        self.assertTrue(self.history.read_bytes().startswith(old))
        self.assertEqual([r['status'] for r in wrapper.rows(self.history)[-2:]],['started','complete'])
        self.assertEqual(len(wrapper.rows(self.history)[-1]['report_sha256']),7)
        self.assertEqual(len(wrapper.read(self.target/wrapper.tests.REPORT_DIR/'metrics.json')),378)
        self.assertEqual(len(wrapper.read(self.target/wrapper.tests.REPORT_DIR/'macro.json')),42)
        self.assertTrue(all(p.read_bytes()==raw for p,raw in b0.items()))
        self.assertAlmostEqual(wrapper.read(self.root/'artifacts/budget.json')['reserved'],3)
        with self.assertRaisesRegex(ValueError,'Prior confirmation'):self.execute()

    def test_actual_post_freeze_budget_stop_preserves_choice_before_any_test(self):
        original = wrapper.freeze.freeze_development
        def freeze(*args):
            result=original(*args); self.held()
            p=self.root/'artifacts/budget.json'; b=wrapper.read(p); b['spent']=196.99; p.write_text(json.dumps(b))
            return result
        with patch.object(wrapper,'verify_ready',return_value=self.checked), \
                patch.object(wrapper,'current_scheduler',return_value=self.checked), \
                patch.object(wrapper,'verify',return_value=self.replica), \
                patch.object(wrapper.freeze,'freeze_development',side_effect=freeze), \
                patch('urllib.request.urlopen',side_effect=self.http) as http:
            with self.assertRaisesRegex(ValueError,'selected-test scenario'):self.execute()
        self.assertEqual(http.call_count,2176)
        receipt=self.target/wrapper.freeze.RECEIPT; before=receipt.read_bytes()
        self.assertFalse((self.target/wrapper.tests.ATTEMPT).exists())
        self.assertFalse((self.target/wrapper.tests.MARKER).exists())
        last=wrapper.rows(self.history)[-1]
        self.assertEqual(last['phase'],'selected_test_gate');self.assertTrue(last['selection_preserved'])
        with self.assertRaisesRegex(ValueError,'Prior confirmation'):self.execute()
        self.assertEqual(receipt.read_bytes(),before)

    def test_concurrent_development_budget_exhaustion_drains_without_freeze(self):
        p=self.root/'artifacts/budget.json'; b=wrapper.read(p);b['spent']=196.99;p.write_text(json.dumps(b));old=p.read_bytes()
        with patch.object(wrapper,'verify_ready',return_value=self.checked), patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            with self.assertRaisesRegex(RuntimeError,'Development failed'):self.execute()
        self.assertEqual(p.read_bytes(),old)
        self.assertFalse((self.target/wrapper.freeze.RECEIPT).exists())
        models=wrapper.rows(self.target/wrapper.ATTEMPT/'development/models.jsonl')
        self.assertEqual(sum(r['event']=='model_end' for r in models),7)
        self.assertEqual(wrapper.rows(self.history)[-1]['phase'],'development')

    def test_initial_actual_gate_with_synthetic_scheduler_prerequisites_and_cost(self):
        self.assertEqual(self.ready()['dependencies'],self.spec['reviewed_dependencies'])

    def test_all_prerequisite_steps_must_be_terminal_successful(self):
        for job in ('42','44','45'):
            original=self.accounts[job]
            for changed in (original.replace('.batch|COMPLETED','.batch|RUNNING'),
                            original.replace('.batch|COMPLETED|0:0','.batch|FAILED|1:0'),
                            original.split(job+'.extern')[0]):
                self.accounts[job]=changed
                with self.assertRaises(ValueError):self.ready()
            self.accounts[job]=original

    def test_core_report_admission_and_full_raw_credit_required(self):
        before=copy.deepcopy(self.spec)
        self.spec['core_stage_end_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'Core completion'):self.ready()
        self.spec=before
        p=self.target/'runs/quality/jev/S1/abt-buy/prompt.jsonl';p.write_text('')
        with self.assertRaisesRegex(ValueError,'core B0 and S1'):self.ready()

    def test_all_joins_scope_and_policy_refreshed(self):
        for value in ('afterany:1','afterok:1?afterok:2','afterok:999(unfulfilled)'):
            self.dependency=value
            with self.assertRaises(ValueError):self.ready()
        self.dependency=self.spec['reviewed_dependencies']['99'];self.other_script='runner --run-id 2'
        with self.assertRaisesRegex(ValueError,'Another writer'):self.ready()

    def test_source_cache_progress_and_unadmitted_entry_are_closed(self):
        self.spec['admitted']=False
        with self.assertRaisesRegex(ValueError,'admission'):self.execute()
        self.assertFalse((self.target/wrapper.ATTEMPT).exists())
        self.spec['admitted']=True
        self.spec['cache_sha256_or_absent']={}
        with self.assertRaisesRegex(ValueError,'cache changed'):self.ready()
        self.spec['cache_sha256_or_absent']=wrapper.cache_inventory(self.target,self.confirmation,protocol)
        (self.root/wrapper.SELF).write_text('drift')
        with self.assertRaisesRegex(ValueError,'source drift'):self.ready()

    def test_initial_budget_uses_full_development_only_and_current_source_hashes(self):
        for rel in (wrapper.COST,wrapper.COST_PROOF):
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
        expected=wrapper.read(self.root/wrapper.COST)['protected_sha256_or_absent']
        real_sha=wrapper.sha
        def hashed(path):
            rel=str(path.relative_to(self.root))
            return expected[rel] if rel.startswith(('runs/','data/')) else real_sha(path)
        with patch.object(wrapper,'sha',side_effect=hashed), patch.object(Path,'exists',return_value=True):
            result=wrapper.planning(self.root,{'spent':100,'reserved':3},self.confirmation,protocol)
            self.assertFalse(result['unknown_choice_envelope_used_as_admission'])
            with self.assertRaisesRegex(ValueError,'development scenario'):
                wrapper.planning(self.root,{'spent':190,'reserved':3},self.confirmation,protocol)
        with self.assertRaisesRegex(ValueError,'source changed'):
            wrapper.planning(self.root,{'spent':100,'reserved':3},self.confirmation,protocol)

    def test_optional_stages_order_provenance_and_all_steps(self):
        prior=[]
        for stage,job,status in [('calibration','46','complete_with_explicit_unavailable_fits'),('stability','47','complete')]:
            marker=self.target/f'artifacts/expanded_{stage}_complete.json';marker.write_text(json.dumps({'status':status}))
            info={'stage':stage,'slurm_job_id':job,'admission_sha256':stage+'-admission',
                  'report_sha256':{str(marker.relative_to(self.target)):wrapper.sha(marker)}}
            prior.append(info)
            with self.history.open('a') as f:
                for s in ('started','complete'):f.write(json.dumps({**info,'run_id':2,'status':s})+'\n')
        self.spec['previous_completed_stages']=prior
        self.spec['prerequisite_stage_journal_sha256']=wrapper.sha(self.history)
        self.assertEqual(len(self.ready()['previous_completed_stages']),2)
        self.accounts['47']=self.accounts['47'].replace('.extern|COMPLETED','.extern|RUNNING')
        with self.assertRaises(ValueError):self.ready()
        self.spec['previous_completed_stages']=list(reversed(prior))
        with self.assertRaisesRegex(ValueError,'calibration must precede'):self.ready()


if __name__=='__main__':unittest.main()
