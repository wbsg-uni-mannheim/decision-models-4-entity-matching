import copy
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import resume_fresh_core_drained as launcher


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.target = self.root/'replications/run02'
        self.target.mkdir(parents=True)
        for p in ('artifacts', 'runs', 'src'):
            (self.target/p).mkdir()
        (self.root/'artifacts/operations').mkdir(parents=True)
        self.rows = [dict(run_id=2, stage=stage, status=status, slurm_job_id=job)
                     for stage, status, job in [('smoke','started','1'), ('smoke','complete','1'),
                                                ('core','started','2'), ('core','failed','2')]]
        self.journal = self.target/'artifacts/stage_execution.jsonl'
        self.journal.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        self.progress = {'smoke': {'status': 'complete'}, 'AP1_baseline': {'status': 'needs_attention'}}
        self.save(self.target/'artifacts/progress.json', self.progress)
        self.save(self.root/'artifacts/budget.json', {'spent': 1, 'reserved': 3})
        self.replica = {'runner_sha256': 'runner', 'replica': str(self.target)}
        sources = {}
        for rel in launcher.REQUIRED:
            p = self.root/rel
            p.write_text('fixture')
            sources[rel] = launcher.sha(p)
        diagnosis = {'run_id': 2, 'predecessor': '2', 'status': 'terminal_partial_baseline_audited_not_recovered',
                     'stage_journal_sha256': launcher.sha(self.journal), 'progress': self.progress,
                     'response_files_sha256': {}}
        self.save(self.root/'diagnosis.json', diagnosis)
        self.save(self.root/'cost_sources.json', {'verified': True})
        self.cost = {'diagnosis_sha256': launcher.sha(self.root/'diagnosis.json'),
                     'source_verification_sha256': launcher.sha(self.root/'cost_sources.json'),
                     'quote_or_hard_bound': False,
                     'phases': {p: {'known_proxy_usd': 1, 'historical_unknown_scenario_usd': .1} for p in launcher.PHASES}}
        self.save(self.root/'cost.json', self.cost)
        self.spec = {'admitted': True, 'run_id': 2, 'stage': 'core', 'slurm_job_id': '3',
                     'budget_cap_usd': 100, 'max_invocations': 1, 'replica': str(self.target),
                     'runner_sha256': 'runner', 'historical_unknown_outcomes_disclosed': True,
                     'source_sha256': sources, 'diagnosis': 'diagnosis.json',
                     'diagnosis_sha256': launcher.sha(self.root/'diagnosis.json'), 'predecessor': '2',
                     'cost_plan': 'cost.json', 'cost_plan_sha256': launcher.sha(self.root/'cost.json'),
                     'cost_source_verification': 'cost_sources.json', 'parallel_policy': 'policy.json',
                     'parallel_policy_sha256': 'policy', 'reviewed_dependencies': {'3': '(null)', '4': 'afterok:3(unfulfilled)'}}
        self.policy = {'policy_sha256': 'policy', 'reviewed_jobs': [
            {'id': '3', 'self': True}, {'id': '4', 'self': False, 'scope': 'independent'}]}
        (self.root/launcher.APPROVAL).write_bytes((Path(__file__).resolve().parents[2]/launcher.APPROVAL).read_bytes())
        self.commands = []
        patcher = patch('socket.socket', side_effect=AssertionError('Network forbidden'))
        patcher.start(); self.addCleanup(patcher.stop)

    def save(self, path, data):
        path.write_text(json.dumps(data))

    def command(self, args, **kwargs):
        self.commands.append(args)
        if args[0] == 'sacct':
            return '2|FAILED|1:0|end\n2.batch|FAILED|1:0|end\n2.extern|COMPLETED|0:0|end\n'
        if args[1:3] == ['show', 'job']:
            return 'Dependency='+self.spec['reviewed_dependencies'][args[3]]
        return 'independent writer --run-id 3'

    def ready(self):
        with patch.object(launcher, 'review', return_value=self.policy), \
                patch.object(launcher.subprocess, 'check_output', side_effect=self.command):
            return launcher.verify_ready(self.root, self.target, 2, self.spec, '3', self.replica)

    def test_full_gate_and_foreign_hold_preserved(self):
        before = (self.root/'artifacts/budget.json').read_bytes()
        checked = self.ready()
        self.assertEqual(checked['budget']['reserved'], 3)
        self.assertEqual((self.root/'artifacts/budget.json').read_bytes(), before)
        self.assertTrue(any(c[0] == 'sacct' for c in self.commands))

    def test_incomplete_cost_scope_and_budget_rejected(self):
        bad = copy.deepcopy(self.cost)
        del bad['phases']['AP1_timing']
        with self.assertRaisesRegex(ValueError, 'phase cost plan'):
            launcher.validate_cost(bad, self.spec['diagnosis_sha256'], {'spent': 1, 'reserved': 3})
        for budget in ({'spent': 97, 'reserved': 3}, {'spent': float('nan'), 'reserved': 0}):
            with self.assertRaises(ValueError):
                launcher.validate_cost(self.cost, self.spec['diagnosis_sha256'], budget)

    def test_unadmitted_and_source_cache_history_drift_rejected(self):
        self.spec['admitted'] = False
        with self.assertRaisesRegex(ValueError, 'admission'):
            self.ready()
        self.spec['admitted'] = True
        (self.target/'runs/extra.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.ready()
        (self.target/'runs/extra.jsonl').unlink()
        self.journal.write_text(self.journal.read_text()+json.dumps(self.rows[-1])+'\n')
        with self.assertRaisesRegex(ValueError, 'initial core'):
            self.ready()

    def test_scheduler_terminal_policy_overlap_and_join_gates(self):
        with patch.object(launcher.subprocess, 'check_output', return_value='2|RUNNING|0:0|Unknown'):
            with self.assertRaises(ValueError):
                launcher.verify_ready(self.root, self.target, 2, self.spec, '3', self.replica)
        self.spec['reviewed_dependencies']['4'] = 'afterok:2(failed)'
        with self.assertRaisesRegex(ValueError, 'failed predecessor'):
            self.ready()
        self.spec['reviewed_dependencies']['4'] = 'afterok:3(unfulfilled)'
        original = self.command
        def overlap(args, **kwargs):
            return '--run-id=2' if args[1:3] == ['write', 'batch_script'] else original(args, **kwargs)
        with patch.object(launcher, 'review', return_value=self.policy), \
                patch.object(launcher.subprocess, 'check_output', side_effect=overlap):
            with self.assertRaisesRegex(ValueError, 'Another queued'):
                launcher.verify_ready(self.root, self.target, 2, self.spec, '3', self.replica)

    def execute_mock(self, fail=False):
        admission = self.root/'admission.json'; self.save(admission, self.spec)
        protocol = types.ModuleType('protocol'); protocol.canonical = json.dumps
        run = types.ModuleType('run'); run.PROJECT = self.target
        orch = types.ModuleType('orchestrate'); orch.PROJECT = self.target; orch.verify = lambda: None
        def gate(*args):
            sys.modules.update(run=run, protocol=protocol, orchestrate=orch)
            return {'synthetic_scheduler': True}
        def phases(*args):
            # Both locks must still be held throughout execution.
            for path, flag in [(self.target/'artifacts/parallel_writer.lock', fcntl.LOCK_EX),
                               (self.root/'artifacts/operations/repetition_writer.lock', fcntl.LOCK_EX)]:
                with path.open('a') as lock:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(lock, flag | fcntl.LOCK_NB)
            if fail:
                raise RuntimeError('synthetic failed phase')
            return {'synthetic_component': True}
        with patch.dict(sys.modules, {}), patch.dict('os.environ', {'SLURM_JOB_ID': '3'}), \
                patch.object(launcher, 'verify', return_value=self.replica), \
                patch.object(launcher, 'verify_ready', side_effect=gate), \
                patch.object(launcher, 'run_phases', side_effect=phases):
            return launcher.execute(self.root, 2, admission)

    def test_append_only_success_and_exclusive_recovery_attempt(self):
        before = self.journal.read_bytes()
        self.assertEqual(self.execute_mock(), 0)
        self.assertTrue(self.journal.read_bytes().startswith(before))
        rows = launcher.history_rows(self.journal)
        self.assertEqual([r['status'] for r in rows[-2:]], ['started', 'complete'])
        self.assertEqual(rows[-1]['predecessor'], '2')
        self.assertFalse(rows[-1]['historical_unjournaled_outcomes_resolved'])
        with self.assertRaisesRegex(ValueError, 'Prior recovery'):
            self.execute_mock()

    def test_real_component_budget_stop_through_wrapper(self):
        from test_fresh_core_drained import CoreTests
        fixture = CoreTests()
        with patch.object(tempfile, 'tempdir', str(Path(tempfile.gettempdir()).resolve())):
            fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.save(fixture.root/'artifacts/budget.json',
                  {'spent': 97., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0})
        (fixture.root/'artifacts/operations').mkdir()
        stage = fixture.target/'artifacts/stage_execution.jsonl'
        stage.write_bytes(self.journal.read_bytes())
        admission = fixture.root/'admission.json'
        self.save(admission, self.spec)
        import protocol
        def gate(*args):
            sys.modules.update(run=fixture.inference, protocol=protocol, orchestrate=fixture.orchestration)
            return {'synthetic_scheduler': True}
        original = stage.read_bytes()
        budget = (fixture.root/'artifacts/budget.json').read_bytes()
        with patch.dict(sys.modules), patch.dict('os.environ', {'SLURM_JOB_ID': '3'}), \
                patch.object(launcher, 'verify', return_value={'replica': str(fixture.target)}), \
                patch.object(launcher, 'verify_ready', side_effect=gate):
            for name in ('run', 'protocol', 'orchestrate', 'analyze'):
                sys.modules.pop(name, None)
            with self.assertRaisesRegex(RuntimeError, 'draining every model'):
                launcher.execute(fixture.root, 2, admission)
        self.assertEqual((fixture.root/'artifacts/budget.json').read_bytes(), budget)
        self.assertTrue(stage.read_bytes().startswith(original))
        self.assertEqual(launcher.history_rows(stage)[-1]['status'], 'failed')
        events = launcher.history_rows(fixture.target/launcher.ATTEMPT/'phases/AP1_baseline.jsonl')
        self.assertEqual(sum(r['event'] == 'model_end' for r in events), 7)
        self.assertEqual(len(list((fixture.target/'runs').rglob('*.operations.jsonl'))), 7)
        self.assertFalse(list((fixture.target/'runs').rglob('*.failures.jsonl')))

    def test_failure_preserves_old_history_and_never_marks_complete(self):
        before = self.journal.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'synthetic failed'):
            self.execute_mock(fail=True)
        self.assertTrue(self.journal.read_bytes().startswith(before))
        self.assertEqual(launcher.history_rows(self.journal)[-1]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
