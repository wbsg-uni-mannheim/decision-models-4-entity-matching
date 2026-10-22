import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import protocol
import run
import resume_variants_drained as recovery
from operational_journal import inspect_operations
from drained_inference import build
from variant_remaining_cost_plan import project


class Response(io.BytesIO):
    status = 200


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for p in ('artifacts/operations', 'configs', 'data/abt-buy'):
            (self.root/p).mkdir(parents=True)
        self.case = {'benchmark': 'abt-buy', 'pair_id': 'a#b', 'label': 1,
                     'records': {'left': {'name': 'x'}, 'right': {'name': 'x'}}}
        self.case['input_hash'] = protocol.digest(self.case['records'])
        (self.root/'data/abt-buy/test.json').write_text(json.dumps([self.case]))
        (self.root/'artifacts/budget.json').write_text(json.dumps({'spent': 0., 'reserved': 0., 'attempts': 0, 'unknown_cost_attempts': 0}))
        self.response = {'model': 'typesafe/jev-1.13', 'provider': 'typesafe',
                         'answers': {'match': {'type': 'noul', 'noul': .8}}, 'usage': {'cost': .01}}
        self.inference = types.SimpleNamespace(**vars(run))
        self.inference.PROJECT = self.root
        self.inference.key = lambda: 'fake-secret'
        self.inference.account = types.FunctionType(run.account.__code__, {**run.account.__globals__, 'PROJECT': self.root},
                                                   argdefs=run.account.__defaults__)
        self.protocol = types.SimpleNamespace(**vars(protocol))
        self.protocol.MODELS = {'jev': protocol.MODELS['jev']}
        self.variants = types.SimpleNamespace(VARIANTS={'abt-buy': 'abt-buy'})
        sock = patch('socket.socket', side_effect=AssertionError('Network forbidden'))
        sock.start()
        self.addCleanup(sock.stop)
        self.spec = {'admitted': True, 'run_id': 1, 'stage': 'variants', 'predecessor': '365279',
                     'slurm_job_id': '42', 'budget_cap_usd': 100, 'max_invocations': 1,
                     'historical_unknown_outcomes_disclosed': True, 'unobserved_worker_causes': ['kev', 'luna', 'pplx'],
                     'source_sha256': {p: 'synthetic' for p in recovery.REQUIRED}}

    def test_admission_identity_cap_unknown_disclosure_and_budget(self):
        recovery.validate(self.spec, '42', {'spent': 96, 'reserved': 3})
        for field, bad in [('admitted', False), ('slurm_job_id', None), ('max_invocations', 2),
                           ('budget_cap_usd', 200), ('historical_unknown_outcomes_disclosed', False),
                           ('unobserved_worker_causes', []), ('source_sha256', {})]:
            changed = copy.deepcopy(self.spec)
            changed[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery.validate(changed, '42', {'spent': 1, 'reserved': 3})
        for budget in ({'spent': 97, 'reserved': 3}, {'spent': float('nan'), 'reserved': 0}):
            with self.assertRaises(ValueError):
                recovery.validate(self.spec, '42', budget)

    def test_real_cell_cache_reuse_single_model_invocation_and_journal(self):
        journal = self.root/'artifacts/operations/models.jsonl'
        with patch('urllib.request.urlopen', side_effect=lambda *a, **k: Response(json.dumps(self.response).encode())) as http:
            outcomes, operations, adapter = recovery.run_once(self.root, self.variants, self.protocol, self.inference, journal)
            self.assertEqual(outcomes, [{'model': 'jev', 'status': 'complete'}])
            self.assertEqual(operations[0]['additional_paid_attempts_counted'], 0)
            self.assertFalse(adapter['admission_granted'])
            with self.assertRaises(FileExistsError):
                recovery.run_once(self.root, self.variants, self.protocol, self.inference, journal)
        self.assertEqual(http.call_count, 1)
        cache = self.root/'runs/quality/jev/B0/abt-buy/test.jsonl'
        before = cache.read_bytes()
        # A separate synthetic cache-only invocation exercises reuse. Production
        # execute refuses a second invocation via its fixed exclusive journal.
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            recovery.run_once(self.root, self.variants, self.protocol, self.inference,
                              self.root/'artifacts/operations/cache-only-test.jsonl')
        self.assertEqual(cache.read_bytes(), before)
        self.assertEqual(json.loads(cache.read_text())['study_run_id'], 1)

    def test_real_budget_rejection_has_no_paid_failure_and_is_audited(self):
        path = self.root/'artifacts/budget.json'
        obj = json.loads(path.read_text()); obj.update(spent=97, reserved=3)
        path.write_text(json.dumps(obj))
        before = path.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            outcomes, operations, _ = recovery.run_once(self.root, self.variants, self.protocol, self.inference,
                                                        self.root/'artifacts/operations/models.jsonl')
        self.assertEqual(outcomes[0]['status'], 'failed')
        self.assertEqual(operations[0]['budget_rejections'], 1)
        self.assertEqual(operations[0]['unknown_outcome_events'], 0)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.root/'runs/quality/jev/B0/abt-buy/test.failures.jsonl').exists())

    def test_operational_join_preserves_paid_attempts_once_and_rejects_tampering(self):
        request, _, _ = build(self.inference, 1)
        def account(action, *args):
            raise RuntimeError('Project spending ceiling reached')
        # Obtain an authentic metadata record via the real adapter, no HTTP.
        self.inference.account = account
        request, _, _ = build(self.inference, 1)
        row = request(self.case, 'jev', 'B0', 'fake-secret')
        row['operational_stop']['previous_attempts_preserved'] = 1
        row['attempts'] = [{'cost': None, 'status': 503, 'error': 'HTTP 503', 'seconds': 1, 'response': {}}]
        op = {k: v for k, v in row.items() if k != 'attempts'}
        path = self.root/'operations.jsonl'
        path.write_text(json.dumps(op)+'\n')
        def inspect(failures):
            return inspect_operations(path, [self.case], failures, protocol, 'jev', 'B0', 1)
        result = inspect([row])
        self.assertEqual(result['previous_attempts_in_failure_journal'], 1)
        self.assertEqual(result['additional_paid_attempts_counted'], 0)
        for failures in ([], [row, row]):
            with self.assertRaises(ValueError):
                inspect(failures)
        for field, value in [('request_hash', 'wrong'), ('study_run_id', 3), ('attempts', [])]:
            changed = {**op, field: value}
            path.write_text(json.dumps(changed)+'\n')
            with self.assertRaises(ValueError):
                inspect([row])
        path.write_text('')
        with self.assertRaises(ValueError):
            inspect([row])

    def test_real_scheduler_gates_via_mocked_terminal_policy_and_join(self):
        for rel in recovery.REQUIRED:
            path = self.root/rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture')
            self.spec['source_sha256'][rel] = recovery.sha(path)
        diagnosis = self.root/'artifacts/operations/diagnosis.json'
        diagnosis.write_text(json.dumps({'job': '365279', 'status': 'terminal_scoped_audit_no_recovery', 'files_sha256': {}, 'totals': {'missing': 1}}))
        self.spec.update(diagnosis=str(diagnosis.relative_to(self.root)), diagnosis_sha256=recovery.sha(diagnosis),
                         parallel_policy='configs/policy.json', parallel_policy_sha256='policy')
        cost = self.root/'cost.json'
        cost.write_text(json.dumps({'diagnosis_sha256': recovery.sha(diagnosis), 'missing_pairs': 1,
                                   'known_proxy_usd': .01, 'historical_unknown_scenario_usd': .05}))
        self.spec.update(cost_plan='cost.json', cost_plan_sha256=recovery.sha(cost))
        (self.root/'configs/expanded_execution.json').write_text(json.dumps({'e6_variants_job': 42, 'statistics_full_job': 99}))
        accounting = '365279|FAILED|1:0\n365279.batch|FAILED|1:0\n365279.extern|COMPLETED|0:0\n'
        policy = {'policy_sha256': 'policy', 'reviewed_jobs': [{'id': '42', 'self': True}, {'id': '51', 'scope': 'schema'}]}
        def command(args, **kwargs):
            if args[0] == 'sacct': return accounting
            if args[1] == 'show': return 'JobId=99 Dependency=afterok:42(unfulfilled):51(unfulfilled)'
            return 'python src/expanded_schema.py run'
        with patch.object(recovery, 'review', return_value=policy), patch.object(recovery.subprocess, 'check_output', side_effect=command):
            recovery.verify_ready(self.root, self.spec, '42', ['jev'], ['abt-buy'])
            budget = self.root/'artifacts/budget.json'
            before_budget = budget.read_text()
            budget.write_text(json.dumps({'spent': 99.94, 'reserved': .005}))
            with self.assertRaisesRegex(ValueError, 'planning scenario'):
                recovery.verify_ready(self.root, self.spec, '42', ['jev'], ['abt-buy'])
            budget.write_text(before_budget)
            accounting = accounting.replace('COMPLETED', 'RUNNING')
            with self.assertRaisesRegex(ValueError, 'active'):
                recovery.verify_ready(self.root, self.spec, '42', ['jev'], ['abt-buy'])
            accounting = accounting.replace('RUNNING', 'COMPLETED')
            with patch.object(recovery.subprocess, 'check_output', side_effect=[accounting, 'Dependency=afterok:365279']):
                with self.assertRaisesRegex(ValueError, 'dependency'):
                    recovery.verify_ready(self.root, self.spec, '42', ['jev'], ['abt-buy'])
            with patch.object(recovery.subprocess, 'check_output', side_effect=[accounting, 'Dependency=afterok:42', 'python src/expanded_variants.py run']):
                with self.assertRaisesRegex(ValueError, 'overlapping'):
                    recovery.verify_ready(self.root, self.spec, '42', ['jev'], ['abt-buy'])
            (self.root/'src/run.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Source drift'):
                recovery.verify_ready(self.root, self.spec, '42', ['jev'], ['abt-buy'])

    def test_family_cost_proxy_uses_missing_count_and_separate_unknown_scenario(self):
        cell = {'model': 'jev', 'benchmark': 'wdc-050un', 'missing': 25, 'responses': 75,
                'expected': 100, 'missing_pair_ids_sha256': 'frozen'}
        source = {'model': 'jev', 'benchmark': 'wdc', 'namespace': 'quality', 'variant': 'B0',
                  'split': 'test', 'responses': 50, 'known_usd': 2., 'unknown_attempts': 4}
        result = project([cell], [source])
        self.assertEqual(result[0]['known_proxy_usd'], 1.)
        self.assertAlmostEqual(result[0]['historical_unknown_scenario_usd'], .1)
        self.assertEqual(project([{**cell, 'missing': 0, 'responses': 100}], [source]), [])
        for cells in ([cell, cell], [{**cell, 'missing': -1}], [{**cell, 'benchmark': 'unknown'}]):
            with self.assertRaises(ValueError):
                project(cells, [source])


if __name__ == '__main__':
    unittest.main()
