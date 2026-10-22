"""Synthetic process-level launcher checks; no scheduler or network access."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from repeated_runs import prepare, replica_path, sha
import repeated_runs_parallel as launcher

SOURCE = Path(__file__).resolve().parents[2] / 'src'
OPERATIONS = Path(__file__).resolve().parent
CHILD = r'''
import io, json, os, sys
from pathlib import Path
from unittest.mock import patch
import repeated_runs_parallel as launcher
root, run_id, admission, policy = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
class Response(io.BytesIO):
    status = 200
def response(*args, **kwargs):
    # The real accounting function must reserve in the shared root BEFORE HTTP.
    budget = json.loads((root/'artifacts/budget.json').read_text())
    assert abs(budget['reserved'] - 3.05) < 1e-8
    body = {'model':'typesafe/jev-1.13', 'provider':'typesafe',
            'answers':{'match':{'type':'noul','noul':.8}}}
    if os.environ['TEST_OUTCOME'] == 'success':
        body['usage'] = {'cost':.012}
    return Response(json.dumps(body).encode())
reviewed = {'policy_sha256':launcher.sha(policy), 'reviewed_jobs':[{'self':True}]}
with patch.object(launcher, 'review', return_value=reviewed), \
        patch('urllib.request.urlopen', side_effect=response), \
        patch('socket.socket', side_effect=AssertionError('Network forbidden')):
    launcher.execute(root, run_id, os.environ['TEST_STAGE'], admission, policy)
'''


class LauncherExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        for directory in ['src', 'configs', 'data/abt-buy', 'artifacts/astra_agent',
                          'artifacts/operations', 'runs']:
            (self.root/directory).mkdir(parents=True)
        for name in ['run.py', 'protocol.py']:
            shutil.copy2(SOURCE/name, self.root/'src'/name)
        # Deliberately tiny synthetic pipeline; real request, cache and accounting.
        (self.root/'src/orchestrate.py').write_text('''
import json
from run import PROJECT, run_cell
def verify():
    return {}
def stamp(stage, status, detail=None):
    path = PROJECT/'artifacts/progress.json'
    value = json.loads(path.read_text())
    value[stage] = {'status':status}
    path.write_text(json.dumps(value))
def parallel(models, variants, splits):
    assert variants == ['B0']
    for model in models:
        for split in splits:
            run_cell(model, 'B0', 'abt-buy', split, workers=1)
if __name__ == '__main__':
    run_cell('jev', 'B0', 'abt-buy', 'prompt', workers=1)
    stamp('smoke', 'complete')
''')
        (self.root/'src/billiger50.py').write_text('''
from run import PROJECT
BENCHMARK = 'billiger-de-50un'
def verify():
    assert (PROJECT/'data'/BENCHMARK/'test.json').exists()
''')
        for name in ['manifest.json', 'learned_rules.json']:
            (self.root/'configs'/name).write_text('{}')
        (self.root/'artifacts/astra_agent/acceptance.json').write_text('{}')
        case = {'benchmark':'abt-buy', 'pair_id':'a#b', 'input_hash':'synthetic',
                'records':{'left':{'name':'item'}, 'right':{'name':'item'}}}
        (self.root/'data/abt-buy/prompt.json').write_text(json.dumps([case]))
        self.budget_path = self.root/'artifacts/budget.json'
        self.budget_path.write_text(json.dumps(
            {'spent':70, 'reserved':3, 'attempts':0, 'unknown_cost_attempts':0}))
        self.policy = self.root/'policy.json'
        self.policy.write_text('{"synthetic_scheduler_fixture":true}')
        self.admissions = {}
        for run_id in [2, 3]:
            spec = prepare(self.root, run_id)
            admission = self.root/f'admission{run_id}.json'
            admission.write_text(json.dumps({
                'run_id':run_id, 'stage':'smoke', 'slurm_job_id':'synthetic',
                'budget_cap_usd':100, 'runner_sha256':spec['runner_sha256'],
                'launcher_sha256':sha(Path(launcher.__file__)),
                'parallel_policy_sha256':sha(self.policy), 'replica':spec['replica']}))
            self.admissions[run_id] = admission

    def launch(self, run_id=2, outcome='success', stage='smoke'):
        env = {**os.environ, 'PYTHONPATH':str(OPERATIONS),
               'SLURM_JOB_ID':'synthetic', 'OPENROUTER_API_KEY':'synthetic-no-network',
               'TEST_OUTCOME':outcome, 'TEST_STAGE':stage}
        return subprocess.run([sys.executable, '-c', CHILD, str(self.root), str(run_id),
                               str(self.admissions[run_id]), str(self.policy)],
                              capture_output=True, text=True, env=env, timeout=20)

    def rows(self, run_id):
        path = replica_path(self.root, run_id)/'artifacts/stage_execution.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_real_request_cache_tags_and_shared_ledger_across_replicas(self):
        for run_id in [2, 3]:
            result = self.launch(run_id)
            self.assertEqual(result.returncode, 0, result.stderr)
            replica = replica_path(self.root, run_id)
            rows = [json.loads(line) for line in
                    (replica/'runs/quality/jev/B0/abt-buy/prompt.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['study_run_id'], run_id)
            self.assertEqual(rows[0]['prediction'], 1)
            self.assertTrue((replica/'artifacts/budget.json').is_symlink())
            self.assertEqual([r['status'] for r in self.rows(run_id)], ['started', 'complete'])
        budget = json.loads(self.budget_path.read_text())
        self.assertAlmostEqual(budget['spent'], 70.024)
        self.assertAlmostEqual(budget['reserved'], 3)
        self.assertEqual(budget['attempts'], 2)
        self.assertEqual(list((self.root/'runs').iterdir()), [])

    def test_failed_stage_keeps_error_cost_and_cannot_blindly_repeat(self):
        result = self.launch(outcome='unknown_usage')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([r['status'] for r in self.rows(2)], ['started', 'failed'])
        budget_before = self.budget_path.read_bytes()
        budget = json.loads(budget_before)
        self.assertAlmostEqual(budget['spent'], 70.05)
        self.assertEqual(budget['unknown_cost_attempts'], 1)
        repeat = self.launch()
        self.assertNotEqual(repeat.returncode, 0)
        self.assertIn('Stage already attempted', repeat.stderr)
        self.assertEqual(self.budget_path.read_bytes(), budget_before)

    def test_successful_stage_cannot_repeat(self):
        self.assertEqual(self.launch().returncode, 0)
        before = self.budget_path.read_bytes()
        repeat = self.launch()
        self.assertNotEqual(repeat.returncode, 0)
        self.assertIn('Stage already attempted', repeat.stderr)
        self.assertEqual(self.budget_path.read_bytes(), before)

    def test_budget_and_source_drift_fail_before_requests(self):
        budget = json.loads(self.budget_path.read_text())
        budget['spent'] = 97
        self.budget_path.write_text(json.dumps(budget))
        before = self.budget_path.read_bytes()
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('budget exhausted', result.stderr)
        self.assertEqual(self.budget_path.read_bytes(), before)
        (replica_path(self.root, 3)/'src/run.py').write_text('raise RuntimeError()')
        result = self.launch(3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('src/run.py', result.stderr)
        self.assertEqual(self.budget_path.read_bytes(), before)

    def test_baseline_branch_reuses_only_own_smoke_and_excludes_ap2(self):
        # Test only one model here by freezing a fixture protocol before preparing
        # a NEW independent fixture; never edit a prepared real replica.
        # The exact seven-model matrix is separately covered by test_baseline_only.
        for run_id in [2, 3]:
            # These are disposable synthetic replicas, not study workspaces.
            shutil.rmtree(replica_path(self.root, run_id))
        protocol_path = self.root/'src/protocol.py'
        protocol_path.write_text(protocol_path.read_text()+"\nMODELS = {'jev': MODELS['jev']}\n")
        case = json.loads((self.root/'data/abt-buy/prompt.json').read_text())
        for split in ['route', 'test']:
            (self.root/'data/abt-buy'/f'{split}.json').write_text(json.dumps(case))
        (self.root/'data/billiger-de-50un').mkdir()
        (self.root/'data/billiger-de-50un/test.json').write_text(json.dumps([
            {**case[0], 'benchmark':'billiger-de-50un'}]))
        prepare(self.root, 2)
        self.assertEqual(self.launch().returncode, 0)
        admission = json.loads(self.admissions[2].read_text())
        admission['stage'] = 'baseline'
        self.admissions[2].write_text(json.dumps(admission))
        result = self.launch(stage='baseline')
        self.assertEqual(result.returncode, 0, result.stderr)
        replica = replica_path(self.root, 2)
        paths = [p for p in (replica/'runs/quality').rglob('*.jsonl')
                 if p.name in {'prompt.jsonl', 'route.jsonl', 'test.jsonl'}]
        self.assertEqual(len(paths), 4)
        self.assertTrue(all('/B0/' in str(path) for path in paths))
        budget = json.loads(self.budget_path.read_text())
        self.assertEqual(budget['attempts'], 4)
        self.assertAlmostEqual(budget['spent'], 70.048)
        progress = json.loads((replica/'artifacts/progress.json').read_text())
        self.assertEqual(set(progress), {'smoke', 'baseline_only'})
        self.assertEqual([r['status'] for r in self.rows(2)],
                         ['started', 'complete', 'started', 'complete'])


if __name__ == '__main__':
    unittest.main()
