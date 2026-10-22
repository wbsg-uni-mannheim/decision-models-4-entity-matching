"""Offline tests for the inputs-study runner: mocked HTTP, temporary root, no network."""
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import drained_inference
import inputs_study
import leased_budget
import prompt12
import protocol
import run
import run_inputs_20261019 as ri

REAL = Path(__file__).resolve().parents[2]


class InputsRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = self.root = Path(self.temp.name)
        for rel in ri.SOURCES:
            if rel not in ('configs/manifest.json', inputs_study.SELECTION):
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(REAL/rel, root/rel)
        manifest = {'benchmarks': {}}
        for b in protocol.BENCHMARKS:
            rows = []
            for i in range(3):
                records = {'left': {'title': f'{b} alpha beta gamma {i}', 'brand': 'x'},
                           'right': {'title': f'{b} alpha beta delta {i}', 'brand': 'x'}}
                rows.append({'benchmark': b, 'pair_id': f'{b}#{i}', 'records': records,
                             'input_hash': protocol.digest(records), 'label': i % 2})
            (root/'data'/b).mkdir(parents=True)
            (root/'data'/b/'test.json').write_text(json.dumps(rows))
            manifest['benchmarks'][b] = {'splits': {'test': {'n': 3, 'sha256': protocol.digest(rows)}}}
        (root/'configs/manifest.json').write_text(json.dumps(manifest))
        selection = {'selected': {m: {b: 'B0' for b in protocol.BENCHMARKS} for m in prompt12.RUN_MODELS}}
        selection['selected']['luna'] = {b: 'R_rules' for b in protocol.BENCHMARKS}
        (root/inputs_study.SELECTION).write_text(json.dumps(selection))
        (root/'artifacts/operations').mkdir(parents=True, exist_ok=True)
        (root/'artifacts/budget.json').write_text(json.dumps({
            'spent': 295., 'reserved': 9.4, 'attempts': 0, 'unknown_cost_attempts': 0,
            'project_cap_usd': 355, 'budget_authority_sha256': 'prompt12', 'foreign': {'keep': 1}}))
        self.calls = []
        ri.STOP.clear(); self.addCleanup(ri.STOP.clear)
        prompt12.register_models()
        mods = (run, protocol, inputs_study, drained_inference, leased_budget)
        for p in [patch.object(ri, 'modules', return_value=mods), patch.object(run, 'key', return_value='mock'),
                  patch.object(ri, 'other_project_jobs', return_value=[]), patch('signal.signal'),
                  patch.dict(os.environ, SLURM_JOB_ID='999'),
                  patch('socket.socket', side_effect=AssertionError('Real network forbidden'))]:
            p.start(); self.addCleanup(p.stop)
        self.addCleanup(inputs_study.SELECTED.clear)

    def approve(self, run_cap=5., cap=400.):
        (self.root/ri.APPROVAL).write_text(json.dumps({
            'approved_by': 'Aaron', 'scope': ri.SCOPE, 'project_cap_usd': cap, 'run_cap_usd': run_cap,
            'workers': 4, 'timeout_s': 30, 'quote': 'synthetic test approval'}))

    def http(self, request, timeout=None, **kw):
        body = json.loads(request.data); self.calls.append((body, timeout))
        if 'questions' in body:
            data = {'model': body['model'], 'provider': body['provider']['only'][0].split('/')[0],
                    'answers': {'match': {'type': 'noul', 'noul': .9}}, 'usage': {'cost': .001}}
        else:
            data = {'model': body['model'], 'provider': 'OpenAI', 'usage': {'cost': .001},
                    'choices': [{'finish_reason': 'stop', 'message': {'content': '{"match": true}'}}]}
        response = io.BytesIO(json.dumps(data).encode()); response.status = 200
        return response

    def test_no_run_without_approval(self):
        ri.freeze(self.root)
        with self.assertRaises(FileNotFoundError):
            ri.execute(self.root)
        self.assertFalse((self.root/'runs').exists())

    def test_full_matrix_selected_prompt_timeout_and_budget(self):
        ri.freeze(self.root); self.approve()
        with self.assertRaisesRegex(ValueError, 'authorize-budget'):
            ri.execute(self.root)
        ri.authorize_budget(self.root)
        with patch('urllib.request.urlopen', side_effect=self.http):
            ri.execute(self.root)
        cells = 7*(9*8+4)  # R3 only on the four benchmarks that are not lowercase already
        self.assertEqual(len(ri.matrix(protocol, inputs_study)), cells)
        self.assertFalse((self.root/'runs'/ri.NAMESPACE/'jev/R3_lowercase/abt-buy').exists())
        self.assertEqual(len(self.calls), cells*3)
        self.assertEqual({t for _, t in self.calls}, {30})
        luna = [b for b, _ in self.calls if 'messages' in b]
        self.assertTrue(all('General rules derived' in b['messages'][0]['content'] for b in luna))
        self.assertTrue((self.root/ri.ATTEMPT/'job-999/complete.json').exists())
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 295+cells*3*.001)
        self.assertAlmostEqual(budget['reserved'], 9.4)
        self.assertEqual(budget['foreign'], {'keep': 1})
        spec = json.loads((self.root/ri.MANIFEST).read_text())
        self.assertEqual(spec['selected_prompts']['luna']['wdc'], 'R_rules')

    def test_authorize_refuses_while_job_runs(self):
        ri.freeze(self.root); self.approve()
        with patch.object(ri, 'other_project_jobs', return_value=[{'job': '1'}]):
            with self.assertRaisesRegex(ValueError, 'project job is running'):
                ri.authorize_budget(self.root)

    def test_failed_cell_continues_then_resume(self):
        ri.freeze(self.root); self.approve(); ri.authorize_budget(self.root)
        state = {'n': 0}

        def flaky(request, timeout=None, **kw):
            if json.loads(request.data)['model'] == 'typesafe/jev-1.13' and state['n'] == 0:
                state['n'] += 1
                import urllib.error
                raise urllib.error.HTTPError(request.full_url, 520, 'unknown', {}, io.BytesIO(b'{}'))
            return self.http(request, timeout)
        with patch('urllib.request.urlopen', side_effect=flaky):
            with self.assertRaisesRegex(ValueError, 'model stopped'):
                ri.execute(self.root)
        ns = self.root/'runs'/ri.NAMESPACE
        self.assertTrue((ns/'jev/R6_typos/semi-heter/test.jsonl').exists())
        failed_cell = [p for p in ns.rglob('test.failures.jsonl')][0].parent
        self.assertEqual(len((failed_cell/'test.jsonl').read_text().splitlines()), 2)  # rest of the cell done
        self.assertEqual(len(ri.archive_failures(self.root)), 1)
        with patch.dict(os.environ, SLURM_JOB_ID='1000'), patch('urllib.request.urlopen', side_effect=self.http):
            ri.execute(self.root, resume=True)
        self.assertTrue((self.root/ri.ATTEMPT/'job-1000/complete.json').exists())

    def test_source_drift_blocks(self):
        ri.freeze(self.root)
        with (self.root/'src/inputs_study.py').open('a') as f:
            f.write('\n# drift\n')
        with self.assertRaisesRegex(ValueError, 'Source drift'):
            ri.preflight(self.root, require_approval=False)


if __name__ == '__main__':
    unittest.main()
