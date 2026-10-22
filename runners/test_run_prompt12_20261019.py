"""Offline tests: mocked HTTP only, temporary project root, no real network."""
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import drained_inference
import leased_budget
import prompt12
import protocol
import run
import run_prompt12_20261019 as p12

REAL = Path(__file__).resolve().parents[2]


class Prompt12RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = self.root = Path(self.temp.name)
        for rel in p12.SOURCES:
            if rel != 'configs/manifest.json':
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(REAL/rel, root/rel)
        manifest = {'benchmarks': {}}
        for b in protocol.BENCHMARKS:
            rows = []
            for i in range(3):
                records = {'left': {'title': f'{b} {i}'}, 'right': {'title': f'{b} {i}'}}
                rows.append({'benchmark': b, 'pair_id': f'{b}#{i}', 'records': records,
                             'input_hash': protocol.digest(records), 'label': i % 2})
            (root/'data'/b).mkdir(parents=True)
            (root/'data'/b/'test.json').write_text(json.dumps(rows))
            manifest['benchmarks'][b] = {'splits': {'test': {'n': 3, 'sha256': protocol.digest(rows)}}}
        (root/'configs/manifest.json').write_text(json.dumps(manifest))
        (root/'artifacts/operations').mkdir(parents=True, exist_ok=True)
        (root/'artifacts/budget.json').write_text(json.dumps({
            'spent': 213.5, 'reserved': 9.4, 'attempts': 0, 'unknown_cost_attempts': 0,
            'project_cap_usd': 250, 'budget_authority_sha256': 'old', 'foreign': {'keep': 1}}))
        self.calls = []
        p12.STOP.clear(); self.addCleanup(p12.STOP.clear)
        prompt12.register_models()
        mods = (run, protocol, prompt12, drained_inference, leased_budget)
        for p in [patch.object(p12, 'modules', return_value=mods), patch.object(run, 'key', return_value='mock'),
                  patch.object(p12, 'other_project_jobs', return_value=[]),
                  patch.dict(os.environ, SLURM_JOB_ID='999'),
                  patch('socket.socket', side_effect=AssertionError('Real network forbidden'))]:
            p.start(); self.addCleanup(p.stop)

    def approve(self, run_cap=5., cap=330.):
        (self.root/p12.APPROVAL).write_text(json.dumps({
            'approved_by': 'Aaron', 'scope': p12.SCOPE, 'project_cap_usd': cap, 'run_cap_usd': run_cap,
            'workers': 4, 'quote': 'synthetic test approval'}))
        (self.root/p12.ADDENDUM).write_text(json.dumps({
            'approved_by': 'Aaron', 'scope': p12.SCOPE, 'model_alias': 'pplx_v12',
            'model_id': 'perplexity/pplx-decider-v1.2', 'own_b0': True, 'quote': 'synthetic'}))

    def http(self, request, **kw):
        body = json.loads(request.data); self.calls.append(body)
        if 'questions' in body:
            text = body['questions']['match']['instructions']
            data = {'model': body['model'], 'provider': body['provider']['only'][0].split('/')[0],
                    'answers': {'match': {'type': 'noul', 'noul': .9}}, 'usage': {'cost': .001}}
        else:
            text = body['messages'][0]['content']
            data = {'model': body['model'], 'provider': 'OpenAI', 'usage': {'cost': .001},
                    'choices': [{'finish_reason': 'stop', 'message': {'content': '{"match": true}'}}]}
        if body['model'] != 'perplexity/pplx-decider-v1.2':  # only pplx_v12 runs its own B0
            self.assertNotEqual(text, protocol.BASE)
        response = io.BytesIO(json.dumps(data).encode()); response.status = 200
        return response

    def test_no_run_without_approval(self):
        p12.freeze(self.root)
        with self.assertRaises(FileNotFoundError):
            p12.execute(self.root)
        self.assertFalse((self.root/'runs').exists())

    def test_budget_authority_required_then_full_matrix(self):
        p12.freeze(self.root); self.approve()
        with self.assertRaisesRegex(ValueError, 'authorize-budget'):
            p12.execute(self.root)
        history = p12.authorize_budget(self.root)
        self.assertEqual(history['previous_cap_usd'], 250); self.assertEqual(history['new_cap_usd'], 330.)
        with patch('urllib.request.urlopen', side_effect=self.http):
            p12.execute(self.root)
        cells = (6*11+12)*8
        self.assertEqual(len(p12.matrix(protocol, prompt12)), cells)
        self.assertEqual(len(self.calls), cells*3)
        attempt = self.root/p12.ATTEMPT/'job-999'
        self.assertTrue((attempt/'complete.json').exists())
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 213.5+cells*3*.001)
        self.assertAlmostEqual(budget['reserved'], 9.4)
        self.assertEqual(budget['foreign'], {'keep': 1})
        rows = (self.root/'runs'/p12.NAMESPACE/'jev/R_rules/abt-buy/test.jsonl').read_text().splitlines()
        self.assertEqual(len(rows), 3)
        self.assertEqual(json.loads(rows[0])['study_run_id'], 1)
        with self.assertRaisesRegex(ValueError, 'Namespace exists'):
            p12.execute(self.root)

    def test_unknown_cost_counted_without_flat_charge(self):
        p12.freeze(self.root); self.approve(); p12.authorize_budget(self.root)
        state = {'n': 0}

        def flaky(request, **kw):
            state['n'] += 1
            if state['n'] % 50 == 1:  # 502 without usage -> unknown cost, then frozen retry
                import urllib.error
                raise urllib.error.HTTPError(request.full_url, 502, 'bad gateway', {}, io.BytesIO(b'{}'))
            return self.http(request, **kw)
        with patch('urllib.request.urlopen', side_effect=flaky), patch('time.sleep'):
            p12.execute(self.root)
        attempt = self.root/p12.ATTEMPT/'job-999'
        validation = json.loads((attempt/'validation.json').read_text())
        self.assertGreater(validation['unknown'], 0)
        self.assertEqual(validation['unknown'], validation['unknown_cost_attempts_this_job'])
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 213.5+(6*11+12)*8*3*.001)  # no .05 per unknown
        self.assertAlmostEqual(budget['reserved'], 9.4)

    def test_sigterm_drains_and_resume_completes_subset(self):
        p12.freeze(self.root); self.approve(); p12.authorize_budget(self.root)
        models = ['jev', 'clef']

        def stopping(request, **kw):
            if len(self.calls) == 40:
                p12.STOP.set()
            return self.http(request, **kw)
        with patch('urllib.request.urlopen', side_effect=stopping), patch('signal.signal'):
            with self.assertRaises(ValueError):
                p12.execute(self.root, models=models)
        failed = json.loads((self.root/p12.ATTEMPT/'job-999/failed.json').read_text())
        self.assertTrue(failed['stopped_by_sigterm']); self.assertEqual(failed['credit']['outstanding'], 0)
        done = sum(len(f.read_text().splitlines()) for f in (self.root/'runs'/p12.NAMESPACE).rglob('test.jsonl'))
        self.assertEqual(done, len(self.calls))  # every paid response journaled
        p12.STOP.clear()
        with patch.dict(os.environ, SLURM_JOB_ID='1000'), patch('urllib.request.urlopen', side_effect=self.http), \
                patch('signal.signal'):
            p12.execute(self.root, resume=True, models=models)
        self.assertEqual(len(self.calls), 2*11*8*3)
        self.assertTrue((self.root/p12.ATTEMPT/'job-1000/complete.json').exists())
        self.assertFalse((self.root/'runs'/p12.NAMESPACE/'luna').exists())

    def test_pplx_v12_needs_addendum_and_runs_own_b0(self):
        p12.freeze(self.root); self.approve(); p12.authorize_budget(self.root)
        (self.root/p12.ADDENDUM).unlink()
        with self.assertRaisesRegex(ValueError, 'pplx_v12 approval'):
            p12.execute(self.root)
        self.approve()
        with patch.dict(os.environ, SLURM_JOB_ID='1001'), patch('urllib.request.urlopen', side_effect=self.http_any):
            p12.execute(self.root, models=['pplx_v12'])
        self.assertTrue((self.root/'runs'/p12.NAMESPACE/'pplx_v12/B0/abt-buy/test.jsonl').exists())
        self.assertFalse((self.root/'runs'/p12.NAMESPACE/'pplx').exists())
        self.assertEqual({c['model'] for c in self.calls}, {'perplexity/pplx-decider-v1.2'})

    def http_any(self, request, **kw):
        body = json.loads(request.data); self.calls.append(body)
        data = {'model': body['model'], 'provider': 'Perplexity',
                'answers': {'match': {'type': 'noul', 'noul': .2}}, 'usage': {'cost': .001}}
        response = io.BytesIO(json.dumps(data).encode()); response.status = 200
        return response

    def test_v1_templates_unchanged(self):
        p12.freeze(self.root)
        spec = json.loads((self.root/p12.MANIFEST).read_text())
        real_v1 = REAL/p12.MANIFEST_V1
        if real_v1.exists():
            self.assertEqual(spec['v1_request_templates_sha256'], json.loads(real_v1.read_text())['request_templates_sha256'])

    def test_close_stale_credit_against_journals(self):
        p12.freeze(self.root); self.approve(); p12.authorize_budget(self.root)

        def stopping(request, **kw):
            if len(self.calls) == 100:
                raise KeyboardInterrupt  # stands in for a killed process: no drain, no credit close
            return self.http(request, **kw)
        with patch('urllib.request.urlopen', side_effect=stopping), patch('signal.signal'):
            with self.assertRaises(BaseException):
                p12.execute(self.root, models=['jev'])
        before = json.loads((self.root/'artifacts/budget.json').read_text())
        with patch('subprocess.check_output', return_value='CANCELLED by 1\n'):
            result = p12.close_stale_credit(self.root, '999')
        after = json.loads((self.root/'artifacts/budget.json').read_text())
        row = after['inference_credits']['prompt12-run1-999']
        self.assertEqual(row['status'], 'closed_stale')
        journaled = sum(len(f.read_text().splitlines()) for f in (self.root/'runs'/p12.NAMESPACE).rglob('test.jsonl'))
        self.assertAlmostEqual(result['charged'], journaled*.001)
        self.assertAlmostEqual(after['reserved'], 9.4)
        self.assertAlmostEqual(after['spent'], 213.5+journaled*.001)
        self.assertLessEqual(before['reserved'], 9.4+8+1e-9)
        self.approve(run_cap=5.)
        self.assertAlmostEqual(p12.run_remaining(self.root, p12.approval(self.root)), 5.-journaled*.001)

    def test_speed_addendum_sets_workers_and_timeout(self):
        p12.freeze(self.root); self.approve(); p12.authorize_budget(self.root)
        (self.root/p12.SPEED).write_text(json.dumps({'approved_by': 'Aaron', 'scope': p12.SCOPE, 'workers': 32,
                                                     'timeout_s': 30, 'quote': 'synthetic'}))
        timeouts = []

        def seen(request, timeout=None, **kw):
            timeouts.append(timeout)
            return self.http(request)
        with patch.dict(os.environ, SLURM_JOB_ID='1003'), patch('urllib.request.urlopen', side_effect=seen), \
                patch('signal.signal'):
            p12.execute(self.root, models=['jev'])
        started = json.loads((self.root/p12.ATTEMPT/'job-1003/started.json').read_text())
        self.assertEqual((started['workers'], started['http_timeout_s']), (32, 30))
        self.assertEqual(set(timeouts), {30})
        (self.root/p12.SPEED).write_text(json.dumps({'approved_by': 'Aaron', 'scope': p12.SCOPE, 'workers': 64,
                                                     'timeout_s': 30, 'quote': 'synthetic'}))
        with self.assertRaisesRegex(ValueError, 'speed addendum'):
            p12.speed(self.root, p12.approval(self.root))

    def test_default_timeout_is_frozen_120(self):
        p12.freeze(self.root); self.approve(); p12.authorize_budget(self.root)
        self.assertEqual(p12.speed(self.root, p12.approval(self.root))['timeout_s'], 120)

    def test_dns_cache_reuses_lookup(self):
        import socket
        calls = []
        with patch('socket.getaddrinfo', side_effect=lambda *a, **k: calls.append(a) or [('x',)]):
            stats = p12.install_dns_cache()
            socket.getaddrinfo('openrouter.ai', 443); socket.getaddrinfo('openrouter.ai', 443)
            self.assertEqual(len(calls), 1); self.assertEqual(stats['hits'], 1)

    def test_failed_cell_does_not_stop_model_then_archive_and_resume(self):
        p12.freeze(self.root); self.approve(); p12.authorize_budget(self.root)
        state = {'n': 0}

        def cf520(request, **kw):
            body = json.loads(request.data)
            if body['model'] == 'perplexity/pplx-decider-v1.2' and state['n'] == 0:
                state['n'] += 1
                import urllib.error
                raise urllib.error.HTTPError(request.full_url, 520, 'unknown', {}, io.BytesIO(b'{}'))
            return self.http(request, **kw)
        with patch('urllib.request.urlopen', side_effect=cf520), patch('signal.signal'):
            with self.assertRaisesRegex(ValueError, 'model stopped'):
                p12.execute(self.root)
        ns = self.root/'runs'/p12.NAMESPACE
        self.assertEqual(len(list(ns.rglob('test.failures.jsonl'))), 1)
        self.assertTrue((ns/'pplx_v12/R_rules/semi-heter/test.jsonl').exists())  # model kept going
        failed = json.loads((self.root/p12.ATTEMPT/'job-999/failed.json').read_text())
        self.assertFalse(failed['hold_retained'])  # drained: credit closed exactly
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['reserved'], 9.4)
        moved = p12.archive_failures(self.root)
        self.assertEqual(len(moved), 1)
        with patch.dict(os.environ, SLURM_JOB_ID='1002'), patch('urllib.request.urlopen', side_effect=self.http), \
                patch('signal.signal'):
            p12.execute(self.root, resume=True)
        self.assertTrue((self.root/p12.ATTEMPT/'job-1002/complete.json').exists())

    def test_run_cap_stops_and_retains_credit(self):
        p12.freeze(self.root); self.approve(run_cap=.06); p12.authorize_budget(self.root)
        with patch('urllib.request.urlopen', side_effect=self.http):
            with self.assertRaises(ValueError):
                p12.execute(self.root)
        failed = json.loads((self.root/p12.ATTEMPT/'job-999/failed.json').read_text())
        self.assertFalse(failed['hold_retained'])  # drained, all attempts settled -> closed exactly
        self.assertEqual(failed['credit']['outstanding'], 0)

    def test_cap_cannot_decrease(self):
        p12.freeze(self.root); self.approve(cap=200.)
        with self.assertRaisesRegex(ValueError, 'only increase'):
            p12.authorize_budget(self.root)

    def test_source_drift_blocks(self):
        p12.freeze(self.root)
        with (self.root/'src/prompt12.py').open('a') as f:
            f.write('\n# drift\n')
        with self.assertRaisesRegex(ValueError, 'Source drift'):
            p12.preflight(self.root, require_approval=False)


if __name__ == '__main__':
    unittest.main()
