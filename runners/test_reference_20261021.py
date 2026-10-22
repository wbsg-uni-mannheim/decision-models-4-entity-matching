"""Offline tests for the reference run (Sol direct flex, Opus batch, Argon live). No network."""
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

REAL = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REAL/'src'), str(REAL/'artifacts/operations')]
import escalation
import protocol
import reference_rest
import reference_sol_direct_20261021 as sol
import reference_opus_batch_20261021 as opus
import run_reference_argon_20261021 as argon
import drained_inference
import leased_budget
import run


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = self.root = Path(self.temp.name)
        for rel in argon.SOURCES:
            if rel not in ('configs/manifest.json', escalation.SELECTION):
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(REAL/rel, root/rel)
        (root/'configs/manifest.json').write_text('{}')
        import prompt12
        (root/escalation.SELECTION).write_text(json.dumps({'selected': {m: {b: 'B0' for b in protocol.BENCHMARKS}
                                                                       for m in prompt12.RUN_MODELS}}))
        for b in protocol.BENCHMARKS:
            rows = []
            for i in range(3):
                rec = {'left': {'title': f'{b} {i} a'}, 'right': {'title': f'{b} {i} b'}}
                rows.append({'benchmark': b, 'pair_id': f'{b}#{i}', 'label': i % 2, 'records': rec,
                             'input_hash': protocol.digest(rec)})
            (root/'data'/b).mkdir(parents=True)
            (root/'data'/b/'test.json').write_text(json.dumps(rows))
            (root/'data'/b/'escalation.json').write_text(json.dumps(rows[:1]))
        (root/'artifacts/operations').mkdir(parents=True, exist_ok=True)
        (root/'artifacts/budget.json').write_text(json.dumps({'spent': 424., 'reserved': 9.38, 'attempts': 0,
            'unknown_cost_attempts': 0, 'project_cap_usd': 525., 'budget_authority_sha256': 'x'}))
        escalation.register_models()


class RestAndBodies(Base):
    def test_rest_is_test_minus_escalation(self):
        report = reference_rest.freeze_rest(self.root)
        self.assertEqual(report['pairs'], 2*8)
        ids = {c['pair_id'] for c in json.loads((self.root/'data/wdc/reference_rest.json').read_text())}
        self.assertEqual(ids, {'wdc#1', 'wdc#2'})

    def test_bodies_match_escalation_requests(self):
        c = json.loads((self.root/'data/wdc/test.json').read_text())[1]
        esc_sol, esc_opus = escalation.payload(c, 'sol61'), escalation.payload(c, 'opus55')
        s = reference_rest.sol_direct_body(c)
        self.assertEqual((s['messages'], s['response_format']), (esc_sol['messages'], esc_sol['response_format']))
        self.assertEqual((s['reasoning_effort'], s['max_completion_tokens'], s['service_tier'], s['model']),
                         ('medium', 2048, 'flex', 'gpt-6.1-sol'))
        o = reference_rest.opus_batch_body(c)
        self.assertNotIn('model', o); self.assertNotIn('provider', o)
        self.assertEqual({k: v for k, v in esc_opus.items() if k not in ('model', 'provider')}, o)


class SolDirect(Base):
    def setUp(self):
        super().setUp()
        reference_rest.freeze_rest(self.root)
        (self.root/sol.APPROVAL).write_text(json.dumps({'approved_by': 'Aaron', 'quote': 'synthetic', 'member': 'sol61',
                                                         'run_cap_usd': 5}))
        self.n = 0
        for p in [patch('astra_discovery_submit.credential', return_value='k'), patch('time.sleep'),
                  patch('socket.socket', side_effect=AssertionError('no network'))]:
            p.start(); self.addCleanup(p.stop)

    def fake(self, req, timeout=None):
        self.n += 1
        if self.n % 5 == 0:
            raise urllib.error.HTTPError(req.full_url, 429, 'flex', {}, io.BytesIO(b'{"error":{"type":"resource_unavailable"}}'))
        body = {'model': 'gpt-6.1-sol-2026', 'service_tier': 'flex',
                'usage': {'prompt_tokens': 300, 'completion_tokens': 40, 'prompt_tokens_details': {'cached_tokens': 0}},
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{"match": false}'}}]}
        r = io.BytesIO(json.dumps(body).encode()); r.status = 200
        return r

    def test_run_retries_flex_and_books_cost(self):
        with patch('urllib.request.urlopen', side_effect=self.fake):
            s = sol.run(self.root)
        self.assertEqual((s['requested'], s['valid'], s['missing_after_this_pass']), (16, 16, 0))
        self.assertAlmostEqual(s['cost_usd'], 16*(300e-6+40*5e-6))
        b = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(b['spent'], 424+s['cost_usd']); self.assertAlmostEqual(b['reserved'], 9.38)

    def test_fill_requests_only_missing(self):
        def broken(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 400, 'bad', {}, io.BytesIO(b'{}'))
        with patch('urllib.request.urlopen', side_effect=broken):
            s = sol.run(self.root)
        self.assertEqual(s['valid'], 0)
        with patch('urllib.request.urlopen', side_effect=self.fake):
            s = sol.run(self.root, fill=True)
        self.assertEqual((s['requested'], s['missing_after_this_pass']), (16, 0))


class OpusBatch(Base):
    def setUp(self):
        super().setUp()
        reference_rest.freeze_rest(self.root)
        (self.root/opus.APPROVAL).write_text(json.dumps({'approved_by': 'Aaron', 'quote': 'synthetic', 'member': 'opus55',
                                                          'run_cap_usd': 5}))
        self.submitted = []
        for p in [patch.object(run, 'key', return_value='k'), patch('socket.socket', side_effect=AssertionError('no network'))]:
            p.start(); self.addCleanup(p.stop)

    def http(self, key, method, url, data=None):
        if method == 'POST':
            batch = json.loads(data)
            self.assertEqual(list(batch)[-1], 'requests')
            import re
            self.assertTrue(all(re.fullmatch(r'[A-Za-z0-9_-]+', r['custom_id']) for r in batch['requests']))
            self.submitted.append(batch)
            return {'id': f'b{len(self.submitted)}', 'status': 'validating'}
        i = int(url.rsplit('b', 1)[1])-1
        batch = self.submitted[i]
        results = [{'id': str(j), 'custom_id': r['custom_id'], 'response': {'status_code': 200, 'body': {
            'model': 'anthropic/claude-opus-5.5-20260921', 'provider': 'Anthropic',
            'choices': [{'finish_reason': 'stop', 'message': {'content': '{"match": true}'}}]}}}
                   for j, r in enumerate(batch['requests'])]
        return {'id': f'b{i+1}', 'status': 'completed', 'request_counts': {'total': len(results)},
                'usage': {'cost': .01}, 'results': results}

    def test_build_submit_collect(self):
        with patch.object(opus, 'CHUNK', 5):
            m = opus.build(self.root)
        self.assertEqual((m['requests'], m['chunks']), (16, 4))
        with patch.object(opus, 'http', side_effect=self.http):
            self.assertEqual(len(opus.submit(self.root)['batches']), 4)
            self.assertEqual(len(opus.submit(self.root)['batches']), 4)  # no resubmission
            self.assertEqual(len(self.submitted), 4)
            s = opus.collect(self.root)
        self.assertEqual((s['valid'], round(s['cost_usd'], 6)), (16, .04))
        b = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(b['spent'], 424.04); self.assertAlmostEqual(b['reserved'], 9.38)
        self.assertEqual(self.submitted[0]['provider'], {'only': ['anthropic'], 'allow_fallbacks': False})


class ArgonLive(Base):
    def test_argon_only_on_rest(self):
        import types
        study = types.SimpleNamespace(**{**vars(escalation), 'RUN_MODELS': ['argon4'], 'freeze_rest': reference_rest.freeze_rest})
        mods = (run, protocol, study, drained_inference, leased_budget)
        for p in [patch.object(argon, 'modules', return_value=mods),
                  patch.object(run, 'key', return_value='mock'), patch.object(argon, 'other_project_jobs', return_value=[]),
                  patch('signal.signal'), patch.dict(os.environ, SLURM_JOB_ID='999'),
                  patch('socket.socket', side_effect=AssertionError('no network'))]:
            p.start(); self.addCleanup(p.stop)
        spec = argon.freeze(self.root)
        self.assertEqual((spec['split'], spec['decisions'], list(spec['models'])), ('reference_rest', 16, ['argon4']))
        (self.root/argon.APPROVAL).write_text(json.dumps({'approved_by': 'Aaron', 'scope': argon.SCOPE,
            'project_cap_usd': 525., 'run_cap_usd': 5., 'workers': 32, 'timeout_s': 120, 'quote': 'synthetic'}))
        argon.authorize_budget(self.root)
        calls = []

        def http(request, timeout=None, **kw):
            body = json.loads(request.data); calls.append(body)
            data = {'model': body['model'], 'provider': 'Google AI Studio', 'usage': {'cost': .001},
                    'choices': [{'finish_reason': 'stop', 'message': {'content': '{"match": true}'}}]}
            r = io.BytesIO(json.dumps(data).encode()); r.status = 200
            return r
        with patch('urllib.request.urlopen', side_effect=http):
            argon.execute(self.root)
        self.assertEqual(len(calls), 16)
        self.assertEqual({c['service_tier'] for c in calls}, {'flex'})
        self.assertTrue((self.root/argon.ATTEMPT/'job-999/complete.json').exists())


if __name__ == '__main__':
    unittest.main()
