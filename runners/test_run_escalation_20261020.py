"""Offline tests for the escalation runner: mocked HTTP, temporary root, no network."""
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import drained_inference
import escalation
import inputs_study
import leased_budget
import prompt12
import protocol
import run
import strong_ensemble
import run_escalation_20261020 as re_

REAL = Path(__file__).resolve().parents[2]
TEST_P = [.5, .99, .01, .3]   # pairs 0 and 3 are uncertain for every model
VALID = [(.99, 1), (.01, 0), (.02, 0), (.45, 1)]  # risk above 0.25% once u reaches 0.45: t = 0.02


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


class EscalationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = self.root = Path(self.temp.name)
        for rel in re_.SOURCES:
            if rel not in ('configs/manifest.json', escalation.SELECTION):
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(REAL/rel, root/rel)
        (root/'configs/manifest.json').write_text('{}')
        selection = {'selected': {m: {b: 'X_P_domain_0' for b in protocol.BENCHMARKS} for m in prompt12.RUN_MODELS}}
        (root/escalation.SELECTION).write_text(json.dumps(selection))
        for b in protocol.BENCHMARKS:
            def case(pid, label):
                records = {'left': {'title': f'{b} {pid} a'}, 'right': {'title': f'{b} {pid} b'}}
                return {'benchmark': b, 'pair_id': pid, 'label': label, 'records': records,
                        'input_hash': protocol.digest(records)}
            test = [case(f'{b}#t{i}', i % 2) for i in range(4)]
            valid = [case(f'{b}#v{i}', y) for i, (_, y) in enumerate(VALID)]
            (root/'data'/b).mkdir(parents=True)
            (root/'data'/b/'test.json').write_text(json.dumps(test))
            (root/'data'/b/'valid.json').write_text(json.dumps(valid))
            (root/'data'/b/'rules_train.json').write_text('[]')
            for m, _ in escalation.DESIGN:
                jsonl(root/'runs/prompt12_run01_20261019'/m/'X_P_domain_0'/b/'test.jsonl',
                      [{'pair_id': c['pair_id'], 'probability_match': p} for c, p in zip(test, TEST_P)])
                jsonl(root/'runs'/escalation.VALID_NS/m/'F0'/b/'valid.jsonl',
                      [{'pair_id': c['pair_id'], 'probability_match': p} for c, (p, _) in zip(valid, VALID)])
        (root/'artifacts/operations').mkdir(parents=True, exist_ok=True)
        (root/'artifacts/budget.json').write_text(json.dumps({
            'spent': 384., 'reserved': 9.4, 'attempts': 0, 'unknown_cost_attempts': 0,
            'project_cap_usd': 420, 'budget_authority_sha256': 'inputs', 'foreign': {'keep': 1}}))
        self.calls = []
        re_.STOP.clear(); self.addCleanup(re_.STOP.clear)
        escalation.register_models()
        mods = (run, protocol, escalation, drained_inference, leased_budget)
        for p in [patch.object(re_, 'modules', return_value=mods), patch.object(run, 'key', return_value='mock'),
                  patch.object(re_, 'other_project_jobs', return_value=[]), patch('signal.signal'),
                  patch.dict(os.environ, SLURM_JOB_ID='999'),
                  patch('socket.socket', side_effect=AssertionError('Real network forbidden'))]:
            p.start(); self.addCleanup(p.stop)
        self.addCleanup(inputs_study.SELECTED.clear)

    def approve(self, run_cap=60., cap=480.):
        (self.root/re_.APPROVAL).write_text(json.dumps({
            'approved_by': 'Aaron', 'scope': re_.SCOPE, 'project_cap_usd': cap, 'run_cap_usd': run_cap,
            'workers': 16, 'timeout_s': 120, 'quote': 'synthetic test approval'}))

    def http(self, request, timeout=None, provider=None, **kw):
        body = json.loads(request.data); self.calls.append((body, timeout))
        alias = {s['id']: a for a, s in strong_ensemble.MEMBERS.items()}[body['model']]
        data = {'model': body['model'], 'provider': provider or escalation.OBSERVED_PROVIDER[alias],
                'usage': {'cost': .002},
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{"match": true}'}}]}
        response = io.BytesIO(json.dumps(data).encode()); response.status = 200
        return response

    def test_escalation_set_is_union_of_deferred_pairs(self):
        report = escalation.freeze_set(self.root)
        self.assertEqual(report['union'], 2*8)
        self.assertEqual({t for t in report['thresholds'].values()}, {.02})
        ids = {c['pair_id'] for c in json.loads((self.root/'data/wdc/escalation.json').read_text())}
        self.assertEqual(ids, {'wdc#t0', 'wdc#t3'})

    def test_payload_is_frozen_luna_b0_with_member_settings(self):
        case = json.loads((self.root/'data/wdc/test.json').read_text())[0]
        luna = protocol.payload(case, 'luna', 'B0')
        body = escalation.payload(case, 'argon4')
        self.assertEqual(body['messages'], luna['messages'])
        self.assertEqual(body['response_format'], luna['response_format'])
        self.assertEqual((body['model'], body['max_tokens'], body['reasoning']),
                         ('google/gemini-4-argon', 2048, {'effort': 'medium'}))
        self.assertEqual(body['provider'], {'only': ['google-ai-studio/flex'], 'order': ['google-ai-studio/flex'],
                                            'allow_fallbacks': False, 'require_parameters': True})
        self.assertEqual(body['service_tier'], 'flex')
        sol = escalation.payload(case, 'sol61')
        self.assertEqual(sol['provider']['only'], ['openai'])
        self.assertNotIn('service_tier', sol)
        opus = escalation.payload(case, 'opus55')
        self.assertNotIn('service_tier', opus)
        self.assertEqual(opus['provider']['only'], ['anthropic'])

    def test_full_run_three_members(self):
        re_.freeze(self.root); self.approve()
        with self.assertRaisesRegex(ValueError, 'authorize-budget'):
            re_.execute(self.root)
        re_.authorize_budget(self.root)
        with patch('urllib.request.urlopen', side_effect=self.http):
            re_.execute(self.root)
        self.assertEqual(len(self.calls), 3*16)
        self.assertEqual({t for _, t in self.calls}, {120})
        self.assertEqual({b['model'] for b, _ in self.calls}, {s['id'] for s in strong_ensemble.MEMBERS.values()})
        self.assertTrue((self.root/re_.ATTEMPT/'job-999/complete.json').exists())
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 384+3*16*.002)
        self.assertAlmostEqual(budget['reserved'], 9.4)

    def test_sample_mode_is_separate_and_stratified(self):
        re_.freeze(self.root)
        saved = {k: getattr(re_, k) for k in re_.SAMPLE}
        self.addCleanup(lambda: [setattr(re_, k, v) for k, v in saved.items()])
        for k, v in re_.SAMPLE.items():
            setattr(re_, k, v)
        original = escalation.sample_set
        with patch.object(escalation, 'sample_set', lambda root: original(root, n=8)):
            spec = re_.freeze(self.root)
        self.assertEqual((spec['split'], spec['decisions']), ('escalation_sample', 3*8))
        self.assertEqual({x['n'] for x in spec['benchmarks'].values()}, {1})
        self.approve()
        re_.authorize_budget(self.root)
        with patch('urllib.request.urlopen', side_effect=self.http):
            re_.execute(self.root)
        self.assertEqual(len(self.calls), 3*8)
        self.assertTrue(list((self.root/'runs/escalation_sample_run01_20261020').rglob('escalation_sample.jsonl')))
        self.assertFalse((self.root/'runs/escalation_run01_20261020').exists())

    def test_opus_slots_are_capped(self):
        self.assertEqual(re_.MEMBER_WORKERS, {'opus55': 12, 'argon4': 16})
        re_.freeze(self.root); self.approve(); re_.authorize_budget(self.root)
        with patch('urllib.request.urlopen', side_effect=self.http):
            re_.execute(self.root)
        rows = [json.loads(x) for p in (self.root/'runs'/re_.NAMESPACE).rglob('escalation.executions.jsonl')
                for x in p.read_text().splitlines()]
        self.assertEqual({r['workers'] for r in rows}, {12, 16})

    def test_wrong_provider_is_a_failure_not_a_vote(self):
        re_.freeze(self.root); self.approve(); re_.authorize_budget(self.root)
        with patch('urllib.request.urlopen', side_effect=lambda r, timeout=None: self.http(r, timeout, provider='Other')):
            with self.assertRaisesRegex(ValueError, 'model stopped'):
                re_.execute(self.root)
        self.assertFalse(any(p.read_text() for p in (self.root/'runs').rglob('escalation.jsonl')))
        self.assertTrue(list((self.root/'runs').rglob('escalation.failures.jsonl')))


if __name__ == '__main__':
    unittest.main()
