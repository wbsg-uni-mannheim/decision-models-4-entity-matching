"""Offline test for the Qwen3.8-27B baseline runner: mocked HTTP, no network."""
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
import qwen38_baseline
import prompt12
import protocol
import run
import run_qwen38_baseline_20261022 as rb

REAL = Path(__file__).resolve().parents[2]


class Qwen38BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = self.root = Path(self.temp.name)
        for rel in rb.SOURCES:
            if rel != 'configs/manifest.json':
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(REAL/rel, root/rel)
        manifest = {'benchmarks': {}}
        for b in protocol.BENCHMARKS:
            rows = []
            for i in range(2):
                rec = {'left': {'title': f'{b} {i} a'}, 'right': {'title': f'{b} {i} b'}}
                rows.append({'benchmark': b, 'pair_id': f'{b}#{i}', 'label': i, 'records': rec, 'input_hash': protocol.digest(rec)})
            (root/'data'/b).mkdir(parents=True)
            (root/'data'/b/'test.json').write_text(json.dumps(rows))
            manifest['benchmarks'][b] = {'splits': {'test': {'n': 2, 'sha256': protocol.digest(rows)}}}
        (root/'configs/manifest.json').write_text(json.dumps(manifest))
        (root/'artifacts/operations').mkdir(parents=True, exist_ok=True)
        (root/'artifacts/budget.json').write_text(json.dumps({'spent': 424., 'reserved': 9.38, 'attempts': 0,
            'unknown_cost_attempts': 0, 'project_cap_usd': 440, 'budget_authority_sha256': 'x'}))
        qwen38_baseline.register_models()
        mods = (run, protocol, qwen38_baseline, drained_inference, leased_budget)
        self.saved = {k: getattr(rb, k) for k in ('RUN_ID', 'NAMESPACE', 'ATTEMPT', 'MANIFEST', 'CREDIT_PREFIX')}
        self.addCleanup(lambda: [setattr(rb, k, v) for k, v in self.saved.items()])
        for p in [patch.object(rb, 'modules', return_value=mods), patch.object(run, 'key', return_value='mock'),
                  patch.object(rb, 'other_project_jobs', return_value=[]), patch('signal.signal'),
                  patch.dict(os.environ, SLURM_JOB_ID='999'),
                  patch('socket.socket', side_effect=AssertionError('Real network forbidden'))]:
            p.start(); self.addCleanup(p.stop)
        self.calls = []

    def http(self, request, timeout=None, **kw):
        body = json.loads(request.data); self.calls.append(body)
        data = {'model': body['model'], 'provider': 'DeepInfra', 'usage': {'cost': .0001},
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{"match": true}'}}]}
        r = io.BytesIO(json.dumps(data).encode()); r.status = 200
        return r

    def test_run2_uses_luna_b0_request_with_qwen(self):
        for k, v in dict(RUN_ID=2, NAMESPACE='qwen38_b0_run02_20261022', ATTEMPT='artifacts/operations/qwen38_b0_run02_20261022',
                         MANIFEST='configs/qwen38_b0_run02_manifest_20261022.json', CREDIT_PREFIX='qwen38-b0-run2-').items():
            setattr(rb, k, v)
        spec = rb.freeze(self.root)
        self.assertEqual((spec['run_id'], spec['decisions'], list(spec['models'])), (2, 16, ['qwen38']))
        (self.root/rb.APPROVAL).write_text(json.dumps({'approved_by': 'Aaron', 'scope': rb.SCOPE, 'project_cap_usd': 440.,
            'run_cap_usd': 1., 'workers': 32, 'timeout_s': 30, 'quote': 'synthetic'}))
        (self.root/rb.SHARED_AUTHORITY).write_text(json.dumps({'project_cap_usd': 440., 'quote': 'synthetic'}))
        rb.authorize_budget(self.root)
        with patch('urllib.request.urlopen', side_effect=self.http):
            rb.execute(self.root)
        self.assertEqual(len(self.calls), 16)
        case = json.loads((self.root/'data/wdc/test.json').read_text())[0]
        self.assertIn(qwen38_baseline.payload(case), self.calls)
        luna = protocol.payload(case, 'luna', 'B0')
        self.assertEqual(self.calls[0]['messages'][0], luna['messages'][0])
        self.assertEqual({c['provider']['only'][0] for c in self.calls}, {'deepinfra/bf16'})
        rows = [json.loads(x) for x in (self.root/'runs/qwen38_b0_run02_20261022/qwen38/B0/wdc/test.jsonl').read_text().splitlines()]
        self.assertEqual({r['study_run_id'] for r in rows}, {2})


if __name__ == '__main__':
    unittest.main()
