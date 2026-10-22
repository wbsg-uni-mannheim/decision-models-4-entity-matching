"""Offline test for the PPLX v1.2 baseline runner (runs 2 and 3): mocked HTTP, no network."""
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
import pplx12_baseline
import prompt12
import protocol
import run
import run_pplx12_baseline_20261021 as rb

REAL = Path(__file__).resolve().parents[2]


class Pplx12BaselineTests(unittest.TestCase):
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
        prompt12.register_models()
        mods = (run, protocol, pplx12_baseline, drained_inference, leased_budget)
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
        data = {'model': body['model'], 'provider': 'Perplexity', 'answers': {'match': {'type': 'noul', 'noul': .8}},
                'usage': {'cost': .0001}}
        r = io.BytesIO(json.dumps(data).encode()); r.status = 200
        return r

    def test_run2_uses_prompt_study_b0_request(self):
        for k, v in dict(RUN_ID=2, NAMESPACE='pplx12_b0_run02_20261021', ATTEMPT='artifacts/operations/pplx12_b0_run02_20261021',
                         MANIFEST='configs/pplx12_b0_run02_manifest_20261021.json', CREDIT_PREFIX='pplx12-b0-run2-').items():
            setattr(rb, k, v)
        spec = rb.freeze(self.root)
        self.assertEqual((spec['run_id'], spec['decisions'], list(spec['models'])), (2, 16, ['pplx_v12']))
        (self.root/rb.APPROVAL).write_text(json.dumps({'approved_by': 'Aaron', 'scope': rb.SCOPE, 'project_cap_usd': 440.,
            'run_cap_usd': 1., 'workers': 32, 'timeout_s': 30, 'quote': 'synthetic'}))
        rb.authorize_budget(self.root)
        with patch('urllib.request.urlopen', side_effect=self.http):
            rb.execute(self.root)
        self.assertEqual(len(self.calls), 16)
        case = json.loads((self.root/'data/wdc/test.json').read_text())[0]
        self.assertIn(prompt12.payload(case, 'pplx_v12', 'B0'), self.calls)
        rows = [json.loads(x) for x in (self.root/'runs/pplx12_b0_run02_20261021/pplx_v12/B0/wdc/test.jsonl').read_text().splitlines()]
        self.assertEqual({r['study_run_id'] for r in rows}, {2})


if __name__ == '__main__':
    unittest.main()
