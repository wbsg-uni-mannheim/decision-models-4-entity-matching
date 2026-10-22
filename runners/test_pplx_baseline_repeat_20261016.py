import http.client
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import protocol
import run
import run_pplx_baseline_repeat_20261016 as repeat


class RepeatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'artifacts/operations').mkdir(parents=True)
        (self.root/'artifacts/budget.json').write_text(json.dumps({'spent': 5., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0, 'foreign': {'keep': 3}}))
        for b in protocol.BENCHMARKS:
            folder = self.root/'data'/b; folder.mkdir(parents=True)
            rows = []
            for i in range(2):
                records = {'left': {'title': str(i)}, 'right': {'title': str(i)}}
                rows.append({'benchmark': b, 'pair_id': str(i), 'records': records,
                             'input_hash': protocol.digest(records), 'label': 1})
            (folder/'test.json').write_text(json.dumps(rows))
        self.spec = {'benchmarks': list(protocol.BENCHMARKS), 'slurm_job_id': '123'}
        self.admission = self.root/'admission.json'; self.admission.write_text(json.dumps(self.spec))
        self.calls = []
        self.original_account = run.account
        for p in [patch.object(run, 'PROJECT', self.root), patch.object(run, 'key', return_value='mock'),
                  patch.dict(os.environ, SLURM_JOB_ID='123'), patch.object(repeat, 'source_check'),
                  patch.object(repeat, 'readiness', side_effect=self.ready),
                  patch('socket.socket', side_effect=AssertionError('Real network forbidden'))]:
            p.start(); self.addCleanup(p.stop)

    def ready(self, root, spec, held=False):
        if (root/repeat.ATTEMPT).exists():
            raise ValueError('Repeat already attempted')
        return {'synthetic_scheduler': True}

    def http(self, request, **kw):
        body = json.loads(request.data); self.calls.append(body)
        self.assertEqual(body['model'], protocol.MODELS['pplx']['id'])
        self.assertEqual(body['questions']['match']['instructions'], protocol.BASE)
        self.assertTrue((self.root/repeat.ATTEMPT/'prior.json').exists())
        obj = repeat.read(self.root/'artifacts/budget.json')
        self.assertEqual(obj['reserved'], 6.)
        response = io.BytesIO(json.dumps({'model': protocol.MODELS['pplx']['id'], 'provider': 'Perplexity',
                                     'answers': {'match': {'type': 'noul', 'noul': .9}},
                                     'usage': {'cost': .001}}).encode())
        response.status = 200
        return response

    def test_full_frozen_pipeline_and_repeat_guard(self):
        with patch('urllib.request.urlopen', side_effect=self.http):
            repeat.execute(self.root, self.spec, self.admission)
        self.assertEqual(len(self.calls), 16)
        proof = repeat.read(self.root/repeat.ATTEMPT/'final_validation.json')
        self.assertEqual(len(proof['cells']), 8); self.assertEqual(proof['macro_f1'], 1.)
        self.assertEqual(proof['totals']['attempts'], 16)
        budget = repeat.read(self.root/'artifacts/budget.json')
        self.assertAlmostEqual(budget['spent'], 5.016); self.assertAlmostEqual(budget['reserved'], 3.)
        self.assertEqual(budget['foreign'], {'keep': 3}); self.assertIs(run.account, self.original_account)
        with self.assertRaisesRegex(ValueError, 'already attempted'):
            repeat.execute(self.root, self.spec, self.admission)

    def test_unknown_zero_outstanding_retains_credit(self):
        with patch('urllib.request.urlopen', side_effect=http.client.IncompleteRead(b'partial')):
            with self.assertRaises(RuntimeError):
                repeat.execute(self.root, self.spec, self.admission)
        obj = repeat.read(self.root/repeat.ATTEMPT/'failed.json')
        self.assertTrue(obj['hold_retained']); self.assertEqual(obj['credit']['outstanding'], 0)
        self.assertFalse((self.root/repeat.ATTEMPT/'complete.json').exists())
        self.assertEqual(repeat.read(self.root/'artifacts/budget.json')['reserved'], 6.)

    def test_undeclared_source_rejected(self):
        # Validate the unmocked implementation in a separate import namespace.
        import importlib.util
        spec = importlib.util.spec_from_file_location('unmocked_repeat', repeat.__file__)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            module.source_check(self.root, {'source_sha256': {}})

    def test_prior_storage_failure_precedes_credit(self):
        with patch.object(repeat, 'save', side_effect=OSError('prior fsync failed')):
            with self.assertRaises(OSError):
                repeat.execute(self.root, self.spec, self.admission)
        self.assertEqual(repeat.read(self.root/'artifacts/budget.json')['reserved'], 3.)
        self.assertNotIn('inference_credits', repeat.read(self.root/'artifacts/budget.json'))

    def test_raw_tampering_blocks_close(self):
        original = repeat.validate_results
        def tamper(root, spec, credit):
            p = root/'runs'/repeat.NAMESPACE/'pplx/B0/abt-buy/test.jsonl'
            rows = [json.loads(x) for x in p.read_text().splitlines()]
            rows[0]['study_run_id'] = 2
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            return original(root, spec, credit)
        with patch('urllib.request.urlopen', side_effect=self.http), patch.object(repeat, 'validate_results', side_effect=tamper):
            with self.assertRaisesRegex(ValueError, 'Raw identity'):
                repeat.execute(self.root, self.spec, self.admission)
        self.assertTrue(repeat.read(self.root/repeat.ATTEMPT/'failed.json')['hold_retained'])
        self.assertFalse((self.root/repeat.ATTEMPT/'complete.json').exists())

    def test_foreign_job_rejected_before_funding(self):
        with patch.dict(os.environ, SLURM_JOB_ID='456'):
            with self.assertRaisesRegex(ValueError, 'Real admitted'):
                repeat.execute(self.root, self.spec, self.admission)
        self.assertFalse((self.root/repeat.ATTEMPT).exists())
        self.assertEqual(repeat.read(self.root/'artifacts/budget.json')['reserved'], 3.)


if __name__ == '__main__':
    unittest.main()
