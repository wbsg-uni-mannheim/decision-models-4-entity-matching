import concurrent.futures
import io
import json
from pathlib import Path
import socket
import tempfile
import types
import unittest
from unittest.mock import patch

import protocol
import run
from drained_inference import build
from leased_budget import Credit, install_barrier, remove_barrier


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); (self.root/'artifacts').mkdir()
        self.path = self.root/'artifacts/budget.json'
        self.path.write_text(json.dumps({'spent': 100., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0}))

    def test_parallel_known_unknown_and_foreign_hold(self):
        credit = Credit(self.root, 'test', {'fixture': True})
        def call(i):
            credit.account('reserve'); credit.account('settle', None if i % 10 == 0 else .001)
        with concurrent.futures.ThreadPoolExecutor(max_workers=56) as pool:
            list(pool.map(call, range(2000)))
        before = json.loads(self.path.read_text())
        self.assertGreater(before['reserved'], 3)
        result = credit.checkpoint(close=True)
        after = json.loads(self.path.read_text())
        self.assertAlmostEqual(after['spent'], 111.8)
        self.assertAlmostEqual(after['reserved'], 3)
        self.assertEqual(after['attempts'], 2000)
        self.assertEqual(after['unknown_cost_attempts'], 200)
        self.assertLess(result['global_transactions'], 10)
        with self.assertRaises(ValueError): Credit(self.root, 'test', {})

    def test_concurrent_leases_cannot_overallocate_and_crash_never_expires(self):
        self.path.write_text(json.dumps({'spent': 196.91, 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0}))
        first = Credit(self.root, 'first', {})
        with self.assertRaises(RuntimeError): Credit(self.root, 'second', {})
        first.account('reserve')
        with self.assertRaises(RuntimeError): first.account('reserve')
        with self.assertRaises(ValueError): first.checkpoint(close=True)
        self.assertAlmostEqual(json.loads(self.path.read_text())['spent'], 196.91)
        self.assertAlmostEqual(json.loads(self.path.read_text())['reserved'], 3.09)
        first.account('settle', .01); first.checkpoint(close=True)
        self.assertAlmostEqual(json.loads(self.path.read_text())['reserved'], 3)

    def test_barrier_drains_loaded_frozen_account_without_blocking_settlements(self):
        old = types.FunctionType(run.account.__code__, {**run.account.__globals__, 'PROJECT': self.root}, 'account', (0, 200))
        for _ in range(56): old('reserve')
        barrier = install_barrier(self.root, 'transition', {})
        self.assertTrue(barrier['nonfinancial_admission_barrier'])
        for i in range(56):
            old('settle', None if i == 0 else .001)
            with self.assertRaises(RuntimeError): old('reserve')
        with self.assertRaises(ValueError): remove_barrier(self.root, 'transition', {})
        remove_barrier(self.root, 'transition', {'all_steps_terminal': True, 'fixture': True})
        d = json.loads(self.path.read_text())
        self.assertAlmostEqual(d['reserved'], 3)
        self.assertAlmostEqual(d['spent'], 100.105)
        self.assertEqual(d['attempts'], 56)
        self.assertEqual(d['unknown_cost_attempts'], 1)

    def test_real_frozen_request_exact_payload_parser_and_credit(self):
        credit = Credit(self.root, 'real-functions', {})
        inference = types.SimpleNamespace(**{**vars(run), 'account': credit.account})
        request, _, _ = build(inference, 2)
        case = {'pair_id': 'x#y', 'benchmark': 'abt-buy', 'records': {'left': {'name': 'A'}, 'right': {'name': 'A'}}}
        case['input_hash'] = protocol.digest(case['records'])
        raw = {'provider': 'typesafe', 'model': protocol.MODELS['jev']['id'],
            'answers': {'match': {'type': 'noul', 'noul': .8}}, 'usage': {'cost': .001}}
        def http(req, **kwargs):
            self.assertEqual(json.loads(req.data), protocol.payload(case, 'jev', 'B0'))
            response = io.BytesIO(json.dumps(raw).encode()); response.status = 200
            return response
        with patch.object(socket, 'socket', side_effect=AssertionError('No network')), \
                patch.object(run.urllib.request, 'urlopen', side_effect=http):
            result = request(case, 'jev', 'B0', 'fake')
        self.assertEqual(result['study_run_id'], 2)
        self.assertEqual(result['prediction'], 1)
        self.assertEqual(result['request_hash'], protocol.digest(protocol.payload(case, 'jev', 'B0')))
        credit.checkpoint(close=True)
        self.assertAlmostEqual(json.loads(self.path.read_text())['spent'], 100.001)


if __name__ == '__main__': unittest.main()
