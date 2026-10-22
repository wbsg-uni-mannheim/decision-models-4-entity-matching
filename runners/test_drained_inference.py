import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import protocol
import run
from drained_inference import build


class Response(io.BytesIO):
    status = 200


class DrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'artifacts').mkdir()
        data = self.root/'data/abt-buy'
        data.mkdir(parents=True)
        self.cases = [{'benchmark': 'abt-buy', 'pair_id': str(i),
                       'records': {'left': {'name': str(i)}, 'right': {'name': str(i)}}} for i in range(3)]
        for c in self.cases:
            c['input_hash'] = protocol.digest(c['records'])
        (data/'prompt.json').write_text(json.dumps(self.cases))
        self.response = {'model': 'typesafe/jev-1.13', 'provider': 'typesafe',
                         'answers': {'match': {'type': 'noul', 'noul': .8}}, 'usage': {'cost': .01}}
        self.module = types.SimpleNamespace(**vars(run))
        self.module.PROJECT = self.root
        self.module.key = lambda: 'fake-secret'
        self.module.account = types.FunctionType(run.account.__code__, {**run.account.__globals__, 'PROJECT': self.root},
                                                 argdefs=run.account.__defaults__)
        self.socket = patch('socket.socket', side_effect=AssertionError('Network forbidden'))
        self.socket.start()
        self.addCleanup(self.socket.stop)
        self.cache = self.root/'runs/quality/jev/B0/abt-buy'

    def budget(self, spent=0):
        (self.root/'artifacts/budget.json').write_text(json.dumps(
            {'spent': spent, 'reserved': 0, 'attempts': 0, 'unknown_cost_attempts': 0}))

    def rows(self, suffix):
        p = self.cache/('prompt.'+suffix)
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def test_real_http_parser_account_and_frozen_semantics_match(self):
        # Compare all response values except runtime timestamps/durations.
        self.budget()
        request, _, proof = build(self.module, 3)
        captured = []
        def http(req, **kwargs):
            captured.append(json.loads(req.data))
            return Response(json.dumps(self.response).encode())
        original = types.FunctionType(run.request.__code__, vars(self.module))
        with patch('urllib.request.urlopen', side_effect=http):
            actual = request(self.cases[0], 'jev', 'B0', 'fake-secret')
            expected = original(self.cases[0], 'jev', 'B0', 'fake-secret')
        self.assertEqual(captured, [protocol.payload(self.cases[0], 'jev')]*2)
        self.assertEqual(actual.pop('study_run_id'), 3)
        for r in (actual, expected):
            r.pop('completed_at')
            for a in r['attempts']:
                a.pop('seconds')
        self.assertEqual(actual, expected)
        self.assertFalse(proof['admission_granted'])
        self.assertIs(self.module.request, run.request)

    def test_cap_drains_paid_sibling_and_stops_before_next_batch(self):
        self.budget(99.94)
        reserve_failed = threading.Event()
        account = self.module.account
        def wrapped(action, *args):
            try:
                return account(action, *args)
            except RuntimeError:
                reserve_failed.set()
                raise
        self.module.account = wrapped
        _, cell, _ = build(self.module, 3)
        def http(*args, **kwargs):
            self.assertTrue(reserve_failed.wait(timeout=3))
            return Response(json.dumps(self.response).encode())
        with patch('urllib.request.urlopen', side_effect=http) as mock:
            with self.assertRaisesRegex(RuntimeError, 'failed batch'):
                cell('jev', 'B0', 'abt-buy', 'prompt', workers=2)
        self.assertEqual(mock.call_count, 1)
        self.assertEqual(len(self.rows('jsonl')), 1)
        self.assertEqual(len(self.rows('operations.jsonl')), 1)
        self.assertEqual(self.rows('failures.jsonl'), [])
        stop = self.rows('operations.jsonl')[0]
        self.assertTrue(stop['operational_stop']['budget_reservation_rejected'])
        self.assertNotIn('attempts', stop)
        self.assertEqual(stop['study_run_id'], 3)
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertEqual((budget['attempts'], budget['reserved']), (1, 0))
        self.assertAlmostEqual(budget['spent'], 99.95)

    def test_paid_retry_attempt_survives_later_reserve_failure(self):
        self.budget(99.94)
        _, cell, _ = build(self.module, 3)
        error = urllib.error.HTTPError('https://unused', 503, 'capacity', {}, io.BytesIO(b'{"error":"overloaded"}'))
        with patch('urllib.request.urlopen', side_effect=error) as mock, patch('time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'failed batch'):
                cell('jev', 'B0', 'abt-buy', 'prompt', workers=1)
        self.assertEqual(mock.call_count, 1)
        sleep.assert_called_once_with(2)
        failure = self.rows('failures.jsonl')[0]
        self.assertEqual(len(failure['attempts']), 1)
        self.assertEqual(failure['attempts'][0]['status'], 503)
        self.assertIsNone(failure['attempts'][0]['cost'])
        self.assertEqual(len(self.rows('operations.jsonl')), 1)
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertEqual((budget['attempts'], budget['unknown_cost_attempts'], budget['reserved']), (1, 1, 0))
        self.assertAlmostEqual(budget['spent'], 99.99)

    def test_original_retry_limit_and_delays_retained(self):
        self.budget()
        request, _, _ = build(self.module, 2)
        def error(*args, **kwargs):
            raise urllib.error.HTTPError('https://unused', 429, 'capacity', {}, io.BytesIO(b'{}'))
        with patch('urllib.request.urlopen', side_effect=error) as mock, patch('time.sleep') as sleep:
            result = request(self.cases[0], 'jev', 'B0', 'fake-secret')
        self.assertEqual(mock.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 4])
        self.assertEqual(len(result['attempts']), 3)
        self.assertEqual(result['error'], 'HTTP 429')
        self.assertNotIn('operational_stop', result)

    def test_unexpected_future_error_does_not_discard_sibling(self):
        self.budget()
        request, cell, _ = build(self.module, 2)
        event = threading.Event()
        def unexpected(case, *args):
            if case['pair_id'] == '0':
                event.set()
                raise RuntimeError('unexpected fake-secret')
            self.assertTrue(event.wait(timeout=3))
            return request(case, *args)
        cell.__globals__['request'] = unexpected
        with patch('urllib.request.urlopen', side_effect=lambda *a, **k: Response(json.dumps(self.response).encode())) as mock:
            with self.assertRaisesRegex(RuntimeError, 'failed batch'):
                cell('jev', 'B0', 'abt-buy', 'prompt', workers=2)
        self.assertEqual(mock.call_count, 1)
        self.assertEqual(len(self.rows('jsonl')), 1)
        stop = self.rows('operations.jsonl')[0]
        self.assertTrue(stop['operational_stop']['outcome_unknown'])
        self.assertNotIn('fake-secret', json.dumps(stop))

    def test_source_drift_and_invalid_run_fail_before_http(self):
        for run_id in (True, 0, 4):
            with self.assertRaises(ValueError):
                build(self.module, run_id)
        drift = self.root/'changed.py'
        drift.write_bytes(Path(run.__file__).read_bytes()+b'\n')
        self.module.__file__ = str(drift)
        with self.assertRaisesRegex(ValueError, 'source changed'):
            build(self.module, 3)

    def test_foreign_hold_is_preserved_on_first_reservation_rejection(self):
        self.budget(96.96)
        path = self.root/'artifacts/budget.json'
        ledger = json.loads(path.read_text())
        ledger['reserved'] = 3
        ledger['foreign_hold'] = {'id': 'separate-package', 'reserved_usd': 3}
        path.write_text(json.dumps(ledger))
        before = path.read_bytes()
        request, _, _ = build(self.module, 2)
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP allowed')):
            result = request(self.cases[0], 'jev', 'B0', 'fake-secret')
        self.assertEqual(result['attempts'], [])
        self.assertTrue(result['operational_stop']['budget_reservation_rejected'])
        self.assertEqual(path.read_bytes(), before)

    def test_cache_reuse_and_hash_guard_no_extra_requests(self):
        self.budget()
        _, cell, _ = build(self.module, 2)
        with patch('urllib.request.urlopen', side_effect=lambda *a, **k: Response(json.dumps(self.response).encode())) as mock:
            cell('jev', 'B0', 'abt-buy', 'prompt', workers=2)
            cell('jev', 'B0', 'abt-buy', 'prompt', workers=2)
        self.assertEqual(mock.call_count, 3)
        self.assertTrue(all(r['study_run_id'] == 2 for r in self.rows('jsonl')))
        rows = copy.deepcopy(self.rows('jsonl'))
        rows[0]['request_hash'] = 'changed'
        (self.cache/'prompt.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP allowed')):
            with self.assertRaisesRegex(AssertionError, 'Cache configuration mismatch'):
                cell('jev', 'B0', 'abt-buy', 'prompt')


if __name__ == '__main__':
    unittest.main()
