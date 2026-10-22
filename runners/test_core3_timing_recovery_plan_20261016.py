"""Offline real frozen requests/account/cache tests; all HTTP is mocked."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import protocol
import run
from drained_inference import build
import core3_timing_recovery_plan_20261016 as plan

ROOT = Path(__file__).resolve().parents[2]


class Response(io.BytesIO):
    status = 200


class TimingPlanTests(unittest.TestCase):
    def evidence(self):
        timing = plan.read(ROOT/'artifacts/verification/core3_timing_terminal_366980_20261016.json')
        proof = plan.read(ROOT/'artifacts/verification/grouped_core3_terminal_366980_20261016.json')
        return timing, proof['remaining_cost_plan']

    def make(self, timing, remaining):
        return plan.plan_from_evidence(timing, remaining, protocol.MODELS, protocol.BENCHMARKS)

    def test_real_saved_evidence_exact_scope_and_independent_cost(self):
        timing, remaining = self.evidence()
        result = self.make(timing, remaining)
        self.assertEqual(len(result['cells']), 112)
        self.assertEqual(sum(c['new_responses'] for c in result['cells']), 420)
        self.assertEqual(sum(c['action'] == 'retain_complete_measurement' for c in result['cells']), 91)
        self.assertEqual(sum(c['action'] == 'measure_unstarted_cell' for c in result['cells']), 20)
        self.assertAlmostEqual(result['known_proxy_usd'], .011552578, places=12)
        self.assertEqual(result['historical_unknown_scenario_usd'], 0)
        self.assertFalse(result['admitted'])
        self.assertEqual(result['new_quality_responses'], 0)

    def test_matrix_gap_and_price_drift_rejected(self):
        t, r = self.evidence()
        for mutate in (
            lambda a, b: a['cells'].pop(),
            lambda a, b: a['cells'].append(copy.deepcopy(a['cells'][0])),
            lambda a, b: b['cells'][0].update(missing=20),
            lambda a, b: b.update(known_proxy_usd=100),
            lambda a, b: b['cells'][0].update(known_proxy_usd=float('nan')),
            lambda a, b: next(c for c in a['cells'] if c['responses_without_execution_walltime']).update(execution_records=1),
        ):
            a, b = copy.deepcopy(t), copy.deepcopy(r)
            mutate(a, b)
            with self.assertRaises(ValueError):
                self.make(a, b)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'artifacts').mkdir()
        data = self.root/'data/billiger-de'
        data.mkdir(parents=True)
        self.cases = [{'benchmark': 'billiger-de', 'pair_id': str(i),
            'records': {'left': {'name': str(i)}, 'right': {'name': str(i)}}} for i in range(20)]
        for c in self.cases:
            c['input_hash'] = protocol.digest(c['records'])
        (data/'prompt.json').write_text(json.dumps(self.cases))
        self.response = {'model': protocol.MODELS['kev']['id'], 'provider': 'siliconflow',
            'answers': {'match': {'type': 'noul', 'noul': .8}}, 'usage': {'cost': .001}}
        module = types.SimpleNamespace(**vars(run))
        module.PROJECT = self.root
        module.key = lambda: 'fake'
        module.account = types.FunctionType(run.account.__code__, {**run.account.__globals__, 'PROJECT': self.root},
                                            argdefs=run.account.__defaults__)
        _, self.run_cell, _ = build(module, 3)
        (self.root/'artifacts/budget.json').write_text(json.dumps({'spent': 0, 'reserved': 3,
            'attempts': 0, 'unknown_cost_attempts': 0, 'foreign_hold': {'reserved_usd': 3}}))
        self.socket = patch('socket.socket', side_effect=AssertionError('Network forbidden'))
        self.socket.start()
        self.addCleanup(self.socket.stop)
        self.cell = {'model': 'kev', 'benchmark': 'billiger-de', 'original_namespace': 'timing_concurrent8',
            'analysis_namespace': plan.REPLACEMENT, 'workers': 8, 'limit': 20, 'variant': 'B0', 'split': 'prompt'}

    def ok(self, *args, **kwargs):
        return Response(json.dumps(self.response).encode())

    def execute(self, namespace):
        self.run_cell('kev', 'B0', 'billiger-de', 'prompt', workers=8, limit=20, namespace=namespace)

    def complete(self):
        with patch('urllib.request.urlopen', side_effect=self.ok) as http:
            self.execute(plan.REPLACEMENT)
        self.assertEqual(http.call_count, 20)
        return plan.base(self.root, plan.REPLACEMENT, 'kev', 'billiger-de')

    def test_failed_nine_preserved_full_twenty_replacement_and_costs(self):
        bodies = {protocol.canonical(protocol.payload(c, 'kev', 'B0')): int(c['pair_id']) for c in self.cases}
        def http(req, **kwargs):
            i = bodies[protocol.canonical(json.loads(req.data))]
            if i in range(9, 16):
                raise urllib.error.HTTPError('https://unused', 503, 'overloaded', {}, io.BytesIO(b'{}'))
            return self.ok()
        with patch('urllib.request.urlopen', side_effect=http) as calls, patch('time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'failed batch'):
                self.execute('timing_concurrent8')
        self.assertEqual(calls.call_count, 30)
        old = plan.base(self.root, 'timing_concurrent8', 'kev', 'billiger-de')
        self.assertEqual(len(plan.rows(old/'prompt.jsonl')), 9)
        self.assertEqual(len(plan.rows(old/'prompt.failures.jsonl')), 7)
        self.assertFalse((old/'prompt.executions.jsonl').exists())
        before = {p.name: p.read_bytes() for p in old.iterdir()}
        self.complete()
        accepted = plan.validate_measurement(self.root, self.cell, protocol)
        self.assertEqual(accepted['responses'], 20)
        self.assertEqual(accepted['attempts'], 20)
        self.assertEqual(before, {p.name: p.read_bytes() for p in old.iterdir()})
        ledger = plan.read(self.root/'artifacts/budget.json')
        self.assertEqual(ledger['attempts'], 50)
        self.assertEqual(ledger['unknown_cost_attempts'], 21)
        self.assertAlmostEqual(ledger['spent'], 1.079)
        self.assertAlmostEqual(ledger['reserved'], 3)
        self.assertEqual(ledger['foreign_hold'], {'reserved_usd': 3})
        # Demonstrate the rejected blind-resume behavior only in this disposable fixture.
        with patch('urllib.request.urlopen', side_effect=self.ok) as resumed:
            self.execute('timing_concurrent8')
        self.assertEqual(resumed.call_count, 11)
        self.assertEqual(len(plan.rows(old/'prompt.jsonl')), 20)
        self.assertEqual(plan.rows(old/'prompt.executions.jsonl')[0]['new_pairs'], 11)
        self.assertTrue((old/'prompt.jsonl').read_bytes().startswith(before['prompt.jsonl']))

    def test_partial_execution_multiple_or_invalid_walltime_rejected(self):
        directory = self.complete()
        path = directory/'prompt.executions.jsonl'
        original = plan.rows(path)[0]
        for field, value in [('new_pairs', 11), ('new_pairs', True), ('workers', 1),
                             ('wall_seconds', 0), ('wall_seconds', float('nan')),
                             ('throughput_pairs_per_second', 1e20)]:
            row = {**original, field: value}
            path.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):
                plan.validate_measurement(self.root, self.cell, protocol)
        path.write_text((json.dumps(original)+'\n')*2)
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            plan.validate_measurement(self.root, self.cell, protocol)

    def test_raw_run_request_and_analysis_source_guards(self):
        directory = self.complete()
        path = directory/'prompt.jsonl'
        original = plan.rows(path)
        for field, value in [('study_run_id', True), ('study_run_id', 2), ('request_hash', 'bad'),
                             ('prediction', 0), ('input_hash', 'bad')]:
            records = copy.deepcopy(original)
            records[0][field] = value
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(ValueError):
                plan.validate_measurement(self.root, self.cell, protocol)
        path.write_text(''.join(json.dumps(r)+'\n' for r in original))
        for changes in ({'analysis_namespace': 'timing_concurrent8'}, {'original_namespace': 'quality'},
                        {'ordered_pair_ids': list(reversed([c['pair_id'] for c in self.cases]))}):
            with self.assertRaises(ValueError):
                plan.validate_measurement(self.root, {**self.cell, **changes}, protocol)
        (directory/'prompt.operations.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'Failed/unknown'):
            plan.validate_measurement(self.root, self.cell, protocol)


if __name__ == '__main__':
    unittest.main()
