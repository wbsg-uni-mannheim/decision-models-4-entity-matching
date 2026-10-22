"""Offline synthetic inference tests. Only identity/data manifests are mocked."""
import copy
import http.client
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import expanded_protocol as expanded
import expanded_schema as schema
import protocol
import same_condition_repetition_runtime_20261016 as component
import narrow_core_scope_20261016 as scope
from fresh_schema_selection import durable_write
from leased_budget import Credit
import test_authorized_recovery_200 as budget_fixture


class Response(io.BytesIO):
    status = 200


class RepetitionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        original = budget_fixture.AuthorizedBudget().setup_root(self.root, 100, 3)
        self.rid = 2
        self.target = self.root / 'replications/run02'
        (self.target / 'artifacts').mkdir(parents=True)
        self.inference = types.SimpleNamespace(**{**vars(original), 'PROJECT': self.target, 'key': lambda: 'mock'})
        self.cases = {}
        for b in scope.BENCHMARKS:
            cases = [dict(benchmark=b, pair_id=str(i), label=i % 2,
                          records={'left': {'title': str(i)}, 'right': {'title': str(i)}}) for i in range(2)]
            for c in cases:
                c['input_hash'] = protocol.digest(c['records'])
            self.cases[b] = cases
            p = self.target / 'data' / b / 'prompt.json'
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(cases))
        self.attempt = self.target / 'artifacts/repetition_fixture'
        self.attempt.mkdir()
        self.prior_path, self.journal = self.attempt / 'prior.json', self.attempt / 'inference'
        self.budget = self.root / 'artifacts/budget.json'
        self.protected = {str(p): component.sha(p) for p in (self.target / 'data').rglob('*.json')}
        def sources(*args):
            component.unchanged(self.protected)
            return self.cases, self.protected
        for name, value in [('sources', sources)]:
            mock = patch.object(component, name, side_effect=value)
            mock.start()
            self.addCleanup(mock.stop)
        import socket
        guard = patch.object(socket, 'socket', side_effect=AssertionError('No real network'))
        guard.start()
        self.addCleanup(guard.stop)
        self.bodies = []

    def small(self):
        actual = component.matrix
        mock = patch.object(component, 'matrix', side_effect=lambda rid: [c for c in actual(rid)
            if c[4] == 'abt-buy' and c[3] in ('S1', 'X_P_minimal_0', 'C_match_reverse', 'C_review')])
        mock.start()
        self.addCleanup(mock.stop)

    def http(self, request, **kwargs):
        body = json.loads(request.data)
        self.bodies.append(body)
        m = next(m for m in scope.MODELS if protocol.MODELS[m]['id'] == body['model'])
        raw = dict(model=body['model'], provider=protocol.MODELS[m]['provider'].split('/')[0], usage={'cost': .001})
        if m == 'luna':
            prop = body['response_format']['json_schema']['schema']['properties']['match']
            answer = 'Match' if 'enum' in prop else True
            if 'enum' in prop and 'Match' not in prop['enum']:
                answer = 'Same entity'
            raw['choices'] = [dict(finish_reason='stop', message={'content': json.dumps({'match': answer})})]
        elif body['questions']['match']['type'] == 'choice':
            labels = list(body['questions']['match']['criteria'])
            answer = 'Match' if 'Match' in labels else 'Same entity'
            raw['answers'] = {'match': dict(type='choice', choice=answer,
                                            probabilities={s: int(s == answer) for s in labels})}
        else:
            raw['answers'] = {'match': dict(type='noul', noul=.8)}
        return Response(json.dumps(raw).encode())

    def prepare(self):
        prior = component.capture_prior(self.root, self.target, self.rid, self.inference,
                                         protocol, expanded, schema, self.journal)
        durable_write(self.prior_path, (json.dumps(prior) + '\n').encode())
        self.prior_sha = component.sha(self.prior_path)
        self.credit = Credit(self.root, 'repeat-fixture', dict(prior_sha256=self.prior_sha,
            run_id=self.rid, stage=component.STAGE, journal_dir=str(self.journal.relative_to(self.target))))
        return prior

    def execute(self):
        return component.execute(self.root, self.target, self.rid, self.inference, protocol,
                                 expanded, schema, self.prior_path, self.prior_sha, self.journal, self.credit)

    def reconcile(self):
        return component.reconcile(self.root, self.target, self.rid, self.inference, protocol,
                                   expanded, schema, self.prior_path, self.prior_sha, self.journal, self.credit)

    def test_full_matrix_real_frozen_requests_credit_checkpoint_no_selection(self):
        before = (expanded.payload, schema.payload, self.inference.account)
        self.prepare()
        with patch('urllib.request.urlopen', side_effect=self.http) as http:
            result = self.execute()
        self.assertTrue(result['inference_complete'])
        self.assertEqual(http.call_count, 2688)
        self.assertEqual(len(result['cells']), 1344)
        self.assertEqual(len(result['models']), 7)
        self.assertGreater(self.credit.global_transactions, 2)  # Real 512 checkpoint boundaries.
        proof = self.reconcile()
        self.assertEqual(proof['attempts'], 2688)
        self.assertAlmostEqual(proof['charged'], 2.688)
        self.assertTrue(self.credit.closed)
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'], 3)
        self.assertEqual(before, (expanded.payload, schema.payload, self.inference.account))
        self.assertFalse((self.target / 'configs').exists())
        self.assertFalse(list((self.target / 'runs').rglob('test*')))
        self.assertFalse((self.target / 'artifacts/results').exists())
        with self.assertRaises(ValueError):
            self.execute()
        with self.assertRaisesRegex(ValueError, 'already complete'):
            component.capture_prior(self.root, self.target, self.rid, self.inference,
                                     protocol, expanded, schema, self.attempt / 'other')

    def test_run3_partial_prior_excluded_exact_requests_and_runid(self):
        self.small()
        target = self.root / 'replications/run03'
        self.target.rename(target)
        self.target, self.rid, self.inference.PROJECT = target, 3, target
        self.attempt = target / 'artifacts/repetition_fixture'
        self.prior_path, self.journal = self.attempt / 'prior.json', self.attempt / 'inference'
        self.protected = {str(p): component.sha(p) for p in (target / 'data').rglob('*.json')}
        # One prior real mocked HTTP; this is synthetic run3, never copied real responses.
        isolated = types.SimpleNamespace(**{**vars(self.inference), 'payload': expanded.payload,
            'account': __import__('repeated_runs').shared_account(self.inference.account, self.root)})
        _, cell, _ = __import__('authorized_recovery_200').make_wrapper(self.root, 'core').authorized_build(isolated, 3)
        with patch('urllib.request.urlopen', side_effect=self.http):
            cell('jev', 'S1', 'abt-buy', 'prompt', limit=1, workers=8)
        path = target / 'runs/quality/jev/S1/abt-buy/prompt.jsonl'
        old = path.read_bytes()
        self.bodies = []
        self.prepare()
        with patch('urllib.request.urlopen', side_effect=self.http) as http:
            self.execute()
        proof = self.reconcile()
        self.assertEqual((http.call_count, proof['attempts'], proof['prior_responses']), (55, 55, 1))
        self.assertTrue(path.read_bytes().startswith(old))
        self.assertTrue(all(json.loads(line)['study_run_id'] == 3
            for p in (target / 'runs').rglob('prompt.jsonl') for line in p.read_text().splitlines()))
        expected = [func(c, m, v) for _, ns, m, v, b, _ in component.matrix(3)
                    for func in [expanded.payload if ns == 'quality' else schema.payload]
                    for c in self.cases[b] if not (m == 'jev' and v == 'S1' and c['pair_id'] == '0')]
        self.assertCountEqual(self.bodies, expected)
        self.assertAlmostEqual(json.loads(self.budget.read_text())['spent'], 100.056)

    def test_unknown_zero_outstanding_keeps_credit(self):
        self.small()
        self.prepare()
        first = True
        def broken(request, **kwargs):
            nonlocal first
            if first:
                first = False
                raise http.client.IncompleteRead(b'partial')
            return self.http(request, **kwargs)
        with patch('urllib.request.urlopen', side_effect=broken), self.assertRaisesRegex(ValueError, 'Unknown'):
            self.execute()
        self.assertEqual(self.credit.outstanding, 0)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            self.reconcile()
        self.assertFalse(self.credit.closed)
        self.assertEqual(json.loads(self.budget.read_text())['reserved'], 11)

    def test_readable_storage_failure_holds_credit(self):
        self.small()
        self.prepare()
        grouped = __import__('grouped_inference')
        real = grouped.write_group
        first = True
        def broken(path, lines):
            nonlocal first
            real(path, lines)
            if first:
                first = False
                raise OSError('synthetic unconfirmed fsync')
        with patch.object(grouped, 'write_group', side_effect=broken), patch('urllib.request.urlopen', side_effect=self.http):
            self.execute()
        with self.assertRaisesRegex(ValueError, 'durability'):
            self.reconcile()
        self.assertFalse(self.credit.closed)

    def test_missing_raw_model_end_foreign_run_and_prior_corruption(self):
        self.small()
        self.prepare()
        with patch('urllib.request.urlopen', side_effect=self.http):
            self.execute()
        p = self.target / 'runs/quality/jev/S1/abt-buy/prompt.jsonl'
        old = p.read_bytes()
        p.unlink()
        with self.assertRaisesRegex(ValueError, 'delta'):
            self.reconcile()
        p.write_bytes(old)
        data = [json.loads(s) for s in old.splitlines()]
        data[0]['study_run_id'] = True
        p.write_text(''.join(json.dumps(x) + '\n' for x in data))
        with self.assertRaisesRegex(ValueError, 'study run'):
            self.reconcile()
        p.write_bytes(old)
        p = self.journal / 'models.jsonl'
        old = p.read_bytes()
        p.write_bytes(old[:old.rfind(b'\n', 0, -1)+1])
        with self.assertRaisesRegex(ValueError, 'model ends'):
            self.reconcile()
        p.write_bytes(old)
        self.prior_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'prior'):
            self.reconcile()
        self.assertFalse(self.credit.closed)

    def test_budget_stop_exact_durable_cost_not_a_success(self):
        self.small()
        value = json.loads(self.budget.read_text())
        value['spent'] = 196.95
        self.budget.write_text(json.dumps(value))
        self.prepare()
        with patch('urllib.request.urlopen', side_effect=self.http) as http:
            result = self.execute()
        self.assertFalse(result['inference_complete'])
        proof = self.reconcile()
        self.assertEqual(proof['attempts'], http.call_count)
        self.assertTrue(self.credit.closed)
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'], 3)

    def test_parser_failure_drains_remaining_models_no_auto_retry(self):
        self.small()
        self.prepare()
        def broken(request, **kwargs):
            response = self.http(request, **kwargs)
            raw = json.loads(response.getvalue())
            if json.loads(request.data)['model'] == protocol.MODELS['jev']['id']:
                raw.pop('answers')
            return Response(json.dumps(raw).encode())
        with patch('urllib.request.urlopen', side_effect=broken):
            result = self.execute()
        self.assertEqual(sum(m['status'] == 'complete' for m in result['models']), 6)
        self.assertFalse(result['inference_complete'])
        proof = self.reconcile()
        self.assertEqual(proof['attempts'], 50)
        self.assertAlmostEqual(proof['charged'], .05)
        self.assertFalse(proof['inference_complete'])

    def test_funding_provenance_and_cell_scope_guards(self):
        self.small()
        self.prepare()
        value = json.loads(self.budget.read_text())
        value['inference_credits'][self.credit.identity]['provenance']['stage'] = 'core'
        self.budget.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Credit does not bind'):
            self.execute()
        self.assertFalse(self.journal.exists())
        for rid in (1, True, '2'):
            with self.assertRaises(ValueError):
                component.matrix(rid)
        # No test/pilot/B0/calibration condition can enter the fixed matrix.
        self.assertTrue(all(c[5] == 'prompt' and c[1] in ('quality', 'schema')
                            and c[3] != 'B0' for c in component.matrix(2)))

    def test_input_change_before_calls_keeps_unused_credit(self):
        self.small()
        self.prepare()
        path = self.target / 'data/abt-buy/prompt.json'
        data = json.loads(path.read_text())
        data[0]['label'] = 1
        path.write_text(json.dumps(data))
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')), self.assertRaisesRegex(ValueError, 'sources'):
            self.execute()
        self.assertEqual(self.credit.attempts, 0)
        self.assertFalse(self.credit.closed)


if __name__ == '__main__':
    unittest.main()
