import concurrent.futures
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
import analyze
import expanded_schema as es
import protocol
import run
import schema_drained as component
from schema_run_export import selected_matrix


class Response(io.BytesIO):
    status = 200


def raw_response(model, variant, positive=True):
    labels = es.options(variant)
    selected = 'Review' if variant == 'C_review' else (
        ('Same entity' if variant.startswith('C_same') else 'Match') if positive else
        ('Different entities' if variant.startswith('C_same') else 'No Match'))
    raw = {'model': protocol.MODELS[model]['id'],
           'provider': protocol.MODELS[model]['provider'].split('/')[0], 'usage': {'cost': .001}}
    if model == 'luna':
        raw['choices'] = [{'finish_reason': 'stop', 'message': {'content': json.dumps({'match': selected})}}]
    else:
        probs = {s: (.8 if s == selected else .2/(len(labels)-1)) for s in labels}
        raw['answers'] = {'match': {'type': 'choice', 'choice': selected, 'probabilities': probs}}
    return raw


class SchemaDrainTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        for d in ('configs', 'artifacts'):
            (self.root/d).mkdir()
        self.cases = {}
        for b in [*es.BENCHMARKS, 'billiger-de-50un']:
            case = {'benchmark': b, 'pair_id': b+'#a', 'label': 1,
                    'records': {'left': {'z': 'Ä', 'a': 'B'}, 'right': {'z': 'A', 'a': 'B'}}}
            case['input_hash'] = protocol.digest(case['records'])
            self.cases[b] = case
            d = self.root/'data'/b; d.mkdir(parents=True)
            for split in ('prompt', 'test', 'schema_pilot'):
                (d/(split+'.json')).write_text(json.dumps([case]))
        self.pilot_families = ['abt-buy', 'dblp-scholar']
        pilot = []
        for mi, m in enumerate(es.MODELS):
            for v in es.CONDITIONS:
                for b in es.BENCHMARKS:
                    self.write_source(m, v, b, 'prompt', 'schema', v == es.BINARY[mi % 4])
                    if b in self.pilot_families:
                        self.write_source(m, v, b, 'schema_pilot', 'schema_pilot', True)
                        pilot.append({'model': m, 'condition': v, 'benchmark': b, 'status': 'passed'})
        (self.root/'artifacts/schema_pilot_status.json').write_text(json.dumps({'complete': True, 'cells': pilot}))
        self.schema = types.SimpleNamespace(**{**vars(es), 'PROJECT': self.root,
            'verify': lambda: {'pilot': {'families': self.pilot_families}}})
        loader = types.FunctionType(es.load.__code__, {**es.load.__globals__, 'PROJECT': self.root}, argdefs=es.load.__defaults__)
        dev = {(m,v,b): loader(m,v,b,'prompt') for m in es.MODELS for v in es.CONDITIONS for b in es.BENCHMARKS}
        selected, self.cells = selected_matrix(es, dev, analyze.metric)
        for name, value in [('expanded_schema_selection', selected), ('expanded_schema_test_cells', self.cells)]:
            (self.root/'configs'/(name+'.json')).write_text(json.dumps(value))
        account = types.FunctionType(run.account.__code__, {**run.account.__globals__, 'PROJECT': self.root}, argdefs=run.account.__defaults__)
        self.inference = types.SimpleNamespace(**{**vars(run), 'PROJECT': self.root, 'account': account, 'key': lambda: 'fake'})
        self.budget = self.root/'artifacts/budget.json'
        self.budget.write_text(json.dumps({'spent': 0., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0}))
        self.journal = self.root/'artifacts/attempt'
        sock = patch('socket.socket', side_effect=AssertionError('No network')); sock.start(); self.addCleanup(sock.stop)

    def write_source(self, m, v, b, split, namespace, positive):
        case = self.cases[b]; raw = raw_response(m, v, positive)
        pred, prob = es.parse(m, raw, v)
        row = {'pair_id': case['pair_id'], 'benchmark': b, 'input_hash': case['input_hash'],
               'model_alias': m, 'variant': v, 'study_run_id': 1, 'prediction': pred, 'probability_match': prob,
               'request_hash': es.wire_hash(es.payload(case, m, v)), 'attempts': [
                   {'status': 200, 'cost': .001, 'seconds': 0., 'error': None, 'response': raw}]}
        d = self.root/'runs'/namespace/m/v/b; d.mkdir(parents=True, exist_ok=True)
        (d/(split+'.jsonl')).write_text(json.dumps(row)+'\n')

    def execute(self):
        return component.run_selected_tests(self.root, self.schema, analyze, self.inference, self.journal)

    def http(self, req, **kwargs):
        body = json.loads(req.data)
        m = next(m for m, s in protocol.MODELS.items() if s['id'] == body['model'])
        labels = (list(body['questions']['match']['criteria']) if m != 'luna' else
                  body['response_format']['json_schema']['schema']['properties']['match']['enum'])
        v = next(v for v in es.CONDITIONS if es.options(v) == labels)
        return Response(json.dumps(raw_response(m, v)).encode())

    def test_complete_real_wire_parser_account_cache_matrix(self):
        original = (run.payload, run.digest, run.parse_response, run.request, es.CONTEXT, es.ORIGINAL_REQUEST)
        before = {p: p.read_bytes() for p in (self.root/'runs').rglob('*.jsonl')}
        sent = []
        def http(req, **kwargs):
            sent.append(req.data); return self.http(req, **kwargs)
        with patch('urllib.request.urlopen', side_effect=http):
            result = self.execute()
        self.assertTrue(all(x['status'] == 'complete' for x in result['models']))
        self.assertEqual(len(sent), 126)
        self.assertCountEqual(sent, [es.wire(es.payload(self.cases[b], m, v)).encode() for m,v,b in self.cells])
        self.assertEqual({v for _,v,_ in self.cells}, set(es.CONDITIONS))
        for m,v,b in self.cells:
            row = json.loads((self.root/'runs/schema'/m/v/b/'test.jsonl').read_text())
            self.assertEqual(row['study_run_id'], 1)
            self.assertEqual(row['request_hash'], es.wire_hash(es.payload(self.cases[b],m,v)))
            self.assertEqual(row['prediction'], -1 if v == 'C_review' else 1)
        self.assertTrue(all(p.read_bytes() == data for p,data in before.items()))
        self.assertEqual(original, (run.payload, run.digest, run.parse_response, run.request, es.CONTEXT, es.ORIGINAL_REQUEST))
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'], 3.)
        self.assertFalse((self.root/'artifacts/expanded_schema_complete.json').exists())
        with self.assertRaisesRegex(ValueError, 'Already attempted'):
            self.execute()

    def test_all_35_model_condition_requests_overlap_with_private_context(self):
        request, _, _ = component.build_schema(self.schema, self.inference)
        barrier = threading.Barrier(35)
        def http(req, **kwargs):
            barrier.wait(timeout=10); return self.http(req, **kwargs)
        keys = [(m,v) for m in es.MODELS for v in es.CONDITIONS]
        with patch('urllib.request.urlopen', side_effect=http), concurrent.futures.ThreadPoolExecutor(max_workers=35) as pool:
            rows = list(pool.map(lambda k: request(self.cases['abt-buy'], *k, 'fake'), keys))
        for (m,v), row in zip(keys, rows):
            self.assertNotIn('error', row)
            self.assertEqual(row['prediction'], -1 if v == 'C_review' else 1)
            self.assertEqual(row['request_hash'], es.wire_hash(es.payload(self.cases['abt-buy'],m,v)))

    def test_budget_stops_all_models_before_http(self):
        self.budget.write_text(json.dumps({'spent': 97., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0}))
        before = self.budget.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')) as http:
            result = self.execute()
        http.assert_not_called()
        self.assertEqual(self.budget.read_bytes(), before)
        self.assertEqual(len(result['models']), 7)
        self.assertTrue(all(r['status'] == 'failed' for r in result['models']))
        self.assertEqual(sum(r['budget_rejections'] for r in result['operations']), 7)
        self.assertFalse(list((self.root/'runs').rglob('*.failures.jsonl')))

    def test_prior_retry_kept_once_when_next_reserve_fails(self):
        self.budget.write_text(json.dumps({'spent': 96.95, 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0}))
        _, cell, _ = component.build_schema(self.schema, self.inference)
        err = urllib.error.HTTPError('https://mock', 503, 'Unavailable', {}, io.BytesIO(b'{}'))
        with patch('urllib.request.urlopen', side_effect=err) as http, patch('time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'failed batch'):
                cell('jev', 'C_same_reverse', 'abt-buy', 'test', namespace='schema')
        self.assertEqual(http.call_count, 1)
        directory = self.root/'runs/schema/jev/C_same_reverse/abt-buy'
        failure = json.loads((directory/'test.failures.jsonl').read_text())
        operation = json.loads((directory/'test.operations.jsonl').read_text())
        self.assertEqual(len(failure['attempts']), 1)
        self.assertEqual(failure['attempts'][0]['status'], 503)
        self.assertEqual({k:v for k,v in failure.items() if k != 'attempts'}, operation)
        self.assertEqual(operation['request_hash'], es.wire_hash(es.payload(self.cases['abt-buy'],'jev','C_same_reverse')))
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'], 3.)

    def test_budget_failure_drains_already_running_schema_sibling(self):
        self.budget.write_text(json.dumps({'spent': 96.95, 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0}))
        cases = [self.cases['abt-buy'], {**self.cases['abt-buy'], 'pair_id': 'second'}]
        (self.root/'data/abt-buy/test.json').write_text(json.dumps(cases))
        rejected = threading.Event()
        original_account = self.inference.account
        def account(*args, **kwargs):
            try:
                return original_account(*args, **kwargs)
            except RuntimeError:
                rejected.set(); raise
        inference = types.SimpleNamespace(**{**vars(self.inference), 'account': account})
        _, cell, _ = component.build_schema(self.schema, inference)
        def http(req, **kwargs):
            self.assertTrue(rejected.wait(timeout=10))
            return self.http(req, **kwargs)
        with patch('urllib.request.urlopen', side_effect=http) as network:
            with self.assertRaisesRegex(RuntimeError, 'failed batch'):
                cell('jev', 'C_same_reverse', 'abt-buy', 'test', workers=2, namespace='schema')
        self.assertEqual(network.call_count, 1)
        directory = self.root/'runs/schema/jev/C_same_reverse/abt-buy'
        success = json.loads((directory/'test.jsonl').read_text())
        operation = json.loads((directory/'test.operations.jsonl').read_text())
        self.assertNotEqual(success['pair_id'], operation['pair_id'])
        self.assertEqual(success['prediction'], 1)
        self.assertEqual(operation['operational_stop']['previous_attempts_preserved'], 0)
        self.assertFalse((directory/'test.failures.jsonl').exists())
        budget = json.loads(self.budget.read_text())
        self.assertAlmostEqual(budget['spent'], 96.951)
        self.assertAlmostEqual(budget['reserved'], 3.)

    def test_sorted_json_cache_hash_cannot_replace_ordered_wire_hash(self):
        self.write_source('jev', 'C_same_reverse', 'abt-buy', 'test', 'schema', True)
        path = self.root/'runs/schema/jev/C_same_reverse/abt-buy/test.jsonl'
        row = json.loads(path.read_text())
        row['request_hash'] = protocol.digest(es.payload(self.cases['abt-buy'], 'jev', 'C_same_reverse'))
        path.write_text(json.dumps(row)+'\n')
        _, cell, _ = component.build_schema(self.schema, self.inference)
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')) as http:
            with self.assertRaisesRegex(AssertionError, 'Cache configuration mismatch'):
                cell('jev', 'C_same_reverse', 'abt-buy', 'test', namespace='schema')
        http.assert_not_called()

    def test_invalid_probabilities_remain_rejected(self):
        request, _, _ = component.build_schema(self.schema, self.inference)
        raw = raw_response('jev', 'C_review')
        raw['answers']['match']['probabilities'] = {'Match': .07, 'No Match': .81, 'Review': .11}
        raw['answers']['match']['choice'] = 'No Match'
        with patch('urllib.request.urlopen', return_value=Response(json.dumps(raw).encode())):
            row = request(self.cases['abt-buy'], 'jev', 'C_review', 'fake')
        self.assertIn('do not sum to one', row['error'])
        self.assertEqual(len(row['attempts']), 1)
        self.assertEqual(row['attempts'][0]['cost'], .001)

    def test_missing_pilot_development_selection_source_and_completion_rejected(self):
        path = self.root/'artifacts/schema_pilot_status.json'; original = path.read_bytes()
        path.write_text('{"complete": false, "cells": []}')
        with self.assertRaisesRegex(ValueError, 'pilot incomplete'): self.execute()
        path.write_bytes(original)
        path = self.root/'runs/schema/jev/C_review/abt-buy/prompt.jsonl'; original = path.read_bytes()
        path.write_text('')
        with self.assertRaisesRegex(ValueError, 'Incomplete schema'): self.execute()
        path.write_bytes(original)
        path = self.root/'configs/expanded_schema_test_cells.json'; original = path.read_bytes()
        path.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'matrix mismatch'): self.execute()
        path.write_bytes(original)
        clone = copy.copy(self.schema); clone.__file__ = __file__
        with self.assertRaisesRegex(ValueError, 'source changed'): component.build_schema(clone, self.inference)
        (self.root/'artifacts/expanded_schema_complete.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'already complete'): self.execute()
        self.assertFalse(self.journal.exists())


if __name__ == '__main__':
    unittest.main()
