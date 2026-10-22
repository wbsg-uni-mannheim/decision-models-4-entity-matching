import copy
import unittest
import threading
from unittest.mock import patch
from diagnose_schema_terminal import audit_schema_rows
import expanded_schema as es


class SchemaAuditTests(unittest.TestCase):
    def setUp(self):
        self.case = {'benchmark': 'wdc', 'pair_id': 'p', 'label': 1,
                     'records': {'left': {'title': 'A'}, 'right': {'title': 'B'}}}
        self.case['input_hash'] = es.protocol.digest(self.case['records'])
        raw = {'model': es.MODELS['jev']['id'], 'provider': es.MODELS['jev']['provider'].split('/')[0],
               'usage': {'cost': .001}, 'answers': {'match': {'type': 'choice', 'choice': 'Match',
                                                           'probabilities': {'No Match': .2, 'Match': .8}}}}
        self.row = {'pair_id': 'p', 'input_hash': self.case['input_hash'], 'model_alias': 'jev',
                    'variant': 'C_match_reverse', 'benchmark': 'wdc',
                    'request_hash': es.wire_hash(es.payload(self.case, 'jev', 'C_match_reverse')),
                    'prediction': 1, 'probability_match': .8,
                    'attempts': [{'cost': .001, 'status': 200, 'seconds': 1, 'response': raw}]}

    def audit(self, successes, failures=()):
        with patch('socket.socket', side_effect=AssertionError('No network')):
            return audit_schema_rows([self.case], successes, list(failures), es, 'jev', 'C_match_reverse', 'wdc')

    def test_real_wire_parser_and_partial_inventory(self):
        original = copy.deepcopy(self.row)
        result = self.audit([self.row])
        self.assertEqual(result['known_usd'], .001)
        self.assertTrue(result['complete'])
        self.assertEqual(self.row, original)
        self.assertEqual(self.audit([])['missing'], 1)
        self.row['request_hash'] = es.wire_hash(es.payload(self.case, 'jev', 'C_match'))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.audit([self.row])

    def test_foreign_run_variant_duplicate_and_raw_mismatch(self):
        for update in ({'study_run_id': 2}, {'study_run_id': True}, {'variant': 'C_match'}, {'prediction': 0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.audit([{**self.row, **update}])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.audit([self.row, self.row])

    def test_real_schema_dispatcher_exposes_only_first_model_exception(self):
        models = list(es.MODELS)
        barrier = threading.Barrier(len(models), timeout=10)
        observed = []
        def cell(model, variant, benchmark, split, **kwargs):
            if split == 'prompt':
                return
            observed.append(model)
            barrier.wait()
            raise RuntimeError('synthetic-'+model)
        with patch('sys.argv', ['expanded_schema.py', 'run']), \
                patch.object(es, 'verify'), patch.object(es, 'install'), \
                patch.object(es, 'pilot'), patch.object(es, 'forecast'), \
                patch.object(es, 'select', return_value=[(m, 'C_match', 'wdc') for m in models]), \
                patch.object(es.run, 'run_cell', side_effect=cell), patch.object(es, 'report') as report, \
                patch('socket.socket', side_effect=AssertionError('No network')):
            with self.assertRaisesRegex(RuntimeError, '^synthetic-'+models[0]+'$'):
                es.main()
        self.assertCountEqual(observed, models)
        report.assert_not_called()

    def test_preserved_unknown_failure_and_paid_parser_failure(self):
        failure = {**self.row, 'error': 'HTTP 499', 'attempts': [
            {'cost': None, 'seconds': 1, 'status': 499, 'error': 'HTTP 499'}]}
        result = self.audit([self.row], [failure])
        self.assertEqual(result['unknown_attempts'], 1)
        self.assertEqual(result['known_usd'], .001)
        failure = copy.deepcopy(self.row)
        failure['error'] = 'ValueError: Choice probabilities do not sum to one'
        failure['attempts'][0]['error'] = failure['error']
        failure['attempts'][0]['response']['answers']['match']['probabilities']['Match'] = .79
        self.assertEqual(self.audit([], [failure])['failure_rows'], 1)


if __name__ == '__main__':
    unittest.main()
