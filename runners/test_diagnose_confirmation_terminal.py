import copy
import threading
import unittest
from unittest.mock import patch
from diagnose_confirmation_terminal import audit_confirmation_rows
import expanded_confirmation as ec
import protocol


class ConfirmationAuditTests(unittest.TestCase):
    def setUp(self):
        self.case = {'benchmark': 'abt-buy', 'pair_id': 'p', 'label': 1,
                     'records': {'left': {'name': 'A'}, 'right': {'name': 'B'}}}
        self.case['input_hash'] = protocol.digest(self.case['records'])
        raw = {'model': protocol.MODELS['jev']['id'], 'provider': 'typesafe',
               'usage': {'cost': .01}, 'answers': {'match': {'type': 'noul', 'noul': .8}}}
        self.row = {'pair_id': 'p', 'model_alias': 'jev', 'variant': 'X_P_detailed_1',
                    'benchmark': 'abt-buy', 'input_hash': self.case['input_hash'],
                    'request_hash': protocol.digest(ec.payload(self.case, 'jev', 'X_P_detailed_1')),
                    'prediction': 1, 'probability_match': .8,
                    'attempts': [{'status': 200, 'cost': .01, 'seconds': 1, 'response': raw}]}

    def audit(self, success, failures=()):
        with patch('socket.socket', side_effect=AssertionError('No network')):
            return audit_confirmation_rows([self.case], success, list(failures), ec, protocol,
                                           'jev', 'X_P_detailed_1', 'abt-buy')

    def test_exact_payload_parser_and_legacy_attribution_preserved(self):
        before = copy.deepcopy(self.row)
        self.assertEqual(self.audit([self.row])['known_usd'], .01)
        self.assertEqual(self.row, before)
        self.assertEqual(self.audit([])['missing'], 1)
        for update in ({'study_run_id': 2}, {'study_run_id': True}, {'variant': 'B0'},
                       {'request_hash': 'wrong'}, {'prediction': 0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.audit([{**self.row, **update}])
        with self.assertRaises(ValueError):
            self.audit([self.row, self.row])

    def test_preserved_retry_costs_and_unknowns(self):
        failure = {**self.row, 'error': 'HTTP 499', 'attempts': [
            {'cost': .003, 'seconds': 1, 'status': 503, 'error': 'HTTP 503'},
            {'cost': None, 'seconds': 1, 'status': 499, 'error': 'HTTP 499'}]}
        result = self.audit([self.row], [failure])
        self.assertAlmostEqual(result['known_usd'], .013)
        self.assertEqual((result['unknown_attempts'], result['failure_rows']), (1, 1))

    def test_actual_dispatcher_only_propagates_first_model_error(self):
        models = list(ec.MODELS)
        barrier = threading.Barrier(len(models), timeout=10)
        seen = []
        def cell(model, *args, **kwargs):
            seen.append(model)
            barrier.wait()
            raise RuntimeError('synthetic-'+model)
        with patch.object(ec.run, 'run_cell', side_effect=cell), \
                patch('socket.socket', side_effect=AssertionError('No network')):
            with self.assertRaisesRegex(RuntimeError, '^synthetic-'+models[0]+'$'):
                ec.run_cells([(m, 'B0', 'abt-buy') for m in models], 'test')
        self.assertCountEqual(seen, models)


if __name__ == '__main__':
    unittest.main()
