import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import protocol
from export_run_metrics import check_response_origin
from live_cascade_export import verify_cell


def fixture():
    case = {'pair_id': 'p', 'input_hash': 'h', 'benchmark': 'wdc',
            'records': {'left': {'title': 'A'}, 'right': {'title': 'B'}}}
    entry = {'model': 'jev', 'benchmark': 'wdc', 'strategy': 'symmetric', 'peer': None}
    response = {'model': protocol.MODELS['jev']['id'],
                'provider': protocol.MODELS['jev']['provider'].split('/')[0],
                'answers': {'match': {'type': 'noul', 'noul': .8}}, 'usage': {'cost': .001}}
    call = {'pair_id': 'p', 'input_hash': 'h', 'benchmark': 'wdc', 'model_alias': 'jev',
            'variant': 'B0', 'request_hash': protocol.digest(protocol.payload(case, 'jev', 'B0')),
            'prediction': 1, 'probability_match': .8, 'study_run_id': 3,
            'completed_at': '2026-10-14T18:00:00+00:00',
            'attempts': [{'response': response, 'cost': .001, 'seconds': 1.0}]}
    chain = {'chain_id': 'c', 'pair_id': 'p', 'input_hash': 'h', 'router_hash': protocol.digest(entry),
             'primary_request_hash': call['request_hash'], 'wall_seconds': 2.0,
             'failed': False, 'routed': False, 'prediction': 1, 'calls': [call]}
    journal = [{'chain_id': 'c', 'pair_id': 'p', 'router_hash': protocol.digest(entry), 'call_index': 0, 'call': call}]
    module = SimpleNamespace(route=lambda *args: [False])
    return entry, [case], [chain], [], journal, module, protocol, lambda rows: check_response_origin(rows, 3, '2026-10-14T17:00:00+00:00')


class LiveExportTests(unittest.TestCase):
    def test_frozen_payload_parser_and_run_identity_without_network(self):
        args = fixture()
        with patch('socket.socket', side_effect=AssertionError('No network')):
            result = verify_cell(*args)
        self.assertEqual(result['known_cost_usd_including_failures'], .001)
        self.assertEqual(result['end_to_end_p95_seconds'], 2)
        args[4][0]['call']['study_run_id'] = 2
        with self.assertRaisesRegex(ValueError, 'run response ID'):
            verify_cell(*args)

    def test_missing_success_and_journal_corruption_rejected(self):
        args = fixture(); args[2].clear()
        with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
            verify_cell(*args)
        args = fixture(); args[4].clear()
        with self.assertRaisesRegex(ValueError, 'durable call journal'):
            verify_cell(*args)

    def test_raw_parser_and_usage_cost_mismatch_rejected(self):
        args = fixture(); args[4][0]['call']['prediction'] = 0
        with self.assertRaisesRegex(ValueError, 'raw response'):
            verify_cell(*args)
        args = fixture(); args[4][0]['call']['attempts'][0]['cost'] = .002
        with self.assertRaisesRegex(ValueError, 'returned usage'):
            verify_cell(*args)

    def test_routing_and_final_decision_checked(self):
        args = fixture(); args[5].route = lambda *args: [True]
        with self.assertRaisesRegex(ValueError, 'route or stage count'):
            verify_cell(*args)
        args = fixture(); args[2][0]['prediction'] = 0
        with self.assertRaisesRegex(ValueError, 'final prediction'):
            verify_cell(*args)

    def test_failure_and_orphan_costs_retained(self):
        args = fixture()
        failed = copy.deepcopy(args[2][0]); failed.update(chain_id='failed', failed=True, prediction=None)
        failed['calls'][0].update(error='provider_unavailable')
        failed['calls'][0]['attempts'] = [{'cost': None, 'seconds': 3, 'response': {'error': {'code': 502}}}]
        args[3].append(failed)
        args[4].append({**args[4][0], 'chain_id': 'failed', 'call': failed['calls'][0]})
        args[4].append({**copy.deepcopy(args[4][0]), 'chain_id': 'orphan'})
        result = verify_cell(*args)
        self.assertEqual(result['failed_chains'], 1)
        self.assertEqual(result['interrupted_chains_with_journaled_calls'], 1)
        self.assertEqual(result['known_cost_usd_including_failures'], .002)
        self.assertEqual(result['unknown_cost_attempts'], 1)
        self.assertEqual(result['journaled_calls'], 3)


if __name__ == '__main__':
    unittest.main()
