import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import protocol
import expanded_protocol
import expanded_stability
from fixed_extension_cost_plan import matrix, key, source


class FixedExtensionCosts(unittest.TestCase):
    def test_only_incremental_requests_and_distinct_repeat_namespaces(self):
        cells = matrix(['jev', 'luna'],
            {'models': ['jev', 'luna'], 'benchmarks': ['a'], 'variants': ['B0', 'X_P_identity_0']},
            {'models': ['jev'], 'benchmarks': {'a': {'extra_n': 3}, 'b': {'extra_n': 0}}},
            {'models': ['jev', 'luna'], 'benchmarks': ['a'],
             'repeats': {'fresh_namespaces': ['e3_repeat_1', 'e3_repeat_2']}})
        self.assertEqual(len(cells), 15)
        self.assertEqual(len({key(c) for c in cells}), 15)
        self.assertEqual(sum(c['stage'] == 'calibration' for c in cells), 1)
        self.assertEqual(sum(c['namespace'].startswith('e3_repeat_') for c in cells), 4)
        self.assertFalse(any(c['split'] in ('prompt', 'route') for c in cells))
        with self.assertRaises(ValueError):
            matrix(['jev', 'jev'], {'models': []}, {'models': []}, {'models': []})

    def fixture(self, root, variant='I3_domain_json'):
        c = {'stage': 'stability', 'namespace': 'quality', 'model': 'jev', 'variant': variant,
             'benchmark': 'abt-buy', 'split': 'expanded_screen'}
        case = {'pair_id': 'one', 'records': {'left': {'title': 'A'}, 'right': {'title': 'B'}},
                'benchmark': 'abt-buy', 'label': 1}
        case['input_hash'] = protocol.digest(case['records'])
        body = expanded_stability.payload(case, 'jev', variant)
        row = {'pair_id': 'one', 'model_alias': 'jev', 'variant': variant, 'benchmark': 'abt-buy',
               'input_hash': case['input_hash'], 'request_hash': protocol.digest(body),
               'prediction': 1, 'probability_match': .8,
               'attempts': [{'status': 200, 'cost': .02, 'seconds': 1,
                    'response': {'model': protocol.MODELS['jev']['id'], 'provider': 'typesafe',
                    'answers': {'match': {'type': 'noul', 'noul': .8}}, 'usage': {'cost': .02}}}]}
        data = root/'data/abt-buy/expanded_screen.json'
        data.parent.mkdir(parents=True)
        data.write_text(json.dumps([case]))
        path = root/'runs/quality/jev'/variant/'abt-buy/expanded_screen.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row)+'\n')
        return c, row, path

    def test_real_frozen_payload_parser_retry_and_failure_costs(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(socket, 'socket', side_effect=AssertionError('HTTP forbidden')):
            root = Path(temp)
            c, row, path = self.fixture(root)
            retry = {'status': 503, 'cost': None, 'seconds': 1, 'error': 'unavailable', 'response': {}}
            row['attempts'].insert(0, retry)
            path.write_text(json.dumps(row)+'\n')
            failed = {**row, 'error': 'failed', 'attempts': [{**retry, 'cost': .01}]}
            path.with_name('expanded_screen.failures.jsonl').write_text(json.dumps(failed)+'\n')
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = source(root, c, expanded_stability, protocol)
            self.assertEqual(result['responses'], 1)
            self.assertEqual(result['attempts'], 3)
            self.assertAlmostEqual(result['known_proxy_usd'], .03)
            self.assertEqual(result['historical_unknown_scenario_usd'], .05)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob('*') if p.is_file()})
            # Plain domain text is a different wire request than domain JSON.
            with self.assertRaises(ValueError):
                source(root, c, expanded_protocol, protocol)

    def test_raw_input_request_parser_identity_and_completion_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c, row, path = self.fixture(root)
            for field, value in [('request_hash', 'drift'), ('input_hash', 'drift'), ('study_run_id', 2),
                                 ('prediction', 0), ('probability_match', .3)]:
                path.write_text(json.dumps({**row, field: value})+'\n')
                with self.subTest(field=field), self.assertRaises(ValueError):
                    source(root, c, expanded_stability, protocol)
            bad = copy.deepcopy(row)
            bad['attempts'][0]['response']['provider'] = 'other'
            path.write_text(json.dumps(bad)+'\n')
            with self.assertRaises(ValueError):
                source(root, c, expanded_stability, protocol)
            for content in ('', (json.dumps(row)+'\n')*2):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    source(root, c, expanded_stability, protocol)

    def test_unknown_operations_block_but_unattempted_reservation_is_not_charged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c, row, path = self.fixture(root)
            event = {k: v for k, v in row.items() if k not in ('attempts', 'prediction', 'probability_match')}
            event.update(study_run_id=1, completed_at='synthetic', error='Project spending ceiling reached',
                         operational_stop={'phase': 'reserve', 'exception_type': 'RuntimeError',
                            'previous_attempts_preserved': 0, 'budget_reservation_rejected': True,
                            'outcome_unknown': False})
            operations = path.with_name('expanded_screen.operations.jsonl')
            operations.write_text(json.dumps(event)+'\n')
            result = source(root, c, expanded_stability, protocol)
            self.assertEqual(result['attempts'], 1)
            self.assertEqual(result['known_proxy_usd'], .02)
            self.assertEqual(result['historical_unknown_scenario_usd'], 0)
            event['error'] = 'worker outcome unknown'
            event['operational_stop'].update(phase='future', budget_reservation_rejected=False, outcome_unknown=True)
            operations.write_text(json.dumps(event)+'\n')
            with self.assertRaisesRegex(ValueError, 'Unresolved source operational outcome'):
                source(root, c, expanded_stability, protocol)


if __name__ == '__main__':
    unittest.main()
