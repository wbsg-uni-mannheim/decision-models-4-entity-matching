import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import protocol
import run
from diagnose_fresh_baseline import audit_rows
from resume_schema_once import terminal_steps


class PartialAuditTests(unittest.TestCase):
    def setUp(self):
        self.case = {'benchmark': 'abt-buy', 'pair_id': 'a#b',
                     'records': {'left': {'name': 'x'}, 'right': {'name': 'x'}}}
        self.case['input_hash'] = protocol.digest(self.case['records'])
        self.response = {'model': 'typesafe/jev-1.13', 'provider': 'typesafe',
                         'answers': {'match': {'type': 'noul', 'noul': .8}}, 'usage': {'cost': .01}}
        self.row = {'pair_id': 'a#b', 'input_hash': self.case['input_hash'],
                    'request_hash': protocol.digest(protocol.payload(self.case, 'jev')),
                    'model_alias': 'jev', 'variant': 'B0', 'benchmark': 'abt-buy', 'study_run_id': 3,
                    'prediction': 1, 'probability_match': .8,
                    'attempts': [{'cost': .01, 'seconds': 1, 'status': 200, 'error': None,
                                  'response': self.response}]}

    def audit(self, successes, failures=None):
        return audit_rows([self.case], successes, failures or [], protocol, 'jev', 'abt-buy', 3)

    def test_partial_and_unknown_failure_are_not_completion(self):
        failure = copy.deepcopy(self.row)
        failure['error'] = 'HTTP 499'
        failure['attempts'][0].update(cost=None, error='HTTP 499', status=499)
        result = self.audit([], [failure])
        self.assertEqual((result['missing'], result['failure_rows'], result['unknown_attempts']), (1, 1, 1))
        self.assertFalse(result['complete'])
        self.assertEqual(self.audit([self.row])['known_usd'], .01)

    def test_identity_duplicate_and_raw_parser_rejections(self):
        for key in ('request_hash', 'input_hash', 'study_run_id', 'model_alias', 'probability_match'):
            with self.subTest(key=key):
                row = copy.deepcopy(self.row)
                row[key] = 'wrong'
                with self.assertRaises(ValueError):
                    self.audit([row])
        with self.assertRaises(ValueError):
            self.audit([self.row, self.row])
        row = copy.deepcopy(self.row)
        row['attempts'][0]['response']['usage']['cost'] = .02
        with self.assertRaises(ValueError):
            self.audit([row])

    def test_all_scheduler_steps_required(self):
        rows = '5|FAILED|1:0\n5.batch|FAILED|1:0\n5.extern|COMPLETED|0:0\n'
        terminal_steps(rows, '5')
        for changed in (rows.replace('COMPLETED', 'RUNNING'), rows.split('5.extern')[0]):
            with self.assertRaises(ValueError):
                terminal_steps(changed, '5')

    def test_frozen_batch_can_finish_a_response_without_journaling_after_raise(self):
        # Reproduce the limitation with the REAL frozen run_cell, no HTTP or money.
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            data = target/'data/abt-buy'
            data.mkdir(parents=True)
            (data/'prompt.json').write_text(json.dumps([self.case, {**self.case, 'pair_id': 'c#d'}]))
            completed = []
            raised = threading.Event()
            def request(case, *args):
                if case['pair_id'] == 'a#b':
                    raised.set()
                    raise RuntimeError('Project spending ceiling reached')
                raised.wait(timeout=2)
                time.sleep(.05)
                completed.append(case['pair_id'])
                return {**self.row, 'pair_id': case['pair_id']}
            with patch.object(run, 'PROJECT', target), patch.object(run, 'key', return_value='fake'), \
                    patch.object(run, 'request', side_effect=request), \
                    patch('socket.socket', side_effect=AssertionError('Network forbidden')):
                with self.assertRaisesRegex(RuntimeError, 'spending ceiling'):
                    run.run_cell('jev', 'B0', 'abt-buy', 'prompt', workers=2)
            self.assertEqual(completed, ['c#d'])
            self.assertEqual(list((target/'runs').rglob('*.jsonl')), [])


if __name__ == '__main__':
    unittest.main()
