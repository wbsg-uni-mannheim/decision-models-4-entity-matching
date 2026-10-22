import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import analyze
import expanded_confirmation as ec
import protocol
import run
import confirmation_drained as component


class Response(io.BytesIO):
    status = 200


class ConfirmationDrainTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        for name in ('artifacts', 'configs'):
            (self.root/name).mkdir()
        self.cases = {}
        for b in list(ec.BENCHMARKS)+['billiger-de-50un']:
            case = {'benchmark': b, 'pair_id': b+'#a', 'label': 1,
                    'records': {'left': {'name': 'A'}, 'right': {'name': 'B'}}}
            case['input_hash'] = protocol.digest(case['records'])
            self.cases[b] = case
            directory = self.root/'data'/b
            directory.mkdir(parents=True)
            for split in ('prompt', 'test'):
                (directory/(split+'.json')).write_text(json.dumps([case]))
        scores = {}
        for m in ec.MODELS:
            for b in ec.BENCHMARKS:
                for v in ec.PROMPT_VARIANTS+ec.SERIALIZATIONS:
                    case = self.cases[b]
                    directory = self.root/'runs/quality'/m/v/b
                    directory.mkdir(parents=True)
                    prediction = int(not ((m == 'jev' and v == 'B0') or
                                          (m == 'clef' and v in ('B0', 'S1'))))
                    row = {'pair_id': case['pair_id'], 'prediction': prediction,
                           'probability_match': None if m == 'luna' else .8,
                           'request_hash': protocol.digest(ec.payload(case, m, v)),
                           'attempts': [{'cost': 0., 'seconds': 0.}]}
                    (directory/'prompt.jsonl').write_text(json.dumps(row)+'\n')
                    scores[m, b, v] = float(prediction)
        selected = ec.make_selection(scores)
        self.cells = ec.confirmation_cells(selected)
        for name, value in [('expanded_selected_variants', selected), ('expanded_confirmation_cells', self.cells)]:
            (self.root/'configs'/(name+'.json')).write_text(json.dumps(value))
        self.confirmation = types.SimpleNamespace(**{**vars(ec), 'PROJECT': self.root, 'verify': lambda: None})
        self.analysis = types.SimpleNamespace(**{**vars(analyze), 'PROJECT': self.root})
        account = types.FunctionType(run.account.__code__, {**run.account.__globals__, 'PROJECT': self.root},
                                     argdefs=run.account.__defaults__)
        self.inference = types.SimpleNamespace(**{**vars(run), 'PROJECT': self.root,
                                                 'account': account, 'key': lambda: 'fake'})
        self.budget = self.root/'artifacts/budget.json'
        self.budget.write_text(json.dumps({'spent': 0., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0}))
        self.socket = patch('socket.socket', side_effect=AssertionError('No network'))
        self.socket.start(); self.addCleanup(self.socket.stop)
        self.journal = self.root/'artifacts/attempt'

    def execute(self):
        return component.run_selected_tests(self.root, self.confirmation, self.analysis,
                                            self.inference, protocol, self.journal)

    def test_complete_synthetic_matrix_real_request_parser_account_and_cache(self):
        seen = []
        def http(req, **kw):
            body = json.loads(req.data)
            seen.append(body)
            m = next(m for m, spec in protocol.MODELS.items() if spec['id'] == body['model'])
            raw = {'model': body['model'], 'provider': protocol.MODELS[m]['provider'].split('/')[0],
                   'usage': {'cost': .001}}
            if m == 'luna':
                raw['choices'] = [{'finish_reason': 'stop', 'message': {'content': '{"match": true}'}}]
            else:
                raw['answers'] = {'match': {'type': 'noul', 'noul': .8}}
            return Response(json.dumps(raw).encode())
        original_payloads = (run.payload, analyze.payload)
        before = {p: p.read_bytes() for p in (self.root/'configs').glob('*.json')}
        with patch('urllib.request.urlopen', side_effect=http):
            result = self.execute()
        self.assertEqual(len(seen), len(self.cells))
        self.assertTrue({'B0', 'S1', 'X_R_aligned'} <= {v for _, v, _ in self.cells})
        self.assertCountEqual([protocol.digest(x) for x in seen],
                             [protocol.digest(ec.payload(self.cases[b], m, v)) for m, v, b in self.cells])
        self.assertEqual((run.payload, analyze.payload), original_payloads)
        self.assertTrue(all(p.read_bytes() == raw for p, raw in before.items()))
        self.assertTrue(all(x['status'] == 'complete' for x in result['models']))
        self.assertFalse(result['completion_marker_written'])
        self.assertFalse((self.root/'artifacts/expanded_confirmation_complete.json').exists())
        budget = json.loads(self.budget.read_text())
        self.assertAlmostEqual(budget['spent'], len(self.cells)*.001)
        self.assertAlmostEqual(budget['reserved'], 3.)
        for m, v, b in self.cells:
            row = json.loads((self.root/'runs/quality'/m/v/b/'test.jsonl').read_text())
            self.assertEqual(row['study_run_id'], 1)
            self.assertEqual(row['request_hash'], protocol.digest(ec.payload(self.cases[b], m, v)))
        with self.assertRaisesRegex(ValueError, 'Already attempted'):
            self.execute()

    def test_all_budget_stops_drained_without_charges_or_foreign_hold_release(self):
        self.budget.write_text(json.dumps({'spent': 97., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0}))
        before = self.budget.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')) as http:
            result = self.execute()
        http.assert_not_called()
        self.assertEqual(self.budget.read_bytes(), before)
        self.assertEqual(len(result['models']), 7)
        self.assertTrue(all(r['status'] == 'failed' for r in result['models']))
        self.assertEqual(sum(r['budget_rejections'] for r in result['operations']), 7)
        self.assertEqual(sum(r['additional_paid_attempts_counted'] for r in result['operations']), 0)
        self.assertEqual(len(list((self.root/'runs').rglob('*.operations.jsonl'))), 7)
        self.assertFalse(list((self.root/'runs').rglob('*.failures.jsonl')))
        rows = [json.loads(x) for x in (self.journal/'models.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['models_drained'], 7)
        self.assertEqual(rows[-1]['status'], 'failed')

    def test_selection_drift_and_incomplete_development_rejected_before_writes(self):
        path = self.root/'configs/expanded_confirmation_cells.json'
        original = path.read_bytes()
        path.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'matrix mismatch'):
            self.execute()
        self.assertFalse(self.journal.exists())
        path.write_bytes(original)
        (self.root/'runs/quality/jev/B0/abt-buy/prompt.jsonl').write_text('')
        with self.assertRaisesRegex(AssertionError, 'Incomplete cell'):
            self.execute()
        self.assertFalse(self.journal.exists())

    def test_wrong_source_workspace_or_completion_refused(self):
        clone = copy.copy(self.confirmation)
        clone.__file__ = __file__
        with self.assertRaisesRegex(ValueError, 'source changed'):
            component.selected_tests(self.root, clone, self.analysis)
        with self.assertRaisesRegex(ValueError, 'workspace'):
            component.selected_tests(self.root/'other', self.confirmation, self.analysis)
        (self.root/'artifacts/expanded_confirmation_complete.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'already complete'):
            self.execute()
        self.assertFalse(self.journal.exists())


if __name__ == '__main__':
    unittest.main()
