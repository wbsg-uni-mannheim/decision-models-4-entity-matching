import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from unittest.mock import Mock, patch
import numpy as np
from schema_run_export import condition_rows, selected_matrix, schema_cell, export_schema


def metric(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return dict(f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0)


def data(pred=(0, 1, 0, 1)):
    return {'cases': [{'pair_id': str(i), 'input_hash': str(i)} for i in range(4)],
            'y': np.array([0, 1, 0, 1]), 'pred': np.array(pred),
            'known_cost_usd': .04, 'unknown_cost_attempts': 1}


def fixture(root):
    for path in ['configs', 'artifacts']:
        (root/path).mkdir()
    module = SimpleNamespace(MODELS=['m'], BENCHMARKS=['billiger-de'],
                             BINARY=['C_same_reverse', 'C_same', 'C_match_reverse', 'C_match'])
    module.CONDITIONS = module.BINARY+['C_review']
    module.verify = Mock(return_value={'pilot': {'families': ['billiger-de']}})
    dev = {('m', v, 'billiger-de'): data() for v in module.CONDITIONS}
    selected, cells = selected_matrix(module, dev, metric)
    for name, value in [('expanded_schema_selection', selected), ('expanded_schema_test_cells', cells)]:
        (root/'configs'/(name+'.json')).write_text(json.dumps(value))
    (root/'artifacts/expanded_schema_complete.json').write_text(json.dumps({'status': 'complete', 'test_cells': len(cells)}))
    (root/'artifacts/schema_pilot_status.json').write_text(json.dumps({'complete': True, 'cells': [
        {'model': 'm', 'condition': v, 'benchmark': 'billiger-de', 'status': 'passed'} for v in module.CONDITIONS]}))
    return module, selected, cells


class SchemaExportTests(unittest.TestCase):
    def test_review_is_abstention_and_empty_subset_is_null(self):
        rows = condition_rows(2, 'm', 'b', 'schema/review', 'C_review', data((-1, -1, -1, -1)), data(), metric)
        values = {r['metric']: r['value'] for r in rows}
        self.assertEqual(values['coverage'], 0)
        self.assertEqual(values['resolved_correct_over_all'], 0)
        self.assertIsNone(values['accepted_f1'])
        self.assertIsNone(values['baseline_same_accepted_subset_f1'])
        self.assertIsNone(values['accepted_risk'])
        values = {r['metric']: r['value'] for r in condition_rows(2, 'm', 'b', 'schema/review', 'C_review', data((-1, 1, 1, -1)), data(), metric)}
        self.assertEqual(values['coverage'], .5)
        self.assertEqual(values['accepted_risk'], .5)
        self.assertAlmostEqual(values['accepted_f1'], 2/3)
        self.assertEqual(values['baseline_same_accepted_subset_f1'], 1)

    def test_alignment_and_binary_abstention_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cannot abstain'):
            condition_rows(1, 'm', 'b', 'schema/global', 'C_match', data((-1, 1, 0, 1)), data(), metric)
        baseline = data(); baseline['cases'].reverse()
        with self.assertRaisesRegex(ValueError, 'alignment'):
            condition_rows(1, 'm', 'b', 'schema/global', 'C_match', data(), baseline, metric)

    def test_lexical_ties_transfer_and_full_matrix(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); module, selected, cells = fixture(root)
            self.assertEqual(selected['m']['global'], 'C_match')
            self.assertEqual(selected['m']['per_benchmark']['billiger-de-50un'], 'C_match')
            self.assertEqual(len(cells), 4)
            baseline = Mock(return_value=(data(), 0))
            calls = []
            def loader(ws, mod, m, v, b, split, ns, origin, provenance):
                calls.append((split, ns)); return data()
            with patch('schema_run_export.schema_cell', side_effect=loader):
                rows = export_schema(root, module, SimpleNamespace(metric=metric), baseline, Mock(), [], 3)
            first_test = next(i for i, c in enumerate(calls) if c[0] == 'test')
            self.assertEqual(first_test, 10)  # all five pilot + all five development
            self.assertEqual(len(rows), 6*13+4)
            self.assertTrue(all(r['run_id'] == 3 for r in rows))

    def test_drift_and_incomplete_pilot_block_test_access(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); module, selected, _ = fixture(root)
            selected['m']['global'] = 'C_same'
            (root/'configs/expanded_schema_selection.json').write_text(json.dumps(selected))
            baseline = Mock()
            with patch('schema_run_export.schema_cell', return_value=data()) as loader:
                with self.assertRaisesRegex(ValueError, 'selection differs'):
                    export_schema(root, module, SimpleNamespace(metric=metric), baseline, Mock(), [], 1)
                self.assertTrue(all(c.args[5] != 'test' for c in loader.call_args_list))
            baseline.assert_not_called()
            (root/'artifacts/schema_pilot_status.json').write_text('{"complete":false}')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                export_schema(root, module, SimpleNamespace(metric=metric), baseline, Mock(), [], 1)

    def test_real_wire_loader_origin_and_order_hash(self):
        import expanded_schema as module
        import protocol
        from export_run_metrics import check_response_origin
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = {'benchmark': 'wdc', 'pair_id': 'p', 'input_hash': 'h', 'label': 1,
                    'records': {'left': {'title': 'A'}, 'right': {'title': 'B'}}}
            source = root/'data/wdc/prompt.json'; source.parent.mkdir(parents=True)
            source.write_text(json.dumps([case]))
            response = {'model': protocol.MODELS['jev']['id'], 'provider': protocol.MODELS['jev']['provider'].split('/')[0],
                        'answers': {'match': {'type': 'choice', 'choice': 'Match',
                                             'probabilities': {'No Match': .2, 'Match': .8}}}}
            row = {'pair_id': 'p', 'input_hash': 'h', 'study_run_id': 3,
                   'completed_at': '2026-10-14T18:00:00+00:00',
                   'request_hash': module.wire_hash(module.payload(case, 'jev', 'C_match_reverse')),
                   'prediction': 1, 'probability_match': .8, 'attempts': [{'cost': .001, 'response': response}]}
            p = root/'runs/schema/jev/C_match_reverse/wdc/prompt.jsonl'; p.parent.mkdir(parents=True)
            p.write_text(json.dumps(row)+'\n')
            fail = {**row, 'attempts': [{'cost': None}]}
            p.with_name('prompt.failures.jsonl').write_text(json.dumps(fail)+'\n')
            def origin(rows):
                check_response_origin(rows, 3, '2026-10-14T17:00:00+00:00')
            with patch.object(module, 'PROJECT', root), patch('socket.socket', side_effect=AssertionError('No network')):
                checked = schema_cell(root, module, 'jev', 'C_match_reverse', 'wdc', 'prompt', 'schema', origin, [])
                self.assertEqual(checked['unknown_cost_attempts'], 1)
                self.assertEqual(checked['known_cost_usd'], .001)
                row['request_hash'] = module.wire_hash(module.payload(case, 'jev', 'C_match'))
                p.write_text(json.dumps(row)+'\n')
                with self.assertRaises(AssertionError):
                    schema_cell(root, module, 'jev', 'C_match_reverse', 'wdc', 'prompt', 'schema', origin, [])

    def test_schema_provenance_checks_failures_and_input_hash(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); p = root/'runs/schema/m/C_match/b'
            p.mkdir(parents=True)
            row = {'pair_id': '0', 'input_hash': '0', 'study_run_id': 3}
            (p/'test.jsonl').write_text(json.dumps(row)+'\n')
            (p/'test.failures.jsonl').write_text(json.dumps(row)+'\n')
            module = SimpleNamespace(load=Mock(return_value=data()))
            origin = Mock(); provenance = []
            schema_cell(root, module, 'm', 'C_match', 'b', 'test', 'schema', origin, provenance)
            self.assertEqual(len(origin.call_args.args[0]), 2)
            self.assertTrue(provenance[0]['failure_sha256'])
            row['input_hash'] = 'drift'
            (p/'test.failures.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                schema_cell(root, module, 'm', 'C_match', 'b', 'test', 'schema', origin, [])


if __name__ == '__main__':
    unittest.main()
