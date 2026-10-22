import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
from export_run_metrics import check_response_origin, normalized_rows, confirmation_policy, export_fixed_extension, export_rules
from summarize_three_runs import summarize


def rule_fixture(root):
    conditions = {'R_rules': 'rules', 'X_P_domain_0': 'quality'}
    entries = [{'model': 'm', 'variant': v, 'benchmark': b, 'split': split, 'namespace': ns}
               for v, ns in conditions.items() for b in ['wdc', 'billiger-de-50un']
               for split in (['test'] if b == 'billiger-de-50un' else ['prompt', 'test'])]
    marker = {'status': 'complete', 'evaluated_conditions': list(conditions),
              'controls': ['B0', 'X_P_domain_0'], 'test_cells_per_condition': 2,
              'development_cells_per_condition': 1}
    (root/'artifacts').mkdir()
    (root/'artifacts/expanded_rules_complete.json').write_text(json.dumps(marker))
    return SimpleNamespace(CONDITIONS=conditions, payload=object(), verify=Mock(return_value={'cells': entries}))


class ExportTests(unittest.TestCase):
    def test_rules_require_development_and_restore_payload_on_missing_cache(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = rule_fixture(root)
            original = object()
            analyze, emit = SimpleNamespace(payload=original), Mock()
            def missing_development(m, v, b, split, ns):
                self.assertIs(analyze.payload, module.payload)
                if split == 'prompt':
                    raise ValueError('Incomplete development cache')
            with self.assertRaisesRegex(ValueError, 'Incomplete development'):
                export_rules(root, module, ['m'], ['wdc'], analyze, missing_development, emit)
            emit.assert_not_called()
            self.assertIs(analyze.payload, original)

    def test_rules_verify_matrix_and_distinguish_shared_control_namespaces(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = rule_fixture(root)
            checked, emitted = [], []
            original = object()
            analyze = SimpleNamespace(payload=original)
            def cell(*args):
                self.assertIs(analyze.payload, module.payload)
                checked.append(args)
            def emit(*args):
                self.assertEqual(len(checked), 6)
                emitted.append(args)
            export_rules(root, module, ['m'], ['wdc'], analyze, cell, emit)
            self.assertIs(analyze.payload, original)
            self.assertEqual(len(emitted), 6)
            self.assertEqual({(r[1], r[4], r[5]) for r in emitted},
                             {('B0', 'rules/B0', 'quality'), ('R_rules', 'rules/R_rules', 'rules'),
                              ('X_P_domain_0', 'rules/X_P_domain_0', 'quality')})
            self.assertEqual(sum(r[3] == 'prompt' for r in checked), 2)
            self.assertTrue(all(r[3] == 'test' for r in emitted))

    def test_rules_reject_partial_or_changed_matrix_before_loading(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = rule_fixture(root)
            cell, emit = Mock(), Mock()
            module.verify.return_value['cells'].pop()
            with self.assertRaisesRegex(ValueError, 'matrix differs'):
                export_rules(root, module, ['m'], ['wdc'], SimpleNamespace(payload=None), cell, emit)
            cell.assert_not_called()
            emit.assert_not_called()
            marker = root/'artifacts/expanded_rules_complete.json'
            marker.write_text('{"status":"running"}')
            module.verify.reset_mock()
            with self.assertRaisesRegex(ValueError, 'Rules incomplete'):
                export_rules(root, module, ['m'], ['wdc'], SimpleNamespace(payload=None), cell, emit)
            module.verify.assert_not_called()

    def test_response_identity_and_time(self):
        prepared = '2026-10-14T08:00:00+00:00'
        row = {'study_run_id': 2, 'completed_at': '2026-10-14T09:00:00+00:00'}
        check_response_origin([row], 2, prepared)
        for bad in [{}, {**row, 'study_run_id': 3}, {**row, 'completed_at': '2026-10-13T09:00:00+00:00'}]:
            with self.assertRaises((ValueError, KeyError)):
                check_response_origin([bad], 2, prepared)
        with self.assertRaises(ValueError): check_response_origin([row], 1)

    def test_cost_scaling_and_empty(self):
        metrics = {'n': 200, 'f1': .5, 'precision': .4, 'recall': .6}
        rows = normalized_rows(1, 'm', 'baseline', 'wdc', metrics, .02, 2, 400, 'B0')
        values = {r['metric']: r['value'] for r in rows}
        self.assertAlmostEqual(values['known_cost_usd_per_1000'], .1)
        self.assertEqual(values['unknown_attempts_per_1000'], 10)
        self.assertEqual(values['request_attempt_seconds_per_pair'], 2)
        with self.assertRaises(ValueError): normalized_rows(1, 'm', 'c', 'b', {**metrics, 'n': 0}, 0, 0, 0, 'B0')

    def test_policies_not_selected_prompts_are_grouped(self):
        rows = []
        for rid, variant, f1 in [(1, 'a', .7), (2, 'b', .8), (3, 'c', .9)]:
            rows.extend(normalized_rows(rid, 'm', confirmation_policy('prompt', 'global'), 'wdc',
                        {'n': 500, 'f1': f1, 'precision': f1, 'recall': f1}, 0, 0, 0, variant))
        f1 = next(r for r in summarize(rows)['rows'] if r['metric'] == 'f1')
        self.assertAlmostEqual(f1['sd'], .1)
        self.assertEqual(f1['n'], 3)

    def test_extension_refuses_incomplete_before_loading(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'artifacts').mkdir()
            marker = root/'artifacts/expanded_variants_complete.json'
            module, emit = Mock(), Mock()
            for status in ['running', 'failed']:
                marker.write_text(json.dumps({'status': status}))
                with self.assertRaises(ValueError):
                    export_fixed_extension(root, 'variants', module, ['m'], Mock(), emit)
            module.verify.assert_not_called()
            emit.assert_not_called()

    def test_interaction_payload_restored_even_on_bad_cache(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'artifacts').mkdir()
            (root/'artifacts/expanded_stability_complete.json').write_text('{"status":"complete"}')
            original, interaction = object(), object()
            analyze = SimpleNamespace(payload=original)
            module = SimpleNamespace(payload=interaction, verify=Mock(return_value={}),
                                     SCREEN_BENCHMARKS=['wdc'], INTERACTIONS={'json': 'text'})
            def bad_cache(*args):
                self.assertIs(analyze.payload, interaction)
                raise ValueError('Wrong request hash')
            with self.assertRaisesRegex(ValueError, 'Wrong request hash'):
                export_fixed_extension(root, 'stability', module, ['m'], analyze, bad_cache)
            self.assertIs(analyze.payload, original)

    def test_small_repeats_do_not_become_three_study_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'artifacts').mkdir()
            (root/'artifacts/expanded_stability_complete.json').write_text('{"status":"complete"}')
            module = SimpleNamespace(payload=object(), verify=Mock(return_value={}),
                                     SCREEN_BENCHMARKS=['wdc'], INTERACTIONS={'json': 'text'})
            rows, namespaces = [], []
            def emit(m, v, b, split, condition, namespace='quality'):
                namespaces.append(namespace)
                rows.extend(normalized_rows(1, m, condition, b,
                            {'n': 100, 'f1': .8, 'precision': .8, 'recall': .8}, 0, 0, 0, v))
            export_fixed_extension(root, 'stability', module, ['m'], SimpleNamespace(payload=None), emit)
            self.assertEqual(namespaces[-2:], ['e3_repeat_1', 'e3_repeat_2'])
            self.assertEqual({row['run_id'] for row in rows}, {1})
            self.assertTrue(all(row['mean'] is None and row['sd'] is None for row in summarize(rows)['rows']))

    def test_supplementary_variants_do_not_enter_core_macro(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'artifacts').mkdir()
            (root/'artifacts/expanded_variants_complete.json').write_text('{"status":"complete"}')
            module = SimpleNamespace(payload=object(), verify=Mock(return_value={
                'conditions': {'wdc-050un': {}, 'billiger-en-en-100un': {}}}))
            rows = []
            def emit(m, v, b, split, condition):
                for rid in [1, 2, 3]:
                    rows.extend(normalized_rows(rid, m, condition, b,
                                {'n': 100, 'f1': .8, 'precision': .8, 'recall': .8}, 0, 0, 0, v))
            export_fixed_extension(root, 'variants', module, ['m'], SimpleNamespace(payload=None), emit)
            self.assertFalse(any(row['benchmark'] == 'macro_8_core_families' for row in summarize(rows)['rows']))


if __name__ == '__main__': unittest.main()
