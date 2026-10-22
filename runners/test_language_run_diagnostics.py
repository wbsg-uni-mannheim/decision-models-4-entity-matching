import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from export_run_metrics import export_paired_languages
from language_run_diagnostics import LANGUAGES, export_language_diagnostics, paired_values


def fixture():
    cases = [{'pair_id': str(i), 'entity_ids': ['a'+str(i), 'b'+str(i)], 'label': y,
              'records': {'left': {'name': 'Produkt'}, 'right': {'name': 'Produkt'}}}
             for i, y in enumerate([1, 0, 1, 0])]
    base = {'cases': cases, 'y': [1, 0, 1, 0], 'pred': [1, 1, 0, 0], 'prob': [.9, .8, .1, .2]}
    other = copy.deepcopy(base)
    other['pred'] = [1, 0, 1, 1]
    other['prob'] = [.8, .2, .7, .8]
    for key in ['cases', 'y', 'pred', 'prob']:
        other[key].reverse()
    return base, other


class PairedLanguage(unittest.TestCase):
    def test_reordered_pairs_have_hand_computed_effects(self):
        base, other = fixture()
        n, values = paired_values('jev', base, other)
        self.assertEqual(n, 4)
        self.assertAlmostEqual(values['f1_delta_vs_de_de'], .8-.5)
        self.assertEqual(values['prediction_flip_rate'], .75)
        self.assertEqual(values['both_correct_fraction'], .25)
        self.assertEqual(values['only_variant_correct_fraction'], .5)
        self.assertEqual(values['only_de_de_correct_fraction'], .25)
        self.assertEqual(values['both_wrong_fraction'], 0)
        self.assertAlmostEqual(values['mean_absolute_probability_shift'], 1.9/4)
        self.assertAlmostEqual(values['mean_signed_probability_shift'], .5/4)

    def test_identity_label_duplicate_and_partial_mismatches_rejected(self):
        base, variant = fixture()
        for mode in ['entity', 'label', 'field', 'duplicate', 'partial']:
            other = copy.deepcopy(variant)
            if mode == 'entity': other['cases'][0]['entity_ids'][0] = 'wrong'
            if mode == 'label': other['cases'][0]['label'] = 1
            if mode == 'field': other['cases'][0]['records']['left']['extra'] = 'x'
            if mode == 'duplicate': other['cases'][0]['pair_id'] = other['cases'][1]['pair_id']
            if mode == 'partial':
                for key in ['cases', 'y', 'pred', 'prob']: other[key].pop()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                paired_values('jev', base, other)

    def test_probabilities_require_full_finite_aligned_values(self):
        base, variant = fixture()
        for scores in [None, [.1], [float('nan')]*4, [1.1]*4]:
            other = {**variant, 'prob': scores}
            with self.assertRaises(ValueError): paired_values('jev', base, other)

    def test_generative_unavailable_and_wdc_rejection(self):
        base, other = fixture()
        base['prob'] = other['prob'] = None
        rows = export_language_diagnostics(2, 'luna', 'billiger-en-en-100un', base, other)
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(not r['available'] for r in rows), 2)
        self.assertTrue(all(r['run_id'] == 2 and r['split'] == 'test' for r in rows))
        with self.assertRaises(ValueError):
            export_language_diagnostics(2, 'luna', 'wdc-050un', base, other)

    def test_completion_gate_before_cache_reads_and_payload_restoration(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'artifacts').mkdir()
            marker = root/'artifacts/expanded_variants_complete.json'
            marker.write_text(json.dumps({'status': 'running', 'language_comparisons': 3}))
            module = SimpleNamespace(LANGUAGES=LANGUAGES, payload=object())
            original = object()
            analyze = SimpleNamespace(payload=original)
            cell = Mock()
            with self.assertRaises(ValueError):
                export_paired_languages(root, module, ['jev'], analyze, cell, 1)
            cell.assert_not_called()
            marker.write_text(json.dumps({'status': 'complete', 'language_comparisons': 3}))
            base, other = fixture()
            def load(m, v, b, split):
                self.assertIs(analyze.payload, module.payload)
                return (base if b == 'billiger-de' else other, 0)
            self.assertEqual(len(export_paired_languages(root, module, ['jev'], analyze, load, 1)), 24)
            self.assertIs(analyze.payload, original)
            with self.assertRaisesRegex(RuntimeError, 'partial'):
                export_paired_languages(root, module, ['jev'], analyze, Mock(side_effect=RuntimeError('partial')), 1)
            self.assertIs(analyze.payload, original)
