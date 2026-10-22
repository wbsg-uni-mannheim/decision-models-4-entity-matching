import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import expanded_confirmation as conf
from protocol import payload as core_payload


class ConfirmationTests(unittest.TestCase):
    def test_selection_is_macro_and_ties_lexical(self):
        scores = {('a', 'z'): .8, ('a', 'a'): .6, ('b', 'z'): .4, ('b', 'a'): .6}
        winner, means = conf.choose_variant(scores, ['z', 'a'], ['a', 'b'])
        self.assertEqual(winner, 'z' if means['z'] > means['a'] else 'a')
        winner, _ = conf.choose_variant({('a', 'z'): .6, ('a', 'a'): .6}, ['z', 'a'], ['a'])
        self.assertEqual(winner, 'a')

    def test_transfer_excludes_target_and_supplement_not_source(self):
        self.assertNotIn('wdc', conf.sources('wdc', 'leave_one_out'))
        self.assertEqual(set(conf.sources('wdc', 'opposite_domain')), conf.PUBLICATION)
        self.assertNotIn('billiger-de-50un', conf.sources('billiger-de', 'global'))

    def test_json_serialization_reuses_exact_core_payload(self):
        c = {'benchmark': 'wdc', 'records': {'left': {'title': 'A'}, 'right': {'title': 'B'}}}
        for model in conf.MODELS:
            self.assertEqual(conf.payload(c, model, 'X_R_json'), core_payload(c, model, 'S1'))

    def test_selection_freezes_before_test_and_deduplicates_confirmations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bmarks = {'billiger-de': {}, 'dblp-acm': {}}
            variants = ['X_P_identity_0', 'X_P_domain_0']
            def load(m, v, b, split):
                if split == 'test':
                    self.assertTrue((root/'configs/expanded_selected_variants.json').exists())
                cases = [{'pair_id': str(i), 'entity_ids': {'left': 'same', 'right': str(i)}} for i in range(6)]
                y = np.array([0, 1, 0, 1, 0, 1])
                return {'cases': cases, 'y': y, 'pred': y, 'costs': np.zeros(6), 'unknown_cost_attempts': 0}
            with patch.object(conf, 'PROJECT', root), patch.object(conf, 'MODELS', {'jev': {}}), \
                    patch.object(conf, 'BENCHMARKS', bmarks), patch.object(conf, 'PROMPT_VARIANTS', variants), \
                    patch.object(conf.analyze, 'load', load):
                selected = conf.select()
                self.assertEqual(len(selected), 18)
                self.assertEqual(len(conf.confirmation_cells(selected)), 6)
                supplement = [e for e in selected if e['benchmark'] == 'billiger-de-50un']
                self.assertEqual(len(supplement), 6)
                self.assertTrue(all(e['variant'] == 'B0' for e in selected if e['family'] == 'serialization'))
                conf.report(selected)
                self.assertEqual(conf.select(), selected)


if __name__ == '__main__':
    unittest.main()
