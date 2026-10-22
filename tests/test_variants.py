import copy
import unittest
from unittest.mock import patch

import numpy as np
import expanded_variants as v
from protocol import digest, payload


def case(pid, label, lang):
    records = {s: {'brand': 'A', 'price': None, 'name': lang+s, 'desc': lang+' text'} for s in ['left', 'right']}
    return {'pair_id': pid, 'entity_ids': {'left': pid+'L', 'right': pid+'R'}, 'label': label,
            'benchmark': 'billiger-de', 'records': records, 'input_hash': digest(records)}


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.de = [case('SECRET1', 1, 'DE'), case('SECRET2', 0, 'DE')]
        self.en = [case('SECRET2', 0, 'EN'), case('SECRET1', 1, 'EN')]

    def test_alignment_and_exact_side_substitution_without_core_mutation(self):
        original = copy.deepcopy(self.de)
        variants = v.language_cases(self.de, self.en)
        for name, rows in variants.items():
            languages = v.LANGUAGES[name]
            for i, row in enumerate(rows):
                for s, lang in zip(['left', 'right'], languages):
                    expected = self.de[i] if lang == 'de' else self.en[1-i]
                    self.assertEqual(row['records'][s], expected['records'][s])
                self.assertEqual(row['input_hash'], digest(row['records']))
                for model in v.MODELS:
                    text = str(payload(row, model))
                    self.assertNotIn('SECRET', text)
                    self.assertNotIn('entity_ids', text)
                    control = copy.deepcopy(row)
                    control['benchmark'] = 'billiger-de'
                    self.assertEqual(payload(row, model), payload(control, model))
        self.assertEqual(self.de, original)

    def test_mismatches_rejected(self):
        mutations = [lambda r: r.pop(), lambda r: r.append(r[0]),
                     lambda r: r[0].update(label=1),
                     lambda r: r[0]['entity_ids'].update(left='WRONG'),
                     lambda r: r[0]['records']['left'].update(price=99),
                     lambda r: r[0]['records']['left'].update(extra='oops')]
        for mutation in mutations:
            rows = copy.deepcopy(self.en)
            mutation(rows)
            with self.assertRaises(AssertionError):
                v.language_cases(self.de, rows)

    def test_overlap_does_not_assert_distinct_splits_paired(self):
        other = [case('DIFFERENT', 0, 'EN')]
        self.assertEqual(v.overlap(other, self.de)['common_pair_ids'], 0)
        with self.assertRaises(AssertionError):
            v.align(self.de, other)

    def test_prediction_alignment(self):
        reference = {'cases': self.de, 'y': np.array([1, 0]), 'pred': np.array([1, 1]), 'prob': np.array([.9, .7])}
        other = {'cases': self.en, 'y': np.array([0, 1]), 'pred': np.array([0, 0]), 'prob': np.array([.2, .3])}
        with patch.object(v.analyze, 'bootstrap_delta', return_value={'interval_tested': True}) as bootstrap:
            out = v.paired_language(reference, other)
        self.assertEqual(out['prediction_flip_rate'], 1.)
        self.assertEqual(out['only_variant_correct'], 1)
        self.assertEqual(out['only_de_correct'], 1)
        np.testing.assert_array_equal(bootstrap.call_args.args[2], [0, 0])
        self.assertAlmostEqual(out['mean_absolute_probability_shift'], .55)


if __name__ == '__main__':
    unittest.main()
