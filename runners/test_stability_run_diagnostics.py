import unittest

from stability_run_diagnostics import (factorial_diagnostics, repeat_diagnostics,
                                       export_diagnostics, export_probability_diagnostics)
from summarize_three_runs import summarize


def cell(predictions, ids=('a', 'b', 'c', 'd'), labels=(1, 1, 0, 0)):
    return {'cases': [{'pair_id': k} for k in ids], 'y': labels, 'pred': predictions}


def factorial():
    return {'I3_identity_json': cell([1, 0, 1, 0]),  # F1 .5
            'I3_domain_json': cell([1, 1, 0, 0]),  # F1 1
            'X_P_identity_0': cell([1, 1, 0, 0]),
            'X_P_domain_0': cell([1, 0, 1, 0])}


class StabilityDiagnosticsTests(unittest.TestCase):
    def test_probability_range_aligned_subset(self):
        base = {**cell([1, 1, 0, 0]), 'prob': [.8, .9, .2, .1]}
        one = {**cell([0, 1], ('c', 'a'), (0, 1)), 'prob': [.3, .6]}
        two = {**cell([1, 0], ('a', 'c'), (1, 0)), 'prob': [.7, .1]}
        rows = export_probability_diagnostics(2, 'jev', 'b', base, one, two)
        self.assertAlmostEqual(rows[0]['value'], .2)
        self.assertAlmostEqual(rows[1]['value'], .2)
        self.assertEqual({r['run_id'] for r in rows}, {2})
        self.assertTrue(all(r['mean'] is None and r['sd'] is None for r in summarize(rows)['rows']))

    def test_probability_missing_is_unavailable_only_for_luna(self):
        data = {**cell([1, 1, 0, 0]), 'prob': None}
        rows = export_probability_diagnostics(1, 'luna', 'b', data, data, data)
        self.assertTrue(all(r['value'] is None and not r['available'] for r in rows))
        with self.assertRaises(ValueError):
            export_probability_diagnostics(1, 'jev', 'b', data, data, data)

    def test_probability_invalid_or_partial_rejected(self):
        good = {**cell([1, 1, 0, 0]), 'prob': [.8, .9, .2, .1]}
        for bad in [None, [.8], [float('nan'), .9, .2, .1], [float('inf'), .9, .2, .1],
                    [-.1, .9, .2, .1], [1.1, .9, .2, .1]]:
            with self.assertRaises(ValueError):
                export_probability_diagnostics(1, 'jev', 'b', good, {**good, 'prob': bad}, good)
        with self.assertRaises(ValueError):
            export_probability_diagnostics(1, 'luna', 'b', good, good, good)

    def test_interaction_direction_and_pair_alignment(self):
        values = factorial()
        self.assertEqual(factorial_diagnostics(values), {'n': 4, 'f1_interaction': 1.0})
        values['I3_domain_json'] = cell([0, 1, 0, 1], ('d', 'b', 'c', 'a'), (0, 1, 0, 1))
        self.assertEqual(factorial_diagnostics(values)['f1_interaction'], 1.0)

    def test_repeat_subset_permutation_and_known_flips(self):
        base = cell([1, 1, 0, 0])
        one = cell([0, 0], ('c', 'a'), (0, 1))
        two = cell([1, 1], ('a', 'c'), (1, 0))
        self.assertEqual(repeat_diagnostics(base, one, two), {
            'n': 2, 'all_three_prediction_agreement': 0,
            'baseline_vs_repeat1_flip_rate': .5, 'baseline_vs_repeat2_flip_rate': .5,
            'repeat1_vs_repeat2_flip_rate': 1})

    def test_duplicate_missing_and_conflicting_pairs_rejected(self):
        base = cell([1, 1, 0, 0])
        for bad in [cell([1, 1], ('a', 'a'), (1, 1)),
                    cell([1], ('unknown',), (1,)),
                    cell([1, 1, 0, 0], labels=(0, 1, 0, 0)),
                    cell([1, 1, 0, 2]), cell([], (), ())]:
            with self.assertRaises(ValueError):
                repeat_diagnostics(base, bad, base)
        values = factorial()
        values['I3_domain_json'] = cell([1, 1, 0, 0], labels=(0, 1, 0, 0))
        with self.assertRaises(ValueError):
            factorial_diagnostics(values)
        values.pop('I3_domain_json')
        with self.assertRaises(ValueError):
            factorial_diagnostics(values)

    def test_single_study_run_keeps_three_measurements_without_sd(self):
        rows = export_diagnostics(1, 'm', 'wdc', factorial(), cell([1, 1, 0, 0]),
                                  cell([0, 1, 0, 0]), cell([1, 1, 1, 0]))
        self.assertEqual(len(rows), 5)
        self.assertEqual({r['run_id'] for r in rows}, {1})
        self.assertEqual({r['split'] for r in rows}, {'development'})
        result = summarize(rows)
        self.assertTrue(all(r['mean'] is None and r['sd'] is None for r in result['rows']))


if __name__ == '__main__':
    unittest.main()
