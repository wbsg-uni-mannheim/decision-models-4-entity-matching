import unittest
import copy
import numpy as np
import expanded_stability as stability
from expanded_protocol import payload as original


class StabilityTests(unittest.TestCase):
    def test_factorial_changes_format_only(self):
        case = {'benchmark': 'dblp-scholar', 'label': 1, 'pair_id': 'secret-id',
                'records': {'left': {'title': 'α\nA', 'year': None}, 'right': {'title': 'B', 'year': 2020}}}
        for m in stability.MODELS:
            self.assertEqual(stability.payload(case, m, 'B0'), original(case, m, 'B0'))
            for condition, base in stability.INTERACTIONS.items():
                before, after = original(case, m, base), stability.payload(case, m, condition)
                expected = copy.deepcopy(before)
                if 'state' in expected:
                    expected['state'] = stability.state(case, 'S1')
                else:
                    expected['messages'][1]['content'] = stability.state(case, 'S1')
                self.assertEqual(after, expected)
                self.assertNotIn('secret-id', str(after))

    def test_repeat_alignment_and_flip_rates(self):
        baseline = {'cases': [{'pair_id': x} for x in ['a', 'b', 'c']], 'y': np.array([0, 1, 0]),
                    'pred': np.array([0, 1, 0]), 'prob': np.array([.1, .9, .2])}
        first = {'cases': [{'pair_id': x} for x in ['b', 'a']], 'y': np.array([1, 0]),
                 'pred': np.array([1, 1]), 'prob': np.array([.85, .6])}
        second = {'cases': [{'pair_id': x} for x in ['a', 'b']], 'y': np.array([0, 1]),
                  'pred': np.array([0, 1]), 'prob': np.array([.1, .9])}
        result = stability.repeat_summary(baseline, first, second)
        self.assertEqual(result['all_three_prediction_agreement'], .5)
        self.assertEqual(result['baseline_vs_repeat2_flip_rate'], 0.)
        self.assertEqual(result['changed_pair_ids'], ['a'])
        self.assertAlmostEqual(result['max_within_pair_probability_range'], .5)
        second['cases'][1]['pair_id'] = 'wrong'
        with self.assertRaises(AssertionError):
            stability.repeat_summary(baseline, first, second)


if __name__ == '__main__':
    unittest.main()
