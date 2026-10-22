import unittest
from prepare_ensemble_pilot import select


class SelectionTests(unittest.TestCase):
    def test_labels_predictions_and_order_do_not_select(self):
        cases = [{'pair_id': str(i), 'records': {'left': {'title': 'x'*i}, 'right': {}},
                  'label': i % 2, 'prediction': 1} for i in range(1, 10)]
        a = [c['pair_id'] for c in select(cases)]
        altered = [{**c, 'label': 1-c['label'], 'prediction': 0} for c in reversed(cases)]
        self.assertEqual(a, [c['pair_id'] for c in select(altered)])
        self.assertEqual(len(set(a)), 2)
        self.assertIn('9', a)

    def test_no_duplicate_or_invented_second_case(self):
        c = {'pair_id': 'x', 'records': {}}
        for cases in [[], [c], [c, c]]:
            with self.assertRaises(ValueError): select(cases)


if __name__ == '__main__': unittest.main()
