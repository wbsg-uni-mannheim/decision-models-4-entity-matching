import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
import rule_training as r


def c(i, label):
    return {'pair_id': str(i), 'label': label, 'records': {'left': {'name': 'item '+str(i)}, 'right': {'name': 'peer '+str(i)}}}


class RuleTrainingTests(unittest.TestCase):
    def test_selection_excludes_ids_and_swapped_inputs_and_is_stable(self):
        rows = [c(i, i % 2) for i in range(160)]
        swapped = copy.deepcopy(rows[2]);swapped['records'] = {'left': swapped['records']['right'], 'right': swapped['records']['left']}
        selected, info = r.select_training(rows, 'x', {'1'}, {r.unordered_input(swapped)})
        other, _ = r.select_training(list(reversed(rows)), 'x', {'1'}, {r.unordered_input(swapped)})
        self.assertEqual(selected, other)
        self.assertEqual(info['positive_selected'], 50)
        self.assertEqual(len(selected), 100)
        self.assertFalse({'1', '2'} & {x['pair_id'] for x in selected})

    def test_missing_class_and_conflicting_input_unavailable(self):
        rows = [c(i, 0) for i in range(110)]
        selected, info = r.select_training(rows, 'x', set(), set())
        self.assertFalse(info['available']);self.assertFalse(selected)
        conflict = copy.deepcopy(rows[0]);conflict['pair_id'] = 'new';conflict['label'] = 1
        _, info = r.select_training(rows+[conflict], 'x', set(), set())
        self.assertEqual(info['ambiguous_input_groups_removed'], 1)
        self.assertFalse(info['available'])

    def test_small_positive_class_is_preserved_without_invention(self):
        rows = [c(i, int(i < 7)) for i in range(110)]
        selected, info = r.select_training(rows, 'x', set(), set())
        self.assertEqual(len(selected), 100)
        self.assertEqual(info['positive_selected'], 7)

    def test_gzip_column_source_and_malformed_indices(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'train.json.gz'
            columns = {'label': {'0': 1, '1': 0}, 'pair_id': {'0': 'a', '1': 'b'}}
            with gzip.open(path, 'wt') as f:
                json.dump(columns, f)
            self.assertEqual(r.training_rows(path), [{'label': 1, 'pair_id': 'a'}, {'label': 0, 'pair_id': 'b'}])
            del columns['pair_id']['1']
            with gzip.open(path, 'wt') as f:
                json.dump(columns, f)
            with self.assertRaises(AssertionError):
                r.training_rows(path)


if __name__ == '__main__':
    unittest.main()
