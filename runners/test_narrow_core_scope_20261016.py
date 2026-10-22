import copy
from pathlib import Path
import tempfile
import unittest

import narrow_core_scope_20261016 as scope


def choices(run, variant):
    return [dict(run_id=run, model=m, benchmark=b, regime=r, variant=variant)
            for m in scope.MODELS for b in scope.BENCHMARKS
            for r in ('global', 'per_benchmark')]


def cases(n=140):
    return [dict(pair_id=str(i), input_hash=f'hash{i}', label=(i // 2) % 2,
                 records={'left': {'title': str(i)}, 'right': {}}) for i in range(n)]


class NarrowScopeTest(unittest.TestCase):
    def test_development_has_only_narrow_axes(self):
        cells = scope.development_cells(2)
        self.assertEqual(len(cells), 1344)
        self.assertEqual(sum(c[1] == 'schema' for c in cells), 280)
        self.assertFalse(any(c[3] in ('B0', 'X_R_aligned') for c in cells))
        self.assertEqual({c[-1] for c in cells}, {'prompt'})
        self.assertNotIn('billiger-de-50un', {c[4] for c in cells})
        for bad in (True, '2', 2.0, 0, 4):
            with self.assertRaises(ValueError):
                scope.development_cells(bad)

    def test_selection_deduplicates_only_within_run(self):
        p, s = choices(2, 'X_P_minimal_0'), choices(2, 'C_match')
        a = scope.selected_test_cells(2, p, s)
        self.assertEqual(len(a), 224)
        p3, s3 = choices(3, 'X_P_minimal_0'), choices(3, 'C_match')
        b = scope.selected_test_cells(3, p3, s3)
        self.assertFalse(a & b)
        for row in p:
            if row['regime'] == 'per_benchmark':
                row['variant'] = 'X_P_identity_0'
        for row in s:
            if row['regime'] == 'per_benchmark':
                row['variant'] = 'C_same'
        self.assertEqual(len(scope.selected_test_cells(2, p, s)), 336)

    def test_selection_rejects_partial_foreign_broad_and_inconsistent(self):
        p, s = choices(2, 'X_P_minimal_0'), choices(2, 'C_match')
        bad = [p[:-1], p + [p[0]]]
        for key, value in [('run_id', 3), ('run_id', True),
                           ('regime', 'leave_one_out'), ('benchmark', 'billiger-de-50un'),
                           ('variant', 'X_R_aligned'), ('variant', 'X_P_identity_0')]:
            c = copy.deepcopy(p)
            c[0][key] = value
            bad.append(c)
        for candidate in bad:
            with self.assertRaises(ValueError):
                scope.selected_test_cells(2, candidate, s)

    def test_nested100_uses_only_required_original_extra_order(self):
        pool = cases()
        route = pool[::3]
        extra = list(reversed([r for i, r in enumerate(pool) if i % 3]))
        selected = scope.calibration_partition(pool, route, extra)
        self.assertTrue(selected['available'])
        self.assertEqual(selected['fit_pair_ids'], [str(i) for i in range(0, 100, 2)])
        self.assertEqual(selected['tune_pair_ids'], [str(i) for i in range(1, 100, 2)])
        self.assertEqual(selected['extra_pair_ids'],
                         [str(i) for i in range(99, -1, -1) if i % 3])
        self.assertEqual(len(selected['route_pair_ids']) + len(selected['extra_pair_ids']), 100)
        unavailable = cases()
        for row in unavailable[:100:2]:
            row['label'] = 0
        x = scope.calibration_partition(unavailable, [], unavailable)
        self.assertFalse(x['available'])
        self.assertEqual(x['extra_pair_ids'], [])

    def test_calibration_refuses_changed_or_overlapping_inputs(self):
        p = cases()
        for route, extra in [(p[:1], p), ([], p[:-1]), ([], p + [p[0]])]:
            with self.assertRaises(ValueError):
                scope.calibration_partition(p, route, extra)
        changed = copy.deepcopy(p)
        changed[1]['records']['left']['title'] = 'changed'
        with self.assertRaises(ValueError):
            scope.calibration_partition(p, [], changed)

    def test_calls_cannot_expand_or_silently_shorten(self):
        allowed = scope.development_cells(3)
        c = scope.cell(3, 'quality', 'jev', 'S1', 'wdc', 'prompt')
        self.assertTrue(scope.validate_call(3, c, ['a', 'b'], ['a', 'b'], allowed))
        for ids in ([], ['b', 'a'], ['a'], ['a', 'b', 'c'], ['a', 'a'], [1, 2]):
            with self.assertRaises(ValueError):
                scope.validate_call(3, c, ids, ['a', 'b'], allowed)
        with self.assertRaises(ValueError):
            scope.validate_call(2, c, ['a'], ['a'], allowed)

    def test_baseline_and_old_gap_stay_blocked_even_in_allowlist(self):
        for c in [scope.cell(2, 'quality', 'jev', 'B0', 'wdc', 'test'),
                  scope.cell(2, 'quality', 'jev', 'X_R_aligned', 'wdc', 'prompt'),
                  scope.cell(2, 'schema_pilot', 'jev', 'C_match', 'wdc', 'schema_pilot'),
                  scope.cell(2, 'quality', 'luna', 'B0', 'wdc', 'calibration_extra'),
                  scope.cell(1, 'quality', 'luna', 'X_P_conservative_0', 'wdc', 'test')]:
            with self.assertRaises(ValueError):
                scope.validate_call(c[0], c, ['a'], ['a'], {c})

    def test_sources_fail_closed_without_changing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in scope.PINS:
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'not frozen')
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.assertRaises(ValueError):
                scope.verify_sources(root)
            self.assertEqual(before, {p: p.read_bytes() for p in before})


if __name__ == '__main__':
    unittest.main()
