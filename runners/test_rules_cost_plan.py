import unittest
from rules_remaining_cost_plan import matrix, validate_matrix, partition_missing, scenario


class RulesCostTests(unittest.TestCase):
    def test_full_fixed_matrix_no_quality_selection(self):
        cells = matrix(['m1', 'm2'], ['a', 'b'])
        self.assertEqual(len(cells), 20)
        self.assertEqual(sum(c['split'] == 'prompt' for c in cells), 8)
        self.assertTrue(all(c['split'] == 'test' for c in cells if c['benchmark'] == 'billiger-de-50un'))
        validate_matrix(list(reversed(cells)), cells)
        for wrong in (cells[:-1], cells+[cells[0]], [{**c, 'namespace': 'wrong'} for c in cells]):
            with self.assertRaises(ValueError):
                validate_matrix(wrong, cells)

    def test_exact_cache_and_future_overlap_are_distinct(self):
        cases = [{'pair_id': p} for p in ['a', 'b', 'c', 'd']]
        missing, shared, extra = partition_missing(cases, [{'pair_id': 'a'}], {'a', 'b', 'c'})
        self.assertEqual((missing, shared, extra), (['b', 'c', 'd'], ['b', 'c'], ['d']))
        self.assertEqual(set(missing), set(shared) | set(extra))
        self.assertFalse(set(shared) & set(extra))

    def test_foreign_or_duplicate_inputs_fail(self):
        cases = [{'pair_id': 'a'}, {'pair_id': 'b'}]
        for inputs, cached, overlap in ((cases+cases, [], set()),
                (cases, [{'pair_id': 'a'}, {'pair_id': 'a'}], set()),
                (cases, [{'pair_id': 'c'}], set()), (cases, [], {'c'})):
            with self.assertRaises(ValueError):
                partition_missing(inputs, cached, overlap)

    def test_rule_and_manual_scenarios_not_combined_with_reused_cost(self):
        source = {'responses': 200, 'known_usd': .4, 'unknown_attempts': 2}
        standalone = scenario(source, 100)
        additional = scenario(source, 40)
        self.assertAlmostEqual(standalone['known_proxy_usd'], .2)
        self.assertAlmostEqual(additional['known_proxy_usd'], .08)
        self.assertAlmostEqual(standalone['historical_unknown_scenario_usd'], .05)
        self.assertEqual(scenario(source, 0)['known_proxy_usd'], 0)

    def test_bad_cost_sources_fail(self):
        for field, value in [('responses', 0), ('responses', True), ('known_usd', float('nan')),
                             ('known_usd', -.1), ('unknown_attempts', -1)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                scenario({**{'responses': 200, 'known_usd': .4, 'unknown_attempts': 2}, field: value}, 1)

