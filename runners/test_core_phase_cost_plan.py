import unittest

from core_phase_cost_plan import aggregate, confirmation_scenarios, costs


class CorePhaseCosts(unittest.TestCase):
    def test_global_choice_and_separate_unknown_maximum(self):
        sources = {}
        for v, values in {'P1': (8, 1), 'P2': (1, 8), 'S1': (4, 4)}.items():
            for b, value in zip(('a', 'b'), values):
                sources['quality', 'm', v, b, 'prompt'] = {
                    'path': v+b, 'responses': 2, 'known_proxy_usd': value,
                    'historical_unknown_scenario_usd': .1 if v == 'S1' else 0}
        candidates, envelopes = confirmation_scenarios(sources, ['m'], ['a', 'b'], {'a': 10, 'b': 10})
        self.assertEqual(len(candidates), 3)
        self.assertEqual(envelopes[0]['known_proxy_usd'], 45)  # Not per-family cherry-picked 80.
        self.assertEqual(envelopes[0]['historical_unknown_scenario_usd'], 1)
        self.assertEqual(sum(c['prospective_responses'] for c in candidates[0]['cells']), 20)

    def test_missing_or_empty_candidate_source_rejected(self):
        with self.assertRaises(KeyError):
            confirmation_scenarios({}, ['m'], ['a'], {'a': 10})
        source = {('quality', 'm', 'P1', 'a', 'prompt'): {'responses': 0}}
        with self.assertRaises(ValueError):
            confirmation_scenarios(source, ['m'], ['a'], {'a': 10})

    def test_unknown_and_paid_attempts_kept_separate(self):
        result = costs([{'cost': .1}, {'cost': None}, {'cost': .2}])
        self.assertAlmostEqual(result['known_proxy_usd'], .3)
        self.assertEqual(result['historical_unknown_scenario_usd'], .05)
        self.assertEqual(aggregate([result, costs([])])['historical_unknown_scenario_usd'], .05)
        for invalid in (True, -1, float('nan'), float('inf'), '0.1'):
            with self.assertRaises(ValueError):
                costs([{'cost': invalid}])


if __name__ == '__main__':
    unittest.main()
