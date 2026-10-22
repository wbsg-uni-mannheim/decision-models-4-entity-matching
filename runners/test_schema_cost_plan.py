import unittest
from schema_remaining_cost_plan import scenario


class ScenarioTests(unittest.TestCase):
    def test_separate_unknown_rate_and_missing_only(self):
        self.assertEqual(scenario({'responses': 200, 'known_usd': .4, 'unknown_attempts': 2}, 100),
                         {'known_proxy_usd': .2, 'historical_unknown_scenario_usd': .05})

    def test_complete_target_zero(self):
        self.assertEqual(scenario({'responses': 200, 'known_usd': .4, 'unknown_attempts': 2}, 0),
                         {'known_proxy_usd': 0, 'historical_unknown_scenario_usd': 0})

    def test_invalid_sources_refused(self):
        for field, value in [('responses', 0), ('responses', True), ('known_usd', float('nan')),
                             ('known_usd', -.1), ('unknown_attempts', -1)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                scenario({**{'responses': 200, 'known_usd': .4, 'unknown_attempts': 2}, field: value}, 100)
        with self.assertRaises(ValueError):
            scenario({'responses': 200, 'known_usd': .4, 'unknown_attempts': 2}, -1)
