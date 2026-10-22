import copy
import unittest

from peeters_cost_plan import project_cells, totals
from peeters_control import ARMS


class PeetersCostTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{'namespace': 'quality', 'variant': 'B0', 'model': 'm', 'benchmark': 'b',
                         'split': 'test', 'responses': 2, 'known_usd': .3, 'unknown_attempts': 1,
                         'attempts': 4, 'cache_sha256': 'cache', 'failure_sha256': 'failure'}]
        self.cells = [{'model': 'm', 'benchmark': 'b', 'split': 'test', 'arm': a, 'pairs': 2,
                       'ordered_payloads_sha256': a} for a in ARMS]

    def test_both_arms_all_runs_failures_and_unknowns_separate(self):
        rows = project_cells(self.sources, self.cells)
        result = totals(rows, 3)
        self.assertEqual(result['prospective_calls_before_retries'], 12)
        self.assertAlmostEqual(result['historical_b0_known_cost_proxy_usd'], 1.8)
        self.assertAlmostEqual(result['separate_historical_unknown_allowance_scenario_usd'], .3)
        self.assertEqual([r['b0_failure_sha256'] for r in rows], ['failure']*2)

    def test_missing_duplicate_extra_cells_rejected(self):
        for cells in [self.cells[:1], self.cells+self.cells[:1], self.cells+[dict(self.cells[0], model='other')]]:
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                project_cells(self.sources, cells)

    def test_mismatched_pairs_and_source_identity_rejected(self):
        cells = copy.deepcopy(self.cells)
        cells[0]['pairs'] = 3
        with self.assertRaisesRegex(ValueError, 'pair count'):
            project_cells(self.sources, cells)
        for sources in [self.sources*2, [dict(self.sources[0], variant='P1')], [dict(self.sources[0], namespace='peeters')]]:
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                project_cells(sources, self.cells)


if __name__ == '__main__':
    unittest.main()
