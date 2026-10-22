import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import protocol
from expanded_protocol import VARIANTS, payload, representation


class ExpandedProtocolTest(unittest.TestCase):
    def setUp(self):
        self.case = {'benchmark': 'wdc', 'pair_id': 'SECRET', 'label': 1,
                     'records': {'left': {'title': 'Ä \n B', 'price': None},
                                 'right': {'title': 'C', 'price': '0'}},
                     'entity_ids': {'left': 'SECRET', 'right': 'SECRET'}}

    def test_no_leakage_and_pinned_provider(self):
        self.assertEqual(len(VARIANTS), 27)
        for m in protocol.MODELS:
            for v in VARIANTS:
                body = payload(self.case, m, v)
                self.assertNotIn('SECRET', json.dumps(body))
                self.assertNotIn('"label"', json.dumps(body))
                self.assertFalse(body['provider']['allow_fallbacks'])

    def test_original_protocol_unchanged(self):
        for m in protocol.MODELS:
            for v in ['B0', 'P1', 'P2', 'S1', 'O1']:
                self.assertEqual(payload(self.case, m, v), protocol.payload(self.case, m, v))
        self.assertEqual(len(protocol.BENCHMARKS), 8)

    def test_equivalent_representations_preserve_values(self):
        doc = json.loads(representation(self.case, 'X_R_json'))
        self.assertEqual(doc['record_a'], self.case['records']['left'])
        aligned = [json.loads(x) for x in representation(self.case, 'X_R_aligned').splitlines()[1:]]
        self.assertEqual({x['attribute']: x['record_a'] for x in aligned}, self.case['records']['left'])
        self.assertEqual({x['attribute']: x['record_b'] for x in aligned}, self.case['records']['right'])
        self.assertIn('Ä \\n B', representation(self.case, 'X_R_order'))
        self.assertTrue(representation(self.case, 'X_R_swap').startswith('Record A:\ntitle: "C"'))


if __name__ == '__main__':
    unittest.main()
