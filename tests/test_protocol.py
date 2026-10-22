import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from protocol import MODELS, payload, parse_response, digest, state
from prepare import choose

class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.case = {'benchmark': 'wdc', 'pair_id': 'SECRET_PAIR', 'label': 1,
                     'entity_ids': {'left': 'SECRET_ID'},
                     'records': {'left': {'title': 'A\nB', 'price': None}, 'right': {'title': 'C', 'price': '0'}}}

    def test_no_labels_or_identifiers_in_requests(self):
        for m in MODELS:
            for v in ['B0', 'P1', 'P2', 'S1', 'O1']:
                b = json.dumps(payload(self.case, m, v))
                self.assertNotIn('SECRET', b)
                self.assertNotIn('"label"', b)
                self.assertFalse(payload(self.case, m, v)['provider']['allow_fallbacks'])

    def test_format_preserves_values_and_swap_is_real(self):
        self.assertIn('A\\nB', state(self.case, 'B0'))
        self.assertEqual(json.loads(state(self.case, 'S1'))['record_a'], self.case['records']['left'])
        self.assertTrue(state(self.case, 'O1').startswith('Record A:\ntitle: "C"'))
        self.assertNotEqual(digest(payload(self.case, 'jev', 'B0')), digest(payload(self.case, 'jev', 'P1')))

    def test_probability_boundary_and_provider(self):
        response = {'model': 'typesafe/jev-1.13', 'provider': 'TypeSafe',
                    'answers': {'match': {'type': 'noul', 'noul': .5}}}
        self.assertEqual(parse_response('jev', response), (1, .5))
        response['answers']['match']['noul'] = True
        with self.assertRaises(ValueError):
            parse_response('jev', response)
        response['provider'] = 'Other'
        with self.assertRaises(ValueError):
            parse_response('jev', response)

    def test_generative_refusal_and_nonboolean_fail(self):
        r = {'model': 'openai/gpt-6-luna', 'provider': 'OpenAI',
             'choices': [{'finish_reason': 'stop', 'message': {'content': '{"match":false}'}}]}
        self.assertEqual(parse_response('luna', r), (0, None))
        r['choices'][0]['message']['content'] = '{"match":"false"}'
        with self.assertRaises(ValueError):
            parse_response('luna', r)

    def test_disjoint_reproducible_stratification(self):
        rows = [{'pair_id': str(i), 'label': int(i < 10)} for i in range(100)]
        a = choose(rows, 20, 'prompt')
        self.assertEqual(a, choose(list(reversed(rows)), 20, 'prompt'))
        self.assertEqual(sum(r['label'] for r in a), 2)
        b = choose([r for r in rows if r not in a], 20, 'route')
        self.assertFalse({r['pair_id'] for r in a} & {r['pair_id'] for r in b})

if __name__ == '__main__':
    unittest.main()
