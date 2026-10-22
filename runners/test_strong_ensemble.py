import unittest
from strong_ensemble import MEMBERS, majority, parse_vote, conditional_cost, payload_from_frozen_b0


class EnsembleTests(unittest.TestCase):
    def test_all_vote_patterns(self):
        import itertools
        for values in itertools.product([False, True], repeat=3):
            result = majority(dict(zip(MEMBERS, values)))
            self.assertEqual(result['match'], sum(values) >= 2)
            self.assertEqual(result['unanimous'], len(set(values)) == 1)

    def test_missing_never_becomes_two_member_ensemble(self):
        for votes in [{'sol61': True, 'opus55': True}, {'sol61': True, 'opus55': True, 'argon4': None},
                      {'sol61': 1, 'opus55': True, 'argon4': True}]:
            with self.assertRaises(ValueError): majority(votes)

    def test_invalid_and_truncated_vote(self):
        def response(content, finish='stop'):
            return {'choices': [{'finish_reason': finish, 'message': {'content': content}}]}
        self.assertIs(parse_vote(response('{"match":true}')), True)
        for r in [response('{"match":1}'), response('{"match":true,"reason":"x"}'),
                  response('{"match":true}', 'length'), {'error': {'code': 502}}]:
            with self.assertRaises(ValueError): parse_vote(r)

    def test_inputs_not_mutated_or_augmented(self):
        base = {'model': 'openai/gpt-6-luna', 'messages': [{'role': 'system', 'content': 'identity'},
                {'role': 'user', 'content': 'record text'}], 'response_format': {'type': 'json_schema'}}
        for m in MEMBERS:
            b = payload_from_frozen_b0(base, m)
            self.assertEqual(b['messages'], base['messages'])
            self.assertEqual(b['response_format'], base['response_format'])
            self.assertFalse(b['provider']['allow_fallbacks'])
            self.assertNotIn('astra', b['model'])
        self.assertEqual(base['model'], 'openai/gpt-6-luna')

    def test_cost_includes_all_members_and_reasoning_allowance(self):
        c = conditional_cost(dict.fromkeys(MEMBERS, 1000))
        self.assertAlmostEqual(c['ensemble_usd'], .08992)
        self.assertFalse(c['hard_cost_bound'])
        with self.assertRaises(ValueError): conditional_cost(dict.fromkeys(MEMBERS, -1))


if __name__ == '__main__': unittest.main()
