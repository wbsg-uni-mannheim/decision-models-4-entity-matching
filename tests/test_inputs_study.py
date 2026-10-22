import json
import unittest

import inputs_study as s
import prompt12
import protocol

CASE = {'benchmark': 'walmart-amazon', 'pair_id': 'w#1', 'label': 1, 'input_hash': 'h',
        'records': {'left': {'brand': 'Da-Lite', 'modelno': '89939', 'price': 2595.0,
                             'title': 'da-lite hc cinema vision tensioned advantage electrol screen'},
                    'right': {'brand': None, 'modelno': '89939', 'price': '2,499.00',
                              'title': 'Da-Lite Tensioned Advantage Electrol HC Cinema Vision'}}}


class InputsStudyTests(unittest.TestCase):
    def setUp(self):
        prompt12.register_models()
        s.SELECTED.clear()
        s.SELECTED.update({m: {b: 'B0' for b in protocol.BENCHMARKS} for m in s.RUN_MODELS})
        s.SELECTED['jev']['walmart-amazon'] = 'R_handwritten'

    def test_ten_variants(self):
        self.assertEqual(len(s.VARIANT_IDS), 10)
        self.assertEqual(sum(c == 'serialization' for _, _, c in s.VARIANTS), 4)

    def test_only_state_changes_and_prompt_is_selected(self):
        for m in s.RUN_MODELS:
            ref = prompt12.payload(CASE, m, s.SELECTED[m][CASE['benchmark']])
            for v in s.VARIANT_IDS:
                body = s.payload(CASE, m, v)
                if 'questions' in body:
                    self.assertEqual(body['questions'], ref['questions'])
                    self.assertNotEqual(body['state'], ref['state'], v)
                else:
                    self.assertEqual(body['messages'][0], ref['messages'][0])
                    self.assertNotEqual(body['messages'][1]['content'], ref['messages'][1]['content'], v)
                self.assertEqual(body['model'], ref['model'])
                self.assertEqual(body['provider'], ref['provider'])
                self.assertNotIn('w#1', json.dumps(body))

    def test_values_only_like_peeters(self):
        text = s.state(CASE, 'F1_values')
        self.assertTrue(text.startswith("Record A: 'Da-Lite 89939 2595 da-lite hc"))
        self.assertNotIn('brand', text)
        self.assertNotIn('None', text)

    def test_perturbations_are_deterministic_and_bounded(self):
        for v in ['R4_attr_del', 'R5_span_del', 'R6_typos']:
            self.assertEqual(s.state(CASE, v), s.state(CASE, v))
        left = s.perturb(CASE, 'R4_attr_del')['left']
        self.assertIn('title', left); self.assertEqual(len(left), 3)
        title = s.perturb(CASE, 'R5_span_del')['left']['title'].split()
        self.assertIn(len(title), (7, 8))
        typo = s.perturb(CASE, 'R6_typos')['right']['title']
        self.assertEqual(sorted(typo.replace(' ', '')), sorted(CASE['records']['right']['title'].replace(' ', '')))
        self.assertEqual(s.perturb(CASE, 'R3_lowercase')['right']['title'], CASE['records']['right']['title'].lower())
        other = {**CASE, 'pair_id': 'w#2'}
        self.assertTrue(any(s.state(other, v) != s.state(CASE, v) for v in ['R4_attr_del', 'R5_span_del', 'R6_typos']))

    def test_title_field_per_benchmark(self):
        case = {**CASE, 'benchmark': 'abt-buy', 'records': {'left': {'name': 'a b c d e', 'price': 1},
                                                             'right': {'name': 'f g h i j', 'price': 2}}}
        self.assertEqual(len(s.perturb(case, 'R5_span_del')['left']['name'].split()) in (3, 4), True)
        self.assertIn('name', s.perturb(case, 'R4_attr_del')['left'])

    def test_swap_and_reverse_preserve_information(self):
        base = protocol.state(CASE, 'B0')
        self.assertEqual(sorted(s.state(CASE, 'R1_swap').splitlines()), sorted(base.splitlines()))
        self.assertEqual(sorted(s.state(CASE, 'R2_attr_reverse').splitlines()), sorted(base.splitlines()))

    def test_r3_skips_lowercase_benchmarks(self):
        self.assertEqual(s.benchmarks('R3_lowercase'), ['wdc', 'dn7-walmart-amazon', 'billiger-de', 'semi-heter'])
        self.assertEqual(len(s.benchmarks('F1_values')), 8)

    def test_selection_per_model_and_benchmark(self):
        scores = {'jev': {v: {b: .5 for b in protocol.BENCHMARKS} for v in prompt12.VARIANTS}}
        scores['jev']['R_rules']['wdc'] = .7
        scores['jev']['NA_simple']['wdc'] = .7  # tie: earlier prompt wins
        scores['jev']['R_rules']['dblp-acm'] = .6
        out = s.select(scores)
        self.assertEqual(out['selected']['jev']['wdc'], 'NA_simple')
        self.assertEqual(out['selected']['jev']['dblp-acm'], 'R_rules')
        self.assertEqual(out['selected']['jev']['abt-buy'], 'B0')  # all tied: B0 first
        self.assertAlmostEqual(out['margin_over_b0']['jev']['dblp-acm'], .1)

    def test_payload_uses_benchmark_specific_prompt(self):
        self.assertIn('rules regarding', json.dumps(s.payload(CASE, 'jev', 'F2_json')))
        other = {**CASE, 'benchmark': 'wdc'}
        self.assertNotIn('rules regarding', json.dumps(s.payload(other, 'jev', 'F2_json')))

if __name__ == '__main__':
    unittest.main()
