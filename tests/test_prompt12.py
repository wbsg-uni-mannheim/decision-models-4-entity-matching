import json
import unittest
from pathlib import Path

import expanded_protocol
import prompt12 as p
import protocol

PROJECT = Path(__file__).resolve().parents[1]
CASE = {'benchmark': 'abt-buy', 'pair_id': 'SECRET_ID', 'label': 1, 'input_hash': 'h',
        'records': {'left': {'name': 'Sony TV', 'price': 499}, 'right': {'name': 'Sony television', 'price': None}}}


def instruction(body):
    return body['questions']['match']['instructions'] if 'questions' in body else body['messages'][0]['content']


class Prompt12Tests(unittest.TestCase):
    def test_twelve_prompts_eleven_new(self):
        self.assertEqual(len(p.PROMPTS), 12)
        self.assertEqual(len(set(p.VARIANTS)), 12)
        self.assertEqual(p.NEW_VARIANTS, p.VARIANTS[1:])

    def test_only_instruction_changes(self):
        for b in protocol.BENCHMARKS:
            case = {**CASE, 'benchmark': b}
            for m in protocol.MODELS:
                base = protocol.payload(case, m, 'B0')
                for v in p.VARIANTS:
                    body = p.payload(case, m, v)
                    self.assertNotIn('SECRET_ID', json.dumps(body))
                    self.assertEqual(body['model'], base['model'])
                    self.assertEqual(body['provider'], base['provider'])
                    if m == 'luna':
                        self.assertEqual(body['messages'][1], base['messages'][1])
                        self.assertEqual(body['response_format'], base['response_format'])
                        self.assertEqual(body['max_tokens'], base['max_tokens'])
                    else:
                        self.assertEqual(body['state'], base['state'])
                        self.assertEqual(body['questions']['match']['criteria'], base['questions']['match']['criteria'])
                        self.assertEqual(body['questions']['match']['type'], 'noul')
                    if v != 'B0':
                        self.assertNotEqual(instruction(body), instruction(base))

    def test_existing_variants_byte_identical(self):
        for m in protocol.MODELS:
            for v in ['B0', 'X_P_domain_0', 'X_P_conservative_0', 'X_P_evidence_0']:
                self.assertEqual(protocol.digest(p.payload(CASE, m, v)),
                                 protocol.digest(expanded_protocol.payload(CASE, m, v)))

    def test_astra_rules_match_frozen_templates(self):
        spec = json.loads((PROJECT/'configs/expanded_rules_manifest.json').read_text())
        empty = {'records': {'left': {}, 'right': {}}}
        for b in protocol.BENCHMARKS:
            for m in [m for m in protocol.MODELS if m not in p.EXTRA_MODELS]:
                self.assertEqual(p.payload({**empty, 'benchmark': b}, m, 'R_rules'),
                                 spec['request_templates'][b][m]['R_rules'])

    def test_pplx_v12_registration(self):
        p.register_models()
        body = p.payload(CASE, 'pplx_v12', 'B0')
        self.assertEqual(body['model'], 'perplexity/pplx-decider-v1.2')
        self.assertEqual(body['provider'], {'only': ['perplexity'], 'allow_fallbacks': False})
        self.assertEqual(body['state'], protocol.payload(CASE, 'pplx', 'B0')['state'])
        self.assertEqual(p.run_variants('pplx_v12'), p.VARIANTS)
        self.assertEqual(p.run_variants('jev'), p.NEW_VARIANTS)
        self.assertNotIn('pplx', p.RUN_MODELS)

    def test_domain_wording(self):
        self.assertIn('publications', p.prompt_text('semi-heter', 'PE_domain_simple'))
        self.assertIn('product descriptions', p.prompt_text('wdc', 'PE_domain_simple'))
        self.assertIn('publication features', p.prompt_text('dblp-acm', 'R_handwritten'))
        self.assertIn('product features', p.prompt_text('billiger-de', 'R_handwritten'))
        for v in p.NEW_VARIANTS:
            self.assertNotIn("Answer with 'Yes'", p.prompt_text('abt-buy', v))


if __name__ == '__main__':
    unittest.main()
