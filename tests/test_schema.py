import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import expanded_schema as s
import protocol
import run

CASE = {'benchmark': 'wdc', 'pair_id': 'SECRET', 'input_hash': 'i', 'label': 1,
        'records': {'left': {'title': 'A'}, 'right': {'title': 'B'}}}


def response(m='jev', variant='C_match', choice=None):
    labels = s.options(variant)
    choice = choice or labels[0]
    probs = {x: (.8 if x == choice else .2/(len(labels)-1)) for x in labels}
    return {'model': protocol.MODELS[m]['id'], 'provider': protocol.MODELS[m]['provider'].split('/')[0],
            'answers': {'match': {'type': 'choice', 'choice': choice, 'probabilities': probs}}, 'usage': {'cost': .001}}


class SchemaTests(unittest.TestCase):
    def test_reverse_wire_hash_and_input_identity(self):
        for m in protocol.MODELS:
            a, b = [s.payload(CASE, m, v) for v in ['C_match', 'C_match_reverse']]
            self.assertNotEqual(s.wire_hash(a), s.wire_hash(b))
            self.assertNotIn('SECRET', s.wire(a))
            if m == 'luna':
                self.assertEqual(a['messages'][1], protocol.payload(CASE, m)['messages'][1])
            else:
                self.assertEqual(a['state'], protocol.payload(CASE, m)['state'])
                self.assertEqual(list(a['questions']['match']['criteria']), ['Match', 'No Match'])
                self.assertEqual(list(b['questions']['match']['criteria']), ['No Match', 'Match'])

    def test_parse_review_and_reject_malformed(self):
        self.assertEqual(s.parse('jev', response(variant='C_review', choice='Review'), 'C_review'), (-1, .1))
        self.assertEqual(s.parse('jev', response(choice='No Match'), 'C_match'), (0, .2))
        for mutation in [lambda d: d.update(provider='wrong'),
                         lambda d: d['answers']['match']['probabilities'].update(Match=float('nan')),
                         lambda d: d['answers']['match']['probabilities'].update(Match=.9),
                         lambda d: d['answers']['match'].update(choice='No Match')]:
            d = response();mutation(d)
            with self.assertRaises(ValueError):
                s.parse('jev', d, 'C_match')

    def test_generative_refusal_and_review(self):
        d = {'model': 'openai/gpt-6-luna', 'provider': 'OpenAI', 'choices': [
            {'finish_reason': 'stop', 'message': {'content': '{"match":"Review"}'}}]}
        self.assertEqual(s.parse('luna', d, 'C_review'), (-1, None))
        d['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(ValueError):
            s.parse('luna', d, 'C_review')

    def test_wire_transport_uses_order_and_shared_account(self):
        captured = []
        class Reply:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps(response(variant='C_match_reverse', choice='Match')).encode()
        def open_request(req, **kw):
            captured.append(req.data.decode())
            return Reply()
        names = ['payload', 'canonical', 'digest', 'parse_response', 'request']
        originals = {k: getattr(run, k) for k in names}
        try:
            s.install()
            with patch.object(run, 'account') as account, patch.object(run.urllib.request, 'urlopen', side_effect=open_request):
                r = run.request(CASE, 'jev', 'C_match_reverse', 'fake')
            self.assertEqual(account.call_args_list[0].args, ('reserve',))
            self.assertEqual(account.call_args_list[1].args, ('settle', .001))
            self.assertEqual(r['request_hash'], s.wire_hash(s.payload(CASE, 'jev', 'C_match_reverse')))
            self.assertLess(captured[0].index('"No Match"'), captured[0].index('"Match"'))
            self.assertEqual(r['prediction'], 1)
        finally:
            for k, v in originals.items():
                setattr(run, k, v)

    def test_review_never_counts_as_correct_gold(self):
        y = np.array([0, 1, 1])
        r = s.selective_metrics(y, np.array([-1, 1, 0]))
        self.assertEqual(r['review_count'], 1)
        self.assertEqual(r['accepted_risk'], .5)
        self.assertEqual(r['resolved_correct_over_all'], 1/3)
        empty = s.selective_metrics(y, np.full(3, -1))
        self.assertIsNone(empty['accepted_risk'])
        self.assertEqual(empty['coverage'], 0.)

    def test_selection_uses_only_development_and_freezes_unique_cells(self):
        def load(m, v, b, split):
            self.assertEqual(split, 'prompt')
            return {'y': np.array([0, 1]), 'pred': np.array([0, 1])}
        with tempfile.TemporaryDirectory() as td, patch.object(s, 'PROJECT', Path(td)), patch.object(s, 'MODELS', ['jev']), patch.object(s, 'BENCHMARKS', {'billiger-de': {}}), patch.object(s, 'load', side_effect=load):
            cells = s.select()
            self.assertEqual(len(cells), 4)
            selected = json.loads((Path(td)/'configs/expanded_schema_selection.json').read_text())
            self.assertEqual(selected['jev']['global'], 'C_match')
            self.assertEqual(selected['jev']['per_benchmark']['billiger-de-50un'], 'C_match')
            self.assertTrue((Path(td)/'configs/expanded_schema_test_cells.json').exists())

    def test_full_report_preserves_review_coverage_and_core_macro(self):
        cases = [{**copy.deepcopy(CASE), 'pair_id': str(i), 'label': i} for i in range(2)]
        def load(m, v, b, split):
            return {'cases': cases, 'y': np.array([0, 1]),
                    'pred': np.array([-1, 1] if v == 'C_review' else [0, 1]),
                    'known_cost_usd': .002, 'unknown_cost_attempts': 1}
        with tempfile.TemporaryDirectory() as td, patch.object(s, 'PROJECT', Path(td)), patch.object(s, 'MODELS', ['jev']), patch.object(s, 'BENCHMARKS', {'billiger-de': {}}), patch.object(s, 'load', side_effect=load), patch.object(s.analyze, 'load', side_effect=load), patch.object(s.analyze, 'bootstrap_delta', return_value={'delta': 0.}):
            s.select()
            p = Path(td)/'artifacts/schema_pilot_status.json';p.parent.mkdir();p.write_text('{"complete":true,"cells":[]}')
            s.report()
            out = Path(td)/'artifacts/results/expanded_schema'
            metrics = json.loads((out/'metrics.json').read_text())
            self.assertEqual(len(metrics), 4)
            self.assertEqual([r['coverage'] for r in metrics if r['condition'] == 'C_review'], [.5, .5])
            macro = json.loads((out/'macro.json').read_text())
            self.assertEqual([r['benchmarks'] for r in macro], [1, 1])
            self.assertEqual(len(json.loads((out/'development.json').read_text())), 5)

    def test_load_counts_failed_attempts_and_checks_order_hash(self):
        with tempfile.TemporaryDirectory() as td, patch.object(s, 'PROJECT', Path(td)):
            case = copy.deepcopy(CASE)
            path = Path(td)/'data/wdc/prompt.json';path.parent.mkdir(parents=True);path.write_text(json.dumps([case]))
            path = Path(td)/'runs/schema/jev/C_match/wdc/prompt.jsonl';path.parent.mkdir(parents=True)
            result = {'pair_id': 'SECRET', 'request_hash': s.wire_hash(s.payload(case, 'jev', 'C_match')),
                      'prediction': 1, 'probability_match': .8, 'attempts': [{'cost': .001, 'response': response()}]}
            path.write_text(json.dumps(result)+'\n')
            failed = {**result, 'attempts': [{'cost': None}]}
            path.with_name('prompt.failures.jsonl').write_text(json.dumps(failed)+'\n')
            d = s.load('jev', 'C_match', 'wdc', 'prompt')
            self.assertEqual(d['known_cost_usd'], .001)
            self.assertEqual(d['unknown_cost_attempts'], 1)
            result['request_hash'] = 'wrong';path.write_text(json.dumps(result)+'\n')
            with self.assertRaises(AssertionError):
                s.load('jev', 'C_match', 'wdc', 'prompt')


if __name__ == '__main__':
    unittest.main()
