import copy
import io
import json
from pathlib import Path
import shutil
import socket
import tempfile
import types
import unittest
from unittest.mock import patch

import analyze
import expanded_rules as er
import protocol
import run
import rules_drained as component
from rules_remaining_cost_plan import matrix


class Response(io.BytesIO):
    status = 200


class RulesDrainTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        for d in ('src', 'configs', 'artifacts/astra_agent'):
            (self.root/d).mkdir(parents=True)
        for name in ('expanded_rules', 'run', 'protocol', 'expanded_protocol'):
            shutil.copy(er.PROJECT/'src'/(name+'.py'), self.root/'src'/(name+'.py'))
        self.rules = types.SimpleNamespace(**{**vars(er), 'PROJECT': self.root})
        self.cells = matrix(protocol.MODELS, protocol.BENCHMARKS)
        learned = {'schema_version': 1, 'generator_model': 'gpt-6-astra',
                   'families': {b: {'rules': ['Compare the complete entity.']} for b in protocol.BENCHMARKS}}
        rule_values = er.validate_rules(learned)
        self.payload = component.bind(er.payload, RULES=rule_values)
        self.cases = {}
        training = {'excluded_split_hashes': {}, 'benchmarks': {}}
        for b in [*protocol.BENCHMARKS, 'billiger-de-50un']:
            for split in ('prompt', 'test'):
                records = {'left': {'name': split+' A'}, 'right': {'name': split+' B'}}
                c = {'benchmark': b, 'pair_id': b+'#'+split, 'label': 1,
                     'records': records, 'input_hash': protocol.digest(records)}
                self.cases[b, split] = c
                self.write('data/'+b+'/'+split+'.json', [c])
                training['excluded_split_hashes']['data/'+b+'/'+split+'.json'] = protocol.digest([c])
            if b in protocol.BENCHMARKS:
                self.write('data/'+b+'/rules_train.json', [])
                training['benchmarks'][b] = {'training_sha256': protocol.digest([])}
        receipt = {'accepted': True, 'fixture': True}
        self.write('configs/learned_rules.json', learned)
        self.write('configs/rule_agent_training_manifest.json', training)
        self.write('artifacts/astra_agent/acceptance.json', receipt)
        spec = {'implementation_sha256': component.sha(self.root/'src/expanded_rules.py'),
                'protocol_sha256': component.sha(self.root/'src/protocol.py'),
                'expanded_protocol_sha256': component.sha(self.root/'src/expanded_protocol.py'),
                'training_manifest_sha256': protocol.digest(training), 'rules_sha256': protocol.digest(learned),
                'acceptance_sha256': protocol.digest(receipt), 'models': protocol.MODELS, 'cells': self.cells,
                'request_templates': {b: {m: {v: self.payload({'benchmark': b, 'records': {'left': {}, 'right': {}}}, m, v)
                                      for v in ('B0', 'R_rules', er.MANUAL)} for m in protocol.MODELS}
                                      for b in [*protocol.BENCHMARKS, 'billiger-de-50un']}}
        self.write('configs/expanded_rules_manifest.json', spec)
        for c in self.cells:
            if c['variant'] == 'R_rules':
                self.cache(dict(c, variant='B0', namespace='quality'))
        # Two actual-format manual cache hits, including a test entry shared with
        # confirmation; every other fixed cell must get exactly one new request.
        self.cached = [c for c in self.cells if c['model'] == 'jev' and
                       c['variant'] == er.MANUAL and c['benchmark'] == 'abt-buy']
        for c in self.cached:
            self.cache(c)
        account = component.bind(run.account, PROJECT=self.root)
        self.inference = types.SimpleNamespace(**{**vars(run), 'PROJECT': self.root,
                                                  'account': account, 'key': lambda: 'fake'})
        self.budget = self.root/'artifacts/budget.json'
        self.write('artifacts/budget.json', {'spent': 0., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0})
        self.journal = self.root/'artifacts/attempt'
        self.original = (run.payload, analyze.payload, copy.deepcopy(er.RULES))
        denied = patch.object(socket, 'socket', side_effect=AssertionError('Network prohibited'))
        denied.start(); self.addCleanup(denied.stop)

    def write(self, rel, obj):
        path = self.root/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj)+'\n')

    def raw(self, m):
        value = {'model': protocol.MODELS[m]['id'], 'provider': protocol.MODELS[m]['provider'].split('/')[0],
                 'usage': {'cost': .001}}
        if m == 'luna':
            value['choices'] = [{'finish_reason': 'stop', 'message': {'content': '{"match": true}'}}]
        else:
            value['answers'] = {'match': {'type': 'noul', 'noul': .8}}
        return value

    def cache(self, c):
        m, v, b, s = (c[k] for k in ('model', 'variant', 'benchmark', 'split'))
        case = self.cases[b, s]
        raw = self.raw(m)
        pred, prob = protocol.parse_response(m, raw)
        row = {'benchmark': b, 'pair_id': case['pair_id'], 'input_hash': case['input_hash'],
               'request_hash': protocol.digest(self.payload(case, m, v)), 'model_alias': m,
               'variant': v, 'prediction': pred, 'probability_match': prob,
               'attempts': [{'status': 200, 'seconds': 0., 'cost': .001, 'error': None, 'response': raw}],
               'completed_at': 'fixture', 'study_run_id': 1}
        self.write(f'runs/{c["namespace"]}/{m}/{v}/{b}/{s}.jsonl', row)

    def execute(self):
        return component.run_fixed_matrix(self.root, self.rules, self.inference, protocol, self.journal)

    def test_full_real_frozen_requests_two_phases_exact_shared_cache_reuse(self):
        seen = []
        frozen, _, _, _ = component.prepare(self.root, self.rules, protocol)
        def http(req, **kw):
            body = json.loads(req.data)
            # Records distinguish phases. Assert every development response is
            # durable and raw-valid at the very first full-test HTTP call.
            if 'test A' in req.data.decode():
                component.audit(self.root, frozen, protocol,
                                [c for c in self.cells if c['split'] == 'prompt'], True)
            seen.append(body)
            m = next(m for m, info in protocol.MODELS.items() if info['id'] == body['model'])
            return Response(json.dumps(self.raw(m)).encode())
        shared_before = {p: p.read_bytes() for p in (self.root/'runs/quality').rglob('*.jsonl')}
        with patch('urllib.request.urlopen', side_effect=http):
            result = self.execute()
        self.assertEqual(result['status'], 'inference_complete_report_pending')
        self.assertEqual(len(seen), 236)
        self.assertCountEqual([protocol.digest(body) for body in seen],
            [protocol.digest(self.payload(self.cases[c['benchmark'], c['split']], c['model'], c['variant']))
             for c in self.cells if c not in self.cached])
        self.assertEqual((run.payload, analyze.payload, er.RULES), self.original)
        self.assertTrue(all(p.read_bytes() == raw for p, raw in shared_before.items()))
        self.assertTrue(all(p['complete'] and len(p['models']) == 7 for p in result['phases']))
        self.assertFalse((self.root/'artifacts/expanded_rules_complete.json').exists())
        budget = json.loads(self.budget.read_text())
        self.assertAlmostEqual(budget['spent'], .236)
        self.assertAlmostEqual(budget['reserved'], 3.)
        for c in self.cells:
            row = component.rows(self.root/f'runs/{c["namespace"]}/{c["model"]}/{c["variant"]}/{c["benchmark"]}/{c["split"]}.jsonl')[0]
            self.assertEqual(row['study_run_id'], 1)
        with self.assertRaisesRegex(ValueError, 'Already attempted'):
            self.execute()

    def test_budget_drains_all_models_and_blocks_entire_test_phase(self):
        self.write('artifacts/budget.json', {'spent': 97., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0})
        before = self.budget.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')) as http:
            result = self.execute()
        http.assert_not_called()
        self.assertEqual(self.budget.read_bytes(), before)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['phases']), 1)
        phase = result['phases'][0]
        self.assertEqual(len(phase['models']), 7)
        self.assertTrue(all(m['status'] == 'failed' for m in phase['models']))
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in phase['cells']), 7)
        self.assertFalse((self.journal/'test_models.jsonl').exists())
        self.assertFalse(list((self.root/'runs').rglob('*.failures.jsonl')))
        self.assertFalse(list((self.root/'runs/rules').rglob('test*')))

    def test_wrong_template_matrix_or_source_rejected_before_attempt(self):
        path = self.root/'configs/expanded_rules_manifest.json'
        original = path.read_bytes()
        spec = json.loads(original)
        spec['cells'] = spec['cells'][:-1]
        path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, 'matrix'):
            self.execute()
        spec = json.loads(original)
        spec['request_templates'] = {}
        path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, 'templates'):
            self.execute()
        path.write_bytes(original)
        rules = copy.copy(self.rules)
        rules.__file__ = __file__
        with self.assertRaisesRegex(ValueError, 'source changed'):
            component.prepare(self.root, rules, protocol)
        self.assertFalse(self.journal.exists())

    def test_complete_cached_development_then_all_test_budget_stops(self):
        for c in self.cells:
            if c['split'] == 'prompt':
                self.cache(c)
        self.write('artifacts/budget.json', {'spent': 97., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0})
        before = self.budget.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')) as http:
            result = self.execute()
        http.assert_not_called()
        self.assertEqual(self.budget.read_bytes(), before)
        self.assertEqual([p['complete'] for p in result['phases']], [True, False])
        self.assertEqual([len(p['models']) for p in result['phases']], [7, 7])
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in result['phases'][1]['cells']), 7)
        self.assertFalse((self.root/'artifacts/expanded_rules_complete.json').exists())

    def test_raw_manual_cache_tampering_and_incomplete_b0_rejected(self):
        p = self.root/'runs/quality/jev/X_P_domain_0/abt-buy/test.jsonl'
        original = p.read_bytes()
        row = json.loads(original)
        row['request_hash'] = 'wrong'
        p.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.execute()
        p.write_bytes(original)
        (self.root/'runs/quality/jev/B0/abt-buy/test.jsonl').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.execute()
        self.assertFalse(self.journal.exists())

    def test_completion_and_workspace_rejected(self):
        with self.assertRaisesRegex(ValueError, 'workspace'):
            component.prepare(self.root/'wrong', self.rules, protocol)
        self.write('artifacts/expanded_rules_complete.json', {})
        with self.assertRaisesRegex(ValueError, 'already complete'):
            self.execute()
        self.assertFalse(self.journal.exists())


if __name__ == '__main__':
    unittest.main()
