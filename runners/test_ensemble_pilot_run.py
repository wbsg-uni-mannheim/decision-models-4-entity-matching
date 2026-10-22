import json
from pathlib import Path
import tempfile
import unittest

import ensemble_pilot_run as pilot
from strong_ensemble import MEMBERS, payload_from_frozen_b0
from prepare_ensemble_pilot import canonical, digest


class PilotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'artifacts').mkdir()
        (self.root/'docs').mkdir()
        (self.root/'docs/APPROVED_TODO_20261014.md').write_text('synthetic test authority')
        self.budget = self.root/'artifacts/budget.json'
        self.budget.write_text(json.dumps({'spent': 60., 'reserved': 0., 'attempts': 0}))
        self.package = self.root/'package'; self.package.mkdir()
        rows = []
        for i in range(16):
            for member in MEMBERS:
                body = payload_from_frozen_b0({'model': 'openai/gpt-6-luna', 'messages': [{},{}]}, member)
                rows.append({'request_id': f'{i}-{member}', 'member': member, 'body': body,
                    'request_sha256': digest(body), 'planning_allowance_usd': .1})
        (self.package/'requests.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows))
        self.hash = pilot.sha(self.package/'requests.jsonl')
        (self.package/'manifest.json').write_text(json.dumps({'requests_sha256': self.hash,
            'pilot_ceiling_usd': 10, 'ensemble_implementation_sha256': pilot.sha(Path(pilot.__file__).parent/'strong_ensemble.py')}))
        self.admission = {'package_sha256': self.hash, 'broad_ensemble_authorization_verified': True,
            'all_project_inference_jobs_reviewed': True,
            'todo_sha256': pilot.sha(self.root/'docs/APPROVED_TODO_20261014.md'),
            'runner_sha256': pilot.sha(Path(pilot.__file__))}

    def response(self, body):
        provider = {'openai/gpt-6.1-sol': 'OpenAI', 'anthropic/claude-opus-5.5': 'Anthropic',
                    'google/gemini-4-argon': 'Google'}[body['model']]
        return {'model': body['model'], 'provider': provider, 'usage': {'cost': .01},
            'choices': [{'finish_reason': 'stop', 'message': {'content': '{"match":true}'}}]}

    def test_success_reserves_before_network_and_no_repeat(self):
        calls = []
        def transport(key, body):
            self.assertEqual(json.loads(self.budget.read_text())['reserved'], 10)
            calls.append(body)
            return 200, self.response(body)
        pilot.run(self.root, self.package, self.admission, 'test', [], transport)
        self.assertEqual(len(calls), 48)
        ledger = json.loads(self.budget.read_text())
        self.assertAlmostEqual(ledger['spent'], 60.48)
        self.assertEqual(ledger['reserved'], 0)
        with self.assertRaises(FileExistsError):
            pilot.run(self.root, self.package, self.admission, 'test', [], transport)
        self.assertEqual(len(calls), 48)

    def test_unknown_cost_keeps_hold_and_stops(self):
        pilot.run(self.root, self.package, self.admission, 'test', [], lambda k,b: (502, {'error':'unknown'}))
        self.assertEqual(json.loads(self.budget.read_text())['reserved'], 10)
        self.assertIn('unknown_cost_hold_retained', (self.package/'execution.jsonl').read_text())

    def test_transport_exception_keeps_hold(self):
        def transport(k,b):
            raise TimeoutError()
        pilot.run(self.root, self.package, self.admission, 'test', [], transport)
        self.assertEqual(json.loads(self.budget.read_text())['reserved'], 10)

    def test_bad_identity_stops_but_accounts_cost(self):
        def transport(k,b):
            value = self.response(b); value['provider'] = 'unexpected'
            return 200, value
        pilot.run(self.root, self.package, self.admission, 'test', [], transport)
        self.assertAlmostEqual(json.loads(self.budget.read_text())['spent'], 60.01)
        self.assertEqual(json.loads(self.budget.read_text())['reserved'], 0)

    def test_active_chain_blocks_before_journal_or_network(self):
        with self.assertRaises(ValueError):
            pilot.run(self.root, self.package, self.admission, 'test', ['365636 running'])
        self.assertFalse((self.package/'execution.jsonl').exists())

    def test_project_budget_and_changed_body_block(self):
        self.budget.write_text('{"spent":91,"reserved":0}')
        with self.assertRaises(ValueError):
            pilot.ledger(self.root, self.hash, 'reserve')
        (self.package/'requests.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):
            pilot.validate(self.root, self.package, self.admission, [])

    def test_cost_validation(self):
        for cost in [None, True, -1, float('nan')]:
            with self.assertRaises(ValueError):
                pilot.money(cost)

    def test_parallel_batch_hold_survives_pilot_and_cap_is_joint(self):
        from batch_budget import update
        approval = {'approved_by': 'Aaron', 'package_sha256': 'batch', 'max_cost_usd': 11,
                    'user_message_reference': 'synthetic'}
        self.budget.write_text('{"spent":60,"reserved":0.5}')
        update(self.root, 'reserve', 'batch', 'batch', 11, approval)
        value = pilot.ledger(self.root, self.hash, 'reserve')
        self.assertEqual(value['reserved'], 21.5)
        value = pilot.ledger(self.root, self.hash, 'reconcile_known', .25)
        self.assertEqual(value['reserved'], 11.5)
        self.assertEqual(value['direct_openai_commitments']['batch']['status'], 'reserved')
        self.budget.write_text('{"spent":80,"reserved":11}')
        with self.assertRaises(ValueError): pilot.ledger(self.root, self.hash, 'reserve')

    def test_racing_pilot_and_batch_cannot_overbook(self):
        from batch_budget import update
        from concurrent.futures import ThreadPoolExecutor
        self.budget.write_text('{"spent":85,"reserved":0}')
        approval = {'approved_by': 'Aaron', 'package_sha256': 'batch', 'max_cost_usd': 11,
                    'user_message_reference': 'synthetic'}
        def batch(): return update(self.root, 'reserve', 'batch', 'batch', 11, approval)
        def ensemble(): return pilot.ledger(self.root, self.hash, 'reserve')
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(batch), pool.submit(ensemble)]
            errors = [f.exception() for f in futures]
        self.assertEqual(sum(e is None for e in errors), 1)
        self.assertEqual(sum(isinstance(e, ValueError) for e in errors), 1)
        value = json.loads(self.budget.read_text())
        self.assertLessEqual(value['spent']+value['reserved'], 100)

    def test_known_prefix_continuation_no_duplicate_call_or_cost(self):
        rows=[json.loads(s) for s in (self.package/'requests.jsonl').read_text().splitlines()]
        events=[]
        for row in rows[:3]:
            events.extend([{'event':'request_start_unknown_until_response','request_id':row['request_id'],
                            'request_sha256':row['request_sha256']},
                           {'event':'response','request_id':row['request_id'],'status':200,
                            'data':self.response(row['body'])}])
        events.append({'event':'stopped_known_cost','known_cost_usd':.03,
                       'reason':'Observed model/provider identity differs; inspect without fallback'})
        prior=self.package/'execution_attempt1.jsonl'
        prior.write_text(''.join(json.dumps(e)+'\n' for e in events))
        pilot.ledger(self.root,self.hash,'reserve')
        pilot.ledger(self.root,self.hash,'reconcile_known',.03)
        self.admission['prior_journal_sha256']=pilot.sha(prior)
        calls=[]
        def transport(k,b):
            calls.append(b)
            return 200,self.response(b)
        pilot.run(self.root,self.package,self.admission,'test',[],transport,prior_journal=prior)
        self.assertEqual(len(calls),45)
        value=json.loads(self.budget.read_text())
        self.assertAlmostEqual(value['spent'],60.48)
        self.assertEqual(value['reserved'],0)
        with self.assertRaises(ValueError): pilot.ledger(self.root,self.hash,'reserve',prior_cost=.48)
        events[-1]['event']='transport_unknown_hold_retained'
        prior.write_text(''.join(json.dumps(e)+'\n' for e in events))
        with self.assertRaises(ValueError): pilot.known_prefix(prior,rows)


if __name__ == '__main__':
    unittest.main()
