import json
from pathlib import Path
import tempfile
import unittest

from fresh_run_cost_plan import inventory, plan, sha


class CostPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root/'runs/quality/m/B0/b/test.jsonl'
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({'pair_id': 'a', 'request_hash': 'req',
                                       'attempts': [{'cost': .2}, {'cost': None}]})+'\n')
        self.fail = self.path.with_name('test.failures.jsonl')
        self.fail.write_text(json.dumps({'pair_id': 'a', 'request_hash': 'req',
                                        'attempts': [{'cost': .1}, {'cost': None}]})+'\n')
        self.source = {'namespace': 'quality', 'model': 'm', 'variant': 'B0', 'benchmark': 'b',
                       'split': 'test', 'response_count': 1, 'request_hashes_verified': True,
                       'cache_sha256': sha(self.path), 'failure_sha256': sha(self.fail)}
        self.proof = self.root/'proof.json'
        self.write_proof()
        (self.root/'artifacts').mkdir()
        (self.root/'artifacts/budget.json').write_text(json.dumps({'spent': 90, 'reserved': 5}))

    def write_proof(self):
        self.proof.write_text(json.dumps({'run_id': 1, 'verified_source_cells': 1,
                                         'sources': [self.source], 'stages': ['baseline']}))

    def test_shared_source_counted_once_unknowns_not_free(self):
        result = plan(self.root, ['proof.json', 'proof.json'])
        self.assertEqual(result['deduplicated_source_cells'], 1)
        self.assertAlmostEqual(result['two_fresh_runs_known_cost_proxy_usd'], .6)
        self.assertEqual(result['two_fresh_runs_unknown_allowance_scenario_usd'], .2)
        self.assertEqual(result['uncommitted_headroom_usd'], 5)
        self.assertFalse(result['execution_admitted'])

    def test_changed_cache_or_failures_rejected(self):
        self.path.write_text(self.path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'changed cache'):
            inventory(self.root, ['proof.json'])
        self.source['cache_sha256'] = sha(self.path)
        self.write_proof()
        self.fail.write_text(self.fail.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'failure journal'):
            inventory(self.root, ['proof.json'])

    def test_duplicate_proof_conflict_rejected(self):
        other = json.loads(self.proof.read_text())
        other['sources'][0]['cache_sha256'] = 'different'
        (self.root/'other.json').write_text(json.dumps(other))
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            inventory(self.root, ['proof.json', 'other.json'])

    def test_failure_must_belong_to_same_request(self):
        self.fail.write_text(json.dumps({'pair_id': 'a', 'request_hash': 'wrong', 'attempts': []})+'\n')
        self.source['failure_sha256'] = sha(self.fail)
        self.write_proof()
        with self.assertRaisesRegex(ValueError, 'Failure outside'):
            inventory(self.root, ['proof.json'])

    def test_wrong_run_and_unverified_proof_rejected(self):
        self.source['request_hashes_verified'] = False
        self.write_proof()
        with self.assertRaises(ValueError):
            inventory(self.root, ['proof.json'])
        wrong = json.loads(self.proof.read_text())
        wrong['run_id'] = 2
        self.proof.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, 'Run1'):
            inventory(self.root, ['proof.json'])


if __name__ == '__main__':
    unittest.main()
