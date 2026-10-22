import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import resume_schema_drained as wrapper
from test_schema_drained import SchemaDrainTests, protocol, analyze


class WrapperExecutionTests(unittest.TestCase):
    write_source = SchemaDrainTests.write_source
    http = SchemaDrainTests.http
    def setUp(self):
        SchemaDrainTests.setUp(self)
        (self.root/'artifacts/operations').mkdir()
        data_check = patch.object(wrapper, 'verify_data')
        data_check.start(); self.addCleanup(data_check.stop)
        self.old = self.root/wrapper.PRIOR
        self.old.write_text('{"event":"end","status":"failed"}\n')
        self.old_bytes = self.old.read_bytes()
        self.admission = self.root/'admission.json'
        self.admission.write_text('{}')
        self.spec = {'admitted': True, 'run_id': 1, 'stage': 'schema', 'predecessor': '365969',
            'slurm_job_id': '999', 'budget_cap_usd': 100, 'max_invocations': 1,
            'historical_unknown_outcomes_disclosed': True, 'unobserved_worker_causes_disclosed': True,
            'source_sha256': {p: 'synthetic-gate-fixture' for p in wrapper.REQUIRED}}
        self.checked = {'prior_journal_sha256': wrapper.sha(self.old), 'policy': {'policy_sha256': 'test-policy'}}
        # Complete B0 report controls, synthetic and outside any real workspace.
        for b, case in self.cases.items():
            case['entity_ids'] = {'left': 'l', 'right': 'r'}
            for split in ('prompt', 'test'):
                (self.root/'data'/b/(split+'.json')).write_text(json.dumps([case]))
            for m in protocol.MODELS:
                raw = self.raw(m)
                row = {'pair_id': case['pair_id'], 'model_alias': m, 'variant': 'B0', 'benchmark': b,
                    'input_hash': case['input_hash'], 'study_run_id': 1,
                    'request_hash': protocol.digest(protocol.payload(case, m, 'B0')),
                    'prediction': 1, 'probability_match': None if m == 'luna' else .8,
                    'attempts': [{'status': 200, 'seconds': .1, 'cost': .001, 'response': raw}]}
                folder = self.root/'runs/quality'/m/'B0'/b
                folder.mkdir(parents=True, exist_ok=True)
                (folder/'test.jsonl').write_text(json.dumps(row)+'\n')

    def raw(self, m):
        r = {'model': protocol.MODELS[m]['id'], 'provider': protocol.MODELS[m]['provider'].split('/')[0],
             'usage': {'cost': .001}}
        if m == 'luna':
            r['choices'] = [{'finish_reason': 'stop', 'message': {'content': '{"match": true}'}}]
        else:
            r['answers'] = {'match': {'type': 'noul', 'noul': .8}}
        return r

    def execute(self):
        return wrapper._execute_bound(self.root, self.admission, self.spec, '999', self.schema,
                                       analyze, protocol, self.inference)

    def test_real_component_and_frozen_report_success_append_only_lock_held(self):
        def http(req, **kwargs):
            with (self.root/'artifacts/operations/schema_recovery.lock').open('a') as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self.http(req, **kwargs)
        with patch.object(wrapper, 'verify_ready', return_value=self.checked), patch('urllib.request.urlopen', side_effect=http):
            self.assertEqual(self.execute(), 0)
        self.assertTrue(self.old.read_bytes().startswith(self.old_bytes))
        end = json.loads(self.old.read_text().splitlines()[-1])
        self.assertEqual(end['status'], 'complete')
        self.assertEqual(end['prior_journal_sha256'], self.checked['prior_journal_sha256'])
        marker = json.loads((self.root/'artifacts/expanded_schema_complete.json').read_text())
        self.assertEqual(marker['test_cells'], len(self.cells))
        self.assertFalse(marker['full_expanded_study_complete'])
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'], 3)
        with self.assertRaisesRegex(ValueError, 'Already attempted'):
            self.execute()

    def test_concurrent_budget_exhaustion_drains_without_completion(self):
        # Simulate a separate authorized writer consuming headroom after preflight.
        def checked(*args):
            self.budget.write_text(json.dumps({'spent': 97., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0}))
            return self.checked
        with patch.object(wrapper, 'verify_ready', side_effect=checked), patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            self.assertEqual(self.execute(), 1)
        self.assertTrue(self.old.read_bytes().startswith(self.old_bytes))
        records = [json.loads(x) for x in self.old.read_text().splitlines()]
        self.assertEqual(records[-1]['status'], 'failed')
        result = records[-2]['result']
        self.assertEqual(len(result['models']), 7)
        self.assertEqual(sum(x['budget_rejections'] for x in result['operations']), 7)
        self.assertFalse((self.root/'artifacts/expanded_schema_complete.json').exists())
        self.assertEqual(json.loads(self.budget.read_text())['reserved'], 3)

    def test_unadmitted_and_report_raw_corruption_do_not_mark_complete(self):
        self.spec['admitted'] = False
        with self.assertRaisesRegex(ValueError, 'admission'):
            self.execute()
        self.assertEqual(self.old.read_bytes(), self.old_bytes)
        self.assertFalse((self.root/wrapper.ATTEMPT).exists())
        self.write_source('jev', 'C_match', 'abt-buy', 'test', 'schema', True)
        p = self.root/'runs/quality/jev/B0/abt-buy/test.jsonl'
        row = json.loads(p.read_text()); row['prediction'] = 0
        p.write_text(json.dumps(row)+'\n')
        with self.assertRaises(ValueError):
            wrapper.finish_report(self.root, self.schema, analyze, protocol, [['jev', 'C_match', 'abt-buy']])
        self.assertFalse((self.root/'artifacts/expanded_schema_complete.json').exists())


class GateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for d in ('artifacts/operations', 'configs'):
            (self.root/d).mkdir(parents=True)
        def save(p, x):
            (self.root/p).write_text(json.dumps(x)); return wrapper.sha(self.root/p)
        self.save = save
        save('artifacts/budget.json', {'spent': 1, 'reserved': 3})
        save('configs/expanded_execution.json', {'e2_schema_job': 999, 'e2_rules_job': 300, 'statistics_full_job': 355})
        save(wrapper.PRIOR, [])
        dh = save('diagnosis.json', {'job': '365969', 'status': 'terminal_partial_schema_audited_no_recovery',
            'source_sha256': {}, 'frozen_artifact_sha256': {wrapper.PRIOR: wrapper.sha(self.root/wrapper.PRIOR)},
            'cells': [], 'files_sha256': {}})
        ph = save('proof.json', {'verified': True})
        ch = save('cost.json', {'diagnosis_sha256': dh, 'missing_pairs': 0, 'quote_or_hard_bound': False,
            'source_verification_sha256': ph, 'known_proxy_usd': 2, 'historical_unknown_scenario_usd': 1})
        self.spec = {'admitted': True, 'run_id': 1, 'stage': 'schema', 'predecessor': '365969',
            'slurm_job_id': '999', 'budget_cap_usd': 100, 'max_invocations': 1,
            'historical_unknown_outcomes_disclosed': True, 'unobserved_worker_causes_disclosed': True,
            'source_sha256': {}, 'diagnosis': 'diagnosis.json', 'diagnosis_sha256': dh,
            'cost_plan': 'cost.json', 'cost_plan_sha256': ch, 'cost_source_verification': 'proof.json',
            'control_files_sha256': {}, 'parallel_policy': 'policy.json', 'parallel_policy_sha256': 'policy',
            'reviewed_dependencies': {'999': '(null)', '300': 'afterok:999', '355': 'afterok:999', '777': '(null)'},
            'reviewed_job_write_cells': {'777': [['quality', 'other', 'B0', 'x', 'test']]}}
        self.policy = {'policy_sha256': 'policy', 'reviewed_jobs': [{'id': i, 'self': i == '999'} for i in ('999', '300', '355', '777')]}
        def command(args, **kwargs):
            if args[0] == 'sacct':
                return '365969|FAILED|1:0|end\n365969.batch|FAILED|1:0|end\n365969.extern|COMPLETED|0:0|end\n'
            return 'JobState=PENDING Dependency='+self.spec['reviewed_dependencies'][args[3]]
        for patcher in (patch.object(wrapper, 'REQUIRED', set()), patch.object(wrapper, 'review', return_value=self.policy),
                        patch.object(wrapper.subprocess, 'check_output', side_effect=command)):
            patcher.start(); self.addCleanup(patcher.stop)

    def test_scheduler_policy_join_and_disjoint_scope(self):
        self.assertIn('accounting_utc', wrapper.verify_ready(self.root, self.spec, '999', []))
        self.spec['reviewed_dependencies']['355'] = 'afterok:365969'
        with self.assertRaisesRegex(ValueError, 'Dependency joins'):
            wrapper.verify_ready(self.root, self.spec, '999', [])

    def test_afterany_or_alternative_join_does_not_authorize_overlap(self):
        for dep in ('afterany:999,afterok:123', 'afterok:999?afterok:123'):
            self.spec['reviewed_dependencies']['355'] = dep
            with self.assertRaisesRegex(ValueError, 'Dependency joins'):
                wrapper.verify_ready(self.root, self.spec, '999', [])

    def test_extra_schema_journal_not_in_diagnosis_rejected(self):
        path = self.root/'runs/schema/unreviewed/C_review/b/test.jsonl'
        path.parent.mkdir(parents=True); path.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'outside diagnosis scope'):
            wrapper.verify_ready(self.root, self.spec, '999', [])

    def test_changed_cost_evidence_and_missing_scope_rejected(self):
        self.spec['reviewed_job_write_cells'] = {}
        with self.assertRaisesRegex(ValueError, 'writer scope'):
            wrapper.verify_ready(self.root, self.spec, '999', [])
        self.save('proof.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'cost evidence'):
            wrapper.verify_ready(self.root, self.spec, '999', [])

    def test_source_drift_and_current_budget_rejected(self):
        self.spec['source_sha256'] = {'proof.json': 'wrong'}
        with self.assertRaisesRegex(ValueError, 'Source drift'):
            wrapper.verify_ready(self.root, self.spec, '999', [])
        self.spec['source_sha256'] = {}
        self.save('artifacts/budget.json', {'spent': 99, 'reserved': 0})
        with self.assertRaisesRegex(ValueError, 'headroom'):
            wrapper.verify_ready(self.root, self.spec, '999', [])

    def test_overlap_with_actual_protected_cell_rejected(self):
        d = json.loads((self.root/'diagnosis.json').read_text())
        d['cells'] = [{'namespace': 'schema', 'model': 'm', 'variant': 'v', 'benchmark': 'b', 'split': 'test', 'missing': 0}]
        dh = self.save('diagnosis.json', d)
        self.spec['diagnosis_sha256'] = dh
        c = json.loads((self.root/'cost.json').read_text()); c['diagnosis_sha256'] = dh
        self.spec['cost_plan_sha256'] = self.save('cost.json', c)
        self.spec['reviewed_job_write_cells']['777'] = [['schema', 'm', 'v', 'b', 'test']]
        with self.assertRaisesRegex(ValueError, 'overlapping writer scope'):
            wrapper.verify_ready(self.root, self.spec, '999', [('m', 'v', 'b')])


if __name__ == '__main__':
    unittest.main()
