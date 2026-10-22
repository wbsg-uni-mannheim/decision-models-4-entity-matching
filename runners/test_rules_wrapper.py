import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_rules_drained as wrapper
from test_rules_drained import RulesDrainTests, Response, protocol, analyze


class WrapperExecutionTests(unittest.TestCase):
    write = RulesDrainTests.write
    raw = RulesDrainTests.raw
    cache = RulesDrainTests.cache

    def setUp(self):
        RulesDrainTests.setUp(self)
        (self.root/'artifacts/operations').mkdir()
        # The real frozen bootstrap uses entity IDs; fixture inputs/manifest are
        # updated consistently before any request or wrapper invocation.
        training_path = self.root/'configs/rule_agent_training_manifest.json'
        training = json.loads(training_path.read_text())
        for (b, split), c in self.cases.items():
            c['entity_ids'] = {'left': 'l', 'right': 'r'}
            self.write(f'data/{b}/{split}.json', [c])
            training['excluded_split_hashes'][f'data/{b}/{split}.json'] = protocol.digest([c])
        self.write('configs/rule_agent_training_manifest.json', training)
        manifest = self.root/'configs/expanded_rules_manifest.json'
        spec = json.loads(manifest.read_text())
        spec['training_manifest_sha256'] = protocol.digest(training)
        manifest.write_text(json.dumps(spec))
        self.spec = {'admitted': True, 'run_id': 1, 'stage': 'rules', 'predecessor': '365300',
            'predecessor_never_started': True, 'slurm_job_id': '999', 'budget_cap_usd': 100,
            'max_invocations_per_model_per_phase': 1, 'historical_unknown_outcomes_disclosed': True,
            'source_sha256': {p: 'fixture-only' for p in wrapper.REQUIRED}}
        self.admission = self.root/'admission.json'
        self.admission.write_text(json.dumps(self.spec))
        self.checked = {'policy': {'policy_sha256': 'fixture-policy'}}
        data_check = patch.object(wrapper, 'verify_data')
        data_check.start(); self.addCleanup(data_check.stop)

    def execute(self):
        return wrapper._execute_bound(self.root, self.admission, self.spec, '999', self.rules,
                                      analyze, protocol, self.inference)

    def assert_locks_held(self):
        for rel in wrapper.LOCKS:
            with (self.root/rel).open('a') as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_full_frozen_inference_report_and_marker_under_both_locks(self):
        seen = []
        def http(req, **kw):
            self.assert_locks_held()
            body = json.loads(req.data); seen.append(body)
            m = next(m for m, info in protocol.MODELS.items() if info['id'] == body['model'])
            return Response(json.dumps(self.raw(m)).encode())
        original_report = wrapper.finish_report
        def report(*args):
            self.assert_locks_held()
            return original_report(*args)
        with patch.object(wrapper, 'verify_ready', return_value=self.checked), \
                patch('urllib.request.urlopen', side_effect=http), \
                patch.object(wrapper, 'finish_report', side_effect=report):
            self.assertEqual(self.execute(), 0)
        self.assertEqual(len(seen), 236)
        self.assertEqual((self.root/wrapper.STAGE).read_bytes(),
                         (self.root/wrapper.ATTEMPT/'execution.jsonl').read_bytes())
        rows = [json.loads(x) for x in (self.root/wrapper.STAGE).read_text().splitlines()]
        self.assertEqual([r['event'] for r in rows], ['start', 'inference_end', 'end'])
        self.assertEqual(rows[-1]['status'], 'complete')
        self.assertEqual(rows[-1]['admission_sha256'], wrapper.sha(self.admission))
        self.assertEqual([len(p['models']) for p in rows[1]['result']['phases']], [7, 7])
        marker = json.loads((self.root/'artifacts/expanded_rules_complete.json').read_text())
        self.assertEqual(marker['test_cells_per_condition'], 63)
        self.assertFalse(marker['full_expanded_study_complete'])
        metrics = json.loads((self.root/'artifacts/results/expanded_rules/metrics.json').read_text())
        self.assertEqual(len(metrics), 357)
        self.assertTrue(all(m['n'] == 1 and m['f1'] == 1 for m in metrics))
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'], 3.)
        from test_rules_drained import er, run
        self.assertEqual((run.payload, analyze.payload, er.RULES), self.original)
        with self.assertRaisesRegex(ValueError, 'Already attempted'):
            self.execute()

    def test_budget_after_gate_drains_without_test_or_report(self):
        def checked(*args):
            self.write('artifacts/budget.json', {'spent': 97., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0})
            return self.checked
        with patch.object(wrapper, 'verify_ready', side_effect=checked), \
                patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            self.assertEqual(self.execute(), 1)
        rows = [json.loads(x) for x in (self.root/wrapper.STAGE).read_text().splitlines()]
        self.assertEqual(rows[-1]['status'], 'failed')
        phase = rows[1]['result']['phases'][0]
        self.assertEqual(len(phase['models']), 7)
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in phase['cells']), 7)
        self.assertFalse((self.root/wrapper.ATTEMPT/'models/test_models.jsonl').exists())
        self.assertFalse((self.root/'artifacts/expanded_rules_complete.json').exists())
        self.assertEqual(json.loads(self.budget.read_text())['reserved'], 3.)

    def test_unadmitted_or_preexisting_stage_never_overwritten(self):
        self.spec['admitted'] = False
        with self.assertRaisesRegex(ValueError, 'admission'):
            self.execute()
        self.assertFalse((self.root/wrapper.STAGE).exists())
        self.spec['admitted'] = True
        path = self.root/wrapper.STAGE
        path.write_text('{"event":"historical"}\n')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Already attempted'):
            self.execute()
        self.assertEqual(path.read_bytes(), before)

    def test_raw_corruption_or_incomplete_matrix_blocks_report(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            wrapper.finish_report(self.root, self.rules, analyze, protocol)
        for c in self.cells:
            self.cache(c)
        path = self.root/'runs/quality/jev/B0/abt-buy/test.jsonl'
        row = json.loads(path.read_text()); row['prediction'] = 0
        path.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError, 'Raw parser'):
            wrapper.finish_report(self.root, self.rules, analyze, protocol)
        self.assertFalse((self.root/'artifacts/expanded_rules_complete.json').exists())


class GateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        def save(p, obj):
            path = self.root/p; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj)); return wrapper.sha(path)
        self.save = save
        self.cells = [{'namespace': 'rules', 'model': 'm', 'variant': 'R_rules', 'benchmark': 'b', 'split': 'test'}]
        self.audited = [dict(self.cells[0], missing=1, missing_pair_ids_sha256='missing')]
        planner = save('artifacts/operations/rules_remaining_cost_plan.py', {})
        verifier = save('artifacts/verification/check_rules_remaining_cost_20261015.py', {})
        save('artifacts/budget.json', {'spent': 1, 'reserved': 3})
        save('configs/expanded_execution.json', {'e2_rules_job': 999, 'e2_confirmation_job': 636,
                                                'e2_schema_job': 969, 'e6_variants_job': 279, 'statistics_full_job': 355})
        prereqs = {s: {'job_id': j, 'completion_sha256': save(f'artifacts/expanded_{s}_complete.json', {'status': 'complete'})}
                   for s, j in (('confirmation', '636'), ('schema', '969'))}
        totals = {'missing': 1, 'standalone_scenario': {'known_proxy_usd': 2., 'historical_unknown_scenario_usd': 1.}}
        ch = save('cost.json', {'script_sha256': planner, 'quote_or_hard_bound': False, 'cells': self.audited,
            'totals': totals, 'journal_sha256_or_absent': {}, 'protected_sha256': {}, 'input_sha256': {}, 'selection_sha256': {}})
        vh = save('proof.json', {'status': 'independent_raw_cost_missing_matrix_deduplication_passed',
            'scenario_sha256': ch, 'script_sha256': verifier, 'totals': totals})
        dh = save('diagnosis.json', {'status': 'real_read_only_binding_and_raw_cache_verified',
            'execution_component_invoked': False, 'cells': self.audited, 'inventory_sha256_or_absent': {}})
        self.spec = {'admitted': True, 'run_id': 1, 'stage': 'rules', 'predecessor': '365300',
            'predecessor_never_started': True, 'slurm_job_id': '999', 'budget_cap_usd': 100,
            'max_invocations_per_model_per_phase': 1, 'historical_unknown_outcomes_disclosed': True,
            'source_sha256': {}, 'cost_scenario': 'cost.json', 'cost_scenario_sha256': ch,
            'cost_verification': 'proof.json', 'cost_verification_sha256': vh,
            'cache_review': 'diagnosis.json', 'cache_review_sha256': dh, 'prerequisites': prereqs,
            'parallel_policy': 'policy.json', 'parallel_policy_sha256': 'policy',
            'reviewed_dependencies': {'999': '(null)', '355': 'afterok:999', '777': '(null)'},
            'reviewed_job_write_cells': {'777': [['quality', 'other', 'B0', 'x', 'test']]}}
        self.policy = {'policy_sha256': 'policy', 'reviewed_jobs': [{'id': j, 'self': j == '999'} for j in ('999', '355', '777')]}
        self.old_accounting = '365300|CANCELLED by 1197|0:0|Unknown|end\n'
        self.prereq_state = 'COMPLETED'
        self.variant_state = 'COMPLETED'
        def command(args, **kwargs):
            if args[0] == 'sacct':
                jid = args[3]
                if jid == '365300':
                    return self.old_accounting
                state = self.variant_state if jid == '279' else self.prereq_state
                return ''.join(f'{jid}{suffix}|{state}|0:0|end\n' for suffix in ('', '.batch', '.extern'))
            state = 'RUNNING' if args[3] == '999' else 'PENDING'
            return 'JobState='+state+' Dependency='+self.spec['reviewed_dependencies'][args[3]]
        for p in (patch.object(wrapper, 'REQUIRED', set()), patch.object(wrapper, 'review', return_value=self.policy),
                  patch.object(wrapper.subprocess, 'check_output', side_effect=command)):
            p.start(); self.addCleanup(p.stop)

    def ready(self):
        return wrapper.verify_ready(self.root, self.spec, '999', self.cells, self.audited)

    def test_current_full_gate_and_independent_scope(self):
        self.assertIn('prerequisite_accounting_utc', self.ready())
        self.spec['reviewed_job_write_cells']['777'] = [['rules', 'm', 'R_rules', 'b', 'test']]
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.ready()

    def test_never_release_pending_or_previously_started_old_job(self):
        for old in ('365300|PENDING|0:0|Unknown|Unknown\n',
                    '365300|CANCELLED|0:0|2026-10-15T00:00:00|end\n'):
            self.old_accounting = old
            with self.assertRaisesRegex(ValueError, 'not retired'):
                self.ready()

    def test_failed_prerequisite_and_marker_mismatch_rejected(self):
        self.prereq_state = 'FAILED'
        with self.assertRaisesRegex(ValueError, 'successfully'):
            self.ready()
        self.prereq_state = 'COMPLETED'
        self.save('artifacts/expanded_schema_complete.json', {'status': 'failed'})
        with self.assertRaisesRegex(ValueError, 'marker changed'):
            self.ready()

    def test_all_joins_require_afterok_no_or_no_old_reference(self):
        for dep in ('afterany:999,afterok:123', 'afterok:999?afterok:123', 'afterok:365300'):
            self.spec['reviewed_dependencies']['355'] = dep
            with self.assertRaisesRegex(ValueError, 'Dependency joins'):
                self.ready()

    def test_unknown_journal_and_source_drift_rejected(self):
        self.spec['source_sha256'] = {'proof.json': 'wrong'}
        with self.assertRaisesRegex(ValueError, 'Source drift'):
            self.ready()
        self.spec['source_sha256'] = {}
        self.save('runs/quality/m/X_P_domain_0/foreign/test.jsonl', {})
        with self.assertRaisesRegex(ValueError, 'outside fixed scope'):
            self.ready()

    def test_independent_variants_must_gate_statistics(self):
        self.variant_state = 'RUNNING'
        with self.assertRaisesRegex(ValueError, 'Variants statistics join'):
            self.ready()
        self.spec['reviewed_dependencies']['355'] = 'afterok:999:279'
        self.ready()
        self.variant_state = 'FAILED'
        with self.assertRaisesRegex(ValueError, 'Variants statistics join'):
            self.ready()

    def test_separate_screen_history_hashes_do_not_expand_target_scope(self):
        cell = {'namespace': 'quality', 'model': 'm', 'variant': 'X_P_domain_0',
                'benchmark': 'b', 'split': 'prompt'}
        rel = 'runs/quality/m/X_P_domain_0/b/expanded_screen.jsonl'
        self.save(rel, {'historical': True})
        reviewed = {rel: wrapper.sha(self.root/rel)}
        self.assertEqual(wrapper.inventory(self.root, [cell], reviewed), {})
        self.save(rel, {'historical': 'changed'})
        with self.assertRaisesRegex(ValueError, 'screening journal changed'):
            wrapper.inventory(self.root, [cell], reviewed)
        reviewed[rel] = wrapper.sha(self.root/rel)
        self.save('runs/quality/m/X_P_domain_0/b/unreviewed.jsonl', {})
        with self.assertRaisesRegex(ValueError, 'outside fixed scope'):
            wrapper.inventory(self.root, [cell], reviewed)
        with self.assertRaisesRegex(ValueError, 'not manual screening'):
            wrapper.inventory(self.root, [cell], {'runs/rules/m/R_rules/b/test.jsonl': 'x'})

    def test_current_missing_and_full_standalone_budget_required(self):
        bad = copy.deepcopy(self.audited); bad[0]['missing_pair_ids_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'missing pairs'):
            wrapper.verify_ready(self.root, self.spec, '999', self.cells, bad)
        self.save('artifacts/budget.json', {'spent': 99, 'reserved': 0})
        with self.assertRaisesRegex(ValueError, 'headroom'):
            self.ready()


if __name__ == '__main__':
    unittest.main()
