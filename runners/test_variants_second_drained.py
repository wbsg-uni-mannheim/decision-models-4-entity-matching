import copy
import json
from pathlib import Path
import shutil
import socket
import tempfile
import types
import unittest
from unittest.mock import patch

import protocol
import resume_variants_second_drained as second
from diagnose_variants_drained_terminal import validate_history, STAGE, MODELS, ADMISSION
import test_authorized_recovery_200 as budget_tests
import test_variant_recovery as old_tests

ROOT = Path(__file__).resolve().parents[2]


class SecondAttempt(unittest.TestCase):
    def test_private_binding_rejects_old_predecessor_and_original_sources_unchanged(self):
        w = second.make_wrapper(ROOT)
        self.assertEqual(w.JOURNAL, second.JOURNAL)
        self.assertIn(second.MODEL_JOURNAL, w.MODEL_JOURNAL)
        spec = dict(admitted=True, run_id=1, stage='variants', predecessor='366105', slurm_job_id='42',
            budget_cap_usd=200, max_invocations=1, historical_unknown_outcomes_disclosed=True,
            unobserved_worker_causes=['kev', 'luna', 'pplx'], source_sha256=dict.fromkeys(w.REQUIRED, 'fixture'))
        w.validate(spec, '42', {'spent': 110, 'reserved': 3})
        for change in ({'predecessor': '365279'}, {'budget_cap_usd': 100}, {'max_invocations': 2}, {'admitted': False}):
            with self.assertRaises(ValueError):
                w.validate({**spec, **change}, '42', {'spent': 110, 'reserved': 3})
        self.assertEqual(second.budget.sha(ROOT/'artifacts/operations/resume_variants_drained.py'),
                         second.budget.PINNED['resume_variants_drained'])

    def fixture(self, root):
        (root/'configs').mkdir(exist_ok=True)
        (root/'artifacts/operations').mkdir(parents=True, exist_ok=True)
        (root/ADMISSION).write_text('{}')
        ends = [{'event': 'model_end', 'model': 'jev', 'status': 'failed', 'utc': 'fixture'}]
        stage = [{'event': 'start', 'invocations_limit': 1},
                 {'event': 'inference_end', 'models': [{'model': 'jev', 'status': 'failed'}]},
                 {'event': 'end', 'status': 'failed', 'report_created': False}]
        for r in stage:
            r.update(job_id='366105', admission_sha256=second.budget.sha(root/ADMISSION))
        models = [{'event': 'start', 'models': ['jev'], 'invocations_per_model': 1}, *ends,
                  {'event': 'end', 'status': 'failed', 'models_drained': 1}]
        for rel, data in ((STAGE, stage), (MODELS, models)):
            (root/rel).write_text(''.join(json.dumps(x)+'\n' for x in data))
        d = {'job': '366105', 'historical_unjournaled_outcomes_resolved': False, 'operations': [],
             'protected_sha256': {rel: second.budget.sha(root/rel) for rel in (STAGE, MODELS, ADMISSION)},
             'absent_journals': ['absent.jsonl']}
        (root/'diagnosis.json').write_text(json.dumps(d))
        (root/'source').write_text('immutable')
        (root/'cost.json').write_text(json.dumps({'source_sha256_or_absent': {'source': second.budget.sha(root/'source')}, 'source_proofs': []}))
        (root/'configs/expanded_execution.json').write_text(json.dumps(dict(e2_rules_job=365300,
            statistics_full_job=365355, e2_confirmation_job=366115, e2_schema_job=366116)))
        deps = {'42': '(null)', '366115': '(null)', '366116': '(null)',
                '365300': 'afterok:366115(unfulfilled),afterok:366116(unfulfilled)',
                '365355': 'afterok:366115(unfulfilled),afterok:366116(unfulfilled),afterok:42(unfulfilled),afterok:365300(unfulfilled)'}
        spec = dict(diagnosis='diagnosis.json', diagnosis_sha256=second.budget.sha(root/'diagnosis.json'),
                    cost_plan='cost.json', reviewed_dependencies=deps,
                    reviewed_job_write_cells={'366115': [], '366116': []})
        checked = {'policy': {'reviewed_jobs': [{'id': j, 'self': j == '42'} for j in deps]}}
        def command(args, **kwargs):
            jid = args[3]
            return f'JobId={jid} Dependency={deps[jid]} Priority=0 JobState=PENDING StartTime=Unknown'
        return spec, checked, command, stage, models

    def test_all_joins_scopes_source_provenance_and_absence_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, checked, command, _, _ = self.fixture(root)
            def run():
                return second.guard(root, spec, '42', checked, ['jev'], ['abt-buy'], command)
            self.assertTrue(run()['scopes_checked'])
            old = copy.deepcopy(spec)
            spec['reviewed_job_write_cells']['366115'] = [('quality', 'jev', 'B0', 'abt-buy', 'test')]
            with self.assertRaisesRegex(ValueError, 'overlapping'): run()
            spec.update(copy.deepcopy(old))
            for dep in ('afterany:366115', 'afterok:366115?afterok:42', 'afterok:42', 'afterok:366105:42:366115:366116:365300'):
                spec['reviewed_dependencies']['365355'] = dep
                with self.assertRaises(ValueError): run()
            spec.update(copy.deepcopy(old))
            (root/'absent.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError, 'absent'): run()
            (root/'absent.jsonl').unlink()
            (root/'source').write_text('drift')
            with self.assertRaisesRegex(ValueError, 'Cost source'): run()

    def test_model_end_completeness_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, stage, models = self.fixture(root)
            digest = second.budget.sha(root/ADMISSION)
            validate_history(stage, models, ['jev'], digest)
            for bad in (models[:-1], [models[0], models[1], models[1], models[-1]]):
                with self.assertRaises(ValueError): validate_history(stage, bad, ['jev'], digest)
            stage[-1]['job_id'] = 'wrong'
            with self.assertRaises(ValueError): validate_history(stage, models, ['jev'], digest)

    def test_real_drained_request_account_cache_exclusive_model_journal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(socket, 'socket', side_effect=AssertionError('No network')):
            root = Path(tmp)
            inference = budget_tests.AuthorizedBudget().setup_root(root, 110, 3)
            for rel in (second.SELF, second.AUDITOR):
                shutil.copyfile(ROOT/rel, root/rel)
            inference.key = lambda: 'fixture'
            case = dict(benchmark='abt-buy', pair_id='a#b', label=1, records={'left': {'name': 'x'}, 'right': {'name': 'x'}})
            case['input_hash'] = protocol.digest(case['records'])
            p = root/'data/abt-buy'; p.mkdir(parents=True)
            (p/'test.json').write_text(json.dumps([case]))
            w = second.make_wrapper(root)
            variants = types.SimpleNamespace(VARIANTS=['abt-buy'])
            proto = types.SimpleNamespace(**{**vars(protocol), 'MODELS': {'jev': protocol.MODELS['jev']}})
            raw = {'model': 'typesafe/jev-1.13', 'provider': 'typesafe',
                   'answers': {'match': {'type': 'noul', 'noul': .8}}, 'usage': {'cost': .01}}
            journal = root/second.MODEL_JOURNAL
            with patch('urllib.request.urlopen', side_effect=lambda *a, **k: old_tests.Response(json.dumps(raw).encode())) as http:
                outcomes, ops, _ = w.run_once(root, variants, proto, inference, journal)
                self.assertEqual(outcomes, [{'model': 'jev', 'status': 'complete'}])
                self.assertEqual(ops[0]['events'], 0)
                with self.assertRaises(FileExistsError): w.run_once(root, variants, proto, inference, journal)
                self.assertEqual(http.call_count, 1)
            self.assertEqual(json.loads((root/'artifacts/budget.json').read_text())['reserved'], 3)
            self.assertEqual(json.loads((root/'artifacts/budget.json').read_text())['spent'], 110.01)


if __name__ == '__main__':
    unittest.main()
