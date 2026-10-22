import copy
import subprocess
import sys
import json
from pathlib import Path
import shutil
import socket
import tempfile
import types
import unittest
from unittest.mock import patch

import protocol
import resume_variants_third_drained as second
from diagnose_variants_second_terminal_366251_20261015 import validate_history, STAGE, MODELS, ADMISSION
import test_authorized_recovery_200 as budget_tests
import test_variant_recovery as old_tests

ROOT = Path(__file__).resolve().parents[2]


class ThirdAttempt(unittest.TestCase):
    def test_private_binding_rejects_old_predecessor_and_original_sources_unchanged(self):
        w = second.make_wrapper(ROOT)
        self.assertEqual(w.JOURNAL, second.JOURNAL)
        self.assertIn(second.MODEL_JOURNAL, w.MODEL_JOURNAL)
        spec = dict(admitted=True, run_id=1, stage='variants', predecessor='366251', slurm_job_id='42',
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
            r.update(job_id='366251', admission_sha256=second.budget.sha(root/ADMISSION))
        models = [{'event': 'start', 'models': ['jev'], 'invocations_per_model': 1}, *ends,
                  {'event': 'end', 'status': 'failed', 'models_drained': 1}]
        for rel, data in ((STAGE, stage), (MODELS, models)):
            (root/rel).write_text(''.join(json.dumps(x)+'\n' for x in data))
        d = {'job': '366251', 'historical_unjournaled_outcomes_resolved': False, 'operations': [],
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
                with patch.object(second, 'history', return_value={'fixture':True}), patch.object(second, 'checked_scope', side_effect=lambda r,s,j: s['reviewed_job_write_cells'].get(j)):
                    return second.guard(root, spec, '42', checked, ['jev'], ['abt-buy'], command)
            self.assertTrue(run()['scopes_checked'])
            old = copy.deepcopy(spec)
            spec['reviewed_job_write_cells']['366115'] = [('quality', 'jev', 'B0', 'abt-buy', 'test')]
            with self.assertRaisesRegex(ValueError, 'overlapping'): run()
            spec.update(copy.deepcopy(old))
            for dep in ('afterany:366115', 'afterok:366115?afterok:42', 'afterok:42', 'afterok:366251:42:366115:366116:365300'):
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


    def test_preserved_complete_history_and_independent_audit_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rel in (*second.HISTORY_FILES, second.PROOF, second.AUDITOR, second.VERIFIER):
                path=root/rel; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('{}')
            def save(rel,obj): (root/rel).write_text(json.dumps(obj))
            def rows(rel,obj): (root/rel).write_text(''.join(json.dumps(x)+'\n' for x in obj))
            first_spec=dict(slurm_job_id='366105',predecessor='365279')
            save(second.HISTORY_FILES[2],first_spec)
            first_models=[dict(event='start',models=['jev'],invocations_per_model=1),
                          dict(event='model_end',model='jev',status='failed'),dict(event='end',status='failed',models_drained=1)]
            first_stage=[dict(event='start',invocations_limit=1),dict(event='inference_end',models=[dict(model='jev',status='failed')]),dict(event='end',status='failed',report_created=False)]
            for r in first_stage:r.update(job_id='366105',admission_sha256=second.budget.sha(root/second.HISTORY_FILES[2]))
            rows(second.HISTORY_FILES[0],first_stage);rows(second.HISTORY_FILES[1],first_models)
            save(ADMISSION,dict(slurm_job_id='366251',predecessor='366105',diagnosis_sha256='prior-diag'))
            stage=copy.deepcopy(first_stage)
            for r in stage:r.update(job_id='366251',admission_sha256=second.budget.sha(root/ADMISSION))
            stage[0].update(predecessor_job='366105',verification={'predecessor_provenance':{
                'predecessor_stage_sha256':second.budget.sha(root/second.HISTORY_FILES[0]),
                'predecessor_models_sha256':second.budget.sha(root/second.HISTORY_FILES[1]),'predecessor_diagnosis_sha256':'prior-diag'}})
            rows(STAGE,stage);rows(MODELS,first_models)
            d=dict(stage_history=stage,script_sha256=second.budget.sha(root/second.AUDITOR),totals={'responses':1},model_endings=first_models[1:2],complete_cells=0,known_journaled_usd=.01)
            proof=dict(script_sha256=second.budget.sha(root/second.VERIFIER),diagnosis_sha256='new-diag',totals=d['totals'],model_endings={'jev':'failed'},complete_cells=0,known_journaled_usd=.01,operations_events=0,cost_plan_sha256='cost')
            save(second.PROOF,proof)
            spec=dict(diagnosis_sha256='new-diag',cost_plan_sha256='cost')
            second.history(root,spec,d,['jev'])
            for key,value in [('totals',{}),('model_endings',[]),('operations_events',1),('cost_plan_sha256','wrong')]:
                save(second.PROOF,{**proof,key:value})
                with self.assertRaises(ValueError):second.history(root,spec,d,['jev'])
            save(second.PROOF,proof)
            rows(second.HISTORY_FILES[1],first_models[:-1])
            with self.assertRaises(ValueError):second.history(root,spec,d,['jev'])

    def test_actual_preserved_scopes_and_separate_core_namespaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir()
            specs={}
            for jid,rel in second.WRITER_ADMISSIONS.items():
                obj=json.loads((ROOT/rel).read_text())
                if jid in ('366520','366545'):obj['replica']=str(root/f"replications/run{obj['run_id']:02d}")
                (root/rel).write_text(json.dumps(obj));specs[jid]=obj
            reference=specs['366115']['reviewed_job_write_cells']
            scopes={j:reference[{'366520':'366118','366545':'366117'}.get(j,j)] for j in specs}
            spec={'reviewed_job_write_cells':scopes}
            for jid in specs:self.assertEqual(second.checked_scope(root,spec,jid),scopes[jid])
            spec['reviewed_job_write_cells']['366520']=[]
            with self.assertRaises(ValueError):second.checked_scope(root,spec,'366520')
            spec['reviewed_job_write_cells']['366520']=reference['366118']
            specs['366520']['replica']=str(root)
            (root/second.WRITER_ADMISSIONS['366520']).write_text(json.dumps(specs['366520']))
            with self.assertRaises(ValueError):second.checked_scope(root,spec,'366520')
            with self.assertRaises(ValueError):second.checked_scope(root,spec,'unreviewed')

    def full_execution(self, stop):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);budget_tests.AuthorizedBudget().setup_root(root,120,3)
            shutil.copyfile(ROOT/'src/protocol.py',root/'src/protocol.py')
            shutil.copyfile(ROOT/second.SELF,root/second.SELF)
            benchmarks=['abt-buy','wdc','dblp-acm','dblp-scholar','walmart-amazon']
            for b in benchmarks:
                path=root/'data'/b;path.mkdir(parents=True)
                case=dict(benchmark=b,pair_id='a#b',label=1,records={'left':{'name':'A'},'right':{'name':'A'}})
                case['input_hash']=protocol.digest(case['records'])
                (path/'test.json').write_text(json.dumps([case]))
            for name in ('expanded_variants','orchestrate'):
                (root/'src'/f'{name}.py').write_text('from pathlib import Path\nPROJECT=Path(__file__).resolve().parents[1]\ndef verify(): return {}\n'+(
                    'VARIANTS='+repr(benchmarks)+'\ndef report(): (PROJECT/"artifacts/mock_report").write_text("fixture report")\n' if name=='expanded_variants' else ''))
            for rel in second.HISTORY_FILES[:2]:(root/rel).write_text('preserved historical fixture\n')
            (root/'admission.json').write_text('{}')
            script=r"""
import fcntl,json,os,socket,sys
from pathlib import Path
from unittest.mock import patch
import resume_variants_third_drained as module
from test_variant_recovery import Response
root=Path(sys.argv[1]);stop=sys.argv[2]=='True'
w=module.make_wrapper(root)
def gate(*args):
    if stop:(root/'artifacts/budget.json').write_text(json.dumps(dict(spent=197,reserved=3,attempts=0,unknown_cost_attempts=0)))
    return {'fixture_scheduler_and_admission':True}
w.verify_ready=gate
calls=[]
def http(req,**kw):
    import protocol
    body=json.loads(req.data);calls.append(body)
    with (root/'artifacts/operations/variants_recovery.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise AssertionError('Writer lock was not held')
    alias=next(m for m,v in protocol.MODELS.items() if v['id']==body['model'])
    raw=dict(model=body['model'],provider=protocol.MODELS[alias]['provider'].split('/')[0],usage={'cost':.001})
    if alias=='luna':raw['choices']=[{'finish_reason':'stop','message':{'content':'{"match":true}'}}]
    else:raw['answers']={'match':{'type':'noul','noul':.8}}
    return Response(json.dumps(raw).encode())
# Original helper imports run/protocol; remove before testing the actual entry.
for name in ('run','protocol','expanded_variants','analyze','orchestrate'):sys.modules.pop(name,None)
os.environ['SLURM_JOB_ID']='42';os.environ['OPENROUTER_API_KEY']='fixture'
with patch('socket.socket',side_effect=AssertionError('No sockets')),patch('urllib.request.urlopen',side_effect=http):
    code=w.execute(root,root/'admission.json')
    try:w.execute(root,root/'admission.json')
    except ValueError:pass
    else:raise AssertionError('Blind repeat allowed')
(root/'test_summary.json').write_text(json.dumps(dict(code=code,calls=len(calls))))
"""
            out=subprocess.run([sys.executable,'-c',script,str(root),str(stop)],text=True,capture_output=True)
            self.assertEqual(out.returncode,0,out.stdout+out.stderr)
            result=json.loads((root/'test_summary.json').read_text())
            self.assertEqual(result,{'code':1 if stop else 0,'calls':0 if stop else 35})
            ends=[json.loads(x) for x in (root/second.MODEL_JOURNAL).read_text().splitlines()]
            self.assertEqual(sum(r['event']=='model_end' for r in ends),7)
            self.assertEqual(ends[-1]['status'],'failed' if stop else 'complete')
            stage=[json.loads(x) for x in (root/second.JOURNAL).read_text().splitlines()]
            self.assertEqual(stage[0]['predecessor_job'],'366251')
            self.assertEqual(stage[-1]['report_created'],not stop)
            ledger=json.loads((root/'artifacts/budget.json').read_text())
            self.assertAlmostEqual(ledger['spent'],197 if stop else 120.035)
            self.assertAlmostEqual(ledger['reserved'],3)
            for rel in second.HISTORY_FILES[:2]:self.assertEqual((root/rel).read_text(),'preserved historical fixture\n')
            self.assertFalse(list((root/'runs').rglob('*.failures.jsonl')))
            self.assertEqual(len(list((root/'runs').rglob('*.operations.jsonl'))),7 if stop else 0)

    def test_full_wrapper_real_frozen_http_and_exclusive_attempt(self):self.full_execution(False)
    def test_full_wrapper_budget_200_stop_all_seven_ends(self):self.full_execution(True)


if __name__ == '__main__':
    unittest.main()
