import copy
import fcntl
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import resume_fresh_core_run02_second_drained as second
from test_fresh_core_drained import CoreTests
from test_drained_inference import Response
import protocol

PROJECT = Path(__file__).resolve().parents[2]


class SecondTests(unittest.TestCase):
    def setUp(self):
        self.fixture = CoreTests()
        with patch.object(tempfile, 'tempdir', str(Path(tempfile.gettempdir()).resolve())):
            self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        old = self.fixture.target
        self.target = old
        self.fixture.target = self.target
        self.fixture.inference.PROJECT = self.target
        self.fixture.orchestration.PROJECT = self.target
        for name in ('verify','stamp','select'):
            fn = getattr(self.fixture.orchestration,name)
            fn.__globals__['PROJECT'] = self.target
        # All frozen runtime sources and the authorizations are real; data and
        # predecessor evidence are a small synthetic fixture. No network allowed.
        for rel in list(second.budget.PINNED):
            self.copy('artifacts/operations/'+rel+'.py')
        for rel in [second.SELF,second.AUDITOR,second.VERIFIER,second.PRIOR_ADMISSION,
                    second.budget.SELF,second.budget.APPROVAL,
                    'artifacts/operations/resume_schema_once.py','artifacts/operations/operational_journal.py',
                    'artifacts/operations/repeated_runs.py',
                    'artifacts/operations/approval_runs23_aaron_20261014.json',
                    'artifacts/operations/resume_variants_second_drained.py']:
            self.copy(rel)
        self.save(self.root/'configs/core_run03_second_admission_20261015.json',dict(slurm_job_id='366520',run_id=3,replica=str(self.root/'replications/run03')))
        self.w = second.make_wrapper(self.root)
        self.history = [dict(run_id=2,stage=stage,status=status,slurm_job_id=job)
                       for stage,status,job in [('smoke','started','1'),('smoke','complete','1'),
                       ('core','started',second.INITIAL),('core','failed',second.INITIAL),
                       ('core','started',second.PREDECESSOR),('core','failed',second.PREDECESSOR)]]
        for r in self.history[-2:]:
            r.update(recovery='drained_v1',predecessor=second.INITIAL,admission_sha256=second.PRIOR_SHA)
        self.stage = self.target/'artifacts/stage_execution.jsonl'
        self.jsonl(self.stage,self.history)
        self.prior = self.target/'artifacts/core_drained_recovery_v1'
        self.jsonl(self.prior/'execution.jsonl',self.history[-2:])
        ends = [{'event':'start','models':list(protocol.MODELS),'invocations_per_model':1}]
        ends += [{'event':'model_end','model':m,'status':'failed'} for m in protocol.MODELS]
        ends += [{'event':'end','models_drained':7,'status':'failed'}]
        self.jsonl(self.prior/'phases/AP1_baseline.jsonl',ends)
        protected = {str(p.relative_to(self.target)):second.budget.sha(p) for p in self.target.rglob('*') if p.is_file()}
        protected['runs/quality/jev/B0/abt-buy/prompt.operations.jsonl'] = None
        self.diag = dict(job=second.PREDECESSOR,run_id=2,
            status='terminal_first_drained_core_recovery_partial_baseline_audited_not_recovered',
            script_sha256=second.budget.sha(self.root/second.AUDITOR),admission_sha256=second.PRIOR_SHA,
            protected_sha256_or_absent=protected,stage_history=self.history,
            model_journal=ends,totals={'missing':168},operations_events=0,
            progress=second.read(self.target/'artifacts/progress.json'))
        self.save(self.root/'diagnosis.json',self.diag)
        self.save(self.root/second.SOURCE_PLAN,dict(source_files_sha256={},frozen_source_sha256={}))
        self.save(self.root/second.SOURCE_PROOF,{'synthetic':True})
        phases={p:dict(known_proxy_usd=1.,historical_unknown_scenario_usd=.1) for p in self.w.PHASES}
        self.proof=dict(job=second.PREDECESSOR,run_id=2,script_sha256=second.budget.sha(self.root/second.VERIFIER),
            diagnosis_sha256=second.budget.sha(self.root/'diagnosis.json'),independent_counts=self.diag['totals'],
            independent_cells=168,operations_events=0,model_ends=ends[1:-1],fresh_remaining_core_scenario=dict(
            phases=phases,historical_source_scenario_sha256=second.budget.sha(self.root/second.SOURCE_PLAN),
            historical_source_proof_sha256=second.budget.sha(self.root/second.SOURCE_PROOF)))
        self.save(self.root/'proof.json',self.proof)
        cost=dict(phases=phases,diagnosis_sha256=second.budget.sha(self.root/'diagnosis.json'),
                  source_verification_sha256=second.budget.sha(self.root/'proof.json'),quote_or_hard_bound=False)
        self.save(self.root/'cost.json',cost)
        self.save(self.root/'artifacts/budget.json',dict(spent=120.,reserved=3.,attempts=0,unknown_cost_attempts=0))
        self.save(self.root/'configs/expanded_execution.json',dict(e2_rules_job=365300,statistics_full_job=365355,
            e2_confirmation_job=366115,e2_schema_job=366116,e6_variants_job=366251))
        self.deps={'999':'(null)','365300':'afterok:366115:366116','365355':'afterok:366115:366251:366116:365300',
                   '366115':'(null)','366116':'(null)','366251':'(null)','366520':'(null)'}
        self.policy=dict(policy_sha256='policy',reviewed_jobs=[dict(id=j,self=j=='999',scope='project') for j in self.deps])
        self.replica=dict(replica=str(self.target),runner_sha256='runner')
        self.spec=dict(admitted=True,run_id=2,stage='core',slurm_job_id='999',budget_cap_usd=200,max_invocations=1,
            replica=str(self.target),runner_sha256='runner',historical_unknown_outcomes_disclosed=True,
            budget_binding_sha256=second.budget.sha(self.root/second.SELF),budget_authority_sha256=second.budget.APPROVAL_SHA,
            predecessor=second.PREDECESSOR,diagnosis='diagnosis.json',diagnosis_sha256=second.budget.sha(self.root/'diagnosis.json'),
            cost_plan='cost.json',cost_plan_sha256=second.budget.sha(self.root/'cost.json'),cost_source_verification='proof.json',
            parallel_policy='policy.json',parallel_policy_sha256='policy',reviewed_dependencies=self.deps,
            reviewed_job_write_roots={j:[str(self.root/'runs')] for j in ('366115','366116','366251')},
            source_sha256={r:second.budget.sha(self.root/r) for r in self.w.REQUIRED})
        self.spec['reviewed_job_write_roots']['366520']=[str(self.root/'replications/run03')]
        self.save(self.root/'admission.json',self.spec)

    def copy(self,rel):
        p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/rel,p)

    def save(self,p,value):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))

    def jsonl(self,p,value):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r)+'\n' for r in value))

    def command(self,args,**kwargs):
        if args[0]=='sacct':
            j=args[3]
            return f'{j}|FAILED|1:0|end\n{j}.batch|FAILED|1:0|end\n{j}.extern|COMPLETED|0:0|end\n'
        if args[1:3]==['show','job']:
            return f'Dependency={self.deps[args[3]]} Priority=0 JobState=PENDING StartTime=Unknown'
        return 'run1 writer'

    def ready(self):
        with patch.object(self.w,'review',return_value=self.policy),patch.object(self.w.subprocess,'check_output',side_effect=self.command):
            return self.w.verify_ready(self.root,self.target,2,self.spec,'999',self.replica)

    def test_full_gate_preserves_old_attempt_and_foreign_hold(self):
        before={p:p.read_bytes() for p in self.prior.rglob('*') if p.is_file()}
        ledger=(self.root/'artifacts/budget.json').read_bytes()
        self.assertEqual(self.ready()['second_attempt_lineage']['all_model_ends'],7)
        self.assertEqual((self.root/'artifacts/budget.json').read_bytes(),ledger)
        self.assertTrue(all(p.read_bytes()==b for p,b in before.items()))

    def test_unadmitted_wrong_run_and_history_rejected(self):
        self.spec['admitted']=False
        with self.assertRaises(ValueError):self.ready()
        self.spec['admitted']=True
        for mutate in [lambda h:h.pop(),lambda h:h[-1].update(status='complete'),
                       lambda h:h[-1].update(predecessor='wrong'),lambda h:h[3].update(slurm_job_id='wrong')]:
            h=copy.deepcopy(self.history);mutate(h)
            with self.assertRaises(ValueError):second.validate_history(h,2,second.PREDECESSOR)
        with self.assertRaises(ValueError):second.validate_history(self.history,3,second.PREDECESSOR)

    def test_protected_absence_source_model_end_and_cost_drift(self):
        absent=self.target/'runs/quality/jev/B0/abt-buy/prompt.operations.jsonl'
        self.jsonl(absent,[{'unknown':True}])
        with self.assertRaisesRegex(ValueError,'protected'):self.ready()
        absent.unlink()
        self.save(self.root/'cost.json',{'phases':{}})
        with self.assertRaisesRegex(ValueError,'phase costs'):self.ready()
        ends=self.prior/'phases/AP1_baseline.jsonl';ends.write_text(ends.read_text()+'{}\n')
        with self.assertRaises(ValueError):self.ready()

    def test_all_steps_joins_scopes_budget_and_extra_cache(self):
        with patch.object(self.w.subprocess,'check_output',return_value='366117|FAILED|1:0|end\n366117.batch|RUNNING|0:0|Unknown'):
            with self.assertRaises(ValueError):self.w.verify_ready(self.root,self.target,2,self.spec,'999',self.replica)
        self.deps['365355']='afterany:366115:366251:366116:365300'
        with self.assertRaises(ValueError):self.ready()
        self.deps['365355']='afterok:366115:366251:366116:365300'
        self.spec['reviewed_job_write_roots']['366115']=[str(self.target)]
        with self.assertRaisesRegex(ValueError,'writer root'):self.ready()
        self.spec['reviewed_job_write_roots']['366115']=[str(self.root/'runs')]
        self.save(self.root/'artifacts/budget.json',dict(spent=197,reserved=3))
        with self.assertRaisesRegex(ValueError,'headroom'):self.ready()
        self.save(self.root/'artifacts/budget.json',dict(spent=120,reserved=3))
        self.jsonl(self.target/'runs/extra.jsonl',[{}])
        with self.assertRaisesRegex(ValueError,'inventory'):self.ready()

    def execute(self,budget_stop=False):
        old=self.w.verify_ready
        def gate(*args):
            checked=old(*args)
            sys.modules.update(run=self.fixture.inference,protocol=protocol,orchestrate=self.fixture.orchestration)
            if budget_stop:self.save(self.root/'artifacts/budget.json',dict(spent=197.,reserved=3.,attempts=0,unknown_cost_attempts=0))
            return checked
        calls=[]
        def http(req,**kwargs):
            body=json.loads(req.data);calls.append(body)
            alias=next(m for m,v in protocol.MODELS.items() if v['id']==body['model'])
            for path in [self.root/'artifacts/operations/repetition_writer.lock',self.target/'artifacts/parallel_writer.lock']:
                with path.open('a') as f:
                    with self.assertRaises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            response=dict(model=body['model'],provider=protocol.MODELS[alias]['provider'].split('/')[0],usage={'cost':.001})
            if alias=='luna':response['choices']=[{'finish_reason':'stop','message':{'content':'{"match":true}'}}]
            else:response['answers']={'match':{'type':'noul','noul':.8}}
            return Response(json.dumps(response).encode())
        with patch.dict(sys.modules),patch.dict('os.environ',{'SLURM_JOB_ID':'999'}), \
                patch.object(self.w,'verify',return_value=self.replica),patch.object(self.w,'verify_ready',side_effect=gate), \
                patch.object(self.w,'review',return_value=self.policy),patch.object(self.w.subprocess,'check_output',side_effect=self.command), \
                patch('urllib.request.urlopen',side_effect=http),patch.object(self.w.subprocess,'run') as analysis:
            for name in ('run','protocol','orchestrate','analyze'):sys.modules.pop(name,None)
            result=self.w.execute(self.root,2,self.root/'admission.json')
            self.assertEqual(analysis.call_count,1)
        return result,calls

    def test_full_real_http_component_through_both_gates_and_locks(self):
        before=self.stage.read_bytes();prior={p:p.read_bytes() for p in self.prior.rglob('*') if p.is_file()}
        result,calls=self.execute()
        self.assertEqual(result,0);self.assertEqual(len(calls),440)
        self.assertTrue(self.stage.read_bytes().startswith(before))
        self.assertTrue(all(p.read_bytes()==b for p,b in prior.items()))
        end=second.rows(self.stage)[-1]
        self.assertEqual((end['status'],end['recovery'],end['predecessor']),('complete','drained_v2',second.PREDECESSOR))
        ledger=second.read(self.root/'artifacts/budget.json')
        self.assertAlmostEqual(ledger['spent'],120.440);self.assertAlmostEqual(ledger['reserved'],3.)
        self.assertTrue(all(r['study_run_id']==2 for p in (self.target/'runs').rglob('*.jsonl') if not p.name.endswith('.executions.jsonl') for r in second.rows(p)))
        with self.assertRaisesRegex(ValueError,'Prior recovery'):self.execute()

    def test_actual_200_budget_stop_drains_all_models_without_http_or_selection(self):
        before=self.stage.read_bytes()
        with self.assertRaisesRegex(RuntimeError,'draining every model'):self.execute(budget_stop=True)
        self.assertTrue(self.stage.read_bytes().startswith(before))
        self.assertEqual(second.rows(self.stage)[-1]['status'],'failed')
        events=second.rows(self.target/second.ATTEMPT/'phases/AP1_baseline.jsonl')
        self.assertEqual(sum(e['event']=='model_end' for e in events),7)
        self.assertEqual(len(list((self.target/'runs').rglob('*.operations.jsonl'))),7)
        self.assertFalse(list((self.target/'runs').rglob('*.failures.jsonl')))
        self.assertFalse((self.target/'configs/selected_variants.json').exists())
        self.assertEqual(second.read(self.root/'artifacts/budget.json'),dict(spent=197.,reserved=3.,attempts=0,unknown_cost_attempts=0))


if __name__=='__main__':unittest.main()
