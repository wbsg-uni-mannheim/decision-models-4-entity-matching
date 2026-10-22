import copy
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import protocol
import analyze  # Load NumPy before temporary sys.modules fixtures.
import resume_grouped_core3_second_20261015 as second
from test_fresh_core_run02_third_drained import SecondTests as Fixture
from test_drained_inference import Response


class SuccessorTests(unittest.TestCase):
    def setUp(self):
        f = Fixture(); f.setUp(); self.addCleanup(f.doCleanups); self.f = f
        self.root = f.root; target = self.root/'replications/run03'
        # The fixture already creates a Run3 sibling admission, not its data.
        f.target.rename(target); f.target = target; self.target = target
        (target/'artifacts/operations').mkdir(exist_ok=True)
        f.fixture.inference.PROJECT = target; f.fixture.orchestration.PROJECT = target
        for name in ('verify','stamp','select'):
            getattr(f.fixture.orchestration,name).__globals__['PROJECT'] = target
        for rel in (second.BASE,second.SELF):
            f.copy(rel)
        self.w = second.make_wrapper(self.root)
        self.spec = {'admitted':True,'stage':'core3','slurm_job_id':'999','predecessor':'366697',
                     'cap_usd':200,'max_invocations':1,'policy_sha256':'policy'}
        self.admission = self.root/'new_admission.json'; self.admission.write_text(json.dumps(self.spec))
        self.calls = []

    def http(self, req, **kwargs):
        body = json.loads(req.data); self.calls.append(body)
        alias = next(m for m,d in protocol.MODELS.items() if d['id'] == body['model'])
        raw = {'model':body['model'],'provider':protocol.MODELS[alias]['provider'].split('/')[0],
               'usage':{'cost':.001}}
        if alias == 'luna':
            raw['choices'] = [{'finish_reason':'stop','message':{'content':'{"match":true}'}}]
        else:
            raw['answers'] = {'match':{'type':'noul','noul':.8}}
        return Response(json.dumps(raw).encode())

    def execute(self, audit):
        f = self.f
        with patch.dict(os.environ,{'SLURM_JOB_ID':'999'}), \
                patch.dict(sys.modules,{'run':f.fixture.inference,'orchestrate':f.fixture.orchestration,'analyze':analyze}), \
                patch.object(self.w,'preflight',return_value=(self.target,{'run_id':3},{'fixture':True})), \
                patch('repeated_runs.verify'), patch.object(self.w,'final_audit',side_effect=audit), \
                patch('urllib.request.urlopen',side_effect=self.http), \
                patch.object(self.w.subprocess,'run'):
            return self.w.execute(self.root,self.admission)

    def test_full_run3_grouped_executor_exact_credit_and_timing_original(self):
        old = (self.target/'artifacts/stage_execution.jsonl').read_bytes()
        def audit(*args,**kwargs):
            # Full raw parser/request/RunID verification independent of credit.
            from diagnose_fresh_baseline import audit_rows
            import types
            for p in (self.target/'runs').rglob('*.jsonl'):
                if p.name.endswith(('.operations.jsonl','.failures.jsonl','.executions.jsonl')):
                    continue
                ns,m,v,b,split = p.relative_to(self.target/'runs').parts
                split = split.removesuffix('.jsonl')
                cases = json.loads((self.target/'data'/b/(split+'.json')).read_text())
                rows = [json.loads(s) for s in p.read_text().splitlines()]
                adapter = types.SimpleNamespace(payload=lambda c,a,_,v=v:protocol.payload(c,a,v),
                                                digest=protocol.digest,parse_response=protocol.parse_response)
                checked = audit_rows(cases,[{**r,'variant':'B0'} for r in rows],[],adapter,m,b,3)
                self.assertEqual(checked['missing'],0)
            return {'totals':{'missing':0}}
        self.assertEqual(self.execute(audit),0)
        self.assertEqual(len(self.calls),440)
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'],120.440); self.assertAlmostEqual(budget['reserved'],3)
        credit = budget['inference_credits']['fast-core3-999']
        self.assertEqual(credit['status'],'closed'); self.assertEqual(credit['attempts'],328)
        self.assertEqual(budget['attempts'],440)
        self.assertTrue((self.target/'artifacts/stage_execution.jsonl').read_bytes().startswith(old))
        self.assertTrue((self.target/second.ATTEMPT/'final_raw_audit.json').exists())
        with self.assertRaises(FileExistsError):
            self.execute(audit)

    def test_unknown_future_holds_credit_despite_zero_outstanding(self):
        from http.client import IncompleteRead
        self.http = lambda *a,**k: (_ for _ in ()).throw(IncompleteRead(b'partial',10))
        with self.assertRaises(RuntimeError):
            self.execute(lambda *a,**k:self.fail('No final audit after failure'))
        events = [json.loads(s) for s in (self.target/second.ATTEMPT/'execution.jsonl').read_text().splitlines()]
        self.assertEqual([e['status'] for e in events],['started','credit_retained','failed'])
        self.assertEqual(events[-1]['credit']['outstanding'],0)
        self.assertGreater(events[-1]['credit']['unknown'],0)
        self.assertEqual(json.loads((self.root/'artifacts/budget.json').read_text())['inference_credits']['fast-core3-999']['status'],'active')

    def test_unadmitted_foreign_stage_and_budget_ceiling_rejected(self):
        for mutate in (lambda s:s.update(admitted=False),lambda s:s.update(stage='core2'),
                       lambda s:s.update(predecessor='366681'),lambda s:s.update(cap_usd=300),
                       lambda s:s.update(max_invocations=2)):
            spec = copy.deepcopy(self.spec); mutate(spec)
            with self.assertRaisesRegex(ValueError,'Exact single'):
                second.preflight(self.root,spec,'999')

    def test_final_audit_requires_all_phase_model_ends(self):
        with self.assertRaisesRegex(ValueError,'Every current phase'):
            second.final_audit(self.root,'core3')

    def test_executor_source_drift_refused(self):
        p = self.root/second.BASE
        p.write_text(p.read_text().replace('from grouped_transition import audit','# deleted import'))
        with self.assertRaisesRegex(ValueError,'binding drift'):
            second.make_wrapper(self.root)

    def gate_fixture(self):
        required = [second.SELF,second.BASE,second.AUDITOR,second.VERIFIER,second.SOURCE_PLAN,
                    'artifacts/operations/grouped_inference.py','artifacts/operations/leased_budget.py',
                    'artifacts/operations/grouped_transition.py',
                    'artifacts/verification/grouped_stop_reconciled_20261015.json',
                    'configs/parallel_grouped_20261015.json','configs/grouped_core3_admission_20261015.json']
        for rel in required:
            self.f.copy(rel)
        self.spec.update(sources={p:second.binding.sha(self.root/p) for p in required},policy='policy.json',diagnosis_sha256='fixture')
        self.f.save(self.root/'configs/expanded_execution.json',{'e2_confirmation_job':366694,'e2_schema_job':366695})
        self.queue = [{'id':'999','self':True,'scope':'core3'}, {'id':'366696','self':False,'scope':'core2'}]
        self.terminal = True; self.join = '366694:366695'

    def ready(self):
        def command(args,**kwargs):
            if args[0] == 'sacct':
                j = args[3]; state = 'COMPLETED' if j == '366563' else 'FAILED' if self.terminal else 'RUNNING'
                exitcode = '0:0' if state == 'COMPLETED' else '1:0'
                return ''.join(f'{j}{suffix}|{state}|{exitcode}|end\n' for suffix in ('','.batch','.extern'))
            dep = self.join+(':365300' if args[3] == '365355' else '')
            return f'Dependency=afterok:{dep} Priority=0 StartTime=Unknown JobState=PENDING'
        policy = types.SimpleNamespace(review=lambda *a:{'policy_sha256':'policy','reviewed_jobs':self.queue})
        with patch.object(second.binding,'authority'),patch.object(second.binding,'private_module',return_value=policy), \
                patch.object(second,'validate_evidence',return_value=(self.target,{})), \
                patch('repeated_runs.verify'),patch.object(second.subprocess,'check_output',side_effect=command):
            return second.preflight(self.root,self.spec,'999')

    def test_scheduler_all_steps_joins_source_budget_and_overlap_gates(self):
        self.gate_fixture(); self.ready()
        self.terminal = False
        with self.assertRaises(ValueError):self.ready()
        self.terminal = True; self.join = '366694'
        with self.assertRaisesRegex(ValueError,'join'):self.ready()
        self.join = '366694:366695'
        self.queue.append({'id':'888','self':False,'scope':'unreviewed'})
        with self.assertRaisesRegex(ValueError,'writer'):self.ready()
        self.queue.pop()
        self.f.save(self.root/'artifacts/budget.json',{'spent':197.,'reserved':3.})
        with self.assertRaisesRegex(ValueError,'headroom'):self.ready()
        (self.root/second.BASE).write_text('changed')
        with self.assertRaisesRegex(ValueError,'Source drift'):self.ready()


if __name__ == '__main__':
    unittest.main()
