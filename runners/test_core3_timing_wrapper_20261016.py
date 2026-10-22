"""Synthetic whole timing matrix, real frozen HTTP/account/cache, network denied.

Prior raw rows are constructed from exact frozen requests and valid raw bodies.
New requests use real frozen functions. Scheduler/history are separately tested
with mocks; no claim of a real held admission or full quality report execution.
"""
import copy
import csv
import fcntl
from http.client import IncompleteRead
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import protocol
import run
import analyze
import core3_timing_recovery_plan_20261016 as plan
import core3_timing_analysis_20261016 as mapped
import run_core3_timing_recovery_20261016 as wrapper


class Response(io.BytesIO):
    status = 200


class TimingWrapperTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.target = self.root/'replications/run03'
        for p in (self.root/'artifacts/operations', self.target/'artifacts/operations',
                  self.target/'configs', self.target/'runs'):
            p.mkdir(parents=True, exist_ok=True)
        self.saved = plan.read(ROOT/mapped.PLAN)
        self.old = {}
        self.cases = {}
        for b in protocol.BENCHMARKS:
            cases = [{'benchmark':b,'pair_id':str(i),'label':i%2,'entity_ids':{'left':str(i),'right':str(i)},
                      'records':{'left':{'name':str(i)},'right':{'name':str(i)}}} for i in range(20)]
            for c in cases: c['input_hash'] = protocol.digest(c['records'])
            self.cases[b] = cases
            p = self.target/'data'/b/'prompt.json'; p.parent.mkdir(parents=True); p.write_text(json.dumps(cases))
        for c in self.saved['cells']:
            cases = self.cases[c['benchmark']]
            for key, value in (
                ('ordered_pair_ids',[x['pair_id'] for x in cases]),
                ('ordered_input_hashes',[x['input_hash'] for x in cases]),
                ('ordered_request_hashes',[protocol.digest(protocol.payload(x,c['model'],'B0')) for x in cases])):
                c[key] = value
            if not c['new_responses']:
                self.write_measurement(c, 20)
        gap = next(c for c in self.saved['cells'] if c['action'] == 'replace_interrupted_measurement')
        self.write_measurement({**gap,'analysis_namespace':'timing_concurrent8'}, 9)
        directory = plan.base(self.target, *plan.GAP)
        failed = []
        for c in self.cases['billiger-de'][9:16]:
            r = self.record(c,'kev',10)
            r.update(error='HTTP503',attempts=[{'status':503,'seconds':1.,'cost':None,
                     'error':'HTTP503','response':{}} for _ in range(3)])
            r.pop('prediction'); r.pop('probability_match'); failed.append(r)
        (directory/'prompt.failures.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in failed))
        (self.target/'configs/manifest.json').write_text('{}')
        (self.target/'configs/selected_variants.json').write_text('{"unchanged":true}\n')
        (self.target/'artifacts/progress.json').write_text('{"AP1_timing":{"status":"needs_attention"}}\n')
        (self.target/'artifacts/stage_execution.jsonl').write_text('{"historical":"preserved"}\n')
        self.d = {'protected_sha256_or_absent': {str(p.relative_to(self.target)):plan.sha(p)
                    for p in self.target.rglob('*') if p.is_file()}}
        self.old = {p:p.read_bytes() for p in self.target.rglob('*') if p.is_file()}
        (self.root/'artifacts/budget.json').write_text(json.dumps({'spent':120.,'reserved':3.,'attempts':0,
            'unknown_cost_attempts':0,'inference_credits':{'foreign':{'status':'active','funded':3}},
            'foreign_hold':{'reserved_usd':3}}))
        self.admission = self.root/'admission.json'
        self.spec = {'admitted':True,'stage':'core3_timing','run_id':3,'slurm_job_id':'999',
                     'predecessor':'366980','cap_usd':200,'max_invocations':1,'policy_sha256':'fixture'}
        self.admission.write_text(json.dumps(self.spec))
        self.inference = types.SimpleNamespace(**{**vars(run),'PROJECT':self.target,'key':lambda:'mock'})
        self.calls = []
        self.socket = patch('socket.socket',side_effect=AssertionError('Network forbidden'))
        self.socket.start(); self.addCleanup(self.socket.stop)

    def response(self,m):
        r = {'model':protocol.MODELS[m]['id'],'provider':protocol.MODELS[m]['provider'].split('/')[0],
             'usage':{'cost':.001}}
        if m == 'luna': r['choices'] = [{'finish_reason':'stop','message':{'content':'{"match":true}'}}]
        else: r['answers'] = {'match':{'type':'noul','noul':.8}}
        return r

    def record(self,c,m,seconds):
        raw = self.response(m)
        return {'benchmark':c['benchmark'],'pair_id':c['pair_id'],'model_alias':m,'variant':'B0',
            'study_run_id':3,'input_hash':c['input_hash'],'request_hash':protocol.digest(protocol.payload(c,m,'B0')),
            'prediction':1,'probability_match':None if m=='luna' else .8,
            'attempts':[{'status':200,'seconds':seconds,'cost':.001,'response':raw}]}

    def write_measurement(self,c,n):
        directory = plan.base(self.target,c['analysis_namespace'],c['model'],c['benchmark'])
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'prompt.jsonl').write_text(''.join(json.dumps(self.record(x,c['model'],i+1))+'\n'
            for i,x in enumerate(self.cases[c['benchmark']][:n])))
        if n == 20:
            (directory/'prompt.executions.jsonl').write_text(json.dumps({'new_pairs':20,'workers':c['workers'],
                'wall_seconds':40.,'throughput_pairs_per_second':.5})+'\n')

    def http(self, req, **kwargs):
        for p in (self.root/'artifacts/operations/repetition_writer.lock',self.target/'artifacts/parallel_writer.lock'):
            with p.open('a') as f:
                with self.assertRaises(BlockingIOError): fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.assertTrue((self.target/plan.ATTEMPT/'prior.json').is_file())
        self.assertTrue((self.target/plan.ATTEMPT/'cells.jsonl').is_file())
        self.assertEqual(set(plan.read(self.root/'artifacts/budget.json')['inference_credits']),{'foreign'})
        body=json.loads(req.data); self.calls.append(body)
        m=next(m for m in protocol.MODELS if protocol.MODELS[m]['id']==body['model'])
        return Response(json.dumps(self.response(m)).encode())

    def execute(self, http=None, mutate_report=None, full_analysis=False):
        original = mapped.bind
        def bind(*args):
            module, accepted = original(*args)
            if full_analysis: return module, accepted
            # Run actual frozen timing aggregation; full quality bootstrap,
            # audit HTML/plots are not exercised in this wrapper fixture.
            def main():
                table=module.timing_table()
                module.csv_save(module.OUT/'timing.csv',table)
                if mutate_report: mutate_report()
            module.main=main
            return module, accepted
        with patch.dict(os.environ,{'SLURM_JOB_ID':'999'}), \
                patch.object(wrapper,'preflight',return_value=(self.target,self.saved,self.d,{'fixture':True})), \
                patch.object(mapped,'bind',side_effect=bind), patch('urllib.request.urlopen',side_effect=http or self.http):
            return wrapper.execute(self.root,self.admission,self.inference,None,protocol,analyze)

    def test_full_112_cell_map_420_new_frozen_calls_no_quality_credit_or_completion(self):
        result=self.execute()
        self.assertEqual(len(self.calls),420)
        self.assertFalse(result['core_complete']); self.assertFalse(result['reports_published'])
        self.assertEqual(result['account']['attempts'],420)
        self.assertAlmostEqual(result['account']['known_usd'],.42)
        b=plan.read(self.root/'artifacts/budget.json')
        self.assertAlmostEqual(b['spent'],120.42);self.assertAlmostEqual(b['reserved'],3.)
        self.assertEqual(b['inference_credits'],{'foreign':{'status':'active','funded':3}})
        for p,data in self.old.items():
            if p.name == 'stage_execution.jsonl': self.assertTrue(p.read_bytes().startswith(data))
            else: self.assertEqual(p.read_bytes(),data)
        attempt=self.target/plan.ATTEMPT
        events=plan.rows(attempt/'cells.jsonl')
        self.assertEqual(len(events),42)
        table=list(csv.DictReader((attempt/'analysis_staging/artifacts/results/timing.csv').open()))
        self.assertEqual(len(table),112)
        row=next(r for r in table if (r['model'],r['benchmark'],r['regime'])==('kev','billiger-de','concurrent8'))
        cell=next(c for c in self.saved['cells'] if c['action']=='replace_interrupted_measurement')
        p=plan.base(self.target,plan.REPLACEMENT,'kev','billiger-de')
        wall=plan.rows(p/'prompt.executions.jsonl')[0]['wall_seconds']
        times=sorted(sum(a['seconds'] for a in r['attempts']) for r in plan.rows(p/'prompt.jsonl'))
        self.assertAlmostEqual(float(row['pairs_per_second']),20/wall)
        self.assertAlmostEqual(float(row['latency_p50']),(times[9]+times[10])/2)
        self.assertEqual(plan.validate_measurement(self.target,cell,protocol)['responses'],20)
        costs=plan.read(attempt/'timing_cost_proof.json')
        self.assertEqual(costs['analysis_sample']['responses'],2240)
        self.assertEqual(costs['all_retained_timing_attempts'],2270)
        self.assertEqual(costs['all_retained_unknown_attempts'],21)
        self.assertAlmostEqual(costs['all_retained_known_usd'],2.249)
        self.assertIsNone(costs['excluded_interrupted_diagnostic']['execution_walltime'])
        with self.assertRaises(FileExistsError): self.execute()

    def test_unknown_future_stops_after_one_batch_no_execution_or_false_reconciliation(self):
        with self.assertRaises(RuntimeError):
            self.execute(lambda *a,**k: (_ for _ in ()).throw(IncompleteRead(b'x',10)))
        events=plan.rows(self.target/plan.ATTEMPT/'execution.jsonl')
        self.assertEqual([r['status'] for r in events],['started','failed'])
        account=events[-1]['per_call_account']
        self.assertEqual(account['settled'],8); self.assertEqual(account['unknown'],8)
        self.assertEqual(account['outstanding'],0)
        self.assertFalse((self.target/plan.ATTEMPT/'staged_result.json').exists())
        self.assertEqual(len(plan.rows(self.target/plan.ATTEMPT/'cells.jsonl')),2)

    def test_durable_prior_failure_precedes_account_and_http(self):
        with patch.object(wrapper,'save_exclusive',side_effect=OSError('fsync')):
            with self.assertRaises(OSError): self.execute()
        self.assertEqual(self.calls,[])
        self.assertEqual(plan.read(self.root/'artifacts/budget.json')['attempts'],0)

    def test_budget_reread_after_prior_before_calls(self):
        original=wrapper.save_exclusive
        def save(p,obj):
            original(p,obj)
            if p.name=='prior.json':
                b=plan.read(self.root/'artifacts/budget.json');b['spent']=197.
                (self.root/'artifacts/budget.json').write_text(json.dumps(b))
        with patch.object(wrapper,'save_exclusive',side_effect=save):
            with self.assertRaisesRegex(ValueError,'headroom'):self.execute()
        self.assertEqual(self.calls,[])

    def test_unadmitted_entry_has_no_lock_or_attempt_side_effects(self):
        self.admission.write_text(json.dumps({'admitted':False}))
        with self.assertRaisesRegex(ValueError,'Unadmitted'):
            wrapper.execute(self.root,self.admission,self.inference,None,protocol,analyze)
        self.assertFalse((self.target/plan.ATTEMPT).exists())
        self.assertFalse((self.target/'artifacts/parallel_writer.lock').exists())

    def test_mapped_analysis_rejects_source_matrix_missing_execution_and_wrong_namespace(self):
        bad=copy.deepcopy(self.saved);bad['cells'].reverse()
        with self.assertRaisesRegex(ValueError,'ordered'):mapped.validate_map(bad,protocol)
        bad=copy.deepcopy(self.saved)
        next(c for c in bad['cells'] if c['action']=='replace_interrupted_measurement')['analysis_namespace']='timing_concurrent8'
        with self.assertRaisesRegex(ValueError,'canonical'):mapped.validate_map(bad,protocol)
        with self.assertRaises(ValueError):mapped.bind(analyze,self.target,self.saved,protocol,self.target/'out')
        fake=types.SimpleNamespace(**{**vars(analyze),'__file__':str(self.admission)})
        with self.assertRaisesRegex(ValueError,'analyzer changed'):mapped.bind(fake,self.target,self.saved,protocol,self.target/'out')

    def test_full_scheduler_history_joins_and_overlap_gates(self):
        spec={**self.spec,'sources':{p:'hash' for p in wrapper.REQUIRED},'policy':'policy.json'}
        (self.root/'configs').mkdir()
        (self.root/'configs/expanded_execution.json').write_text(json.dumps({'e2_confirmation_job':366694,'e2_schema_job':366695}))
        seen=set(); bad={'step':False,'join':False,'writer':False}
        def command(args,**kw):
            if args[0]=='sacct':
                j=args[3];seen.add(j)
                return ''.join(f'{j}{s}|'+('RUNNING' if bad['step'] and j=='365894' and s=='.extern' else 'FAILED')+'|1:0|end\n' for s in ('','.batch','.extern'))
            deps='366694:366695'+(':365300' if args[3]=='365355' else '')
            if bad['join']:deps+=':666'
            return f'Dependency=afterok:{deps} Priority=0 StartTime=Unknown JobState=PENDING'
        def review(*a):
            return {'policy_sha256':'fixture','reviewed_jobs':[{'id':'999','self':True,'scope':'core3'}]
                +([{'id':'666','scope':'core3'}] if bad['writer'] else [])}
        with patch.object(wrapper.authority,'authority'),patch.object(plan,'sha',return_value='hash'), \
                patch.object(wrapper,'readiness',return_value=(self.target,self.saved,self.d,{})), \
                patch.object(wrapper.authority,'private_module',return_value=types.SimpleNamespace(review=review)), \
                patch.object(wrapper.subprocess,'check_output',side_effect=command):
            wrapper.preflight(self.root,spec,'999',protocol,None)
            self.assertEqual(seen,set(wrapper.HISTORY))
            for field in bad:
                bad[field]=True
                with self.assertRaises((ValueError,AssertionError)):wrapper.preflight(self.root,spec,'999',protocol,None)
                bad[field]=False
        for key,value in [('run_id',True),('predecessor','366955'),('stage','core3'),('cap_usd',201),('max_invocations',2)]:
            with self.assertRaisesRegex(ValueError,'Separate exact'):
                wrapper.preflight(self.root,{**spec,key:value},'999',protocol,None)

    def test_readable_storage_failure_stops_without_staged_success(self):
        original=wrapper.fsync_dir
        def fsync(p):
            if p.name=='billiger-de' and 'runs' in p.parts:raise OSError('unconfirmed directory persistence')
            original(p)
        with patch.object(wrapper,'fsync_dir',side_effect=fsync):
            with self.assertRaises(OSError):self.execute()
        self.assertEqual(len(self.calls),20)
        self.assertFalse((self.target/plan.ATTEMPT/'staged_result.json').exists())

    def test_partial_report_failure_never_completes_or_retries(self):
        def fail(): raise OSError('partial report publication forbidden')
        with self.assertRaises(OSError):self.execute(mutate_report=fail)
        attempt=self.target/plan.ATTEMPT
        self.assertTrue((attempt/'analysis_staging/artifacts/results/timing.csv').exists())
        self.assertFalse((attempt/'staged_result.json').exists())
        with self.assertRaises(FileExistsError):self.execute()

    def test_full_frozen_analysis_in_private_workspace(self):
        # Construct complete quality inputs/records; this does not claim prior
        # HTTP or a true full-quality replay against real study caches.
        for b,cases in self.cases.items():
            for split in ('route','test'):
                (self.target/'data'/b/(split+'.json')).write_text(json.dumps(cases))
        selection={m:{'variant':'P1'} for m in protocol.ABLATION_MODELS}
        (self.target/'configs/selected_variants.json').write_text(json.dumps(selection))
        (self.target/'configs/manifest.json').write_bytes((ROOT/'configs/manifest.json').read_bytes())
        for m in protocol.MODELS:
            for b,cases in self.cases.items():
                for variant,splits in [('B0',('prompt','route','test'))]+(
                        [(v,('prompt','test') if v=='P1' else ('prompt',)) for v in ('P1','P2','S1','O1')]
                        if m in protocol.ABLATION_MODELS else []):
                    for split in splits:
                        p=self.target/'runs/quality'/m/variant/b/(split+'.jsonl');p.parent.mkdir(parents=True,exist_ok=True)
                        records=[{**self.record(c,m,i+1),'variant':variant,
                                  'request_hash':protocol.digest(protocol.payload(c,m,variant))} for i,c in enumerate(cases)]
                        p.write_text(''.join(json.dumps(r)+'\n' for r in records))
        (self.target/'artifacts/budget.json').symlink_to(self.root/'artifacts/budget.json')
        self.d['protected_sha256_or_absent']={str(p.relative_to(self.target)):plan.sha(p)
            for p in self.target.rglob('*') if p.is_file() and p.name!='budget.json'}
        result=self.execute(full_analysis=True)
        reports=result['reports_sha256']
        self.assertIn('manuscript/draft.md',reports)
        self.assertIn('artifacts/results/figures/quality_cost.png',reports)
        out=self.target/plan.ATTEMPT/'analysis_staging/artifacts/results'
        metrics=plan.read(out/'main_metrics.json')
        self.assertEqual(len(metrics),56)
        for r in metrics:
            self.assertEqual(r['n'],20)
            self.assertAlmostEqual(r['f1'],2/3)
            self.assertAlmostEqual(r['known_cost_usd'],.02)
        cascades=plan.read(out/'cascades.json')
        self.assertEqual(len(cascades),144)
        for r in cascades:self.assertAlmostEqual(r['f1'],2/3)
        ablations=list(csv.DictReader((out/'ablations.csv').open()))
        self.assertEqual(len(ablations),160)
        self.assertTrue(all(float(r['delta_f1'])==0 and float(r['decision_flip_rate'])==0 for r in ablations))
        self.assertEqual(len(list(csv.DictReader((out/'timing.csv').open()))),112)
        self.assertFalse((self.target/'artifacts/results').exists())
        self.assertFalse((self.target/'manuscript/draft.md').exists())


if __name__=='__main__':unittest.main()
