"""Guard tests and real frozen request/cache/accounting with synthetic HTTP only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import peeters_run as runner

ROOT = Path(__file__).resolve().parents[2]
CHILD = r'''
import io, json, os, socket, sys
from pathlib import Path
from unittest.mock import patch
import peeters_run as r
root, rid = Path(sys.argv[1]), int(sys.argv[2])
orig_load = r.load_modules
cells = [{'benchmark':'abt-buy', 'split':s, 'model':'jev', 'arm':a}
         for s in ['prompt','route','test'] for a in r.control.ARMS]
def load(target):
    run, proto, orch, bill = orig_load(target)
    orch.verify = bill.verify = lambda: None
    return run, proto, orch, bill
class Response(io.BytesIO):
    status = 200
def reply(req, **kw):
    b = json.loads((root/'artifacts/budget.json').read_text())
    assert abs(b['reserved']-3.05)<1e-8
    body = json.loads(req.data)
    assert body['questions']['match']['instructions'] == r.control.QUESTION
    assert 'private-id' not in req.data.decode()
    assert body['provider'] == {'only':['typesafe'],'allow_fallbacks':False}
    result = {'model':'typesafe/jev-1.13','provider':'typesafe',
              'answers':{'match':{'type':'noul','noul':.8}}}
    if os.environ['OUTCOME'] == 'success': result['usage']={'cost':.012}
    return Response(json.dumps(result).encode())
policy = root/'policy.json'
with patch.object(r,'review',return_value={'policy_sha256':r.sha(policy),'reviewed_jobs':[{'self':True}]}), \
     patch.object(r,'verify_replica',return_value={}), \
     patch.object(r,'verify_matrix',return_value=cells), \
     patch.object(r,'load_modules',side_effect=load), \
     patch('urllib.request.urlopen',side_effect=reply), \
     patch.object(socket,'socket',side_effect=AssertionError('Network forbidden')):
    r.execute(root,rid,root/f'admission{rid}.json',policy)
'''


class PeetersRunTests(unittest.TestCase):
    def test_admission_and_budget_guards(self):
        expected = {'slurm_job_id':'123','run_id':2,'stage':'peeters','target':'r2',
                    'execution_manifest_sha256':'a','parallel_policy_sha256':'b','budget_cap_usd':100}
        runner.validate_admission(expected, expected, {'spent':50,'reserved':3})
        for k in expected:
            with self.assertRaises(ValueError):
                runner.validate_admission({**expected,k:'wrong'},expected,{'spent':50,'reserved':3})
        for budget in [{'spent':97,'reserved':3},{'spent':float('nan'),'reserved':0}]:
            with self.assertRaises(ValueError):
                runner.validate_admission(expected,expected,budget)

    def test_all_development_precedes_test_and_failure_blocks_test(self):
        from types import SimpleNamespace
        cells = [{'model':m,'arm':a,'benchmark':'abt-buy','split':s}
                 for s in ['test','route','prompt'] for m in ['jev','luna'] for a in runner.control.ARMS]
        seen=[]
        def cell(m,a,b,s,**kwargs):
            self.assertEqual(kwargs,{'workers':8,'namespace':'peeters'})
            seen.append(s)
        runner.run_matrix(SimpleNamespace(run_cell=cell),cells)
        self.assertEqual(seen,['prompt']*4+['route']*4+['test']*4)
        seen.clear()
        def fail(m,a,b,s,**kw):
            seen.append(s)
            raise ValueError('Invalid provider response')
        with self.assertRaises(ValueError):
            runner.run_matrix(SimpleNamespace(run_cell=fail),cells)
        self.assertNotIn('test',seen)

    def test_full_matrix_detects_missing_duplicate_data_and_wire_drift(self):
        import copy
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'src'))
        import protocol
        proto = SimpleNamespace(**{k:getattr(protocol,k) for k in ['digest','canonical','payload','state']},
                                MODELS={'jev':protocol.MODELS['jev']},BENCHMARKS={'abt-buy':{}})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rel in ['src','artifacts/operations','data/abt-buy','data/billiger-de-50un']:
                (root/rel).mkdir(parents=True)
            for rel in ['src/protocol.py','artifacts/operations/peeters_control.py']:
                shutil.copy2(ROOT/rel,root/rel)
            cells=[]
            for b in ['abt-buy','billiger-de-50un']:
                for s in (['test'] if b=='billiger-de-50un' else ['prompt','route','test']):
                    records={'left':{'name':'x'},'right':{'name':'y'}}
                    cases=[{'benchmark':b,'pair_id':'private-id','records':records,'input_hash':proto.digest(records)}]
                    (root/'data'/b/(s+'.json')).write_text(json.dumps(cases))
                    for a in runner.control.ARMS:
                        wire=proto.canonical(runner.control.payload(cases[0],'jev',a,proto)).encode()+b'\n'
                        cells.append({'benchmark':b,'split':s,'model':'jev','arm':a,'pairs':1,
                                      'input_data_sha256':proto.digest(cases),'ordered_payloads_sha256':hashlib.sha256(wire).hexdigest()})
            manifest={'question':runner.control.QUESTION,'arms':list(runner.control.ARMS),
                      'implementation_sha256':runner.sha(root/'artifacts/operations/peeters_control.py'),
                      'protocol_sha256':runner.sha(root/'src/protocol.py'),'source_manifest_sha256':'source',
                      'cells':cells,'calls_per_run_before_retries':8}
            with patch.object(runner.control,'verify_source',return_value='source'):
                self.assertEqual(len(runner.verify_matrix(root,root,manifest,proto)),8)
                for badcells in [cells[:-1],cells+[cells[0]]]:
                    with self.assertRaises(ValueError):
                        runner.verify_matrix(root,root,{**manifest,'cells':badcells},proto)
                bad=copy.deepcopy(manifest)
                bad['cells'][0]['ordered_payloads_sha256']='wrong'
                with self.assertRaisesRegex(ValueError,'request drift'):
                    runner.verify_matrix(root,root,bad,proto)
                (root/'data/abt-buy/prompt.json').write_text('[]')
                with self.assertRaisesRegex(ValueError,'dataset drift'):
                    runner.verify_matrix(root,root,manifest,proto)


class PeetersProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        for d in ['src','configs','artifacts/operations','runs']:
            (self.root/d).mkdir(parents=True)
        for rel in runner.SOURCES:
            shutil.copy2(ROOT/rel,self.root/rel)
        (self.root/runner.MANIFEST).write_text('{}')
        (self.root/runner.EXECUTION).write_text(json.dumps({'payload_manifest_sha256':runner.sha(self.root/runner.MANIFEST),
            'source_sha256':{p:runner.sha(self.root/p) for p in runner.SOURCES}}))
        self.budget=self.root/'artifacts/budget.json'
        self.budget.write_text(json.dumps({'spent':70,'reserved':3,'attempts':0,'unknown_cost_attempts':0}))
        (self.root/'policy.json').write_text('{}')
        for rid in [2,3]:
            target=runner.target_path(self.root,rid)
            for d in ['src','artifacts','runs','data/abt-buy']:
                (target/d).mkdir(parents=True)
            for src in (self.root/'src').glob('*.py'):
                shutil.copy2(src,target/'src'/src.name)
            (target/'artifacts/progress.json').write_text('{"smoke":{"status":"complete"}}')
            (target/'artifacts/stage_execution.jsonl').write_text(json.dumps({'run_id':rid,'stage':'core','status':'complete'})+'\n')
            case={'benchmark':'abt-buy','pair_id':'private-id','input_hash':'fixture',
                  'records':{'left':{'name':'item'},'right':{'name':'item'}}}
            for s in ['prompt','route','test']:
                (target/'data/abt-buy'/(s+'.json')).write_text(json.dumps([case]))
            (self.root/f'admission{rid}.json').write_text(json.dumps({'run_id':rid,'slurm_job_id':'fixture','stage':'peeters',
                'target':str(target),'budget_cap_usd':100,'parallel_policy_sha256':runner.sha(self.root/'policy.json'),
                'execution_manifest_sha256':runner.sha(self.root/runner.EXECUTION)}))

    def launch(self,rid=2,outcome='success'):
        return subprocess.run([sys.executable,'-c',CHILD,str(self.root),str(rid)],capture_output=True,text=True,timeout=30,
            env={**os.environ,'PYTHONPATH':str(Path(runner.__file__).parent),'SLURM_JOB_ID':'fixture',
                 'OPENROUTER_API_KEY':'synthetic-no-network','OUTCOME':outcome})

    def test_fresh_runs_arms_and_shared_budget_no_cross_reuse(self):
        for rid in [2,3]:
            result=self.launch(rid)
            self.assertEqual(result.returncode,0,result.stderr)
            target=runner.target_path(self.root,rid)
            for a in runner.control.ARMS:
                rows=[json.loads(x) for x in (target/f'runs/peeters/jev/{a}/abt-buy/test.jsonl').read_text().splitlines()]
                self.assertEqual(rows[0]['study_run_id'],rid)
                self.assertEqual(rows[0]['variant'],a)
                self.assertEqual(rows[0]['prediction'],1)
                self.assertEqual(rows[0]['peeters_payload_manifest_sha256'],runner.sha(self.root/runner.MANIFEST))
                sys.path.insert(0,str(ROOT/'src'))
                import protocol
                case=json.loads((target/'data/abt-buy/test.json').read_text())[0]
                expected=runner.control.payload(case,'jev',a,protocol)
                self.assertEqual(rows[0]['request_hash'],protocol.digest(expected))
                self.assertEqual(rows[0]['input_hash'],case['input_hash'])
                self.assertEqual(rows[0]['attempts'][0]['cost'],.012)
            before=self.budget.read_bytes()
            self.assertNotEqual(self.launch(rid).returncode,0)
            self.assertEqual(self.budget.read_bytes(),before)
        b=json.loads(self.budget.read_text())
        self.assertEqual(b['attempts'],12)
        self.assertAlmostEqual(b['spent'],70.144)
        self.assertAlmostEqual(b['reserved'],3)
        self.assertFalse(list((self.root/'runs').rglob('*.jsonl')))

    def test_failure_keeps_unknown_cost_and_blocks_test_and_repeat(self):
        out=self.launch(outcome='missing_usage')
        self.assertNotEqual(out.returncode,0)
        target=runner.target_path(self.root,2)
        rows=[json.loads(x) for x in (target/'artifacts/peeters_stage.jsonl').read_text().splitlines()]
        self.assertEqual([x['status'] for x in rows],['started','failed'])
        self.assertFalse(list((target/'runs').rglob('test.jsonl')))
        b=json.loads(self.budget.read_text())
        self.assertAlmostEqual(b['spent'],70.05)
        self.assertAlmostEqual(b['reserved'],3)
        self.assertEqual(b['unknown_cost_attempts'],1)
        before=self.budget.read_bytes()
        self.assertNotEqual(self.launch().returncode,0)
        self.assertEqual(self.budget.read_bytes(),before)

    def test_cap_source_and_incomplete_core_block_before_http(self):
        b=json.loads(self.budget.read_text()); b['spent']=97
        self.budget.write_text(json.dumps(b))
        before=self.budget.read_bytes()
        self.assertIn('ceiling',self.launch().stderr)
        self.assertEqual(before,self.budget.read_bytes())
        target=runner.target_path(self.root,2)
        (target/'src/run.py').write_text('changed')
        self.assertIn('source drift',self.launch().stderr)
        target=runner.target_path(self.root,3)
        (target/'artifacts/stage_execution.jsonl').write_text(json.dumps({'run_id':3,'stage':'core','status':'failed'})+'\n')
        self.assertIn('Core incomplete',self.launch(3).stderr)
        self.assertEqual(before,self.budget.read_bytes())


if __name__ == '__main__':
    unittest.main()
