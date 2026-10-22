"""Targeted budget250 and bounded launch integration tests; no real network."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import leased_budget
import same_condition_launch_20261016 as launch
import test_same_condition_repetition_runtime_20261016 as fixture


class LaunchTests(unittest.TestCase):
    def setup_budget(self, spent=205):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        root=Path(tmp.name);(root/'artifacts/operations').mkdir(parents=True);(root/'configs').mkdir();(root/'artifacts/verification').mkdir()
        launch.save(root/launch.AUTH,dict(approved_by='Aaron',project_cap_usd=250,same_conditions_only=True,new_selection=False))
        launch.save(root/launch.BUDGET_POLICY,dict(project_cap_usd=250,approval_sha256=launch.sha(root/launch.AUTH)))
        launch.save(root/'artifacts/budget.json',dict(spent=spent,reserved=3,attempts=0,unknown_cost_attempts=0,
            inference_credits={'foreign':{'status':'active','funded':3}},project_cap_usd=250,budget_authority_sha256=launch.sha(root/launch.AUTH)))
        return root

    def test_private250_above_old_cap_original_unchanged(self):
        root=self.setup_budget();old_default=leased_budget.Credit.account.__defaults__
        credit=launch.credit_class(root)(root,'new',{},chunk=8)
        credit.account('reserve');credit.account('settle',.1);credit.checkpoint(close=True)
        b=launch.read(root/'artifacts/budget.json')
        self.assertAlmostEqual(b['spent'],205.1);self.assertAlmostEqual(b['reserved'],3)
        self.assertEqual(b['inference_credits']['foreign'],{'status':'active','funded':3})
        self.assertEqual(leased_budget.CAP,200);self.assertEqual(leased_budget.Credit.account.__defaults__,old_default)

    def test250_ceiling_and_authority_fail_closed(self):
        root=self.setup_budget(247)
        with self.assertRaises(RuntimeError):launch.credit_class(root)(root,'new',{},chunk=8)
        b=launch.read(root/'artifacts/budget.json');b['project_cap_usd']=350
        (root/'artifacts/budget.json').write_text(json.dumps(b))
        with self.assertRaises(ValueError):launch.credit_class(root)(root,'new',{},chunk=8)

    def test_budget_storage_failure_no_false_close(self):
        root=self.setup_budget();credit=launch.credit_class(root)(root,'new',{},chunk=8)
        credit.account('reserve');credit.account('settle',.1)
        with patch('leased_budget.os.replace',side_effect=OSError('fixture')),self.assertRaises(OSError):credit.checkpoint(close=True)
        self.assertFalse(credit.closed)
        self.assertEqual(launch.read(root/'artifacts/budget.json')['inference_credits']['new']['status'],'active')

    def test_whole_history_numbered_step_and_missing_failure(self):
        root=self.setup_budget();raw='1|FAILED|1:0|end|\n1.batch|FAILED|1:0|end|\n1.extern|COMPLETED|0:0|end|\n1.0|COMPLETED|0:0|end|\n'
        with patch.object(launch,'cmd',return_value=raw):expected=launch.steps(root,['1'])
        with patch.object(launch,'cmd',return_value=raw.replace('1.0|COMPLETED|0:0|end|\n','')),self.assertRaises(ValueError):launch.steps(root,['1'],expected)
        with patch.object(launch,'cmd',return_value=raw.replace('1.batch|FAILED','1.batch|RUNNING')),self.assertRaises(ValueError):launch.steps(root,['1'])

    def test_scheduler_alljoins_held_and_foreign_overlap(self):
        root=self.setup_budget(); names={'365300':'decision-em-rules','365355':'statistics-full','9':launch.NAME}
        deps={'365300':'afterok:366694(failed),afterok:366695(failed)','365355':'afterok:366694(failed),afterok:366695(failed),afterok:365300(unfulfilled)','9':'(null)'}
        import hashlib
        policy={'jobs':{j:dict(name=n,dependency=deps[j],batch_script_sha256=hashlib.sha256(b'batch').hexdigest()) for j,n in names.items()}}
        def command(*args):
            if args[0]=='squeue':return '\n'.join(j+'|'+n for j,n in names.items())
            if args[1]=='write':return 'batch'
            j=args[3];return f'WorkDir={root} JobState=PENDING StartTime=Unknown Priority=0 Dependency={deps[j]}'
        with patch.object(launch,'cmd',side_effect=command):self.assertEqual(launch.queue(root,policy),policy['jobs'])
        deps['365300']='afterok:366694'
        with patch.object(launch,'cmd',side_effect=command),self.assertRaises(ValueError):launch.queue(root,policy)

    def test_runtime_launch_prior_before_funding_real250_mock_requests(self):
        f=fixture.RepetitionTests();f.setUp();self.addCleanup(f.doCleanups);f.small()
        root,target=f.root,f.target
        (root/'artifacts/verification').mkdir(exist_ok=True)
        (root/'configs').mkdir(exist_ok=True);(root/'artifacts/operations').mkdir(exist_ok=True)
        launch.save(root/launch.AUTH,dict(approved_by='Aaron',project_cap_usd=250,same_conditions_only=True,new_selection=False))
        launch.save(root/launch.BUDGET_POLICY,dict(project_cap_usd=250,approval_sha256=launch.sha(root/launch.AUTH)))
        b=launch.read(f.budget);b.setdefault('inference_credits',{});b.update(project_cap_usd=250,budget_authority_sha256=launch.sha(root/launch.AUTH));f.budget.write_text(json.dumps(b))
        attempt=target/launch.ATTEMPT;attempt.parent.mkdir(exist_ok=True)
        modules=(f.inference,fixture.protocol,fixture.expanded,fixture.schema)
        prior=launch.runtime.capture_prior(root,target,2,*modules,attempt/'inference')
        launch.save(root/launch.PREP,{'priors':{'2':prior},'budget_before':copy.deepcopy(b)})
        launch.save(root/launch.ADMISSION,{'slurm_job_id':'9'})
        cls=launch.credit_class(root)
        def fund(*args,**kwargs):
            self.assertEqual(launch.read(attempt/'prior.json'),prior)
            self.assertTrue((attempt/'start_gate.json').exists())
            return cls(*args,**kwargs)
        with patch.dict(os.environ,SLURM_JOB_ID='9'),patch.object(launch,'load_modules',return_value=modules),patch.object(launch,'locks') as locks,patch.object(launch,'gate',return_value={'fixture':True}),patch.object(launch,'protected'),patch.object(launch,'credit_class',return_value=fund),patch('urllib.request.urlopen',side_effect=f.http):
            launch.execute(root,2)
            self.assertTrue(locks.called)
        self.assertTrue((attempt/'inference_complete.json').exists())
        self.assertEqual(launch.read(attempt/'credit_reconciliation.json')['attempts'],56)
        self.assertTrue(launch.read(attempt/'inference_complete.json')['independent_report_and_slurm_steps_pending'])
        with patch.dict(os.environ,SLURM_JOB_ID='not9'),self.assertRaises(ValueError):launch.execute(root,2)

    def test_preflight_runs_full_scoped_child_each_time(self):
        root=self.setup_budget();launch.save(root/launch.ADMISSION,{'admitted':True})
        launch.save(root/launch.PREP,{'priors':{'2':{'x':2},'3':{'x':3}}})
        class Result:
            def __init__(self,rid):self.stdout=json.dumps({'x':int(rid)})
        def child(args,**kwargs):return Result(args[-1])
        with patch.object(launch,'locks'),patch.object(launch,'gate',return_value={}) as gate,patch.object(launch,'protected'),patch.object(launch.subprocess,'run',side_effect=child) as run:
            launch.preflight(root,{},root/'held.json');launch.preflight(root,{},root/'prerelease.json')
            self.assertEqual(gate.call_count,2);self.assertEqual(run.call_count,4)
        self.assertTrue(launch.read(root/'held.json')['full_scoped_raw_readback'])


if __name__=='__main__':unittest.main()
