"""Synthetic Run3 supplement through real frozen IO/report and exact historical prefix.

Core numerical evidence/source bytes and scheduler policy are mocked here;
the separate real anchor readback verifies those actual prerequisites remotely.
"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import analyze
import orchestrate
import protocol
import core3_committed_history_20261016 as history
import run_core3_supplement_grouped_20261016 as wrapper
import test_run_fresh_supplement_grouped as fixture

ROOT=Path(__file__).resolve().parents[2]
old=wrapper.original


class Run3WrapperTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.GroupedWrapperTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.root=self.f.root;self.target=self.root/history.TARGET
        self.f.target.rename(self.target)
        self.f.target=self.target;self.f.f.target=self.target
        (self.target/'artifacts/operations').mkdir(exist_ok=True)
        self.f.inference.PROJECT=self.target;self.f.supplement.PROJECT=self.target
        self.f.history=self.target/history.STAGE;self.f.progress=self.target/history.PROGRESS
        self.f.f.history=self.f.history;self.f.f.progress=self.f.progress
        anchor=self.root/history.ANCHOR;anchor.mkdir(parents=True)
        for rel,name in ((history.STAGE,'stage_execution.jsonl'),(history.PROGRESS,'progress.json')):
            raw=(ROOT/history.TARGET/rel).read_bytes()
            (self.target/rel).write_bytes(raw);(anchor/name).write_bytes(raw)
        for rel in wrapper.REQUIRED:
            if rel.startswith(history.ANCHOR):continue
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
        manifest=dict(adapter=history.ADAPTER,run_id=3,history_jobs=list(history.HISTORY),
            source_sha256=history.sha(self.root/history.SELF),admission_granted=False)
        (anchor/'manifest.json').write_text(json.dumps(manifest))
        self.spec=copy.deepcopy(self.f.spec)
        self.spec.update(run_id=3,replica=str(self.target),history_adapter=history.ADAPTER,
            core_job_id='367108',core_stage_journal_sha256=history.STAGE_SHA,progress_sha256=history.PROGRESS_SHA,
            core_commit_anchor_sha256=history.sha(anchor/'manifest.json'),admission_path='admission.json',
            source_sha256={p:history.sha(self.root/p) for p in wrapper.REQUIRED},
            parallel_policy_sha256='b'*64,reviewed_dependencies={'43':'(null)',
                '365300':'afterok:366694(unfulfilled),afterok:366695(unfulfilled)',
                '365355':'afterok:366694(unfulfilled),afterok:366695(unfulfilled),afterok:365300(unfulfilled)'})
        self.spec.pop('prerequisites',None)
        self.admission=self.root/'admission.json';self.save()
        self.policy={'policy_sha256':'b'*64,'reviewed_jobs':[{'id':j,'self':j=='43'} for j in self.spec['reviewed_dependencies']]}
        self.priority='0';self.other='independent Run1 writer'
        self.plan={'full_stage_scenario':{'known_proxy_usd':.014,'historical_unknown_scenario_usd':0},
                   'quote_or_hard_bound':False,'future_cache_discount':False}

    def save(self):self.admission.write_text(json.dumps(self.spec))

    def scheduler(self,args,**kwargs):
        if args[:3]==['scontrol','show','job']:
            return ('Dependency='+self.spec['reviewed_dependencies'][args[3]]+
                    ' JobState=PENDING StartTime=Unknown Priority='+self.priority)
        if args[:3]==['scontrol','write','batch_script']:return self.other
        raise AssertionError(args)

    @contextmanager
    def gates(self):
        with patch.object(history,'bound_core',return_value=({'report_sha256':{}},{'raw_sha256':{}},{})), \
                patch.object(history,'history_steps',return_value={'actual_steps_mocked':True}), \
                patch.object(old.old,'planning',return_value=self.plan), \
                patch.object(old.common,'make_wrapper') as policy, \
                patch.object(old.common.subprocess,'check_output',side_effect=self.scheduler), \
                patch.object(old,'verify',return_value=self.f.replica):
            policy.return_value.review.return_value=self.policy
            yield

    def execute(self):
        return wrapper.private_wrapper().execute_bound(self.root,self.target,3,self.admission,self.spec,'43',
            self.f.replica,self.f.inference,self.f.supplement,protocol,analyze,orchestrate)

    def ready(self):
        return wrapper.private_wrapper().verify_ready(self.root,self.target,3,self.spec,'43',self.f.replica,
                                                     self.f.supplement,protocol)

    def credit(self):return old.read(self.root/'artifacts/budget.json')['inference_credits']['fresh-supplement-3-43']

    def test_full_frozen_pipeline_and_private_module_isolation(self):
        original_functions={n:getattr(old,n) for n in ('validate','prerequisites','verify_ready','execute_bound','execute')}
        prior=self.f.history.read_bytes();before_progress=old.read(self.f.progress)
        calls=[]
        def http(req,**kw):
            self.f.assert_locks()
            record=self.credit()
            self.assertEqual(record['provenance']['prior_sha256'],old.sha(self.target/wrapper.ATTEMPT/'prior.json'))
            self.assertEqual(record['provenance']['run_id'],3)
            calls.append(req);return self.f.f.http(req,**kw)
        with self.gates(),patch('urllib.request.urlopen',side_effect=http):self.assertEqual(self.execute(),0)
        self.assertEqual(len(calls),14);self.assertEqual(self.credit()['status'],'closed')
        self.assertEqual(self.credit()['settled'],14);self.assertAlmostEqual(self.credit()['charged'],.014)
        self.assertAlmostEqual(old.read(self.root/'artifacts/budget.json')['reserved'],3)
        self.assertTrue(self.f.history.read_bytes().startswith(prior))
        progress=old.read(self.f.progress);self.assertEqual({k:progress[k] for k in before_progress},before_progress)
        self.assertEqual(progress['supplement_billiger50']['status'],'complete_automated')
        for p in (self.target/'runs').rglob('test.jsonl'):
            self.assertTrue(all(type(r['study_run_id']) is int and r['study_run_id']==3 for r in old.old.rows(p)))
        independent=self.root/'independent'
        old.bind(self.f.supplement.analyze,PROJECT=independent,verify=self.f.supplement.verify,
            load=old.bind(analyze.load,PROJECT=self.target))()
        for name in old.REPORT_NAMES:
            self.assertEqual((self.target/old.REPORT_DIR/name).read_bytes(),(independent/old.REPORT_DIR/name).read_bytes())
        for name,fn in original_functions.items():self.assertIs(getattr(old,name),fn)
        with self.gates(),self.assertRaises(ValueError):self.execute()

    def test_two_separate_full_preflights_no_attempt_or_credit(self):
        before=(self.root/'artifacts/budget.json').read_bytes()
        with self.gates():
            for name in ('held.json','prerelease.json'):
                self.assertEqual(wrapper.preflight_bound(self.root,self.target,3,self.admission,self.spec,'43',
                    self.f.replica,self.f.inference,self.f.supplement,protocol,analyze,orchestrate,
                    output=self.root/name),0)
                self.assertFalse(old.read(self.root/name)['inference_started'])
        self.assertFalse((self.target/wrapper.ATTEMPT).exists())
        self.assertEqual((self.root/'artifacts/budget.json').read_bytes(),before)

    def test_unadmitted_wrong_run_anchor_and_canonical_history_rejected(self):
        base=copy.deepcopy(self.spec)
        for key,value in [('admitted',False),('run_id',3.0),('history_adapter','canonical_success_only_v1'),
                          ('core_commit_anchor_sha256','0'*64),('core_job_id','366980')]:
            self.spec={**base,key:value};self.save()
            with self.gates(),self.assertRaises(ValueError):self.execute()
            self.assertFalse((self.target/wrapper.ATTEMPT).exists())

    def test_foreign_history_source_and_queue_scope_rejected(self):
        with self.gates():
            self.f.history.write_bytes(self.f.history.read_bytes()+b'{}\n')
            with self.assertRaises(ValueError):self.ready()
            self.f.history.write_bytes((ROOT/history.TARGET/history.STAGE).read_bytes())
            self.priority='1'
            with self.assertRaises(ValueError):self.ready()
            self.priority='0';self.other=str(self.target)
            with self.assertRaises(ValueError):self.ready()
            self.other='independent'
            p=self.root/history.SELF;p.write_bytes(p.read_bytes()+b'\n')
            with self.assertRaises(ValueError):self.ready()
        self.assertFalse((self.target/wrapper.ATTEMPT).exists())

    def test_old_grouped_attempt_is_not_reused(self):
        (self.target/old.ATTEMPT).mkdir()
        with self.gates(),self.assertRaises(ValueError):self.ready()
        self.assertFalse((self.target/wrapper.ATTEMPT).exists())

    def test_budget_reread_and_prior_fsync_before_funding(self):
        budget=old.read(self.root/'artifacts/budget.json');budget['spent']=190
        (self.root/'artifacts/budget.json').write_text(json.dumps(budget))
        with self.gates(),self.assertRaises(ValueError):self.execute()
        self.assertFalse((self.target/wrapper.ATTEMPT).exists())
        budget['spent']=100;(self.root/'artifacts/budget.json').write_text(json.dumps(budget))
        with self.gates(),patch.object(old.files,'durable_write',side_effect=OSError('fsync')),self.assertRaises(OSError):self.execute()
        self.assertNotIn('fresh-supplement-3-43',old.read(self.root/'artifacts/budget.json').get('inference_credits',{}))

    def test_unknown_incomplete_read_retains_credit_with_zero_outstanding(self):
        import http.client as client
        def fail(req,**kw):
            if json.loads(req.data)['model']==protocol.MODELS['jev']['id']:raise client.IncompleteRead(b'')
            return self.f.f.http(req,**kw)
        with self.gates(),patch('urllib.request.urlopen',side_effect=fail),self.assertRaises(ValueError):self.execute()
        self.assertEqual(self.credit()['status'],'active')
        self.assertFalse((self.target/old.MARKER).exists())
        retained=[r for r in old.old.rows(self.target/wrapper.ATTEMPT/'execution.jsonl') if r['status']=='credit_retained']
        self.assertTrue(retained);self.assertEqual(retained[-1]['credit']['outstanding'],0)


if __name__=='__main__':unittest.main()
