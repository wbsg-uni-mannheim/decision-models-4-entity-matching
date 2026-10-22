import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import grouped_transition as transition
from leased_budget import Credit


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'artifacts/operations').mkdir(parents=True)
        (self.root/'artifacts/verification').mkdir()
        self.budget=self.root/'artifacts/budget.json'
        self.budget.write_text(json.dumps({'spent':100.,'reserved':3.,'attempts':0,'unknown_cost_attempts':0,'foreign_hold':3}))
        self.authority=self.root/'approval.json';self.authority.write_text('{}')
        self.jobs=patch.object(transition,'JOBS',{'confirmation':'123'})
        self.jobs.start();self.addCleanup(self.jobs.stop)
        self.credit=Credit(self.root,'fast-confirmation-123',{})
        self.attempt=self.root/'artifacts/operations/fast_resume_confirmation_20261015'
        self.attempt.mkdir();(self.attempt/'prior_journals.json').write_text('{}')

    def drain(self):
        before=json.loads(self.budget.read_text())
        transition.request_stop(self.root,self.authority)
        stopped=json.loads(self.budget.read_text())
        for key in ('spent','reserved','attempts','unknown_cost_attempts','foreign_hold'):
            self.assertEqual(before[key],stopped[key])
        rows=[]
        for i in range(512):
            self.credit.account('reserve');self.credit.account('settle',None if i==0 else .001)
            rows.append({'attempts':[{'cost':None if i==0 else .001}]})
        for _ in range(56):
            with self.assertRaisesRegex(ValueError,'Concurrent or altered credit state'):
                self.credit.account('reserve')
        with self.assertRaises(ValueError):self.credit.checkpoint(close=True)
        p=self.root/'runs/quality/jev/B0/abt-buy/prompt.jsonl';p.parent.mkdir(parents=True)
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (self.attempt/'execution.jsonl').write_text(json.dumps({'status':'failed','slurm_job_id':'123','utc':transition.utc(),'credit':self.credit.summary()})+'\n')
        return p

    def test_loaded_credit_stops_without_charge_change_then_exact_reconciliation(self):
        p=self.drain();raw=p.read_bytes()
        with patch.object(transition,'terminal',return_value={'confirmation':'terminal fixture'}):
            result=transition.reconcile(self.root)
        actual=json.loads(self.budget.read_text())
        self.assertAlmostEqual(actual['spent'],100.561)
        self.assertAlmostEqual(actual['reserved'],3.)
        self.assertEqual(actual['attempts'],512);self.assertEqual(actual['unknown_cost_attempts'],1)
        self.assertEqual(actual['foreign_hold'],3);self.assertEqual(p.read_bytes(),raw)
        self.assertTrue(result['all_steps_terminal'])
        self.assertEqual(actual['inference_credits'][self.credit.identity]['status'],'closed')

    def test_missing_paid_answer_retains_full_hold(self):
        p=self.drain();p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
        before=self.budget.read_bytes()
        with patch.object(transition,'terminal',return_value={'confirmation':'terminal fixture'}):
            with self.assertRaisesRegex(ValueError,'Unreconciled raw credit'):
                transition.reconcile(self.root)
        self.assertEqual(self.budget.read_bytes(),before)

    def test_no_reconciliation_before_terminal_and_no_blind_second_stop(self):
        self.drain();before=self.budget.read_bytes()
        with self.assertRaises(ValueError):transition.request_stop(self.root,self.authority)
        with patch.object(transition,'terminal',side_effect=ValueError('Active writer')):
            with self.assertRaises(ValueError):transition.reconcile(self.root)
        self.assertEqual(self.budget.read_bytes(),before)


if __name__=='__main__':unittest.main()
