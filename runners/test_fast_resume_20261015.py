import json
from pathlib import Path
from unittest.mock import patch
import unittest

import confirmation_drained
import schema_drained
import protocol
from fast_resume_20261015 import credit_build, core_component, verify_accounting, new_attempt_costs
from leased_budget import Credit
from test_confirmation_drained import ConfirmationDrainTests
from test_schema_drained import SchemaDrainTests
from test_fresh_core_run02_third_drained import SecondTests
from test_drained_inference import Response


class FastTests(unittest.TestCase):
    def test_confirmation_exact_requests_resume_and_reconciliation(self):
        f=ConfirmationDrainTests();f.setUp();self.addCleanup(f.doCleanups)
        credit=Credit(f.root,'fast-confirmation-test',{})
        original=f.execute
        # Exclude pre-existing development rows from incremental accounting.
        from fast_transition_audit import sha
        before={str(p.relative_to(f.root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (f.root/'runs').rglob('*.jsonl')}
        def execute():
            result=original();verify_accounting(credit,f.root,before)
            credit.checkpoint(close=True);return result
        f.execute=execute
        with patch.object(confirmation_drained,'build',side_effect=credit_build(credit,f.root)):
            f.test_complete_synthetic_matrix_real_request_parser_account_and_cache()
        self.assertEqual(credit.attempts,126)
        self.assertEqual(credit.global_transactions,2)

    def test_schema_wire_tls_all_conditions_and_no_cross_namespace_charges(self):
        f=SchemaDrainTests();f.setUp();self.addCleanup(f.doCleanups)
        credit=Credit(f.root,'fast-schema-test',{})
        from fast_transition_audit import sha
        before={str(p.relative_to(f.root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (f.root/'runs').rglob('*.jsonl')}
        original=f.execute
        def execute():
            result=original();verify_accounting(credit,f.root,before)
            credit.checkpoint(close=True);return result
        f.execute=execute
        with patch.object(schema_drained,'build',side_effect=credit_build(credit,f.root)):
            f.test_complete_real_wire_parser_account_cache_matrix()
        other=f.root/'runs/quality/other.jsonl';other.parent.mkdir(parents=True,exist_ok=True)
        other.write_text(json.dumps({'attempts':[{'cost':99}]})+'\n')
        self.assertEqual(new_attempt_costs(f.root,before,{'schema'})['attempts'],126)
        self.assertEqual(credit.global_transactions,2)

    def test_full_core_440_real_requests_timing_original_account_and_dev_resume(self):
        f=SecondTests();f.setUp();self.addCleanup(f.doCleanups)
        comp=core_component(f.root)
        credit=Credit(f.root,'fast-core2-test',{})
        comp.build=credit_build(credit,f.root)
        calls=[]
        def http(req,**kw):
            body=json.loads(req.data);calls.append(body)
            m=next(m for m,d in protocol.MODELS.items() if d['id']==body['model'])
            raw={'model':body['model'],'provider':protocol.MODELS[m]['provider'].split('/')[0],'usage':{'cost':.001}}
            if m=='luna':raw['choices']=[{'finish_reason':'stop','message':{'content':'{"match":true}'}}]
            else:raw['answers']={'match':{'type':'noul','noul':.8}}
            return Response(json.dumps(raw).encode())
        # The transition can encounter a completed baseline and interrupted dev.
        progress=f.target/'artifacts/progress.json'
        status=json.loads(progress.read_text());status['AP1_baseline']['status']='complete';status['AP2_development']={'status':'needs_attention'};progress.write_text(json.dumps(status))
        with patch('urllib.request.urlopen',side_effect=http),patch.object(comp.subprocess,'run'):
            comp.run_phases(f.root,f.target,2,f.fixture.inference,f.fixture.orchestration,protocol,f.target/'artifacts/fast')
        self.assertEqual(len(calls),440)
        self.assertEqual(credit.attempts,328)
        verify_accounting(credit,f.target,{})
        credit.checkpoint(close=True)
        budget=json.loads((f.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'],120.440)
        self.assertAlmostEqual(budget['reserved'],3)
        self.assertEqual(budget['attempts'],440)
        self.assertEqual(len(list((f.target/'runs/timing_serial').rglob('prompt.jsonl'))),56)

    def test_changed_raw_cost_or_lost_result_never_releases_credit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'artifacts').mkdir();(root/'runs').mkdir()
            p=root/'artifacts/budget.json';p.write_text(json.dumps({'spent':0,'reserved':3,'attempts':0,'unknown_cost_attempts':0}))
            c=Credit(root,'fast-core2-test',{});c.account('reserve');c.account('settle',.001)
            with self.assertRaises(ValueError):verify_accounting(c,root,{})
            self.assertAlmostEqual(json.loads(p.read_text())['reserved'],11)
            self.assertFalse(c.closed)


if __name__=='__main__':unittest.main()
