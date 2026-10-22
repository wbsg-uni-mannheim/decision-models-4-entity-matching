import concurrent.futures
import importlib.util
import io
import json
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

import protocol
from authorized_recovery_200 import APPROVAL, APPROVAL_SHA, PINNED, SELF, make_wrapper, sha

ROOT = Path(__file__).resolve().parents[2]


class AuthorizedBudget(unittest.TestCase):
    def setup_root(self, root, spent, reserved):
        for rel in [APPROVAL, SELF, 'src/run.py', *[f'artifacts/operations/{n}.py' for n in PINNED]]:
            p = root/rel; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/rel, p)
        (root/'artifacts/budget.json').write_text(json.dumps(
            {'spent': spent, 'reserved': reserved, 'attempts': 0, 'unknown_cost_attempts': 0}))
        spec = importlib.util.spec_from_file_location('isolated_inference', root/'src/run.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_original_modules_and_frozen_sources_unchanged_all_four_gates(self):
        before = {n: sha(ROOT/f'artifacts/operations/{n}.py') for n in PINNED}
        for stage in ('variants', 'confirmation', 'schema', 'core'):
            w = make_wrapper(ROOT, stage)
            self.assertIn(SELF, w.REQUIRED)
            if stage != 'core':
                spec = dict(admitted=True, run_id=1, stage=stage, slurm_job_id='42', budget_cap_usd=200,
                    max_invocations=1, historical_unknown_outcomes_disclosed=True,
                    unobserved_worker_causes_disclosed=True, unobserved_worker_causes=['kev', 'luna', 'pplx'],
                    predecessor={'variants':'365279','confirmation':'365636','schema':'365969'}[stage],
                    source_sha256=dict.fromkeys(w.REQUIRED,'synthetic'))
                w.validate(spec, '42', {'spent':100,'reserved':3})
                with self.assertRaises(ValueError): w.validate({**spec,'budget_cap_usd':201},'42',{'spent':100,'reserved':0})
                with self.assertRaises(ValueError): w.validate(spec,'42',{'spent':199.99,'reserved':0})
        self.assertEqual(before, {n: sha(ROOT/f'artifacts/operations/{n}.py') for n in PINNED})

    def test_shared_atomic_ceiling_foreign_hold_and_no_frozen_default_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inference=self.setup_root(root,196.91,3)
            w=make_wrapper(root,'variants')
            _, cell, provenance=w.authorized_build(inference,1)
            account=cell.__globals__['account']
            def reserve(_):
                try: account('reserve'); return True
                except RuntimeError: return False
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                self.assertEqual(sum(pool.map(reserve,range(8))),1)
            account('settle',.01)
            d=json.loads((root/'artifacts/budget.json').read_text())
            self.assertAlmostEqual(d['reserved'],3)
            self.assertEqual(d['attempts'],1)
            self.assertAlmostEqual(d['spent'],196.92)
            self.assertEqual(inference.account.__defaults__,(0,100))
            self.assertEqual(provenance['budget_authority_sha256'],APPROVAL_SHA)

    def test_real_request_parser_account_crosses_old_cap_then_blocks_new_cap(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(socket,'socket',side_effect=AssertionError('No sockets')):
            root=Path(tmp); inference=self.setup_root(root,100,3)
            request, _, _=make_wrapper(root,'confirmation').authorized_build(inference,1)
            case={'pair_id':'synthetic','benchmark':'abt-buy','records':{'left':{'title':'A'},'right':{'title':'A'}}}
            case['input_hash']=protocol.digest(case['records'])
            raw={'provider':'typesafe','model':protocol.MODELS['jev']['id'],
                 'answers':{'match':{'type':'noul','noul':.8}},'usage':{'cost':.01}}
            response=io.BytesIO(json.dumps(raw).encode());response.status=200
            with patch.object(inference.urllib.request,'urlopen',return_value=response) as http:
                r=request(case,'jev','B0','synthetic-not-a-key')
                self.assertEqual(r['prediction'],1);self.assertEqual(http.call_count,1)
            self.assertEqual(json.loads((root/'artifacts/budget.json').read_text())['spent'],100.01)
            (root/'artifacts/budget.json').write_text(json.dumps({'spent':196.99,'reserved':3,'attempts':1,'unknown_cost_attempts':0}))
            with patch.object(inference.urllib.request,'urlopen',side_effect=AssertionError('No HTTP at cap')):
                r=request(case,'jev','B0','synthetic-not-a-key')
                self.assertTrue(r['operational_stop']['budget_reservation_rejected'])
                self.assertEqual(r['attempts'],[])

    def test_authority_drift_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.setup_root(root,0,0)
            (root/APPROVAL).write_text('{}')
            with self.assertRaises(ValueError):make_wrapper(root,'schema')


if __name__ == '__main__':unittest.main()
