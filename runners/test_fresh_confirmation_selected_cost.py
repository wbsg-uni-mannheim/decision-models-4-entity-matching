import copy
import json
import math
import unittest
import tempfile
import types
from pathlib import Path

import expanded_protocol
import protocol
from test_fixed_extension_cost_plan import FixedExtensionCosts
from unittest.mock import patch

import fresh_confirmation_selected_cost as costs
import fresh_confirmation_selection as freeze
import fresh_confirmation_test_drained as tests
from test_fresh_confirmation_test_drained import SelectedTestTests


class ArithmeticTests(unittest.TestCase):
    def test_exact_missing_reuse_transfer_and_separate_unknown_costs(self):
        sources = [{'model': 'm', 'variant': v, 'benchmark': 'billiger-de',
                    'responses': 200, 'complete': True, 'missing': 0, 'known_usd': 2.,
                    'unknown_attempts': 4} for v in ('B0', 'S1')]
        cells = [{'model': 'm', 'variant': v, 'benchmark': b, 'expected': 10,
                  'responses': cached, 'missing': 10-cached, 'missing_pair_ids_sha256': 'digest'}
                 for v,b,cached in [('B0','billiger-de',10), ('S1','billiger-de',3),
                                   ('S1','billiger-de-50un',0)]]
        result = costs.calculate(sources,cells)
        self.assertEqual(result['missing_test_responses'],17)
        self.assertAlmostEqual(result['known_proxy_usd'],.17)
        self.assertAlmostEqual(result['historical_unknown_scenario_usd'],.017)
        self.assertEqual(result['cells'][0]['known_proxy_usd'],0)
        self.assertEqual(result['cells'][2]['source_benchmark'],'billiger-de')
        for damaged in [sources+[sources[0]], [{**sources[0],'complete':False}],
                        [{**sources[0],'known_usd':math.nan}], [{**sources[0],'unknown_attempts':True}],
                        [{**sources[0],'responses':0}]]:
            with self.assertRaises(ValueError): costs.calculate(damaged,cells)
        with self.assertRaises(ValueError): costs.calculate(sources,cells+cells[:1])
        with self.assertRaises(ValueError): costs.calculate(sources,[{**cells[0],'missing':1}])
        with self.assertRaises(ValueError): costs.calculate(sources,[])

    def test_own_run_raw_retry_failure_operations_and_unknown_costs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            _,row,path=FixedExtensionCosts().fixture(root, 'X_P_domain_0')
            data=root/'data/abt-buy/expanded_screen.json'
            data.rename(data.with_name('prompt.json'))
            path.unlink();path=path.with_name('prompt.jsonl')
            row['study_run_id']=2
            retry={'status':503,'cost':None,'seconds':1,'error':'unavailable','response':{}}
            row['attempts'].insert(0,retry)
            path.write_text(json.dumps(row)+'\n')
            failed={**row,'error':'failed','attempts':[{**retry,'cost':.01}]}
            path.with_name('prompt.failures.jsonl').write_text(json.dumps(failed)+'\n')
            event={k:v for k,v in row.items() if k not in ('attempts','prediction','probability_match')}
            event.update(completed_at='synthetic',error='Project spending ceiling reached',
                operational_stop={'phase':'reserve','exception_type':'RuntimeError',
                    'previous_attempts_preserved':0,'budget_reservation_rejected':True,'outcome_unknown':False})
            op=path.with_name('prompt.operations.jsonl');op.write_text(json.dumps(event)+'\n')
            confirmation=types.SimpleNamespace(MODELS=['jev'],BENCHMARKS=['abt-buy'],
                PROMPT_VARIANTS=['X_P_domain_0'],SERIALIZATIONS=[],payload=expanded_protocol.payload)
            dev=costs.development.inventory(root,2,confirmation,protocol)
            self.assertEqual(dev[0]['attempts'],3)
            self.assertEqual(dev[0]['operations']['budget_rejections'],1)
            cells=[{'model':'jev','variant':'X_P_domain_0','benchmark':'abt-buy',
                    'expected':10,'responses':0,'missing':10,'missing_pair_ids_sha256':'synthetic'}]
            plan=costs.calculate(dev,cells)
            self.assertAlmostEqual(plan['known_proxy_usd'],.3)
            self.assertAlmostEqual(plan['historical_unknown_scenario_usd'],.5)
            event['operational_stop'].update(phase='future',budget_reservation_rejected=False,outcome_unknown=True)
            event['error']='worker outcome unknown';op.write_text(json.dumps(event)+'\n')
            with self.assertRaisesRegex(ValueError,'Unknown development outcomes'):
                costs.development.inventory(root,2,confirmation,protocol)

    def test_budget_snapshot_foreign_holds_no_saved_boolean_authorization(self):
        plan={'known_proxy_usd':.17,'historical_unknown_scenario_usd':.017}
        budget={'spent':196.9,'reserved':3,'unknown_cost_attempts':3}
        out=costs.budget_review(plan,budget)
        self.assertFalse(out['scenario_fits_current_headroom'])
        self.assertEqual(budget['reserved'],3)
        for value in (math.nan,math.inf,-1,True,None):
            with self.assertRaises(ValueError): costs.budget_review(plan,{**budget,'spent':value})
        self.assertTrue(costs.budget_review(plan,{'spent':100,'reserved':3})['scenario_fits_current_headroom'])


class RawGateTests(unittest.TestCase):
    http = SelectedTestTests.http
    execute = SelectedTestTests.execute
    args = SelectedTestTests.args
    setUp = SelectedTestTests.setUp

    def test_full_raw_gate_independent_cost_and_missing_reconstruction(self):
        # The fixture has all 1176 own-run development cells and a real freeze,
        # plus 63 complete B0 test controls; only replica/data manifests mocked.
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            result=costs.review(*self.args())
        self.assertEqual(result['development_cells_raw_verified'],1176)
        self.assertEqual(result['unique_test_cells'],126)
        self.assertEqual(result['b0_test_control_cells'],63)
        self.assertEqual(result['missing_test_responses'],126)
        self.assertEqual(result['valid_cached_test_responses'],126)
        # Independently rebuild all selected cells directly from data and raw
        # attempts, without calculate(), inventories or the selector's metrics.
        cells=json.loads((self.target/'configs/expanded_confirmation_cells.json').read_text())
        known=[]; unknown=[]; missing_total=0
        for m,v,b in cells:
            source_b='billiger-de' if b=='billiger-de-50un' else b
            directory=self.target/'runs/quality'/m/v/source_b
            source=[json.loads(s) for s in (directory/'prompt.jsonl').read_text().splitlines()]
            failure=directory/'prompt.failures.jsonl'
            failures=[json.loads(s) for s in failure.read_text().splitlines()] if failure.exists() else []
            attempts=[a for row in source+failures for a in row['attempts']]
            target_cases=json.loads((self.target/'data'/b/'test.json').read_text())
            path=self.target/'runs/quality'/m/v/b/'test.jsonl'
            valid=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
            missing={c['pair_id'] for c in target_cases}-{r['pair_id'] for r in valid}
            missing_total+=len(missing)
            known.append(math.fsum(a['cost'] for a in attempts if a['cost'] is not None)*len(missing)/len(source))
            unknown.append(sum(a['cost'] is None for a in attempts)*.05*len(missing)/len(source))
        self.assertEqual(missing_total,result['missing_test_responses'])
        self.assertAlmostEqual(math.fsum(known),result['known_proxy_usd'])
        self.assertAlmostEqual(math.fsum(unknown),result['historical_unknown_scenario_usd'])
        self.assertAlmostEqual(result['known_proxy_usd'],.126)
        self.assertFalse(result['admission_granted']);self.assertFalse(result['reservation_created'])
        self.assertFalse((self.target/tests.ATTEMPT).exists())
        self.assertEqual(before,{p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        costs.require_headroom(self.root,result)

    def test_changed_sources_controls_attempts_and_actual_budget_stop(self):
        path=self.target/freeze.RECEIPT;original=path.read_bytes();path.unlink()
        with self.assertRaisesRegex(ValueError,'No committed'): costs.review(*self.args())
        path.write_bytes(original)
        path=self.target/'runs/quality/jev/B0/abt-buy/test.jsonl';original=path.read_bytes();path.write_text('')
        with self.assertRaisesRegex(ValueError,'B0 test controls'): costs.review(*self.args())
        rows=[json.loads(s) for s in original.splitlines()];rows[0]['study_run_id']=3
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'study run'): costs.review(*self.args())
        path.write_bytes(original)
        extra=path.with_name('test.foreign.jsonl');extra.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'Unreviewed'): costs.review(*self.args())
        extra.unlink()
        p=self.root/'artifacts/budget.json';d=json.loads(p.read_text());d['spent']=196.99;p.write_text(json.dumps(d))
        before=p.read_bytes()
        result=costs.review(*self.args())
        result['scenario_fits_current_headroom']=True
        result['budget_snapshot']['spent']=0  # stale or forged saved snapshot must not pass
        with self.assertRaisesRegex(ValueError,'headroom'): costs.require_headroom(self.root,result)
        self.assertEqual(p.read_bytes(),before)
        self.assertFalse((self.target/tests.ATTEMPT).exists())
        (self.target/tests.ATTEMPT).mkdir()
        with self.assertRaisesRegex(ValueError,'Existing selected-test attempt'): costs.review(*self.args())
        (self.target/tests.ATTEMPT).rmdir()
        original_snapshot=costs.test_snapshot;calls=[]
        def drift(*args):
            value=original_snapshot(*args);calls.append(1)
            if len(calls)>1: value=copy.deepcopy(value);value['changed']='changed'
            return value
        with patch.object(costs,'test_snapshot',side_effect=drift):
            with self.assertRaisesRegex(ValueError,'sources changed'): costs.review(*self.args())
        path=self.target/'runs/quality/jev/X_P_domain_2/abt-buy/prompt.jsonl'
        path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'provenance changed'): costs.review(*self.args())


if __name__=='__main__': unittest.main()
