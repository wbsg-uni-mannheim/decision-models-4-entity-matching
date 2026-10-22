import copy
import json
import socket
import unittest
from unittest.mock import patch

import protocol
import fresh_rules_cost_plan as plan
import test_fresh_rules_drained as fixtures
import test_fixed_extension_cost_plan as source_fixtures


class FreshRulesCosts(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.FreshRulesTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.sources = [dict(namespace='quality', model=m, variant=v, benchmark=b, split='prompt',
                            responses=2, known_proxy_usd=.02 if v=='B0' else .04,
                            historical_unknown_scenario_usd=.05 if v=='B0' else 0)
                        for m in protocol.MODELS for v in ('B0', 'X_P_domain_0') for b in protocol.BENCHMARKS]

    def inventory(self):
        return plan.inventory(self.f.root,self.f.target,self.f.rid,self.f.rules,protocol)

    def test_full_matrix_exact_own_manual_cache_and_billiger50_transfer(self):
        before={p:p.read_bytes() for p in self.f.root.rglob('*') if p.is_file()}
        with patch.object(socket,'socket',side_effect=AssertionError('No HTTP')):
            inv=self.inventory()
            result=plan.price(self.sources,inv['cells'],list(protocol.MODELS),list(protocol.BENCHMARKS))
        self.assertEqual(len(result['cells']),238)
        self.assertEqual(result['totals']['expected'],238)
        self.assertEqual(result['totals']['responses'],2)
        self.assertEqual(result['totals']['missing'],236)
        self.assertEqual(result['by_variant']['R_rules']['missing'],119)
        self.assertEqual(result['by_variant']['X_P_domain_0']['missing'],117)
        self.assertAlmostEqual(result['totals']['known_proxy_usd'],119*.01+117*.02)
        self.assertAlmostEqual(result['totals']['historical_unknown_scenario_usd'],119*.025)
        self.assertTrue(all(c['source_key'][3]=='billiger-de' and c['billiger50_transfer'] for c in result['cells'] if c['benchmark']=='billiger-de-50un'))
        self.assertEqual(before,{p:p.read_bytes() for p in self.f.root.rglob('*') if p.is_file()})
        self.assertFalse(inv['b0_controls_replayed']);self.assertFalse(inv['admission'])

    def test_no_hypothetical_reuse_and_independent_run3(self):
        for p in (self.f.target/'runs/quality').rglob('*.jsonl'):
            if 'X_P_domain_0' in p.parts:p.unlink()
        # A completion marker alone never earns a cache discount.
        inv=self.inventory();self.assertEqual(sum(c['missing'] for c in inv['cells']),238)
        target=self.f.root/'replications/run03';self.f.target.rename(target)
        self.f.target=target;self.f.rules.PROJECT=target;self.f.rid=3
        inv=self.inventory();self.assertEqual(sum(c['missing'] for c in inv['cells']),238)
        with self.assertRaises(ValueError):plan.inventory(self.f.root,target,2,self.f.rules,protocol)

    def test_bad_matrix_costs_and_cache_quantities(self):
        inv=self.inventory();cells=inv['cells']
        for sources in (self.sources[:-1],self.sources+[self.sources[0]]):
            with self.assertRaises(ValueError):plan.price(sources,cells,list(protocol.MODELS),list(protocol.BENCHMARKS))
        for field,value in [('known_proxy_usd',float('nan')),('responses',0),('historical_unknown_scenario_usd',-1)]:
            sources=copy.deepcopy(self.sources);sources[0][field]=value
            with self.assertRaises(ValueError):plan.price(sources,cells,list(protocol.MODELS),list(protocol.BENCHMARKS))
        for change in ({'missing':2},{'responses':True},{'expected':1,'responses':1,'missing':0}):
            bad=copy.deepcopy(cells);bad[0].update(change)
            with self.assertRaises(ValueError):plan.price(self.sources,bad,list(protocol.MODELS),list(protocol.BENCHMARKS))

    def test_foreign_or_corrupt_manual_cache_unknown_operations_and_old_rule_attempt(self):
        p=self.f.target/'runs/quality/jev/X_P_domain_0/abt-buy/prompt.jsonl';old=p.read_bytes();row=json.loads(old)
        for field,value in [('study_run_id',1),('request_hash','bad'),('input_hash','bad'),('prediction',0)]:
            p.write_text(json.dumps({**row,field:value})+'\n')
            with self.assertRaises(ValueError):self.inventory()
        p.write_bytes(old)
        event={k:v for k,v in row.items() if k not in ('attempts','prediction','probability_match')}
        event.update(error='Operational future exception: IncompleteRead',operational_stop=dict(previous_attempts_preserved=0,
            budget_reservation_rejected=False,outcome_unknown=True,phase='future',exception_type='IncompleteRead'))
        op=p.with_name('prompt.operations.jsonl');op.write_text(json.dumps(event)+'\n')
        with self.assertRaisesRegex(ValueError,'Unknown'):self.inventory()
        op.unlink()
        oldrule=self.f.target/'runs/rules/jev/R_rules/abt-buy/prompt.jsonl';oldrule.parent.mkdir(parents=True);oldrule.touch()
        with self.assertRaisesRegex(ValueError,'absent rule'):self.inventory()

    def test_manual_source_real_frozen_parser_retained_retry_failure_costs(self):
        import tempfile
        from pathlib import Path
        import expanded_protocol
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c,row,p=source_fixtures.FixedExtensionCosts().fixture(root,variant='X_P_domain_0')
            row['attempts'].insert(0,dict(status=503,cost=None,seconds=1,error='unavailable',response={}))
            p.write_text(json.dumps(row)+'\n')
            failed={**row,'error':'failed','attempts':[dict(status=503,cost=.01,seconds=1,error='unavailable',response={})]}
            p.with_name('expanded_screen.failures.jsonl').write_text(json.dumps(failed)+'\n')
            checked=plan.source(root,c,expanded_protocol,protocol)
            self.assertEqual(checked['attempts'],3)
            self.assertAlmostEqual(checked['known_proxy_usd'],.03)
            self.assertAlmostEqual(checked['historical_unknown_scenario_usd'],.05)
            row['study_run_id']=3;p.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):plan.source(root,c,expanded_protocol,protocol)


if __name__=='__main__':unittest.main()
