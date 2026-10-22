import copy
import itertools
import unittest

from fresh_confirmation_cost_plan import scenarios


class FreshConfirmationCosts(unittest.TestCase):
    def fixture(self):
        models=['a','b'];benchmarks=['billiger-de','other'];prompts=['p'+str(i) for i in range(6)]
        serial=['B0','S1','X_R_aligned'];sources=[]
        for m,v,b in itertools.product(models,prompts+serial,benchmarks):
            rank=(prompts+serial).index(v)+1
            sources.append(dict(model=m,variant=v,benchmark=b,responses=2,
                                known_proxy_usd=rank,historical_unknown_scenario_usd=10-rank))
        return [sources,models,benchmarks,prompts,serial,['a'],{'billiger-de':10,'other':20,'billiger-de-50un':5}]

    def test_development_reuse_and_transfer(self):
        args=self.fixture();r=scenarios(*args);d=r['development_alternatives_not_additive']
        self.assertEqual([d[k]['cells'] for k in d],[36,32,30])
        self.assertEqual([d[k]['prospective_responses'] for k in d],[72,64,60])
        self.assertEqual(r['new_test_response_scenario_max'],2*35*6)
        candidates=r['test_candidates']
        self.assertEqual(len(candidates),36)
        for c in candidates:
            n=15 if c['source_benchmark']=='billiger-de' else 20
            self.assertEqual(c['target_responses'],n)
            if c['variant']=='B0':
                self.assertEqual(c['known_proxy_usd'],0)
                self.assertEqual(c['historical_unknown_scenario_usd'],0)
            if c['variant']=='S1':
                self.assertEqual(c['known_proxy_usd'],8*n/2)  # no hypothetical S1 test credit
        self.assertFalse(any(c.get('benchmark')=='billiger-de-50un' for c in candidates))

    def test_envelope_exhaustive_combinations(self):
        r=scenarios(*self.fixture())
        for e in r['test_envelopes']:
            candidates=[c for c in r['test_candidates'] if all(c[k]==e[k] for k in ('model','source_benchmark','family'))]
            for field in ('known_proxy_usd','historical_unknown_scenario_usd'):
                sums=[sum(c[field] for c in combo) for n in range(e['maximum_distinct_choices']+1)
                      for combo in itertools.combinations(candidates,n)]
                self.assertEqual(e[field],max(sums))
        prompts=[e for e in r['test_envelopes'] if e['family']=='prompt']
        self.assertNotEqual(prompts[0]['known_proxy_usd_variants'],prompts[0]['historical_unknown_scenario_usd_variants'])
        for f in ('known_proxy_usd','historical_unknown_scenario_usd'):
            self.assertEqual(r['conditional_two_runs_total'][f],2*(r['development_alternatives_not_additive']['conditional_successful_core'][f]+sum(e[f] for e in r['test_envelopes'])))

    def test_reject_incomplete_duplicate_bad_cost_and_sizes(self):
        args=self.fixture()
        for change in ('missing','duplicate','nan','negative','empty','testmissing','testnegative'):
            a=copy.deepcopy(args)
            if change=='missing':a[0].pop()
            if change=='duplicate':a[0].append(a[0][0])
            if change=='nan':a[0][0]['known_proxy_usd']=float('nan')
            if change=='negative':a[0][0]['historical_unknown_scenario_usd']=-1
            if change=='empty':a[0][0]['responses']=0
            if change=='testmissing':a[-1].pop('other')
            if change=='testnegative':a[-1]['other']=-1
            with self.subTest(change=change),self.assertRaises(ValueError):scenarios(*a)


if __name__=='__main__':unittest.main()
