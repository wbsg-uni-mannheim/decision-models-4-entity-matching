import copy
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import expanded_schema as es
import protocol
import fresh_schema_cost_plan as planner
from test_schema_drained import raw_response


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.models = ['a', 'b']; self.families = ['abt-buy', 'billiger-de', 'dblp-scholar']
        self.pilot = ['abt-buy', 'dblp-scholar']; self.counts = dict(zip(self.families+[ 'billiger-de-50un'], [7, 11, 13, 17]))
        self.sources = []
        for ns, bb, n in [('schema_pilot', self.pilot, 2), ('schema', self.families, 200)]:
            for mi, m in enumerate(self.models):
                for vi, v in enumerate(es.CONDITIONS):
                    for bi, b in enumerate(bb):
                        self.sources.append(dict(namespace=ns, model=m, variant=v, benchmark=b,
                            responses=n, known_proxy_usd=(vi+mi+bi+1)**2/100,
                            historical_unknown_scenario_usd=(4-vi+bi)%3*.05))

    def call(self, sources=None):
        return planner.scenarios(self.sources if sources is None else sources, self.models,
                                 self.families, es.BINARY, self.pilot, self.counts)

    def test_all_global_local_combinations_transfer_review_and_totals(self):
        result = self.call()
        for field in planner.FIELDS:
            prices = {(s['model'], s['variant'], s['benchmark']):s[field]/200
                      for s in self.sources if s['namespace']=='schema'}
            maxima = []
            for m in self.models:
                brute = []
                for global_ in es.BINARY:
                    for local in itertools.product(es.BINARY, repeat=len(self.families)):
                        cost = 0.
                        for b, l in zip(self.families, local):
                            n = self.counts[b]+(self.counts['billiger-de-50un'] if b=='billiger-de' else 0)
                            cost += sum(prices[m,v,b]*n for v in {global_,l,'C_review'})
                        brute.append(cost)
                maxima.append(max(brute))
            self.assertAlmostEqual(result['test_scenario_maximum'][field], math.fsum(maxima))
            development = math.fsum(s[field] for s in self.sources)
            self.assertAlmostEqual(result['two_runs_total'][field], 2*(development+sum(maxima)))
        self.assertEqual(result['maximum_test_responses'], 3*2*sum(self.counts.values()))
        self.assertEqual(result['maximum_distinct_test_cells'], 24)
        self.assertEqual(len(result['test_candidates']), 30)
        for x in result['test_candidates']:
            if x['source_benchmark']=='billiger-de':
                self.assertEqual(x['target_responses'],28)
                self.assertEqual(x['target_benchmarks'],['billiger-de','billiger-de-50un'])

    def test_incomplete_duplicate_invalid_cost_counts_rejected(self):
        for bad in [self.sources[:-1],self.sources+[self.sources[0]]]:
            with self.assertRaises(ValueError):self.call(bad)
        for k,v in [('responses',True),('responses',0),('known_proxy_usd',float('nan')),
                    ('historical_unknown_scenario_usd',-.1),('responses',3)]:
            bad=copy.deepcopy(self.sources);bad[0][k]=v
            with self.subTest(k=k,v=v),self.assertRaises(ValueError):self.call(bad)
        self.counts['billiger-de-50un']=0
        with self.assertRaises(ValueError):self.call()

    def test_one_shared_global_is_stricter_than_unrelated_top_two_per_family(self):
        a,b,c,d=es.BINARY
        expensive=dict(zip(self.families,[{a,b},{c,d},{a,c}]))
        for s in self.sources:
            if s['namespace']=='schema':
                family=s['benchmark'];n=self.counts[family]+(17 if family=='billiger-de' else 0)
                price=0 if s['variant']=='C_review' else (100 if s['variant'] in expensive[family] else 1)
                s['known_proxy_usd']=price*200/n
        result=self.call()
        self.assertAlmostEqual(result['test_scenario_maximum']['known_proxy_usd'],2*501)
        self.assertLess(result['test_scenario_maximum']['known_proxy_usd'],2*600)


class RawSourceTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.case=dict(benchmark='abt-buy',pair_id='p',label=1,
                       records={'left':{'z':'Ä','a':'B'},'right':{'z':'A','a':'B'}})
        self.case['input_hash']=protocol.digest(self.case['records'])
        self.input=self.root/'data/abt-buy/prompt.json';self.input.parent.mkdir(parents=True)
        self.input.write_text(json.dumps([self.case]))
        self.base=self.root/'runs/schema/jev/C_match_reverse/abt-buy/prompt'
        self.base.parent.mkdir(parents=True)
        raw=raw_response('jev','C_match_reverse');pred,prob=es.parse('jev',raw,'C_match_reverse')
        self.row=dict(pair_id='p',benchmark='abt-buy',input_hash=self.case['input_hash'],model_alias='jev',
            variant='C_match_reverse',study_run_id=1,prediction=pred,probability_match=prob,completed_at='done',
            request_hash=es.wire_hash(es.payload(self.case,'jev','C_match_reverse')),
            attempts=[dict(status=503,cost=.02,seconds=0,error='503',response={}),
                      dict(status=200,cost=.001,seconds=0,error=None,response=raw)])
        self.write('.jsonl',[self.row])
        sock=patch('socket.socket',side_effect=AssertionError('No network'));sock.start();self.addCleanup(sock.stop)

    def write(self,suffix,rows):
        Path(str(self.base)+suffix).write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def call(self):
        return planner.source(self.root,es,'jev','C_match_reverse','abt-buy')

    def operation(self,unknown=False):
        return {k:v for k,v in dict(self.row,error='unknown' if unknown else 'Project spending ceiling reached',
            completed_at='reserve',operational_stop=dict(previous_attempts_preserved=0,
                budget_reservation_rejected=not unknown,outcome_unknown=unknown,phase='future' if unknown else 'reserve',
                exception_type='ValueError' if unknown else 'RuntimeError')).items()
            if k not in ('attempts','prediction','probability_match')}

    def test_real_wire_parser_retry_failure_cost_and_zero_attempt_operation(self):
        failed={**self.row,'completed_at':'failed','error':'520',
                'attempts':[dict(status=520,cost=None,seconds=0,error='520',response={})]}
        self.write('.failures.jsonl',[failed]);self.write('.operations.jsonl',[self.operation()])
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        s=self.call()
        self.assertEqual(s['attempts'],3);self.assertEqual(s['unknown_attempts'],1)
        self.assertAlmostEqual(s['known_proxy_usd'],.021)
        self.assertAlmostEqual(s['historical_unknown_scenario_usd'],.05)
        self.assertEqual(s['operations']['events'],1)
        self.assertEqual(s['operations']['additional_paid_attempts_counted'],0)
        self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))

    def test_foreign_run_wire_provider_parser_input_incomplete_and_unknown_operation(self):
        for field,value in [('study_run_id',2),('study_run_id',True),('request_hash',protocol.digest(es.payload(self.case,'jev','C_match_reverse'))),
                            ('input_hash','bad'),('variant','C_match')]:
            row=copy.deepcopy(self.row);row[field]=value;self.write('.jsonl',[row])
            with self.subTest(field=field),self.assertRaises(ValueError):self.call()
        for field,value in [('model','wrong'),('provider','wrong')]:
            row=copy.deepcopy(self.row);row['attempts'][-1]['response'][field]=value;self.write('.jsonl',[row])
            with self.assertRaises(ValueError):self.call()
        row=copy.deepcopy(self.row);row['attempts'][-1]['response']['answers']['match']['probabilities']={'Match':.8,'No Match':.19}
        self.write('.jsonl',[row])
        with self.assertRaises(ValueError):self.call()
        self.write('.jsonl',[])
        with self.assertRaises(ValueError):self.call()
        self.write('.jsonl',[self.row]);self.write('.operations.jsonl',[self.operation(True)])
        with self.assertRaisesRegex(ValueError,'unknown operational'):self.call()


if __name__=='__main__':unittest.main()
