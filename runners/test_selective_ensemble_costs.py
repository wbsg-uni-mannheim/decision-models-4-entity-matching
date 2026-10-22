import unittest
from selective_ensemble_costs import union_by_fraction

class Routes(unittest.TestCase):
    def test_overlap_reused_only_within_run_and_benchmark(self):
        rows=[{'fraction':.25,'benchmark':'a','pair_ids':['1','2']},
              {'fraction':.25,'benchmark':'a','pair_ids':['2','3']},
              {'fraction':.25,'benchmark':'b','pair_ids':['1']}]
        self.assertEqual(union_by_fraction(rows)['0.25'],{'pairs':4,'member_calls_per_run':12,'member_calls_three_run_projection':36})

    def test_quotas_not_summed(self):
        rows=[{'fraction':.1,'benchmark':'a','pair_ids':['1']},
              {'fraction':.5,'benchmark':'a','pair_ids':['1','2']}]
        result=union_by_fraction(rows)
        self.assertEqual(result['0.1']['pairs'],1)
        self.assertEqual(result['0.5']['pairs'],2)

if __name__=='__main__': unittest.main()

class PilotQuote(unittest.TestCase):
    def test_real_shape_complete_and_missing_rejected(self):
        import tempfile
        import json
        import hashlib
        from pathlib import Path
        from selective_ensemble_costs import pilot_costs, quote
        from strong_ensemble import MEMBERS
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); rows=[]; events=[]
            for benchmark in range(8):
                for pair in range(2):
                    for member in MEMBERS:
                        rid=f'{benchmark}-{pair}-{member}'
                        rows.append({'request_id':rid,'benchmark':str(benchmark),'pair_id':str(pair),'member':member})
                        events.append({'event':'valid_vote','request_id':rid,'cost_usd':.01})
            wire=''.join(json.dumps(r)+'\n' for r in rows)
            (root/'requests.jsonl').write_text(wire)
            (root/'manifest.json').write_text(json.dumps({'requests_sha256':hashlib.sha256(wire.encode()).hexdigest()}))
            events.append({'event':'complete_technical_pilot_only'})
            (root/'execution.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            self.assertAlmostEqual(pilot_costs(root)['0']['mean_usd'],.03)
            inv={'rows':[{'benchmark':'0','model':'a','fraction':.25,'pair_ids':['1','2']},
                         {'benchmark':'0','model':'b','fraction':.25,'pair_ids':['2','3']}],
                'sources':[{'known_cost_usd':1.,'unknown_cost_attempts':2}]}
            (root/'inventory.json').write_text(json.dumps(inv))
            quote(root/'inventory.json',root,root/'quote.json')
            out=json.loads((root/'quote.json').read_text())
            self.assertAlmostEqual(out['all_quotas_union']['ensemble_cost_three_runs']['mean_usd'],.27)
            events.pop(0)
            (root/'execution.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            with self.assertRaises(ValueError): pilot_costs(root)
