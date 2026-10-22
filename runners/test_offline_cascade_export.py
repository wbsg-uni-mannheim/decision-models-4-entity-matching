import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import analyze
import expanded_cascades as module
import protocol
from offline_cascade_export import reconstruct, evaluate, normalized, compare, export_cascades


def data(prob=(.1, .4, .6, .9), pred=None):
    return dict(cases=[{'pair_id': str(i), 'records': {'left': {'title': 'x'}, 'right': {'title': 'y'}}} for i in range(4)],
                y=np.array([0, 1, 0, 1]), prob=np.array(prob), pred=np.array(pred if pred is not None else np.array(prob)>=.5),
                costs=np.array([.01]*4), unknown_cost_by_pair=np.array([0, 1, 0, 0]), responses=[])


def small_module():
    return SimpleNamespace(**{**vars(module), 'BENCHMARKS': ['billiger-de'], 'MODELS': ['jev', 'luna'],
                              'DECISIONS': ['jev'], 'PEERS': {'jev': 'jev'}, 'STRATEGIES': ['symmetric'],
                              'FRACTIONS': [.25], 'verify': Mock()})


def selection(d):
    return [dict(model='jev', benchmark='billiger-de', strategy='symmetric', status='frozen', peer=None,
                 fraction=.25, cutoff=module.cutoff(d['cases'], np.minimum(d['prob'], 1-d['prob']), 1),
                 actual_dev_routes=1, dev_n=4)]


class CascadeExportTests(unittest.TestCase):
    def test_selection_ties_and_alignment_drift(self):
        d = data(); saved=selection(d); mod=small_module()
        dev={(m,'billiger-de',s): copy.deepcopy(d) for m in mod.MODELS for s in ['prompt','route']}
        reconstruct(mod,dev,saved)
        saved[0]['cutoff']['tie']='tampered'
        with self.assertRaisesRegex(ValueError,'differs'):
            reconstruct(mod,dev,saved)
        dev['luna','billiger-de','prompt']['cases'].reverse()
        with self.assertRaisesRegex(ValueError,'alignment'):
            reconstruct(mod,dev,selection(d))

    def test_peer_costs_random_controls_and_retrospective_oracle(self):
        d=data(); fallback=data(pred=[0,1,0,1]); peer=data(prob=(.9,.6,.4,.1))
        entry=dict(model='jev',peer='clef',strategy='disagreement',cutoff=None)
        scores=module.score(entry,d['cases'],d['prob'],peer['prob'])
        entry['cutoff']=module.cutoff(d['cases'],scores,2)
        result,oracle=evaluate(entry,{'jev':d,'clef':peer,'luna':fallback},module,analyze)
        self.assertAlmostEqual(result['known_cost_usd'], .1)  # both first stages plus two fallback calls
        self.assertEqual(len(result['random_same_route_count']),20)
        self.assertTrue(all(abs(r['known_cost_usd']-.1)<1e-12 for r in result['random_same_route_count']))
        self.assertEqual(oracle['f1'],1)
        self.assertLessEqual(oracle['oracle_routes'],2)

    def test_unavailable_not_zero_and_seed_not_run(self):
        entry={**selection(data())[0],'reason':'unidentifiable gain'}
        rows=normalized(3,entry,'billiger-de',None,None)
        self.assertEqual(len(rows),71)
        self.assertTrue(all(r['value'] is None and r['run_id']==3 for r in rows))
        self.assertEqual(sum('/random/' in r['condition'] for r in rows),60)

    def test_full_selection_must_pass_before_test_read(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs').mkdir();(root/'artifacts').mkdir()
            d=data();saved=selection(d);saved[0]['actual_dev_routes']=2
            (root/'configs/expanded_cascade_selection.json').write_text(json.dumps(saved))
            (root/'artifacts/expanded_cascades_offline_complete.json').write_text(json.dumps(dict(status='complete_with_explicit_unavailable_routers',selection_sha256=protocol.digest(saved))))
            calls=[]
            def cell(m,v,b,s):
                calls.append(s);return copy.deepcopy(d),0
            with patch('socket.socket',side_effect=AssertionError('No network')):
                with self.assertRaisesRegex(ValueError,'differs'):
                    export_cascades(root,small_module(),analyze,protocol,cell,Mock(),1)
            self.assertEqual(calls,['prompt','route','prompt','route'])

    def test_fit_comparison_tolerance_is_bounded(self):
        compare({'coef':[.5]}, {'coef':[.50000001]}, tolerance=1e-6)
        with self.assertRaisesRegex(ValueError,'drift'):
            compare({'coef':[.5]}, {'coef':[.51]}, tolerance=1e-6)
        with self.assertRaisesRegex(ValueError,'drift'):
            compare(.5,float('nan'))


if __name__ == '__main__':
    unittest.main()
