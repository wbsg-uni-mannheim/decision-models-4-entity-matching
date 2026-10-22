import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from analyze import calibration, components, metric, routing_mask, bootstrap_delta

class AnalysisTest(unittest.TestCase):
    def test_positive_class_metric(self):
        result=metric([1,1,0,0],[1,0,1,0])
        self.assertEqual(result['f1'],.5)
        self.assertEqual((result['tp'],result['fp'],result['fn'],result['tn']),(1,1,1,1))

    def test_calibration_and_log_loss_finite_at_boundaries(self):
        r=calibration(np.array([0,1]),np.array([0.,1.]))
        self.assertEqual(r['brier'],0.)
        self.assertEqual(r['ece_10_equal_width'],0.)
        self.assertTrue(np.isfinite(r['log_loss']))
        self.assertEqual(sum(b['n'] for b in r['bins']),2)

    def test_routing_uses_development_only_with_ties(self):
        d={'prob':np.array([.5,.5,.1,.9]), 'cases':[{'pair_id':str(i)} for i in range(4)]}
        a,t=routing_mask(d,d,.5)
        self.assertEqual(int(a.sum()),2)
        self.assertEqual(t['dev_routed'],2)
        changed={**d,'y':np.array([1,0,0,1])}
        b,_=routing_mask(d,changed,.5)
        self.assertTrue(np.array_equal(a,b))

    def test_cluster_graph_captures_shared_entities(self):
        c=[{'entity_ids':{'left':'a','right':'x'}}, {'entity_ids':{'left':'a','right':'y'}},
           {'entity_ids':{'left':'b','right':'z'}}]
        groups=components(c)
        self.assertEqual(groups[0],groups[1])
        self.assertNotEqual(groups[0],groups[2])
        ci=bootstrap_delta(c,np.array([1,0,1]),np.array([1,0,1]),np.array([0,0,1]),repeats=10)
        self.assertIsNone(ci['entity_component_bootstrap']['ci95'])

if __name__=='__main__':
    unittest.main()
