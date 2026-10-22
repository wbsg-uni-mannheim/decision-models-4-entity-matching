import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import expanded_statistics as s


def cases(n=60, shared=False):
    return [{'pair_id':str(i),'entity_ids':{'left':'shared' if shared else str(i),'right':str(i)}} for i in range(n)]


class StatisticsTests(unittest.TestCase):
    def test_holm_known_values_unavailable_keeps_family_size(self):
        out=s.holm([.01,.04,.03,None])
        np.testing.assert_allclose(out[:3],[.04,.09,.09])
        self.assertIsNone(out[3])
        self.assertEqual(s.holm([1.,0.]),[1.,0.])
        with self.assertRaises(AssertionError):s.holm([float('nan')])

    def test_f1_aggregation_and_cancelled_interaction(self):
        y=np.tile([0,1],30);pred=y.copy();pred[:10]=1-pred[:10]
        self.assertAlmostEqual(s.estimate(np.array([s.f1_parts(y,pred)]),[1],0.),s.analyze.metric(y,pred)['f1'])
        result=s.contrast(cases(),[s.f1_parts(y,pred)]*4,[1,-1,-1,1],repeats=99)
        self.assertEqual(result['estimate'],0.)
        self.assertEqual(result['entity_component']['ci95'],[0.,0.])
        self.assertEqual(result['entity_component']['p_approx'],1.)
        self.assertEqual(result,s.contrast(cases(),[s.f1_parts(y,pred)]*4,[1,-1,-1,1],repeats=99))

    def test_giant_component_does_not_use_pair_interval_as_primary(self):
        y=np.tile([0,1],30)
        d=s.contrast(cases(shared=True),[s.f1_parts(y,y),s.f1_parts(y,1-y)],[1,-1],repeats=99)
        self.assertIsNone(d['entity_component']['ci95'])
        self.assertIsNone(d['entity_component']['p_approx'])
        self.assertEqual(d['pair_sensitivity']['ci95'],[1.,1.])

    def test_abstention_zero_denominator_is_unavailable(self):
        x=s.mean_parts(np.ones(60),np.zeros(60,dtype=bool))
        d=s.contrast(cases(),[x],[1],np.nan,repeats=99,test_null=False)
        self.assertIsNone(d['estimate'])
        self.assertIsNone(d['entity_component']['ci95'])
        with self.assertRaises(AssertionError):s.f1_parts([1,0],[-1,0])

    def test_alignment_reorders_and_rejects_different_split(self):
        c=cases(4);a={'cases':c,'y':np.array([0,1,0,1]),'pred':np.array([0,1,1,1])}
        order=[3,2,1,0];b={'cases':[c[i] for i in order],'y':a['y'][order],'pred':a['pred'][order]}
        _,aligned=s.align(a,b);np.testing.assert_array_equal(a['pred'],aligned['pred'])
        b['y'][0]=1-b['y'][0]
        with self.assertRaisesRegex(AssertionError,'Labels'):s.align(a,b)
        b={**a,'cases':[{**c[0],'pair_id':'other'},*c[1:]]}
        with self.assertRaisesRegex(AssertionError,'different full splits'):s.align(a,b)

    def test_report_checkpoint_resume_and_holm(self):
        y=np.tile([0,1],30);data={'cases':cases(),'y':y,'pred':y}
        row=s.f1_row('baseline','jev','wdc','B0','luna_B0',data,{**data,'pred':1-y})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(s,'PROJECT',root),patch.object(s,'comparisons',lambda _:iter([row])):
                report=s.report('baseline',{'frozen':True})
                self.assertEqual(report['contrasts'],1)
                result=json.loads((root/'artifacts/results/expanded_statistics/baseline.json').read_text())
                self.assertEqual(result[0]['holm_family_size'],1)
                with patch.object(s,'contrast',side_effect=AssertionError('must reuse')):
                    self.assertEqual(s.report('baseline',{'frozen':True}),report)
                with self.assertRaises(AssertionError):s.report('baseline',{'frozen':False})

    def test_rules_and_schema_control_matrix(self):
        y=np.tile([0,1],30);d={'cases':cases(),'y':y,'pred':y}
        def load(*a,**kw):return d
        with patch.object(s,'MODELS',{'jev':{},'luna':{}}),patch.object(s,'BENCHMARKS',{'wdc':{}}), \
                patch.object(s,'load',load),patch.object(s.rules,'verify',lambda:None):
            rr=list(s.comparisons('rules'));self.assertEqual(len(rr),8)
            self.assertEqual({r[0]['control'] for r in rr},{'B0','X_P_domain_0'})
        with patch.object(s,'read',lambda _: [('jev','C_review','wdc')]),patch.object(s,'load',load), \
                patch.object(s.schema,'load',lambda *a:{**d,'pred':np.full(60,-1)}):
            rows=list(s.comparisons('schema'));self.assertEqual(len(rows),2)
            self.assertEqual({r[0]['metric'] for r in rows},{'accepted_error_difference','coverage'})
            result=s.contrast(*rows[0][1:5],repeats=99,test_null=rows[0][5])
            self.assertIsNone(result['estimate'])

    def test_unavailable_calibration_expands_planned_comparisons(self):
        entry={'model':'jev','benchmark':'semi-heter','regime':'within','labels_per_source':500,
               'status':'unavailable','method':'all','reason':'Only211pairs'}
        with patch.object(s,'read',lambda _: [entry]):
            rows=list(s.comparisons('calibration'))
            self.assertEqual(len(rows),9)
            self.assertTrue(all(r[1] is None for r in rows))
            with tempfile.TemporaryDirectory() as directory, patch.object(s,'PROJECT',Path(directory)):
                summary=s.report('calibration',{'frozen':True})
                self.assertEqual(summary['contrasts'],9)
        # Canonical serialized unavailable contrast tables must not contain NaN.

    def test_factorial_generator_and_repeat_pair_alignment(self):
        y=np.tile([0,1],30)
        def load(m,v,b,split='test',namespace='quality'):
            ids=list(range(60)) if split=='expanded_screen' else list(range(29,-1,-1))
            return {'cases':[cases()[i] for i in ids],'y':y[ids],'pred':y[ids]}
        with patch.object(s.stability,'SCREEN_BENCHMARKS',['wdc']),patch.object(s,'MODELS',{'jev':{},'luna':{}}),patch.object(s,'load',load):
            rows=list(s.comparisons('stability'))
            self.assertEqual(len(rows),9)
            self.assertEqual(sum(r[0]['condition']=='prompt_x_format_vs_luna' for r in rows),1)
            for meta,c,t,w,z,test in rows:
                result=s.contrast(c,t,w,z,repeats=29,test_null=test)
                self.assertEqual(result['estimate'],0.)


if __name__=='__main__':unittest.main()
