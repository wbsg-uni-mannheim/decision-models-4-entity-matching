import unittest
from summarize_three_runs import CORE, summarize


class SummaryTests(unittest.TestCase):
    def rows(self):
        return [{'run_id':r,'model':'m','condition':'global_prompt','benchmark':b,
                 'metric':'f1','value':.5+.1*r} for r in (1,2,3) for b in CORE]

    def test_macro_inside_run_excludes_supplement(self):
        rows=self.rows()+[{'run_id':r,'model':'m','condition':'global_prompt',
                         'benchmark':'billiger-de-50un','metric':'f1','value':0} for r in (1,2,3)]
        macro=next(x for x in summarize(rows)['rows'] if x['benchmark']=='macro_8_core_families')
        self.assertAlmostEqual(macro['mean'],.7)
        self.assertAlmostEqual(macro['sd'],.1)

    def test_missing_and_unavailable_do_not_get_zero_or_partial_macro(self):
        rows=self.rows();rows[0]['value']=None
        macro=next(x for x in summarize(rows)['rows'] if x['benchmark']=='macro_8_core_families')
        self.assertEqual(macro['status'],'incomplete');self.assertIsNone(macro['sd'])
        self.assertNotIn(1,macro['run_values'])

    def test_duplicate_cache_and_fake_run_ids_rejected(self):
        rows=self.rows()
        with self.assertRaises(ValueError):summarize(rows+[rows[0]])
        rows[0]['run_id']=True
        with self.assertRaises(ValueError):summarize(rows)


if __name__=='__main__':unittest.main()
