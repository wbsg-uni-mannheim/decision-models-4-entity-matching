import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from resume_confirmation_parallel import budget_ready, scoped_failures


class ParallelRecovery(unittest.TestCase):
    def test_other_job_failures_do_not_trigger_confirmation_retry(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for benchmark in ['wdc','wdc-050un']:
                p=root/'runs/quality/luna/B0'/benchmark/'test.failures.jsonl'
                p.parent.mkdir(parents=True)
                p.write_text(json.dumps({'benchmark':benchmark})+'\n')
            result=scoped_failures(root,{('luna','B0','wdc','test')})
            self.assertEqual(len(result),1)
            self.assertEqual(next(iter(result.values()))[0]['benchmark'],'wdc')

    def test_other_holds_allowed_without_ignoring_cap(self):
        budget_ready({'spent':63,'reserved':25})
        with self.assertRaises(ValueError): budget_ready({'spent':80,'reserved':20})
