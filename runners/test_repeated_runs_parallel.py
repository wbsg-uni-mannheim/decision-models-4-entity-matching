import fcntl
from pathlib import Path
import tempfile
import unittest

from repeated_runs_parallel import validate_admission, validate_history


class ParallelRunsTests(unittest.TestCase):
    def setUp(self):
        self.spec = {'runner_sha256': 'frozen', 'replica': '/root/replications/run02'}
        self.admission = {'run_id': 2, 'stage': 'smoke', 'slurm_job_id': '10', 'budget_cap_usd': 100,
                          'runner_sha256': 'frozen', 'launcher_sha256': 'new',
                          'parallel_policy_sha256': 'policy', 'replica': self.spec['replica']}

    def check(self, admission=None, budget=None, job='10'):
        validate_admission(self.admission if admission is None else admission, 2, 'smoke', job,
                           self.spec, 'new', 'policy', budget or {'spent': 70, 'reserved': 5})

    def test_other_holds_allowed_cap_enforced(self):
        self.check()
        for budget in [{'spent': 99, 'reserved': 1}, {'spent': float('nan'), 'reserved': 0},
                       {'spent': 1, 'reserved': -1}]:
            with self.assertRaises(ValueError):
                self.check(budget=budget)

    def test_job_run_source_and_policy_identity(self):
        for field in self.admission:
            with self.assertRaises(ValueError):
                self.check({**self.admission, field: 'wrong'})
        with self.assertRaises(ValueError):
            self.check(job=None)

    def test_history_requires_completed_prerequisites_and_no_repeat(self):
        validate_history([], 2, 'smoke')
        rows = [{'run_id': 2, 'stage': 'smoke', 'status': 'started'},
                {'run_id': 2, 'stage': 'smoke', 'status': 'complete'}]
        validate_history(rows, 2, 'core')
        with self.assertRaises(ValueError):
            validate_history(rows, 2, 'smoke')
        with self.assertRaises(ValueError):
            validate_history(rows[:-1], 2, 'core')
        with self.assertRaises(ValueError):
            validate_history(rows, 3, 'core')
        with self.assertRaises(ValueError):
            validate_history(rows, 2, 'rules')
        with self.assertRaises(ValueError):
            validate_history(rows+[{'run_id': 2, 'stage': 'variants', 'status': 'started'}], 2, 'core')

    def test_shared_parallel_lock_excludes_legacy_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'lock'
            with path.open('a') as first, path.open('a') as second, path.open('a') as legacy:
                fcntl.flock(first, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(second, fcntl.LOCK_SH | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(first, fcntl.LOCK_UN)
                fcntl.flock(second, fcntl.LOCK_UN)
                fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(first, fcntl.LOCK_SH | fcntl.LOCK_NB)

    def test_replica_lock_excludes_duplicate_but_other_replica_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder)/'run02').open('a') as first, (Path(folder)/'run02').open('a') as duplicate, \
                    (Path(folder)/'run03').open('a') as other:
                fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(duplicate, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == '__main__':
    unittest.main()
