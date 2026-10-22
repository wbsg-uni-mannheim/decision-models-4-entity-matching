import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from repeated_runs import prepare, replica_path, sha, shared_account, three_run_summary, verify

SOURCE = Path(__file__).resolve().parents[2]/'src'


class RepetitionTests(unittest.TestCase):
    def fixture(self, root):
        for d in ['src', 'configs', 'data', 'artifacts/astra_agent']:
            (root/d).mkdir(parents=True)
        (root/'src/demo.py').write_text('VALUE = 1\n')
        for f in ['manifest.json', 'learned_rules.json', 'selected_variants.json', 'calibration_selection.json']:
            (root/'configs'/f).write_text('{}')
        (root/'artifacts/astra_agent/acceptance.json').write_text('{}')
        (root/'artifacts/budget.json').write_text('{"spent":1,"reserved":0,"attempts":1,"unknown_cost_attempts":0}')

    def test_fresh_isolation_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            for n in [2, 3]:
                prepare(root, n); verify(root, n)
                replica = replica_path(root, n)
                self.assertEqual(list((replica/'runs').iterdir()), [])
                self.assertFalse((replica/'configs/selected_variants.json').exists())
                self.assertFalse((replica/'configs/calibration_selection.json').exists())
                self.assertEqual(json.loads((replica/'artifacts/progress.json').read_text()), {})
            with self.assertRaises(FileExistsError): prepare(root, 2)
            with self.assertRaises(ValueError): replica_path(root, 1)
            (replica_path(root, 2)/'src/demo.py').write_text('VALUE=2')
            with self.assertRaises(AssertionError): verify(root, 2)
            verify(root, 3)

    def test_shared_real_budget_atomic_replace_keeps_link(self):
        # Use the actual frozen accounting function, without a network request.
        import sys
        sys.path.insert(0, str(SOURCE))
        spec = importlib.util.spec_from_file_location('accounting_under_test', SOURCE/'run.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root); prepare(root, 2); prepare(root, 3)
            account = shared_account(module.account, root)
            prior_project = module.PROJECT
            account('reserve'); account('settle', .012)
            for n in [2, 3]:
                path = replica_path(root, n)/'artifacts/budget.json'
                self.assertTrue(path.is_symlink())
                self.assertAlmostEqual(json.loads(path.read_text())['spent'], 1.012)
            self.assertEqual(module.PROJECT, prior_project)
            with self.assertRaises(RuntimeError): account('reserve', cap=1)

    def test_summary_requires_three_real_runs(self):
        self.assertEqual(three_run_summary({1: .5})['status'], 'incomplete')
        row = three_run_summary({1: .7, 2: .8, 3: .9})
        self.assertAlmostEqual(row['mean'], .8); self.assertAlmostEqual(row['sd'], .1)
        with self.assertRaises(ValueError): three_run_summary({1: .7, 2: float('nan'), 3: .9})

    def test_no_paid_admission_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            spec = prepare(root, 2)
            self.assertFalse(spec['execution_enabled'])
            self.assertFalse((replica_path(root, 2)/'execution_admission.json').exists())
            self.assertEqual(spec['immutable_sha256']['src/demo.py'], sha(root/'src/demo.py'))


if __name__ == '__main__': unittest.main()
