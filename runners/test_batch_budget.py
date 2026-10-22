import json
from pathlib import Path
import tempfile
import unittest
from batch_budget import update


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'artifacts').mkdir()
        (self.root/'artifacts/budget.json').write_text(json.dumps({'spent': 90., 'reserved': .1, 'attempts': 2, 'unknown_cost_attempts': 0}))
        self.approval = {'approved_by': 'Aaron', 'package_sha256': 'hash', 'max_cost_usd': 9., 'user_message_reference': 'synthetic-test-only'}

    def test_reserve_settle_and_duplicate_rejection(self):
        obj = update(self.root, 'reserve', 'a', 'hash', 8., self.approval)
        self.assertAlmostEqual(obj['reserved'], 8.1)
        with self.assertRaises(ValueError):
            update(self.root, 'reserve', 'a', 'hash', 8., self.approval)
        with self.assertRaises(ValueError):
            update(self.root, 'reconcile', 'a', 'hash', 3.)
        obj = update(self.root, 'reconcile', 'a', 'hash', 3., terminal=True, usage_complete=True)
        self.assertAlmostEqual(obj['reserved'], .1)
        self.assertEqual(obj['spent'], 93.)
        self.assertEqual(obj['attempts'], 2)

    def test_frozen_openrouter_preserves_batch_hold(self):
        import importlib.util
        import sys
        from repeated_runs import shared_account
        source = Path(__file__).resolve().parents[2]/'src'
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location('batch_accounting_test', source/'run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        account = shared_account(module.account, self.root)
        update(self.root, 'reserve', 'a', 'hash', 8., self.approval)
        account('reserve')
        account('settle', .01)
        obj = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(obj['reserved'], 8.1)
        self.assertEqual(obj['direct_openai_commitments']['a']['status'], 'reserved')
        self.assertAlmostEqual(obj['spent'], 90.01)

    def test_approval_cap_unknown_and_bad_money(self):
        for approval, amount in [(None, 1.), (self.approval, float('nan')), (self.approval, -1.)]:
            with self.assertRaises(ValueError):
                update(self.root, 'reserve', 'a', 'hash', amount, approval)
        update(self.root, 'reserve', 'a', 'hash', 8., self.approval)
        with self.assertRaises(ValueError):
            update(self.root, 'reserve', 'b', 'hash', 3., self.approval)
        with self.assertRaises(ValueError):
            update(self.root, 'reconcile', 'a', 'hash', 9., terminal=True, usage_complete=True)
        obj = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(obj['reserved'], 8.1)


if __name__ == '__main__':
    unittest.main()
