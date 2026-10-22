import json
from pathlib import Path
import tempfile
import unittest

from fresh_stage_readiness import EXTENSIONS, MARKERS, REQUIRES, readiness


class StageReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        self.budget = {'spent': 90, 'reserved': 3}

    def complete(self, stage):
        self.rows.extend({'run_id': 2, 'stage': stage, 'status': status}
                         for status in ('started', 'complete'))
        rel, key, status = MARKERS[stage]
        path = self.root/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = json.loads(path.read_text()) if path.exists() else {}
        if key:
            obj[key] = {'status': status}
        else:
            obj['status'] = status
        path.write_text(json.dumps(obj))

    def inspect(self, stage):
        return readiness(self.root, 2, stage, self.rows, self.budget)

    def test_frozen_entrypoint_prerequisites_beyond_launcher(self):
        self.complete('core')
        self.assertFalse(self.inspect('screen')['local_prerequisites_satisfied'])
        self.assertFalse(self.inspect('calibration')['local_prerequisites_satisfied'])
        self.complete('supplement')
        self.assertTrue(self.inspect('screen')['local_prerequisites_satisfied'])
        self.complete('screen')
        self.assertTrue(self.inspect('calibration')['local_prerequisites_satisfied'])

    def test_every_extension_requires_its_results_and_journal(self):
        for stage in EXTENSIONS:
            with self.subTest(stage=stage):
                self.rows = []
                for prerequisite in REQUIRES[stage]:
                    self.complete(prerequisite)
                result = self.inspect(stage)
                self.assertTrue(result['local_prerequisites_satisfied'], result)
                self.assertFalse(result['execution_admitted'])
                self.assertEqual(result['paid_inference'], stage not in ('cascades', 'errors'))
                self.rows.pop()
                self.assertFalse(self.inspect(stage)['local_prerequisites_satisfied'])

    def test_complete_journal_does_not_replace_required_marker(self):
        self.complete('core')
        (self.root/'artifacts/progress.json').unlink()
        self.assertFalse(self.inspect('supplement')['local_prerequisites_satisfied'])

    def test_active_failed_or_previously_attempted_stage_not_admitted(self):
        self.complete('core')
        for state in ('started', 'failed', 'complete'):
            with self.subTest(state=state):
                row = {'run_id': 2, 'stage': 'supplement', 'status': state}
                self.rows.append(row)
                self.assertFalse(self.inspect('supplement')['local_prerequisites_satisfied'])
                self.rows.pop()
        self.rows.append({'run_id': 2, 'stage': 'variants', 'status': 'started'})
        self.assertFalse(self.inspect('supplement')['local_prerequisites_satisfied'])

    def test_mixed_run_and_budget_guards_preserve_foreign_holds(self):
        self.complete('core')
        self.assertTrue(self.inspect('supplement')['local_prerequisites_satisfied'])
        self.assertEqual(self.budget, {'spent': 90, 'reserved': 3})
        self.rows[0]['run_id'] = 3
        self.assertFalse(self.inspect('supplement')['local_prerequisites_satisfied'])
        self.rows[0]['run_id'] = 2
        self.budget = {'spent': 99.96, 'reserved': 0}
        self.assertFalse(self.inspect('supplement')['local_prerequisites_satisfied'])
        for invalid in (float('nan'), -1, True):
            self.budget = {'spent': invalid, 'reserved': 0}
            with self.assertRaises(ValueError):
                self.inspect('supplement')


if __name__ == '__main__':
    unittest.main()
