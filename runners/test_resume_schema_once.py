import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from resume_schema_once import failure_hashes, terminal_steps, verify_ready


class SchemaRecovery(unittest.TestCase):
    def test_all_steps_required_and_terminal(self):
        rows = '42|FAILED|1:0\n42.batch|FAILED|1:0\n42.extern|COMPLETED|0:0\n'
        terminal_steps(rows, 42)
        for bad in [rows.replace('FAILED', 'RUNNING'), rows.split('42.extern')[0],
                    rows+'42.0|RUNNING|0:0\n']:
            with self.assertRaises(ValueError):
                terminal_steps(bad, 42)

    def test_scoped_failure_inventory_preserves_other_jobs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for ns in ['quality', 'schema', 'schema_pilot']:
                p = root/'runs'/ns/'model/case.failures.jsonl'
                p.parent.mkdir(parents=True)
                p.write_text('{}\n')
            before = failure_hashes(root)
            self.assertEqual(len(before), 2)
            (root/'runs/quality/model/case.failures.jsonl').write_text('changed')
            self.assertEqual(before, failure_hashes(root))
            (root/'runs/schema/model/case.failures.jsonl').write_text('changed')
            self.assertNotEqual(before, failure_hashes(root))

    def test_other_holds_allowed_cap_drift_and_overlap_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'configs').mkdir()
            (root/'artifacts').mkdir()
            (root/'configs/expanded_execution.json').write_text(json.dumps({'e2_schema_job': 43}))
            budget = root/'artifacts/budget.json'
            budget.write_text(json.dumps({'spent': 79, 'reserved': 3}))
            spec = {'job_id': 43, 'predecessor': 42, 'source_sha256': {},
                    'diagnosed_failure_sha256': {}}
            accounting = '42|FAILED|1:0\n42.batch|FAILED|1:0\n42.extern|COMPLETED|0:0\n'
            with patch('resume_schema_once.review'), patch('resume_schema_once.subprocess.check_output') as call:
                call.side_effect = [accounting, '']
                self.assertEqual(verify_ready(root, spec, '43')['reserved'], 3)
                budget.write_text(json.dumps({'spent': 98, 'reserved': 3}))
                call.side_effect = [accounting, '']
                with self.assertRaisesRegex(ValueError, 'ceiling'):
                    verify_ready(root, spec, '43')
                call.side_effect = [accounting, '44|RUNNING\n', 'python src/expanded_schema.py run']
                with self.assertRaisesRegex(ValueError, 'overlapping'):
                    verify_ready(root, spec, '43')
                spec['diagnosed_failure_sha256'] = {'missing': 'x'}
                call.side_effect = [accounting]
                with self.assertRaisesRegex(ValueError, 'journals changed'):
                    verify_ready(root, spec, '43')
