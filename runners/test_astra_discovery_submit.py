import copy
import json
from pathlib import Path
import unittest
from astra_discovery_submit import sha, validate

PACKAGE = Path(__file__).resolve().parents[2]/'artifacts/astra_error_analysis/preparation_v1'


class SubmitTests(unittest.TestCase):
    def setUp(self):
        self.approval = {'stage': 'discovery_observations_only', 'approved_by': 'Aaron', 'package_sha256': sha(PACKAGE/'discovery.jsonl'),
            'max_cost_usd': 11, 'user_message_reference': 'synthetic-test-only', 'quoted_user_authorization': 'synthetic-test-only'}
        self.admission = {'human_approval_verified': True, 'package_sha256': self.approval['package_sha256'], 'all_project_inference_jobs_reviewed': True}

    def test_actual_payload_validation_without_network(self):
        self.assertEqual(validate(PACKAGE, self.approval, self.admission, []), self.approval['package_sha256'])
        self.assertEqual(len((PACKAGE/'discovery.jsonl').read_text().splitlines()), 130)

    def test_no_approval_no_busy_execution(self):
        with self.assertRaises(ValueError):
            validate(PACKAGE, {}, self.admission, [])
        with self.assertRaises(ValueError):
            validate(PACKAGE, self.approval, self.admission, ['365636 running'])
        for key in ['package_sha256', 'quoted_user_authorization', 'max_cost_usd']:
            bad = copy.deepcopy(self.approval)
            bad[key] = None
            with self.assertRaises(ValueError):
                validate(PACKAGE, bad, self.admission, [])

    def test_mocked_submit_reserves_before_network_and_no_repeat(self):
        import tempfile
        import shutil
        from unittest.mock import patch
        from astra_discovery_submit import submit
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'artifacts').mkdir()
            (root/'artifacts/budget.json').write_text(json.dumps({'spent': 50., 'reserved': 0., 'attempts': 0, 'unknown_cost_attempts': 0}))
            package = root/'package'
            shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns('submission_journal.jsonl', 'batch_receipt.json'))
            approval = root/'approval.json'; approval.write_text(json.dumps(self.approval))
            admission = root/'admission.json'; admission.write_text(json.dumps(self.admission))
            def fake_post(key, endpoint, data, content_type):
                budget = json.loads((root/'artifacts/budget.json').read_text())
                self.assertEqual(budget['reserved'], 11)
                return {'id': 'file_test'} if endpoint == 'files' else {'id': 'batch_test', 'status': 'validating'}
            with patch('astra_discovery_submit.subprocess.check_output', return_value=''), patch('astra_discovery_submit.credential', return_value='test-only'), patch('astra_discovery_submit.post', side_effect=fake_post):
                self.assertEqual(submit(root, package, approval, admission), 'batch_test')
                with self.assertRaises(FileExistsError):
                    submit(root, package, approval, admission)

    def test_unknown_upload_retains_hold(self):
        import tempfile
        import shutil
        from unittest.mock import patch
        from astra_discovery_submit import submit
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'artifacts').mkdir()
            (root/'artifacts/budget.json').write_text(json.dumps({'spent': 50., 'reserved': 0., 'attempts': 0, 'unknown_cost_attempts': 0}))
            package = root/'package'; shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns('submission_journal.jsonl', 'batch_receipt.json'))
            approval = root/'approval.json'; approval.write_text(json.dumps(self.approval))
            admission = root/'admission.json'; admission.write_text(json.dumps(self.admission))
            with patch('astra_discovery_submit.subprocess.check_output', return_value=''), patch('astra_discovery_submit.credential', return_value='test-only'), patch('astra_discovery_submit.post', side_effect=TimeoutError):
                with self.assertRaises(TimeoutError): submit(root, package, approval, admission)
            self.assertEqual(json.loads((root/'artifacts/budget.json').read_text())['reserved'], 11)
            self.assertIn('unknown_until_receipt', (package/'submission_journal.jsonl').read_text())

    def test_no_persisted_synthetic_approval(self):
        self.assertFalse((PACKAGE/'approval.json').exists())
        self.assertEqual(json.loads((PACKAGE/'manifest.json').read_text())['status'], 'offline_prepared_no_upload_no_approval')


if __name__ == '__main__':
    unittest.main()
