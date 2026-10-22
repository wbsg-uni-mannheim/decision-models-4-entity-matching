import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import astra_codebook_submit as submitter
from astra_codebook_collect import collect, import_codebook
from astra_discovery_submit import sha

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/astra_error_analysis/codebook_synthesis_v1'
DISCOVERY = ROOT / 'artifacts/astra_error_analysis/preparation_v1'
RETRIEVAL = DISCOVERY / 'retrieval_20261014T1308Z'


class CodebookTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.package = self.root / 'package'
        shutil.copytree(SOURCE, self.package, ignore=shutil.ignore_patterns('submission_journal.jsonl', 'batch_receipt.json', 'approval_aaron*.json'))
        (self.root / 'artifacts').mkdir()
        self.budget = self.root / 'artifacts/budget.json'
        self.budget.write_text(json.dumps({'spent': 65, 'reserved': 5, 'attempts': 1, 'unknown_cost_attempts': 0}))
        policy = self.root / 'policy.json'
        policy.write_text('{}')
        self.approval = {'stage': submitter.STAGE, 'approved_by': 'Aaron', 'max_cost_usd': 3,
                         'package_sha256': sha(self.package / 'requests.jsonl'),
                         'user_message_reference': 'synthetic test only', 'quoted_user_authorization': 'synthetic test only'}
        self.admission = {'human_approval_verified': True, 'package_sha256': self.approval['package_sha256'],
                          'parallel_policy': 'policy.json', 'parallel_policy_sha256': sha(policy),
                          'submitter_sha256': sha(Path(submitter.__file__))}
        self.approval_path = self.root / 'approval.json'
        self.admission_path = self.root / 'admission.json'
        self.save(self.approval_path, self.approval)
        self.save(self.admission_path, self.admission)
        request = json.loads((self.package / 'requests.jsonl').read_text())
        self.case_id = json.loads(request['body']['input'])['discovery_cases'][0]['case_id']
        self.book = {'categories': [{'code': 'C01', 'name': 'Variant', 'definition': 'Observed variant discrepancy',
                     'include_when': 'Different explicit variants', 'exclude_when': 'No stated variant',
                     'supporting_discovery_case_ids': [self.case_id]}],
                     'overlap_policy': 'Allow supported overlaps', 'uncertainty_policy': 'Use unclear when evidence is insufficient'}
        self.row = {'custom_id': 'synthesize-codebook-v1', 'response': {'status_code': 200, 'body': {
            'status': 'completed', 'model': 'gpt-6-astra', 'usage': {'input_tokens': 100, 'output_tokens': 50},
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(self.book)}]}]}}}
        self.batch = {'id': 'batch_mock', 'input_file_id': 'file-mock', 'status': 'completed',
                      'endpoint': '/v1/responses', 'metadata': {'study': 'decision-em-20261013',
                      'stage': submitter.STAGE, 'package_sha256': self.approval['package_sha256']},
                      'request_counts': {'total': 1, 'completed': 1, 'failed': 0},
                      'output_file_id': 'file-output', 'error_file_id': None}

    @staticmethod
    def save(path, obj):
        path.write_text(json.dumps(obj)+'\n')

    def submit(self, network):
        with patch.object(submitter, 'review', return_value={'reviewed_jobs': ['independent inference']}), \
             patch.object(submitter, 'credential', return_value='mock'), patch.object(submitter, 'post', side_effect=network):
            return submitter.submit(self.root, self.package, DISCOVERY, RETRIEVAL, self.approval_path, self.admission_path)

    def test_actual_packet_reconstructed_and_separate_approval_required(self):
        self.assertEqual(submitter.validate(self.package, DISCOVERY, RETRIEVAL, self.approval, self.admission), self.approval['package_sha256'])
        wrong = copy.deepcopy(self.approval)
        wrong.update(stage='discovery_observations_only', max_cost_usd=11)
        with self.assertRaisesRegex(ValueError, 'Separate codebook'):
            submitter.validate(self.package, DISCOVERY, RETRIEVAL, wrong, self.admission)

    def test_request_content_cannot_change_even_with_matching_file_hash(self):
        path = self.package / 'requests.jsonl'
        row = json.loads(path.read_text())
        row['body']['instructions'] = 'Use additional held-out data'
        self.save(path, row)
        manifest = json.loads((self.package / 'manifest.json').read_text())
        manifest['requests_sha256'] = sha(path)
        self.save(self.package / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'reconstructed'):
            submitter.validate(self.package, DISCOVERY, RETRIEVAL, self.approval, self.admission)

    def test_hold_before_network_preserves_other_holds_and_no_double_submit(self):
        calls = []
        def network(key, endpoint, data, content_type):
            calls.append(endpoint)
            self.assertEqual(json.loads(self.budget.read_text())['reserved'], 8)
            return {'id': 'file-mock'} if endpoint == 'files' else self.batch
        self.assertEqual(self.submit(network), 'batch_mock')
        with self.assertRaisesRegex(ValueError, 'never submit twice'):
            self.submit(network)
        self.assertEqual(calls, ['files', 'batches'])

    def test_unknown_upload_cannot_be_repeated_and_retains_hold(self):
        with self.assertRaises(TimeoutError):
            self.submit(TimeoutError())
        self.assertEqual(json.loads(self.budget.read_text())['reserved'], 8)
        with self.assertRaises(FileExistsError):
            self.submit(AssertionError('Must not call network twice'))

    def test_cap_rejects_before_any_network(self):
        self.budget.write_text(json.dumps({'spent': 94, 'reserved': 5}))
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            self.submit(AssertionError('No paid/upload call permitted'))
        self.assertEqual(json.loads(self.budget.read_text())['reserved'], 5)

    def test_unknown_batch_create_retains_uploaded_id_and_hold(self):
        def network(key, endpoint, data, content_type):
            if endpoint == 'files':
                return {'id': 'file-known'}
            raise TimeoutError('Unknown create outcome')
        with self.assertRaises(TimeoutError):
            self.submit(network)
        self.assertIn('file-known', (self.package / 'submission_journal.jsonl').read_text())
        self.assertEqual(json.loads(self.budget.read_text())['reserved'], 8)
        with self.assertRaises(FileExistsError):
            self.submit(AssertionError('No retry'))

    def test_import_validates_discovery_references_without_human_acceptance(self):
        result = import_codebook(self.row, [self.case_id])
        self.assertTrue(result['schema_and_discovery_references_valid'])
        self.assertFalse(result['accepted_for_test_categorization'])
        with self.assertRaisesRegex(ValueError, 'discovery cases'):
            import_codebook(self.row, ['different-discovery-id'])

    def test_missing_usage_incomplete_wrong_model_rejected(self):
        for mutation in [{'usage': {}}, {'status': 'incomplete'}, {'model': 'another-model'}]:
            row = copy.deepcopy(self.row)
            row['response']['body'].update(mutation)
            with self.assertRaises(ValueError):
                import_codebook(row, [self.case_id])

    def test_get_only_collector_preserves_raw_and_budget(self):
        self.save(self.package / 'batch_receipt.json', self.batch)
        calls = []
        def fetch(key, endpoint):
            calls.append(endpoint)
            return json.dumps(self.batch if endpoint.startswith('batches/') else self.row).encode()
        before = self.budget.read_bytes()
        with patch('astra_codebook_collect.credential', return_value='mock'):
            result = collect(self.root, self.package, self.root / 'retrieval', fetch)
        self.assertTrue(result['technical_import_valid'])
        self.assertFalse(result['accepted_for_test_categorization'])
        self.assertEqual(calls, ['batches/batch_mock', 'files/file-output/content'])
        self.assertEqual(before, self.budget.read_bytes())

    def test_pending_collector_only_gets_batch(self):
        self.batch['status'] = 'in_progress'
        self.save(self.package / 'batch_receipt.json', self.batch)
        with patch('astra_codebook_collect.credential', return_value='mock'):
            result = collect(self.root, self.package, self.root / 'retrieval', lambda key, path: json.dumps(self.batch).encode())
        self.assertNotIn('technical_import_valid', result)
        self.assertFalse((self.root / 'retrieval/codebook_candidate.json').exists())

    def test_duplicate_output_rejected_after_raw_preservation(self):
        self.save(self.package / 'batch_receipt.json', self.batch)
        def fetch(key, endpoint):
            return (json.dumps(self.batch) if endpoint.startswith('batches/') else
                    json.dumps(self.row)+'\n'+json.dumps(self.row)).encode()
        with patch('astra_codebook_collect.credential', return_value='mock'):
            with self.assertRaisesRegex(ValueError, 'request count'):
                collect(self.root, self.package, self.root / 'retrieval', fetch)
        self.assertTrue((self.root / 'retrieval/output_file_id.jsonl').exists())
        self.assertFalse((self.root / 'retrieval/codebook_candidate.json').exists())


if __name__ == '__main__':
    unittest.main()
