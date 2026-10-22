import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from astra_discovery_collect import collect, usage_inventory


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.package = self.root/'package'
        self.package.mkdir()
        cases = {'discovery': [{'case_id': 'x', 'records': {'left': {'title': 'A'}, 'right': {'title': 'B'}}}]}
        (self.package/'cases_private.json').write_text(json.dumps(cases))
        (self.package/'discovery.jsonl').write_text('{"custom_id":"discover-x"}\n')
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.package.iterdir()}
        (self.package/'manifest.json').write_text(json.dumps({'files_sha256': hashes}))
        self.batch = {'id': 'batch_test', 'input_file_id': 'file-input', 'endpoint': '/v1/responses',
            'metadata': {'package_sha256': hashes['discovery.jsonl'], 'study': 'decision-em-20261013', 'stage': 'discovery_only'},
            'status': 'completed', 'output_file_id': 'file-output', 'error_file_id': None, 'request_counts': {'total': 1}}
        (self.package/'batch_receipt.json').write_text(json.dumps(self.batch))
        self.row = {'custom_id': 'discover-x', 'response': {'status_code': 200, 'body': {
            'status': 'completed', 'model': 'gpt-6-astra', 'usage': {'input_tokens': 10, 'output_tokens': 20},
            'output': [{'type': 'message', 'content': [{'type': 'output_text',
                'text': json.dumps({'observations': [], 'uncertainty': 'Unclear'})}]}]}}}
        self.called = []

    def run_collect(self, lines=None):
        def fake(key, endpoint):
            self.called.append(endpoint)
            if endpoint == 'batches/batch_test': return json.dumps(self.batch).encode()
            if endpoint == 'files/file-output/content':
                return ('\n'.join(json.dumps(r) for r in ([self.row] if lines is None else lines))+'\n').encode()
            raise AssertionError('Unexpected request')
        with patch('astra_discovery_collect.credential', return_value='mock'):
            return collect(self.root, self.package, self.root/'retrieval', fake)

    def test_complete_no_ledger_write(self):
        ledger = self.root/'artifacts'; ledger.mkdir()
        (ledger/'budget.json').write_text('{"reserved":11}')
        result = self.run_collect()
        self.assertEqual(result['available'], 1)
        self.assertTrue(result['usage']['all_request_usage_present'])
        self.assertFalse(result['usage']['release_commitment'])
        self.assertEqual((ledger/'budget.json').read_text(), '{"reserved":11}')
        with self.assertRaises(FileExistsError): self.run_collect()

    def test_pending_no_download(self):
        self.batch['status'] = 'in_progress'
        result = self.run_collect()
        self.assertNotIn('available', result)
        self.assertEqual(self.called, ['batches/batch_test'])

    def test_wrong_package_or_identity(self):
        self.batch['metadata']['package_sha256'] = 'wrong'
        with self.assertRaises(ValueError): self.run_collect()
        self.assertFalse((self.root/'retrieval').exists())

    def test_duplicates_rejected_and_raw_kept(self):
        with self.assertRaises(ValueError): self.run_collect([self.row, self.row])
        self.assertTrue((self.root/'retrieval/output_file_id.jsonl').exists())

    def test_missing_or_wrong_model_not_valid_observation(self):
        del self.row['response']['body']['model']
        result = self.run_collect()
        self.assertEqual(result['available'], 0)
        self.assertEqual(result['reasons'], {'unexpected_or_missing_model': 1})

    def test_partial_usage_never_zero_filled(self):
        rows = [{'custom_id': 'a', 'response': None}, {'custom_id': 'b', 'response': {'body': None}}]
        result = usage_inventory(rows, {'a', 'b', 'c'})
        self.assertEqual(result['without_usage'], ['a', 'b'])
        self.assertEqual(result['missing_rows'], ['c'])
        self.assertFalse(result['all_request_usage_present'])


if __name__ == '__main__': unittest.main()
