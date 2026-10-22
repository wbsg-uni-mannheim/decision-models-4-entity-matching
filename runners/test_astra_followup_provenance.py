import json
import unittest

from astra_discovery_submit import sha
from astra_followup_provenance import prepare_synthesis, verified_discovery
import test_astra_discovery_collect as collection_fixture


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = collection_fixture.CollectionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.package = self.fixture.package
        self.retrieval = self.fixture.root / 'retrieval'
        cases = json.loads((self.package / 'cases_private.json').read_text())
        cases['discovery'][0].update(source_split='prompt', gold_label=0, prediction=1,
                                     error_type='FP', models=['hidden-model-identity'])
        self.save(self.package / 'cases_private.json', cases)
        self.save(self.package / 'discovery.jsonl', {'custom_id': 'discover-x', 'method': 'POST',
                  'url': '/v1/responses', 'body': {'model': 'gpt-6-astra'}})
        hashes = {name: sha(self.package / name) for name in ['cases_private.json', 'discovery.jsonl']}
        self.save(self.package / 'manifest.json', {'files_sha256': hashes})
        self.fixture.batch['metadata']['package_sha256'] = hashes['discovery.jsonl']
        self.fixture.batch['request_counts'] = {'total': 1, 'completed': 1, 'failed': 0}
        self.save(self.package / 'batch_receipt.json', self.fixture.batch)

    @staticmethod
    def save(path, value):
        path.write_text(json.dumps(value) + '\n')

    def test_raw_provenance_to_immutable_offline_packet(self):
        self.fixture.run_collect()
        output = self.fixture.root / 'followup'
        result = prepare_synthesis(self.package, self.retrieval, output)
        self.assertTrue(result['provenance']['raw_import_recomputed'])
        self.assertFalse(result['submitted'])
        self.assertFalse(result['covered_by_discovery_USD11_approval'])
        self.assertEqual(result['max_output_tokens_including_reasoning'], 8192)
        self.assertNotIn('hidden-model-identity', (output / 'requests.jsonl').read_text())
        with self.assertRaises(FileExistsError):
            prepare_synthesis(self.package, self.retrieval, output)
        self.assertFalse((self.fixture.root / 'artifacts/budget.json').exists())

    def test_in_progress_cannot_create_followup(self):
        self.fixture.batch['status'] = 'in_progress'
        self.fixture.run_collect()
        with self.assertRaisesRegex(ValueError, 'completed'):
            verified_discovery(self.package, self.retrieval)

    def test_edited_normalized_observation_cannot_enter_synthesis(self):
        self.fixture.run_collect()
        path = self.retrieval / 'observations_imported.json'
        data = json.loads(path.read_text())
        data['discover-x']['observations']['uncertainty'] = 'Injected unsupported observation'
        self.save(path, data)
        with self.assertRaisesRegex(ValueError, 'recomputed'):
            verified_discovery(self.package, self.retrieval)

    def test_raw_hash_change_rejected(self):
        self.fixture.run_collect()
        path = self.retrieval / 'output_file_id.jsonl'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            verified_discovery(self.package, self.retrieval)

    def test_missing_response_cannot_be_hidden_by_report(self):
        self.fixture.run_collect(lines=[])
        report_path = self.retrieval / 'retrieval_report.json'
        report = json.loads(report_path.read_text())
        report.update(available=1, unavailable=0, all_observations_available=True, reasons={})
        self.save(report_path, report)
        with self.assertRaisesRegex(ValueError, 'raw-response validation'):
            verified_discovery(self.package, self.retrieval)

    def test_usage_and_provider_counts_recomputed(self):
        self.fixture.run_collect()
        path = self.retrieval / 'retrieval_report.json'
        report = json.loads(path.read_text())
        report['usage']['input_tokens'] = 999
        self.save(path, report)
        with self.assertRaisesRegex(ValueError, 'usage inventory'):
            verified_discovery(self.package, self.retrieval)
        self.fixture.batch['request_counts']['completed'] = 0
        self.save(self.retrieval / 'batch_snapshot.json', self.fixture.batch)
        with self.assertRaisesRegex(ValueError, 'every discovery request'):
            verified_discovery(self.package, self.retrieval)

    def test_package_and_receipt_identity_checked(self):
        self.fixture.run_collect()
        receipt = json.loads((self.package / 'batch_receipt.json').read_text())
        receipt['id'] = 'batch_wrong'
        self.save(self.package / 'batch_receipt.json', receipt)
        with self.assertRaisesRegex(ValueError, 'identity'):
            verified_discovery(self.package, self.retrieval)


if __name__ == '__main__':
    unittest.main()
