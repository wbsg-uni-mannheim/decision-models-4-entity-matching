import json
import unittest
from astra_observation_import import import_rows


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{'case_id': 'x', 'records': {'left': {'title': 'Alpha'}, 'right': {'title': 'Beta'}}}]
        self.content = {'observations': [{'attribute': 'title', 'left_evidence': 'Alpha', 'right_evidence': 'Beta', 'description': 'Titles differ.'}], 'uncertainty': ''}

    def row(self):
        return {'custom_id': 'discover-x', 'response': {'status_code': 200, 'body': {'status': 'completed',
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(self.content)}]}],
            'usage': {'input_tokens': 100, 'output_tokens': 50}}}}

    def test_success_and_unsupported_evidence(self):
        self.assertTrue(import_rows([self.row()], self.cases)['discover-x']['available'])
        self.content['observations'][0]['left_evidence'] = 'invented'
        self.assertFalse(import_rows([self.row()], self.cases)['discover-x']['available'])

    def test_missing_duplicate_wrong_id(self):
        self.assertEqual(import_rows([], self.cases)['discover-x']['reason'], 'missing_batch_result')
        with self.assertRaises(ValueError):
            import_rows([self.row(), self.row()], self.cases)
        row = self.row()
        row['custom_id'] = 'unknown'
        with self.assertRaises(ValueError):
            import_rows([row], self.cases)

    def test_incomplete_and_missing_usage(self):
        row = self.row()
        row['response']['body']['status'] = 'incomplete'
        self.assertFalse(import_rows([row], self.cases)['discover-x']['available'])
        row = self.row()
        del row['response']['body']['usage']
        self.assertTrue(import_rows([row], self.cases)['discover-x']['cost_unknown'])

    def test_abstract_attribute_keeps_all_exact_source_fields(self):
        self.content['observations'][0]['attribute'] = 'product designation'
        self.cases[0]['records']['left']['description'] = 'Alpha product'
        self.assertFalse(import_rows([self.row()], self.cases)['discover-x']['available'])
        result = import_rows([self.row()], self.cases, 'record_with_field_provenance')['discover-x']
        self.assertTrue(result['available'])
        self.assertEqual(result['observations'], self.content)
        self.assertEqual(result['evidence_validation']['source_field_matches'][0]['left'], ['title', 'description'])

    def test_record_policy_still_rejects_wrong_side_and_nonliteral_evidence(self):
        self.content['observations'][0]['attribute'] = 'product designation'
        for quote in ['Beta', 'alpha', 'invented']:
            self.content['observations'][0]['left_evidence'] = quote
            self.assertFalse(import_rows([self.row()], self.cases, 'record_with_field_provenance')['discover-x']['available'])

    def test_actual_field_key_cannot_borrow_evidence_elsewhere(self):
        self.cases[0]['records']['left']['title'] = 'Other'
        self.cases[0]['records']['left']['description'] = 'Alpha'
        self.assertFalse(import_rows([self.row()], self.cases, 'record_with_field_provenance')['discover-x']['available'])

    def test_null_brand_does_not_hide_literal_brand_in_title(self):
        self.content['observations'][0]['attribute'] = 'brand'
        self.cases[0]['records']['left']['brand'] = None
        result = import_rows([self.row()], self.cases, 'record_with_field_provenance')['discover-x']
        self.assertTrue(result['available'])
        self.assertEqual(result['evidence_validation']['source_field_matches'][0]['left'], ['title'])


if __name__ == '__main__':
    unittest.main()
