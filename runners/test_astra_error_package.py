import unittest
from astra_error_package import observation_request, record_key, select


class PackageTests(unittest.TestCase):
    def row(self, pair, records=None):
        records = records or {'left': {'title': pair}, 'right': {'title': 'right'}}
        return {'benchmark': 'b', 'pair_id': pair, 'records': records, 'record_key': record_key(records),
                'case_id': pair, 'models': ['secret_model'], 'gold_label': 0, 'prediction': 1, 'error_type': 'FP'}

    def test_mirrored_records_excluded(self):
        a = self.row('a')
        b = self.row('b', {'left': a['records']['right'], 'right': a['records']['left']})
        self.assertEqual(record_key(a['records']), record_key(b['records']))
        self.assertEqual(select([b], 10, {a['record_key']}), ([], 0))

    def test_deterministic_dedup_shortfall(self):
        rows = [self.row(str(i)) for i in range(5)]
        x, n = select(rows+rows, 10)
        y, _ = select(list(reversed(rows)), 10)
        self.assertEqual(x, y)
        self.assertEqual((len(x), n), (5, 5))

    def test_discovery_wire_blinding(self):
        request = observation_request(self.row('a'))
        import json
        wire = json.dumps(request)
        for forbidden in ['secret_model', 'gold_label', 'prediction', 'error_type', 'pair_id']:
            self.assertNotIn(forbidden, wire)
        self.assertNotIn('tools', request['body'])
        self.assertEqual(request['body']['max_output_tokens'], 2048)
        self.assertEqual(request['url'], '/v1/responses')


if __name__ == '__main__':
    unittest.main()
