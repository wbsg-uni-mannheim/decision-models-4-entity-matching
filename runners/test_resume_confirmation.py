"""Exercise transient-error boundaries without calling providers."""
import copy
import unittest
from resume_confirmation import delta, retryable


class Guards(unittest.TestCase):
    def setUp(self):
        self.jev = {'model_alias': 'jev', 'attempts': [{'status': 499, 'error': 'HTTP 499',
                    'response': {'error': {'message': 'HTTP 499: {"detail":{"error_type":"internal_error"}}'}}}]}
        self.flash = {'model_alias': 'clef_flash', 'attempts': [{'status': 429, 'error': 'HTTP 429',
                      'response': {'error': {'message': 'Capacity temporarily exceeded'}}}]}
        self.luna = {'model_alias': 'luna', 'attempts': [{'status': 200, 'cost': None,
                     'error': 'ValueError: Missing actual usage cost',
                     'response': {'error': {'code': 502, 'metadata': {'error_type': 'provider_unavailable'}}}}]}

    def test_observed_transients(self):
        self.assertTrue(retryable([self.jev, self.flash, self.luna]))
        self.assertFalse(retryable([]))

    def test_missing_cost_without_provider_error_stops(self):
        r = copy.deepcopy(self.luna)
        r['attempts'][0]['response'] = {'choices': []}
        self.assertFalse(retryable([self.jev, r]))

    def test_auth_budget_and_other_provider_errors_stop(self):
        for row in [self.jev, self.flash, self.luna]:
            for code in [400, 401, 402, 403, 500, 503]:
                r = copy.deepcopy(row)
                r['attempts'][0]['status'] = code
                self.assertFalse(retryable([r]))
        r = copy.deepcopy(self.flash)
        r['attempts'][0]['response']['error']['message'] = 'Rate limit'
        self.assertFalse(retryable([r]))

    def test_journal_is_append_only(self):
        before = {'file': [self.jev]}
        self.assertEqual(delta(before, before), [])
        self.assertEqual(delta(before, {'file': [self.jev, self.luna]}), [self.luna])
        with self.assertRaises(AssertionError):
            delta(before, {'file': [self.flash]})


if __name__ == '__main__':
    unittest.main()
