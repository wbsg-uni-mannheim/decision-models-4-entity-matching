"""Offline guard tests: no API calls or real inference."""
import copy
import unittest
from resume_screen import delta, retryable


class RecoveryGuards(unittest.TestCase):
    def setUp(self):
        self.failure = {"model_alias": "jev", "attempts": [{
            "status": 499, "error": "HTTP 499", "response": {
                "error": {"message": 'HTTP 499: {"detail":{"error_type":"internal_error"}}'}}}]}

    def test_only_diagnosed_jev_error(self):
        self.assertTrue(retryable([self.failure]))
        for status in [400, 401, 402, 403, 429, 500]:
            row = copy.deepcopy(self.failure)
            row["attempts"][0]["status"] = status
            self.assertFalse(retryable([row]))
        row = copy.deepcopy(self.failure)
        row["model_alias"] = "clef_flash"
        self.assertFalse(retryable([row]))

    def test_empty_and_other_499_stop(self):
        self.assertFalse(retryable([]))
        row = copy.deepcopy(self.failure)
        row["attempts"][0]["response"] = {"error": {"message": "client closed request"}}
        self.assertFalse(retryable([row]))
        self.assertFalse(retryable([self.failure, row]))

    def test_only_new_failures_count(self):
        before = {"a": [self.failure]}
        self.assertEqual(delta(before, before), [])
        self.assertEqual(delta(before, {"a": [self.failure, self.failure]}), [self.failure])
        with self.assertRaises(AssertionError):
            delta(before, {})


if __name__ == "__main__":
    unittest.main()
