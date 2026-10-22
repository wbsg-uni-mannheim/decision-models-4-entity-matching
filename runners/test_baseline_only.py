"""Verify the restricted stage cannot trigger AP2 or an analysis subprocess."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from repeated_runs_parallel import baseline_only, validate_history


class BaselineOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'artifacts').mkdir()
        self.progress = self.root/'artifacts/progress.json'
        self.progress.write_text(json.dumps({'smoke': {'status': 'complete'}}))
        models = dict.fromkeys(['jev', 'clef', 'clef_flash', 'luna_decisions', 'pplx', 'kev', 'luna'])
        self.modules = {
            'protocol': types.SimpleNamespace(MODELS=models),
            'run': types.SimpleNamespace(PROJECT=self.root, run_cell=Mock()),
            'orchestrate': types.SimpleNamespace(PROJECT=self.root, verify=Mock(),
                                                 parallel=Mock(), stamp=Mock()),
            'billiger50': types.SimpleNamespace(PROJECT=self.root, verify=Mock(),
                                                BENCHMARK='billiger-de-50un')}

    def execute(self):
        with patch.dict('sys.modules', self.modules), \
                patch('subprocess.run', side_effect=AssertionError('No subprocess allowed')):
            baseline_only(self.root)

    def test_exact_models_b0_splits_and_supplement_only(self):
        self.execute()
        self.modules['orchestrate'].parallel.assert_called_once_with(
            list(self.modules['protocol'].MODELS), ['B0'], ['prompt', 'route', 'test'])
        calls = self.modules['run'].run_cell.call_args_list
        self.assertEqual(len(calls), 7)
        self.assertEqual({call.args[0] for call in calls}, set(self.modules['protocol'].MODELS))
        for call in calls:
            self.assertEqual(call.args[1:], ('B0', 'billiger-de-50un', 'test'))
            self.assertEqual(call.kwargs, {'workers':16})
        self.modules['orchestrate'].verify.assert_called_once()
        self.modules['billiger50'].verify.assert_called_once()
        self.assertEqual(self.modules['orchestrate'].stamp.call_args.args[1], 'complete')

    def test_all_data_verification_and_smoke_precede_requests(self):
        self.modules['billiger50'].verify.side_effect = ValueError('changed input')
        with self.assertRaisesRegex(ValueError, 'changed input'):
            self.execute()
        self.modules['billiger50'].verify.side_effect = None
        self.progress.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'smoke'):
            self.execute()
        self.modules['orchestrate'].parallel.assert_not_called()
        self.modules['run'].run_cell.assert_not_called()

    def test_failure_does_not_mark_complete_or_start_supplement(self):
        self.modules['orchestrate'].parallel.side_effect = RuntimeError('worker failed')
        with self.assertRaisesRegex(RuntimeError, 'worker failed'):
            self.execute()
        self.modules['run'].run_cell.assert_not_called()
        self.assertEqual(self.modules['orchestrate'].stamp.call_args.args[1], 'needs_attention')

    def test_wrong_workspace_rejected_before_requests(self):
        self.modules['run'].PROJECT = self.root/'wrong'
        with self.assertRaisesRegex(RuntimeError, 'workspace'):
            self.execute()
        self.modules['orchestrate'].parallel.assert_not_called()

    def test_overlap_and_smoke_prerequisites(self):
        smoke = [{'run_id':2, 'stage':'smoke', 'status':'complete'}]
        validate_history(smoke, 2, 'baseline')
        with self.assertRaisesRegex(ValueError, 'prerequisite'):
            validate_history([], 2, 'baseline')
        for other in ['core', 'supplement']:
            with self.assertRaisesRegex(ValueError, 'Overlapping'):
                validate_history(smoke+[{'run_id':2, 'stage':other, 'status':'complete'}], 2, 'baseline')
            with self.assertRaisesRegex(ValueError, 'Overlapping'):
                validate_history(smoke+[{'run_id':2, 'stage':'baseline', 'status':'complete'}], 2, other)


if __name__ == '__main__':
    unittest.main()
