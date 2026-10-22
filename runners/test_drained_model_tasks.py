import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import expanded_variants
import protocol
from drained_inference import drain_model_tasks


class ModelDrainTests(unittest.TestCase):
    def test_actual_frozen_main_exposes_only_first_model_exception(self):
        models = list(protocol.MODELS)
        barrier = threading.Barrier(len(models))
        completed = set()
        def task(model):
            barrier.wait(timeout=3)
            completed.add(model)
            if model == 'jev':
                raise ValueError('first model failure')
            if model in ('pplx', 'kev', 'luna'):
                raise RuntimeError('Project spending ceiling reached')
        with patch('socket.socket', side_effect=AssertionError('No network')), \
                patch.object(expanded_variants, 'verify'), patch.object(expanded_variants, 'forecast'), \
                patch.object(expanded_variants, 'report') as report, \
                patch.object(expanded_variants, 'model_task', side_effect=task), \
                patch.object(sys, 'argv', ['expanded_variants.py', 'run']):
            with self.assertRaisesRegex(ValueError, 'first model failure'):
                expanded_variants.main()
        self.assertEqual(completed, set(models))
        report.assert_not_called()

    def test_new_collector_journals_all_models_once_and_refuses_journal_reuse(self):
        models = list(protocol.MODELS)
        barrier = threading.Barrier(len(models))
        calls = []
        def task(model):
            calls.append(model)
            barrier.wait(timeout=3)
            if model in ('jev', 'pplx', 'kev', 'luna'):
                raise RuntimeError('Project spending ceiling reached')
        with tempfile.TemporaryDirectory() as folder, \
                patch('socket.socket', side_effect=AssertionError('No network')):
            path = Path(folder)/'models.jsonl'
            result = drain_model_tasks(task, models, path, protocol.canonical)
            self.assertEqual(len(result), 7)
            self.assertEqual(sum(r['status'] == 'failed' for r in result), 4)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows], ['start']+['model_end']*7+['end'])
            self.assertEqual(rows[-1]['models_drained'], 7)
            self.assertEqual(rows[-1]['status'], 'failed')
            with self.assertRaises(FileExistsError):
                drain_model_tasks(task, models, path, protocol.canonical)
            self.assertCountEqual(calls, models)


if __name__ == '__main__':
    unittest.main()
