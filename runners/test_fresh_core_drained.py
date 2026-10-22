import copy
import json
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import orchestrate
import protocol
import run
import fresh_core_drained as core


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root/'replications/run02'
        for p in ('artifacts', 'configs', 'data', 'runs'):
            (self.target/p).mkdir(parents=True)
        (self.root/'artifacts').mkdir()
        self.status = {'smoke': {'status': 'complete'}, 'AP1_baseline': {'status': 'needs_attention'}}
        (self.target/'artifacts/progress.json').write_text(json.dumps(self.status))
        manifest = {'benchmarks': {}}
        for benchmark in protocol.BENCHMARKS:
            cases = [{'pair_id': 'a#b', 'label': 1, 'benchmark': benchmark,
                      'records': {'left': {'name': 'x'}, 'right': {'name': 'x'}}}]
            cases[0]['input_hash'] = protocol.digest(cases[0]['records'])
            (self.target/'data'/benchmark).mkdir()
            manifest['benchmarks'][benchmark] = {'splits': {}}
            for split in ('prompt', 'route', 'test'):
                (self.target/'data'/benchmark/(split+'.json')).write_text(json.dumps(cases))
                manifest['benchmarks'][benchmark]['splits'][split] = {'sha256': protocol.digest(cases)}
        (self.target/'configs/manifest.json').write_text(json.dumps(manifest))
        scope = {**vars(orchestrate), 'PROJECT': self.target}
        self.orchestration = types.SimpleNamespace(**scope)
        for name in ('verify', 'stamp', 'select'):
            f = getattr(orchestrate, name)
            setattr(self.orchestration, name, types.FunctionType(f.__code__, scope, argdefs=f.__defaults__))
        self.inference = types.SimpleNamespace(**{**vars(run), 'PROJECT': self.target, 'key': lambda: 'fake'})
        self.calls = []
        sock = patch.object(socket, 'socket', side_effect=AssertionError('Network forbidden'))
        sock.start(); self.addCleanup(sock.stop)

    def cell(self, model, variant, benchmark, split, **kw):
        self.calls.append((model, variant, benchmark, split, kw))
        d = self.target/'runs'/kw.get('namespace', 'quality')/model/variant/benchmark
        d.mkdir(parents=True, exist_ok=True)
        (d/(split+'.jsonl')).write_text(json.dumps({'pair_id': 'a#b', 'prediction': 1})+'\n')

    def test_full_phase_matrix_frozen_selection_and_timing_order(self):
        initial_account = run.account
        journal = self.target/'artifacts/drained'
        with patch.object(core, 'build', return_value=(None, self.cell, {'admission_granted': False})), \
                patch.object(core.subprocess, 'run') as analysis:
            result = core.run_phases(self.root, self.target, 2, self.inference, self.orchestration, protocol, journal)
        self.assertIs(run.account, initial_account)
        self.assertFalse(result['outer_core_completion_written'])
        self.assertFalse((self.target/'artifacts/stage_execution.jsonl').exists())
        selection = json.loads((self.target/'configs/selected_variants.json').read_text())
        self.assertEqual(set(selection), set(protocol.ABLATION_MODELS))
        self.assertTrue(all(r['variant'] == 'P1' for r in selection.values()))
        b = len(protocol.BENCHMARKS); m = len(protocol.MODELS); a = len(protocol.ABLATION_MODELS)
        self.assertEqual(len(self.calls), 3*m*b+4*a*b+a*b+2*m*b)
        development = self.calls[3*m*b:3*m*b+4*a*b]
        self.assertEqual({c[1] for c in development}, {'P1', 'P2', 'S1', 'O1'})
        self.assertTrue(all(c[3] == 'prompt' for c in development))
        confirmation = self.calls[3*m*b+4*a*b:3*m*b+5*a*b]
        self.assertTrue(all(c[1] == 'P1' and c[3] == 'test' for c in confirmation))
        timing = self.calls[-2*m*b:]
        expected = [(model, benchmark, workers, namespace) for model in protocol.MODELS
                    for benchmark in protocol.BENCHMARKS
                    for workers, namespace in [(1, 'timing_serial'), (8, 'timing_concurrent8')]]
        self.assertEqual([(c[0], c[2], c[4]['workers'], c[4]['namespace']) for c in timing], expected)
        self.assertTrue(all(c[4]['limit'] == 20 for c in timing))
        analysis.assert_called_once()
        for phase, count in [('AP1_baseline', m), ('AP2_development', a), ('AP2_confirmation', a)]:
            rows = [json.loads(s) for s in (journal/(phase+'.jsonl')).read_text().splitlines()]
            self.assertEqual(sum(r['event'] == 'model_end' for r in rows), count)
            self.assertEqual(rows[-1]['status'], 'complete')
        with self.assertRaisesRegex(ValueError, 'initial core failure'):
            core.run_phases(self.root, self.target, 2, self.inference, self.orchestration, protocol, journal)

    def test_real_budget_stop_drains_models_and_blocks_selection_test_and_analysis(self):
        budget = self.root/'artifacts/budget.json'
        budget.write_text(json.dumps({'spent': 97., 'reserved': 3., 'attempts': 0, 'unknown_cost_attempts': 0}))
        before = budget.read_bytes()
        with patch.object(core.subprocess, 'run') as analysis, \
                patch.object(self.orchestration, 'select', side_effect=AssertionError('No selection')):
            with self.assertRaisesRegex(RuntimeError, 'draining every model'):
                core.run_phases(self.root, self.target, 2, self.inference, self.orchestration, protocol,
                                self.target/'artifacts/drained')
        self.assertEqual(budget.read_bytes(), before)
        analysis.assert_not_called()
        rows = [json.loads(s) for s in (self.target/'artifacts/drained/AP1_baseline.jsonl').read_text().splitlines()]
        self.assertEqual(sum(r['event'] == 'model_end' for r in rows), len(protocol.MODELS))
        ops = list((self.target/'runs').rglob('*.operations.jsonl'))
        self.assertEqual(len(ops), len(protocol.MODELS))
        self.assertTrue(all(json.loads(p.read_text())['study_run_id'] == 2 for p in ops))
        self.assertFalse(list((self.target/'runs').rglob('*.failures.jsonl')))
        self.assertFalse((self.target/'configs/selected_variants.json').exists())

    def test_wrong_run_workspace_and_noninitial_phase_refused(self):
        for run_id in (1, True, 4):
            with self.assertRaises(ValueError):
                core.run_phases(self.root, self.target, run_id, self.inference, self.orchestration, protocol,
                                self.target/'artifacts/drained')
        status = copy.deepcopy(self.status); status['AP2_development'] = {'status': 'needs_attention'}
        (self.target/'artifacts/progress.json').write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError, 'initial core failure'):
            core.run_phases(self.root, self.target, 2, self.inference, self.orchestration, protocol,
                            self.target/'artifacts/drained')



if __name__ == '__main__':
    unittest.main()
