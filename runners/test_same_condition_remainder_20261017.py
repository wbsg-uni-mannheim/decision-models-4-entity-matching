"""Synthetic missing-only continuation tests; no real HTTP or scheduler changes."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import same_condition_remainder_20261017 as missing
import same_condition_launch_20261016 as old
import test_same_condition_repetition_runtime_20261016 as fixture


class MissingTests(unittest.TestCase):
    def setup_failed(self):
        f = fixture.RepetitionTests(); f.setUp(); self.addCleanup(f.doCleanups); f.small()
        f.prepare()
        def broken(request, **kwargs):
            response = f.http(request, **kwargs); raw = json.loads(response.getvalue())
            if json.loads(request.data)['model'] == fixture.protocol.MODELS['jev']['id']:
                raw.pop('answers')
            return fixture.Response(json.dumps(raw).encode())
        with patch('urllib.request.urlopen', side_effect=broken):
            f.execute()
        f.reconcile()
        rt = missing.runtime
        cells = rt.inventory(f.target, f.rid, f.cases, fixture.protocol, fixture.expanded, fixture.schema)
        journals = {str(p.relative_to(f.target)): rt.prefix(p) for p in rt.paths(f.target, rt.matrix(f.rid))}
        result = dict(run_id=f.rid, remaining=sum(c['missing'] for c in cells), cells=cells, journals=journals,
                      protected_sha256=f.protected, attempt_sha256={})
        proof_path = f.root / 'terminal-run.json'; old.save(proof_path, result)
        terminal = dict(run_proofs={'2': {'path': 'terminal-run.json', 'sha256': old.sha(proof_path)}})
        mocked = patch.object(missing, 'terminal', return_value=terminal); mocked.start(); self.addCleanup(mocked.stop)
        journal = f.target / missing.ATTEMPT / 'inference'; journal.parent.mkdir(parents=True)
        modules = (f.inference, fixture.protocol, fixture.expanded, fixture.schema)
        prior = missing.capture_prior(f.root, f.target, 2, *modules, journal)
        p = journal.parent / 'prior.json'; old.save(p, prior)
        credit = fixture.Credit(f.root, 'new-missing-fixture', dict(prior_sha256=old.sha(p), run_id=2,
                                stage=rt.STAGE, journal_dir=str(journal.relative_to(f.target))))
        return f, result, modules, journal, p, credit

    def test_paid_failures_preserved_only_missing_sent_and_reconciled(self):
        f, result, modules, journal, prior, credit = self.setup_failed()
        before = {p: p.read_bytes() for p in (f.target / 'runs').rglob('*.jsonl')}
        f.bodies = []
        with patch('urllib.request.urlopen', side_effect=f.http) as http:
            executed = missing.execute_runtime(f.root, f.target, 2, *modules, prior, old.sha(prior), journal, credit)
        self.assertTrue(executed['inference_complete'])
        self.assertEqual(http.call_count, result['remaining'])
        for path, raw in before.items():
            self.assertTrue(path.read_bytes().startswith(raw))
        for path, raw in before.items():
            if path.name.endswith('.failures.jsonl'):
                self.assertEqual(path.read_bytes(), raw)
        recon = missing.runtime.reconcile(f.root, f.target, 2, *modules, prior, old.sha(prior), journal, credit)
        self.assertEqual(recon['attempts'], result['remaining']); self.assertTrue(credit.closed)
        expected = [fn(c, m, v) for cell in result['cells'] for _, ns, m, v, b, _ in [cell['cell']]
                    for c in f.cases[b] if c['pair_id'] in cell['missing_pair_ids']
                    for fn in [fixture.expanded.payload if ns == 'quality' else fixture.schema.payload]]
        self.assertCountEqual(f.bodies, expected)
        with self.assertRaises(ValueError):
            missing.execute_runtime(f.root, f.target, 2, *modules, prior, old.sha(prior), journal, credit)

    def test_raw_drift_blocks_before_http(self):
        f, result, modules, journal, prior, credit = self.setup_failed()
        path = next((f.target / 'runs').rglob('prompt.jsonl'))
        path.write_bytes(path.read_bytes() + path.read_bytes().splitlines(keepends=True)[0])
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')), self.assertRaises(ValueError):
            missing.execute_runtime(f.root, f.target, 2, *modules, prior, old.sha(prior), journal, credit)
        self.assertEqual(credit.attempts, 0); self.assertFalse(credit.closed)

    def test_missing_list_cannot_expand_or_drop(self):
        f, result, modules, journal, prior, credit = self.setup_failed()
        corrupted = copy.deepcopy(result); corrupted['cells'][0]['missing_pair_ids'].append('foreign')
        (f.root / 'terminal-run.json').write_text(json.dumps(corrupted))
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')), self.assertRaisesRegex(ValueError, 'drift'):
            missing.execute_runtime(f.root, f.target, 2, *modules, prior, old.sha(prior), journal, credit)
        self.assertEqual(credit.attempts, 0)

    def test_two_full_separate_gates(self):
        f, result, modules, journal, prior, credit = self.setup_failed()
        (f.root / 'configs').mkdir(exist_ok=True); (f.root / 'artifacts/verification').mkdir(exist_ok=True)
        old.save(f.root / missing.ADMISSION, {'admitted': True})
        old.save(f.root / missing.PREP, {'priors': {'2': {'run': 2}, '3': {'run': 3}}})
        with patch.object(old, 'locks'), patch.object(missing, 'gate', return_value={}) as gate, patch.object(missing, 'protect'), patch.object(missing, 'child_inventory', side_effect=lambda root, rid: {'run': rid}) as inv:
            missing.preflight(f.root, f.root / 'held.json')
            missing.preflight(f.root, f.root / 'prerelease.json')
        self.assertEqual(gate.call_count, 2); self.assertEqual(inv.call_count, 4)

    def test_sequential_processes_and_no_retry(self):
        from types import SimpleNamespace
        with patch.dict('os.environ', SLURM_JOB_ID='fixture'), patch.object(old, 'read', return_value={'slurm_job_id': 'fixture'}), patch.object(missing.subprocess, 'run', side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]) as run:
            with self.assertRaises(ValueError): missing.launch(Path('/fixture'))
        self.assertEqual(run.call_count, 2)
        self.assertEqual([c.args[0][-1] for c in run.call_args_list], ['2', '3'])


if __name__ == '__main__':
    unittest.main()
