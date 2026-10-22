"""Offline tests for the validation runner: mocked HTTP, temporary root, no network."""
import csv
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import drained_inference
import leased_budget
import prompt12
import protocol
import run
import valsel
import run_valsel_20261019 as rv

REAL = Path(__file__).resolve().parents[2]


def source_row(b, i, label, fields):
    row = {'pair_id': f'{b}#v{i}', 'label': label, 'id_left': f'L{i}', 'id_right': f'R{i}'}
    for s, word in (('left', 'gamma'), ('right', 'delta')):
        for f in fields:
            row[f+'_'+s] = f'{b} alpha beta {word} {i}' if f in ('title', 'name') else 'x'
    return row


class ValselRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = self.root = Path(self.temp.name)
        for rel in rv.SOURCES:
            if rel not in ('configs/manifest.json', valsel.SELECTION):
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(REAL/rel, root/rel)
        manifest = {'benchmarks': {}}
        for b, spec in protocol.BENCHMARKS.items():
            fields = spec['fields']
            test = [{'benchmark': b, 'pair_id': f'{b}#t', 'label': 1, 'records': {'left': {}, 'right': {}}, 'input_hash': 'test-input'}]
            rows = [source_row(b, i, i % 2, fields) for i in range(4)]
            rows.append({**rows[0], 'pair_id': f'{b}#dup'})  # same input as row 0: removed
            directory = root/'data/source'/b
            directory.mkdir(parents=True)
            if b == 'wdc':
                with gzip.open(directory/'wdcproducts80cc20rnd000un_valid_large.json.gz', 'wt') as f:
                    f.writelines(json.dumps(r)+'\n' for r in rows)
            else:
                with (directory/(b+'-valid.csv')).open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            (root/'data'/b).mkdir(parents=True)
            (root/'data'/b/'test.json').write_text(json.dumps(test))
            for name in ('prompt', 'route'):
                (root/'data'/b/(name+'.json')).write_text('[]')
            (root/'data'/b/'rules_train.json').write_text(json.dumps([{'pair_id': f'{b}#v1'}]))
            manifest['benchmarks'][b] = {'valid_unique_disjoint_inputs': 4, 'splits': {'test': {'n': 1}}}
        (root/'configs/manifest.json').write_text(json.dumps(manifest))
        selection = {'selected': {m: {b: 'B0' for b in protocol.BENCHMARKS} for m in prompt12.RUN_MODELS}}
        selection['selected']['jev'] = {b: 'R_rules' for b in protocol.BENCHMARKS}
        (root/valsel.SELECTION).write_text(json.dumps(selection))
        (root/'artifacts/operations').mkdir(parents=True, exist_ok=True)
        (root/'artifacts/budget.json').write_text(json.dumps({
            'spent': 385., 'reserved': 9.4, 'attempts': 0, 'unknown_cost_attempts': 0,
            'project_cap_usd': 420, 'budget_authority_sha256': 'inputs', 'foreign': {'keep': 1}}))
        self.calls = []
        rv.STOP.clear(); self.addCleanup(rv.STOP.clear)
        prompt12.register_models()
        mods = (run, protocol, valsel, drained_inference, leased_budget)
        for p in [patch.object(rv, 'modules', return_value=mods), patch.object(run, 'key', return_value='mock'),
                  patch.object(rv, 'other_project_jobs', return_value=[]), patch('signal.signal'),
                  patch.dict(os.environ, SLURM_JOB_ID='999'),
                  patch('socket.socket', side_effect=AssertionError('Real network forbidden'))]:
            p.start(); self.addCleanup(p.stop)
        self.addCleanup(valsel.SELECTED.clear)

    def approve(self, run_cap=10., cap=420.):
        (self.root/rv.APPROVAL).write_text(json.dumps({
            'approved_by': 'Aaron', 'scope': rv.SCOPE, 'project_cap_usd': cap, 'run_cap_usd': run_cap,
            'workers': 32, 'timeout_s': 30, 'quote': 'synthetic test approval'}))

    def http(self, request, timeout=None, **kw):
        body = json.loads(request.data); self.calls.append((body, timeout))
        data = {'model': body['model'], 'provider': body['provider']['only'][0].split('/')[0],
                'answers': {'match': {'type': 'noul', 'noul': .9}}, 'usage': {'cost': .001}}
        response = io.BytesIO(json.dumps(data).encode()); response.status = 200
        return response

    def test_export_filters_like_prepare_and_is_write_once(self):
        report = valsel.export_valid(self.root)
        self.assertEqual({r['n'] for r in report.values()}, {4})
        self.assertEqual({r['rules_train_in_valid'] for r in report.values()}, {1})
        rows = json.loads((self.root/'data/wdc/valid.json').read_text())
        self.assertNotIn('wdc#dup', {r['pair_id'] for r in rows})
        self.assertEqual(valsel.export_valid(self.root), report)

    def test_payload_is_selected_prompt_f0(self):
        valsel.load_selection(self.root)
        case = json.loads((self.root/'data/wdc/test.json').read_text())[0]
        self.assertEqual(valsel.payload(case, 'jev'), prompt12.payload(case, 'jev', 'R_rules'))
        self.assertEqual(valsel.payload(case, 'kev'), prompt12.payload(case, 'kev', 'B0'))
        self.assertNotIn('luna', valsel.RUN_MODELS)

    def test_full_run_on_valid_split(self):
        rv.freeze(self.root); self.approve()
        with self.assertRaisesRegex(ValueError, 'authorize-budget'):
            rv.execute(self.root)
        rv.authorize_budget(self.root)
        with patch('urllib.request.urlopen', side_effect=self.http):
            rv.execute(self.root)
        cells = 6*8
        self.assertEqual(len(rv.matrix(protocol, valsel)), cells)
        self.assertEqual(len(self.calls), cells*4)
        self.assertEqual({t for _, t in self.calls}, {30})
        self.assertTrue((self.root/'runs'/rv.NAMESPACE/'kev/F0/wdc/valid.jsonl').exists())
        self.assertFalse(list((self.root/'runs').rglob('test.jsonl')))
        self.assertTrue((self.root/rv.ATTEMPT/'job-999/complete.json').exists())
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 385+cells*4*.001)
        self.assertAlmostEqual(budget['reserved'], 9.4)
        spec = json.loads((self.root/rv.MANIFEST).read_text())
        self.assertEqual((spec['split'], spec['decisions']), ('valid', cells*4))

    def test_run_cap_hold_fits_32_workers_per_model(self):
        # 6 models x 32 open requests must fit a 10 USD run cap after 5 USD spent.
        self.assertLess(5+rv.RUN_HOLD*6*32, 10)

    def test_failed_pair_then_resume(self):
        rv.freeze(self.root); self.approve(); rv.authorize_budget(self.root)
        state = {'n': 0}

        def flaky(request, timeout=None, **kw):
            if json.loads(request.data)['model'] == 'typesafe/jev-1.13' and state['n'] == 0:
                state['n'] += 1
                import urllib.error
                raise urllib.error.HTTPError(request.full_url, 520, 'unknown', {}, io.BytesIO(b'{}'))
            return self.http(request, timeout)
        with patch('urllib.request.urlopen', side_effect=flaky):
            with self.assertRaisesRegex(ValueError, 'model stopped'):
                rv.execute(self.root)
        self.assertEqual(len(rv.archive_failures(self.root)), 1)
        with patch.dict(os.environ, SLURM_JOB_ID='1000'), patch('urllib.request.urlopen', side_effect=self.http):
            rv.execute(self.root, resume=True)
        self.assertTrue((self.root/rv.ATTEMPT/'job-1000/complete.json').exists())


if __name__ == '__main__':
    unittest.main()
