import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import peeters_run_export as ex
from export_run_metrics import check_response_origin, normalized_rows

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import protocol
from analyze import metric


class PeetersExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for d in ['configs', 'artifacts']:
            (self.root/d).mkdir()
        (self.root/ex.MANIFEST).write_text('{}')
        self.manifest_sha = ex.sha(self.root/ex.MANIFEST)
        (self.root/ex.EXECUTION).write_text(json.dumps({'payload_manifest_sha256': self.manifest_sha,
                                                     'source_sha256': {}}))
        self.event = {'run_id': 2, 'stage': 'peeters', 'target': str(self.root), 'slurm_job_id': 'fixture',
                      'budget_cap_usd': 100, 'parallel_policy_sha256': 'policy', 'admission_sha256': 'admission',
                      'execution_manifest_sha256': ex.sha(self.root/ex.EXECUTION)}
        self.journal = self.root/'artifacts/peeters_stage.jsonl'
        self.journal.write_text('\n'.join(json.dumps(dict(self.event, status=status, utc=utc))
            for status, utc in [('started', '2026-01-01T00:00:00+00:00'), ('complete', '2026-12-31T00:00:00+00:00')])+'\n')
        self.matrix = []
        self.paths = []
        for b in ['abt-buy', 'billiger-de-50un']:
            for split in (['prompt', 'route', 'test'] if b == 'abt-buy' else ['test']):
                records = {'left': {'name': 'Sony TV', 'price': 499, 'brand': None},
                           'right': {'name': 'Sony television', 'price': 499, 'brand': 'Sony'}}
                case = {'benchmark': b, 'pair_id': 'private-id', 'records': records,
                        'input_hash': protocol.digest(records), 'label': 1,
                        'entity_ids': {'left': 'l', 'right': 'r'}}
                data = self.root/'data'/b/(split+'.json')
                data.parent.mkdir(parents=True, exist_ok=True)
                data.write_text(json.dumps([case]))
                for m in ['jev', 'luna']:
                    for v in [*ex.control.ARMS, *(['B0'] if split == 'test' else [])]:
                        ns = 'quality' if v == 'B0' else 'peeters'
                        body = protocol.payload(case, m, v) if v == 'B0' else ex.control.payload(case, m, v, protocol)
                        spec = protocol.MODELS[m]
                        raw = {'model': spec['id'], 'provider': spec['provider'].split('/')[0], 'usage': {'cost': .01}}
                        if m == 'jev':
                            raw['answers'] = {'match': {'type': 'noul', 'noul': .8}}
                        else:
                            raw['choices'] = [{'finish_reason': 'stop', 'message': {'content': '{"match":true}'}}]
                        pred, prob = protocol.parse_response(m, raw)
                        row = {'pair_id': case['pair_id'], 'benchmark': b, 'model_alias': m, 'variant': v,
                               'input_hash': case['input_hash'], 'request_hash': protocol.digest(body),
                               'study_run_id': 2, 'peeters_payload_manifest_sha256': self.manifest_sha,
                               'completed_at': '2026-06-01T00:00:00+00:00', 'prediction': pred, 'probability_match': prob,
                               'attempts': [{'status': 200, 'cost': .01, 'seconds': 2, 'error': None, 'response': raw}]}
                        path = self.root/'runs'/ns/m/v/b/(split+'.jsonl')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(row)+'\n')
                        if ns == 'peeters':
                            self.matrix.append({'model': m, 'arm': v, 'benchmark': b, 'split': split})
                            self.paths.append(path)
        self.origin = lambda rows: check_response_origin(rows, 2, '2025-12-01T00:00:00+00:00')

    def export(self):
        provenance = []
        with patch.object(ex, 'verify_matrix', return_value=self.matrix):
            rows, summary = ex.export_peeters(self.root, self.root, protocol, metric,
                normalized_rows, self.origin, provenance, 2)
        return rows, summary, provenance

    def test_real_payload_parser_and_complete_export_order(self):
        seen = []
        original = ex.load_cell
        def load(*args, **kwargs):
            seen.append(args[5])
            return original(*args, **kwargs)
        with patch.object(ex, 'load_cell', side_effect=load):
            rows, summary, provenance = self.export()
        self.assertEqual(len(rows), 144)
        self.assertEqual(len(provenance), 20)
        self.assertEqual(summary['peeters_cells'], 16)
        self.assertEqual(seen[:8], ['prompt']*4+['route']*4)
        self.assertTrue(all(r['value'] == 0 for r in rows if r['metric'] in ['f1_delta', 'prediction_flip_rate']))

    def test_completion_and_provenance_gate_before_data(self):
        original = self.journal.read_text()
        for text in [original.splitlines()[0]+'\n', original.replace('complete', 'failed'),
                     original.replace('fixture', ''), original.replace('"run_id": 2', '"run_id": 3')]:
            self.journal.write_text(text)
            with patch.object(ex, 'load_cell', side_effect=AssertionError('No data read')), self.assertRaises(ValueError):
                self.export()

    def test_raw_prediction_run_manifest_and_usage_drift_rejected(self):
        p = self.paths[0]
        original = json.loads(p.read_text())
        variants = [dict(original, prediction=0), dict(original, study_run_id=3),
                    dict(original, peeters_payload_manifest_sha256='wrong'),
                    dict(original, completed_at='2027-01-01T00:00:00+00:00')]
        wrong = copy.deepcopy(original)
        wrong['attempts'][0]['response']['usage']['cost'] = .02
        variants.append(wrong)
        for row in variants:
            p.write_text(json.dumps(row)+'\n')
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.export()

    def test_failure_costs_unknowns_duration_retained_and_orphans_rejected(self):
        p = self.paths[0]
        row = json.loads(p.read_text())
        row.update(error='HTTP 529', attempts=[{'status': 529, 'error': 'HTTP 529', 'seconds': 3, 'cost': None, 'response': {}},
                                              {'status': 200, 'error': 'Invalid', 'seconds': 1, 'cost': .02, 'response': {}}])
        failure = p.with_name('prompt.failures.jsonl')
        failure.write_text(json.dumps(row)+'\n')
        rows, _, _ = self.export()
        values = {r['metric']: r['value'] for r in rows if r['condition'] == 'peeters/peeters_fields/prompt' and r['model'] == 'jev'}
        self.assertAlmostEqual(values['known_cost_usd_per_1000'], 30)
        self.assertEqual(values['unknown_attempts_per_1000'], 1000)
        self.assertEqual(values['request_attempt_seconds_per_pair'], 6)
        row['pair_id'] = 'orphan'
        failure.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError, 'outside frozen'):
            self.export()

    def test_incomplete_duplicate_and_invalid_costs_rejected(self):
        p = self.paths[0]
        original = p.read_text()
        row = json.loads(original)
        row['attempts'][0]['cost'] = -1
        for text in ['', original+original, json.dumps(row)+'\n']:
            p.write_text(text)
            with self.assertRaises(ValueError):
                self.export()

    def test_wrong_baseline_origin_and_source_drift_rejected(self):
        p = self.root/'runs/quality/jev/B0/abt-buy/test.jsonl'
        row = json.loads(p.read_text())
        row['study_run_id'] = 1
        p.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError, 'run response ID'):
            self.export()
        (self.root/ex.MANIFEST).write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'manifest drift'):
            self.export()


if __name__ == '__main__':
    unittest.main()
