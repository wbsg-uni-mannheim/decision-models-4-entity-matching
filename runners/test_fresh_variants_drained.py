import copy
import json
from pathlib import Path
import socket
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import expanded_variants
import protocol
import fresh_variants_drained as component
import test_authorized_recovery_200 as budget_fixture
from test_fresh_supplement_drained import Response


class VariantsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.original = budget_fixture.AuthorizedBudget().setup_root(self.root, 100, 3)
        self.target = self.root/'replications/run02'
        (self.target/'artifacts').mkdir(parents=True)
        self.cases = {}
        for b in [*component.CONTROLS, *component.VARIANTS]:
            cases = [{'benchmark': b, 'pair_id': str(i), 'label': 1,
                      'records': {'left': {'title': b+str(i)}, 'right': {'title': b+str(i)}}} for i in range(2)]
            for c in cases: c['input_hash'] = protocol.digest(c['records'])
            data = self.target/'data'/b; data.mkdir(parents=True)
            (data/'test.json').write_text(json.dumps(cases)); self.cases[b] = cases
        self.progress = self.target/'artifacts/progress.json'
        self.progress.write_text(json.dumps({'AP3_AP4_analysis': {'status': 'complete_automated'}}))
        self.inference = types.SimpleNamespace(**{**vars(self.original), 'PROJECT': self.target, 'key': lambda: 'synthetic'})
        self.manifest = {'models': protocol.MODELS, 'new_decisions_before_cache': 156387, 'conditions': {
            b: {'family': f, 'request_templates': {m: protocol.payload({'benchmark': b, 'records': {'left': {}, 'right': {}}}, m, 'B0')
                                                   for m in protocol.MODELS}} for b, f in component.VARIANTS.items()}}
        self.variants = types.SimpleNamespace(**{**vars(expanded_variants), 'PROJECT': self.target,
                                                'verify': Mock(return_value=self.manifest)})
        self.journal = self.target/'artifacts/variants_attempt'
        self.rid = 2
        self.bodies = []
        guard = patch.object(socket, 'socket', side_effect=AssertionError('Network forbidden'))
        guard.start(); self.addCleanup(guard.stop)
        self.seed_controls()
        self.bodies = []

    def http(self, request, **kwargs):
        body = json.loads(request.data); self.bodies.append(body)
        m = next(m for m in protocol.MODELS if protocol.MODELS[m]['id'] == body['model'])
        raw = {'model': body['model'], 'provider': protocol.MODELS[m]['provider'].split('/')[0], 'usage': {'cost': .001}}
        if m == 'luna': raw['choices'] = [{'finish_reason': 'stop', 'message': {'content': '{"match": true}'}}]
        else: raw['answers'] = {'match': {'type': 'noul', 'noul': .8}}
        return Response(json.dumps(raw).encode())

    def seed_controls(self):
        isolated = types.SimpleNamespace(**{**vars(self.inference), 'account': component.shared_account(self.inference.account, self.root)})
        _, cell, _ = component.make_wrapper(self.root, 'core').authorized_build(isolated, self.rid)
        with patch('urllib.request.urlopen', side_effect=self.http):
            for m in protocol.MODELS:
                for b in component.CONTROLS: cell(m, 'B0', b, 'test', workers=8)

    def execute(self):
        return component.run_variants(self.root, self.target, self.rid, self.inference, self.variants, protocol, self.journal)

    def test_full_matrix_real_requests_parser_account_and_unchanged_controls(self):
        controls = component.control_hashes(self.target, protocol); progress = self.progress.read_bytes()
        original_drain = component.drain_model_tasks
        invocations = []
        def drain(task, models, *args):
            def counted(m): invocations.append(m); return task(m)
            return original_drain(counted, models, *args)
        original_make = component.make_wrapper
        workers = []
        def factory(*args):
            wrapper = original_make(*args); build = wrapper.authorized_build
            def bound(*args):
                mod, cell, adapter = build(*args)
                def checked(*args, **kw):
                    workers.append(kw['workers']); return cell(*args, **kw)
                return mod, checked, adapter
            wrapper.authorized_build = bound; return wrapper
        with patch('urllib.request.urlopen', side_effect=self.http) as http, patch.object(component, 'drain_model_tasks', drain), patch.object(component, 'make_wrapper', factory):
            result = self.execute()
        self.assertTrue(result['inference_complete']); self.assertEqual(http.call_count, 70)
        self.assertCountEqual(invocations, list(protocol.MODELS)); self.assertEqual(workers, [8]*35)
        self.assertEqual(len(result['cells']), 35); self.assertEqual(len(result['controls']), 14)
        self.assertCountEqual(self.bodies, [protocol.payload(c, m, 'B0') for m in protocol.MODELS for b in component.VARIANTS for c in self.cases[b]])
        self.assertEqual(component.control_hashes(self.target, protocol), controls)
        self.assertEqual(self.progress.read_bytes(), progress)
        budget = json.loads((self.root/'artifacts/budget.json').read_text())
        self.assertAlmostEqual(budget['spent'], 100.098); self.assertAlmostEqual(budget['reserved'], 3)
        self.assertFalse(result['admission_granted']); self.assertFalse((self.target/'artifacts/results').exists())
        self.assertEqual(self.original.account.__defaults__, (0, 100))
        with self.assertRaises(ValueError): self.execute()

    def test_budget_zero_http_all_models_drained(self):
        p = self.root/'artifacts/budget.json'; d = json.loads(p.read_text()); d['spent'] = 196.99; p.write_text(json.dumps(d)); old = p.read_bytes()
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')): result = self.execute()
        self.assertFalse(result['inference_complete']); self.assertEqual(len(result['models']), 7)
        self.assertTrue(all(m['status'] == 'failed' for m in result['models']))
        self.assertEqual(sum(c['operations']['budget_rejections'] for c in result['cells']), 14)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']), 0); self.assertEqual(p.read_bytes(), old)

    def test_parser_failure_preserves_sibling_and_drains_other_models(self):
        count = 0
        def bad(request, **kwargs):
            nonlocal count
            response = self.http(request, **kwargs)
            if json.loads(request.data)['model'] == protocol.MODELS['jev']['id']:
                count += 1
                if count == 1:
                    raw = json.loads(response.getvalue()); raw['answers']['match']['noul'] = 2
                    return Response(json.dumps(raw).encode())
            return response
        with patch('urllib.request.urlopen', side_effect=bad): result = self.execute()
        self.assertFalse(result['inference_complete'])
        self.assertEqual(sum(m['status'] == 'complete' for m in result['models']), 6)
        self.assertEqual(sum(c['responses'] for c in result['cells']), 61)
        self.assertEqual(sum(c['failure_rows'] for c in result['cells']), 1)
        self.assertAlmostEqual(sum(c['known_usd'] for c in result['cells']), .062)

    def test_prerequisite_source_templates_and_empty_existing_journal(self):
        old = self.progress.read_bytes(); self.progress.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'prerequisite'): self.execute()
        self.progress.write_bytes(old)
        self.variants.__file__ = str(self.progress)
        with self.assertRaisesRegex(ValueError, 'source'): self.execute()
        self.variants.__file__ = expanded_variants.__file__
        original = copy.deepcopy(self.manifest)
        self.manifest['conditions']['wdc-050un']['request_templates']['jev'] = {}
        with self.assertRaisesRegex(ValueError, 'templates'): self.execute()
        self.manifest.clear(); self.manifest.update(original)
        p = self.target/'runs/quality/jev/B0/wdc-050un/test.jsonl'; p.parent.mkdir(parents=True); p.touch()
        with self.assertRaisesRegex(ValueError, 'absent journals'): self.execute()
        self.assertFalse(self.journal.exists())

    def test_controls_incomplete_foreign_run_and_unknown_operation_block_start(self):
        p = self.target/'runs/quality/jev/B0/wdc/test.jsonl'; old = p.read_bytes(); rows = [json.loads(x) for x in old.splitlines()]
        p.write_text(json.dumps(rows[0])+'\n')
        with self.assertRaisesRegex(ValueError, 'controls incomplete'): self.execute()
        for key, value in [('study_run_id', 1), ('request_hash', 'bad'), ('input_hash', 'bad'), ('prediction', 0)]:
            changed = copy.deepcopy(rows); changed[0][key] = value; p.write_text(''.join(json.dumps(x)+'\n' for x in changed))
            with self.assertRaises(ValueError): self.execute()
        p.write_bytes(old)
        row = {k: v for k, v in rows[0].items() if k not in ('attempts','prediction','probability')}
        row.update(error='Operational future exception: IncompleteRead', operational_stop=dict(
            previous_attempts_preserved=0, budget_reservation_rejected=False, outcome_unknown=True, phase='future', exception_type='IncompleteRead'))
        p.with_name('test.operations.jsonl').write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError, 'Unknown'): self.execute()
        self.assertFalse(self.journal.exists())

    def test_run_three_real_request_runid_and_post_raw_guards(self):
        # A separate fixture's controls are relabelled only for this synthetic test.
        old = self.target; self.target = self.root/'replications/run03'; old.rename(self.target)
        self.rid = 3; self.inference.PROJECT = self.target; self.variants.PROJECT = self.target
        self.journal = self.target/'artifacts/variants_attempt'
        for p in (self.target/'runs').rglob('test.jsonl'):
            rows = [json.loads(s) for s in p.read_text().splitlines()]
            for r in rows: r['study_run_id'] = 3
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with patch('urllib.request.urlopen', side_effect=self.http): result = self.execute()
        self.assertTrue(result['inference_complete'])
        p = self.target/'runs/quality/jev/B0/wdc-050un/test.jsonl'
        rows = [json.loads(s) for s in p.read_text().splitlines()]
        self.assertTrue(all(r['study_run_id'] == 3 for r in rows))
        rows[0]['study_run_id'] = 2; p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(ValueError): component.inventory(self.target, 3, protocol)


if __name__ == '__main__': unittest.main()
