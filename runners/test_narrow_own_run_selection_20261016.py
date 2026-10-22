"""Synthetic receipts/records only, zero HTTP. No empirical run selection."""
import copy
import fcntl
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import expanded_confirmation as confirmation
import expanded_protocol as expanded
import expanded_schema as schema
import protocol
import narrow_core_scope_20261016 as scope
import narrow_own_run_selection_20261016 as component


def cases(benchmark, n=2):
    result = []
    for i in range(n):
        c = dict(benchmark=benchmark, pair_id=str(i), label=i % 2,
                 records={'left': {'title': str(i)}, 'right': {'title': str(i)}})
        c['input_hash'] = protocol.digest(c['records'])
        result.append(c)
    return result


def response(case, model, variant, namespace, run_id=2):
    positive = case['label'] == 1
    raw = {'model': protocol.MODELS[model]['id'],
           'provider': protocol.MODELS[model]['provider'].split('/')[0], 'usage': {'cost': .001}}
    if namespace == 'schema':
        labels = schema.options(variant)
        selected = next(s for s in labels if s in ('Match', 'Same entity')) if positive else next(
            s for s in labels if s in ('No Match', 'Different entities'))
        if model == 'luna':
            raw['choices'] = [{'finish_reason': 'stop', 'message': {'content': json.dumps({'match': selected})}}]
        else:
            raw['answers'] = {'match': {'type': 'choice', 'choice': selected,
                                      'probabilities': {s: int(s == selected) for s in labels}}}
        prediction, probability = schema.parse(model, raw, variant)
        digest = schema.wire_hash(schema.payload(case, model, variant))
    else:
        if model == 'luna':
            raw['choices'] = [{'finish_reason': 'stop', 'message': {'content': json.dumps({'match': positive})}}]
        else:
            raw['answers'] = {'match': {'type': 'noul', 'noul': .8 if positive else .2}}
        prediction, probability = protocol.parse_response(model, raw)
        digest = protocol.digest(expanded.payload(case, model, variant))
    return dict(pair_id=case['pair_id'], study_run_id=run_id, model_alias=model,
                variant=variant, benchmark=case['benchmark'], input_hash=case['input_hash'],
                request_hash=digest, prediction=prediction, probability_match=probability,
                completed_at='synthetic', attempts=[dict(status=200, seconds=.1, cost=.001, response=raw)])


class NarrowSelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.target = self.root / 'replications/run02'
        (self.root / 'artifacts/operations').mkdir(parents=True)
        (self.target / 'artifacts').mkdir(parents=True)
        (self.target / 'configs').mkdir()
        (self.root / 'data').mkdir()
        (self.target / 'data').symlink_to(self.root / 'data')
        guard = patch.object(socket, 'socket', side_effect=AssertionError('No network'))
        guard.start()
        self.addCleanup(guard.stop)

    def store(self, model='jev', variant='X_P_minimal_0', ns='quality', benchmark='abt-buy', n=2):
        data = cases(benchmark, n)
        path = self.target / 'runs' / ns / model / variant / benchmark / 'prompt.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [response(c, model, variant, ns) for c in data]
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return data, path, rows

    def inspect(self, data, path, snapshot=None):
        ns, model, variant, b = path.relative_to(self.target / 'runs').parts[:4]
        return component.read_cell(self.target, 2, ns, model, variant, b, data,
                                   protocol, expanded, schema, snapshot or component.Snapshot(self.root))

    def test_all_narrow_cells_frozen_raw_parser_and_selector(self):
        matrix = set(scope.development_cells(2)) | {
            scope.cell(2, 'quality', m, 'B0', b, 'prompt') for m in scope.MODELS for b in scope.BENCHMARKS}
        snap, metrics = component.Snapshot(self.root), []
        for _, ns, model, variant, benchmark, _ in sorted(matrix):
            data, path, _ = self.store(model, variant, ns, benchmark)
            metrics.append(self.inspect(data, path, snap))
        selected = component.choose(2, metrics, confirmation)
        self.assertEqual(len(metrics), 1400)
        self.assertEqual(sum(c['responses'] for c in metrics), 2800)
        self.assertEqual({c['variant'] for c in selected['prompt_choices']}, {min(scope.PROMPTS)})
        self.assertEqual({c['variant'] for c in selected['binary_choices']}, {min(scope.BINARY)})
        self.assertEqual(len(selected['test_cells']), 224)
        snap.recheck()
        self.assertFalse(list((self.target / 'runs').rglob('test*')))

    def test_bad_raw_request_parser_run_and_missing_block(self):
        data, path, rows = self.store()
        for field, value in [('request_hash', 'bad'), ('study_run_id', True), ('study_run_id', 3),
                             ('variant', 'X_R_aligned'), ('prediction', 1), ('input_hash', 'bad')]:
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            path.write_text(''.join(json.dumps(r) + '\n' for r in changed))
            with self.assertRaises(ValueError):
                self.inspect(data, path)
        path.write_text(json.dumps(rows[0]) + '\n')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.inspect(data, path)

    def test_schema_wire_order_and_review_not_binary_winner(self):
        data, path, rows = self.store(variant='C_match_reverse', ns='schema')
        self.assertEqual(self.inspect(data, path)['f1'], 1)
        rows[0]['request_hash'] = protocol.digest(schema.payload(data[0], 'jev', 'C_match_reverse'))
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with self.assertRaises(ValueError):
            self.inspect(data, path)
        data, path, rows = self.store(variant='C_review', ns='schema')
        raw = rows[0]['attempts'][-1]['response']
        raw['answers']['match'].update(choice='Review', probabilities={'Match': 0, 'No Match': 0, 'Review': 1})
        rows[0].update(prediction=-1, probability_match=0.)
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.assertEqual(self.inspect(data, path)['review'], 1)

    def test_retained_unknown_attempt_counted_unrecorded_outcome_blocks(self):
        data, path, rows = self.store()
        rows[0]['attempts'].insert(0, dict(status=None, cost=None, seconds=.2, error='timeout'))
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        audit = self.inspect(data, path)
        self.assertEqual((audit['attempts'], audit['unknown_attempts']), (3, 1))
        operation = {k: rows[0][k] for k in ('pair_id', 'study_run_id', 'model_alias', 'variant',
                                           'benchmark', 'input_hash', 'request_hash', 'completed_at')}
        operation.update(error='unconfirmed store', operational_stop=dict(previous_attempts_preserved=0,
            budget_reservation_rejected=False, outcome_unknown=True, phase='future', exception_type='OSError'))
        path.with_name('prompt.operations.jsonl').write_text(json.dumps(operation) + '\n')
        with self.assertRaisesRegex(ValueError, 'unrecorded'):
            self.inspect(data, path)

    def metrics(self):
        matrix = set(scope.development_cells(2)) | {
            scope.cell(2, 'quality', m, 'B0', b, 'prompt') for m in scope.MODELS for b in scope.BENCHMARKS}
        return [dict(zip(('run_id', 'namespace', 'model', 'variant', 'benchmark', 'split'), c),
                     complete=True, f1=.5) for c in sorted(matrix)]

    def test_global_macro_local_winners_matrix_and_no_test_scores(self):
        metrics = self.metrics()
        for r in metrics:
            if r['variant'] == 'X_P_identity_0':
                r['f1'] = 1 if r['benchmark'] == 'abt-buy' else .49
        selection = component.choose(2, metrics, confirmation)
        self.assertEqual({r['variant'] for r in selection['prompt_choices'] if r['regime'] == 'global'},
                         {'X_P_identity_0'})
        self.assertEqual({r['variant'] for r in selection['prompt_choices']
                          if r['regime'] == 'per_benchmark' and r['benchmark'] == 'wdc'}, {min(scope.PROMPTS)})
        for bad in [metrics[:-1], metrics + [metrics[0]]]:
            with self.assertRaises(ValueError):
                component.choose(2, bad, confirmation)
        for key, value in [('split', 'test'), ('run_id', 3), ('f1', float('nan'))]:
            bad = copy.deepcopy(metrics)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                component.choose(2, bad, confirmation)

    def freeze_fixture(self):
        # Real freeze/locks/file durability/selector, mocked identity and cell
        # audit only. Raw audit has its own full 1400-cell synthetic test.
        manifest = {'benchmarks': {}}
        for b in scope.BENCHMARKS:
            rows = cases(b, 200)
            p = self.root / 'data' / b / 'prompt.json'
            p.parent.mkdir()
            p.write_text(json.dumps(rows))
            manifest['benchmarks'][b] = {'splits': {'prompt': {'sha256': protocol.digest(rows)}}}
        (self.target / 'configs/manifest.json').write_text(json.dumps(manifest))
        def identity(root, run, *args):
            self.assertEqual(run, 2)
            args[-1].read(self.target / 'configs/manifest.json')
            return self.target
        def cell(target, run, ns, m, v, b, *args):
            return dict(run_id=run, namespace=ns, model=m, variant=v, benchmark=b,
                        split='prompt', complete=True, f1=.5)
        for name, function in [('identity', identity), ('read_cell', cell)]:
            mock = patch.object(component, name, side_effect=function)
            mock.start()
            self.addCleanup(mock.stop)

    def freeze(self):
        return component.freeze(self.root, 2, protocol, expanded, schema, confirmation)

    def test_freeze_readback_repeat_and_sources_drift(self):
        self.freeze_fixture()
        result = self.freeze()
        self.assertFalse(result['admission_granted'])
        selected = component.require_frozen(self.root, 2)
        self.assertEqual(len(selected['test_cells']), 224)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.freeze()
        (self.root / 'data/abt-buy/prompt.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            component.require_frozen(self.root, 2)

    def test_partial_publication_no_receipt_no_retry(self):
        self.freeze_fixture()
        original = component.write_exclusive
        def fail(path, value):
            if path.name == 'sources.json':
                raise OSError('synthetic storage failure')
            return original(path, value)
        with patch.object(component, 'write_exclusive', side_effect=fail), self.assertRaises(OSError):
            self.freeze()
        output = self.target / component.OUTPUT
        self.assertTrue((output / 'selection.json').exists())
        self.assertFalse((output / 'receipt.json').exists())
        with self.assertRaises(ValueError):
            component.require_frozen(self.root, 2)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.freeze()

    def test_locks_raw_symlinks_and_absence_drift(self):
        for p in (self.root / 'artifacts/operations/repetition_writer.lock',
                  self.target / 'artifacts/parallel_writer.lock'):
            with p.open('a') as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError), component.locks(self.root, 2):
                    pass
        data, path, _ = self.store()
        snap = component.Snapshot(self.root)
        self.inspect(data, path, snap)
        path.with_name('prompt.failures.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            snap.recheck()
        old = path.with_name('old.jsonl')
        path.rename(old)
        path.symlink_to(old)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.inspect(data, path)

    def test_run1_and_unbound_source_identity_rejected_before_output(self):
        for bad in (1, True, '2', 2.0):
            with self.assertRaises(ValueError):
                component.freeze(self.root, bad, protocol, expanded, schema, confirmation)
        with self.assertRaises((ValueError, FileNotFoundError)):
            component.identity(self.root, 2, protocol, expanded, schema, confirmation,
                               component.Snapshot(self.root))
        self.assertFalse((self.target / component.OUTPUT).exists())


if __name__ == '__main__':
    unittest.main()
