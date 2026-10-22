import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import analyze
import expanded_confirmation as ec
import expanded_protocol as ep
import protocol
import fresh_confirmation_selection as freeze
from test_fresh_confirmation_development_drained import DevelopmentTests


class SelectionTests(unittest.TestCase):
    setUp = DevelopmentTests.setUp
    http = DevelopmentTests.http
    execute = DevelopmentTests.execute

    def args(self):
        return (self.root, self.target, 2, self.confirmation, ep, self.analysis, protocol)

    def complete(self):
        with patch('urllib.request.urlopen', side_effect=self.http):
            result = self.execute()
        self.assertTrue(result['development_complete'])

    def test_full_frozen_selector_equivalence_durable_readback_and_tamper_guards(self):
        self.complete()
        controls = {p: p.read_bytes() for p in (self.target/'runs').rglob('*') if p.is_file()}
        budget = self.root/'artifacts/budget.json'; old_budget = budget.read_bytes()
        # Execute the actual frozen select() independently into a reference
        # directory, reading the same synthetic own-run development cache.
        reference = self.root/'reference'; (reference/'configs').mkdir(parents=True)
        loader = types.FunctionType(analyze.load.__code__,
            {**analyze.load.__globals__, 'PROJECT': self.target, 'payload': ec.payload},
            argdefs=analyze.load.__defaults__)
        frozen_select = types.FunctionType(ec.select.__code__,
            {**ec.select.__globals__, 'PROJECT': reference,
             'analyze': types.SimpleNamespace(**{**vars(analyze), 'load': loader})})
        frozen_select()
        with patch.object(freeze, 'verify'), patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            receipt = freeze.freeze_development(*self.args(), self.journal/'models.jsonl')
            selected = freeze.verify_frozen(*self.args())
            self.assertEqual(receipt['selected_regime_rows'], 378)
            self.assertEqual(receipt['unique_test_cells'], 126)
            self.assertEqual(receipt['development_cells'], 1176)
            self.assertEqual(len(selected['selected']), 378)
            for name in freeze.OUTPUTS:
                self.assertEqual((self.target/name).read_bytes(), (reference/name).read_bytes())
            with self.assertRaisesRegex(ValueError, 'Existing freeze'):
                freeze.freeze_development(*self.args(), self.journal/'models.jsonl')
            p = self.target/freeze.OUTPUTS[0]; old=p.read_bytes(); p.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'differs'):
                freeze.verify_frozen(*self.args())
            p.write_bytes(old)
            # A receipt cannot be transplanted to another run or edited to
            # admit tests, even while all response files remain untouched.
            p=self.target/freeze.RECEIPT; old=p.read_bytes(); d=json.loads(old);d['run_id']=3;p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError, 'identity'):
                freeze.verify_frozen(*self.args())
            p.write_bytes(old)
            newop=self.target/'runs/quality/jev/B0/abt-buy/prompt.operations.jsonl'
            newop.write_text('')
            with self.assertRaisesRegex(ValueError, 'provenance changed'):
                freeze.verify_frozen(*self.args())
            newop.unlink()
            self.confirmation.SERIALIZATIONS=['B0']
            with self.assertRaisesRegex(ValueError, 'matrix'):
                freeze.verify_frozen(*self.args())
            self.confirmation.SERIALIZATIONS=ec.SERIALIZATIONS
        self.assertTrue(all(p.read_bytes()==raw for p,raw in controls.items()))
        self.assertEqual(budget.read_bytes(),old_budget)
        self.assertFalse(list((self.target/'runs').rglob('test*')))
        self.assertFalse((self.target/'artifacts/expanded_confirmation_complete.json').exists())
        self.assertFalse(receipt['admission_granted'])

    def test_budget_failed_development_and_missing_cache_cannot_freeze(self):
        p=self.root/'artifacts/budget.json';d=json.loads(p.read_text());d['spent']=196.99;p.write_text(json.dumps(d))
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            result=self.execute()
        self.assertFalse(result['development_complete'])
        with patch.object(freeze,'verify'):
            with self.assertRaisesRegex(ValueError,'All development model completions'):
                freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
            # Even fabricated all-complete model ends do not replace raw coverage.
            journal=self.journal/'models.jsonl';events=[json.loads(s) for s in journal.read_text().splitlines()]
            for e in events[1:]: e['status']='complete'
            journal.write_text(''.join(json.dumps(e)+'\n' for e in events))
            with self.assertRaisesRegex(ValueError,'Incomplete development'):
                freeze.freeze_development(*self.args(),journal)
            with self.assertRaisesRegex(ValueError,'No committed'):
                freeze.verify_frozen(*self.args())
        self.assertFalse((self.target/freeze.ATTEMPT).exists())
        self.assertFalse(any((self.target/p).exists() for p in freeze.OUTPUTS))

    def test_interrupted_publication_blocks_test_consumption_and_blind_retry(self):
        self.complete()
        original=freeze.durable_write; calls=[]
        def interrupted(path,raw):
            calls.append(path)
            if len(calls)==2: raise OSError('Synthetic storage failure')
            original(path,raw)
        with patch.object(freeze,'verify'), patch('urllib.request.urlopen',side_effect=AssertionError('No HTTP')):
            with patch.object(freeze,'durable_write',side_effect=interrupted):
                with self.assertRaisesRegex(OSError,'storage failure'):
                    freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
            self.assertTrue((self.target/freeze.OUTPUTS[0]).exists())
            self.assertFalse((self.target/freeze.RECEIPT).exists())
            with self.assertRaisesRegex(ValueError,'No committed'):
                freeze.verify_frozen(*self.args())
            with self.assertRaisesRegex(ValueError,'Existing freeze'):
                freeze.freeze_development(*self.args(),self.journal/'models.jsonl')
        self.assertFalse(list((self.target/'runs').rglob('test*')))


if __name__ == '__main__': unittest.main()
