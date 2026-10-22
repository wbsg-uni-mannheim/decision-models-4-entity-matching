import json
import math
import unittest
from unittest.mock import patch

import fresh_schema_selected_cost as costs
import fresh_schema_selection as freeze
import fresh_schema_test_drained as tests
from test_fresh_schema_test_drained import SchemaTestTests


class SelectedCostTests(unittest.TestCase):
    identify = SchemaTestTests.identify
    http = SchemaTestTests.http
    execute = SchemaTestTests.execute
    setUp = SchemaTestTests.setUp
    prepare = SchemaTestTests.prepare
    args = SchemaTestTests.args

    def test_full_run3_raw_gate_independent_matrix_costs_and_no_writes(self):
        self.prepare(3)
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with patch('urllib.request.urlopen', side_effect=AssertionError('No HTTP')):
            result = costs.review(*self.args())
            costs.require_headroom(self.root, result)
        self.assertEqual(result['pilot_cells_raw_verified'], 70)
        self.assertEqual(result['development_cells_raw_verified'], 280)
        self.assertEqual(result['b0_test_control_cells'], 63)
        self.assertEqual(result['unique_test_cells'], 126)
        self.assertEqual(result['missing_test_responses'], 252)
        self.assertEqual(result['valid_cached_test_responses'], 0)
        # Independent reconstruction from all 280 source files, not inventories
        # or calculate(). Pilot is verified separately and never priced again.
        sources = {}
        for path in (self.target/'runs/schema').rglob('prompt.jsonl'):
            m, v, b = path.parts[-4:-1]
            source = [json.loads(s) for s in path.read_text().splitlines()]
            fail = path.with_name('prompt.failures.jsonl')
            failures = [json.loads(s) for s in fail.read_text().splitlines()] if fail.exists() else []
            attempts = [a for r in source + failures for a in r['attempts']]
            sources[m, v, b] = (len(source), math.fsum(a['cost'] for a in attempts if a['cost'] is not None),
                                sum(a['cost'] is None for a in attempts))
        self.assertEqual(len(sources), 280)
        selected = json.loads((self.target/freeze.OUTPUTS[1]).read_text())
        independently = []
        for m, v, b in selected:
            n, known, unknown = sources[m, v, 'billiger-de' if b == 'billiger-de-50un' else b]
            pairs = json.loads((self.target/'data'/b/'test.json').read_text())
            missing = len({c['pair_id'] for c in pairs})
            independently.append((missing, known*missing/n, unknown*.05*missing/n))
        self.assertEqual(sum(r[0] for r in independently), result['missing_test_responses'])
        self.assertAlmostEqual(math.fsum(r[1] for r in independently), result['known_proxy_usd'])
        self.assertAlmostEqual(math.fsum(r[2] for r in independently), result['historical_unknown_scenario_usd'])
        self.assertAlmostEqual(result['known_proxy_usd'], .252)
        self.assertFalse(result['admission_granted'])
        self.assertFalse(result['reservation_created'])
        self.assertFalse((self.target/tests.ATTEMPT).exists())
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertAlmostEqual(json.loads(self.budget.read_text())['reserved'], 3)

    def test_run2_commit_controls_partial_tests_and_attempt_guards(self):
        self.prepare()
        path = self.target/freeze.RECEIPT
        old = path.read_bytes(); path.unlink()
        with self.assertRaisesRegex(ValueError, 'No committed'): costs.review(*self.args())
        path.write_bytes(old)
        path = self.target/'runs/quality/jev/B0/abt-buy/test.jsonl'
        old = path.read_bytes(); path.write_text('')
        with self.assertRaisesRegex(ValueError, 'B0 test controls'): costs.review(*self.args())
        rows = [json.loads(s) for s in old.splitlines()]; rows[0]['study_run_id'] = 3
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'study run'): costs.review(*self.args())
        path.write_bytes(old)
        extra = path.with_name('test.foreign.jsonl'); extra.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'Unreviewed'): costs.review(*self.args())
        extra.unlink()
        path = self.target/'runs/schema/jev/C_match/abt-buy/test.executions.jsonl'; path.write_text('')
        with self.assertRaisesRegex(ValueError, 'Existing schema test'): costs.review(*self.args())
        path.unlink()
        (self.target/tests.ATTEMPT).mkdir()
        with self.assertRaisesRegex(ValueError, 'Existing selected-test attempt'): costs.review(*self.args())
        (self.target/tests.ATTEMPT).rmdir()
        path = self.target/'runs/schema/jev/C_match/abt-buy/prompt.jsonl'
        path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'provenance changed'): costs.review(*self.args())

    def test_fresh_budget_and_drift_no_cached_fit_authority(self):
        self.prepare()
        result = costs.review(*self.args())
        budget = json.loads(self.budget.read_text()); budget['spent'] = 196.99
        self.budget.write_text(json.dumps(budget)); old = self.budget.read_bytes()
        result['scenario_fits_current_headroom'] = True
        result['budget_snapshot']['spent'] = 0
        with self.assertRaisesRegex(ValueError, 'headroom'): costs.require_headroom(self.root, result)
        self.assertEqual(old, self.budget.read_bytes())
        snapshot = costs.test_snapshot; calls = []
        def drift(*args):
            value = snapshot(*args); calls.append(1)
            if len(calls) > 1: value['changed'] = 'changed'
            return value
        with patch.object(costs, 'test_snapshot', side_effect=drift):
            with self.assertRaisesRegex(ValueError, 'sources changed'): costs.review(*self.args())
        self.assertFalse((self.target/tests.ATTEMPT).exists())

    def test_raw_ordered_wire_retry_failure_costs_and_operation_join(self):
        self.prepare()
        path = self.target/'runs/schema/jev/C_match/abt-buy/prompt.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        retry = {'status': 503, 'cost': None, 'seconds': 1, 'error': 'unavailable', 'response': {}}
        rows[0]['attempts'].insert(0, retry)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        failed = {**rows[0], 'error': 'failed', 'attempts': [{**retry, 'cost': .01}]}
        path.with_name('prompt.failures.jsonl').write_text(json.dumps(failed)+'\n')
        event = {k:v for k,v in rows[0].items() if k not in ('attempts','prediction','probability_match')}
        event.update(completed_at='synthetic', error='Project spending ceiling reached',
            operational_stop={'phase':'reserve','exception_type':'RuntimeError',
                'previous_attempts_preserved':0,'budget_reservation_rejected':True,'outcome_unknown':False})
        op = path.with_name('prompt.operations.jsonl'); op.write_text(json.dumps(event)+'\n')
        # This mutation deliberately invalidates the old freeze. Raw audit can
        # price retained attempts, but the full gate must still reject drift.
        dev = costs.development.inventory(self.target, self.rid, self.schema)
        cell = next(c for c in dev if (c['model'],c['variant'],c['benchmark']) == ('jev','C_match','abt-buy'))
        self.assertEqual(cell['attempts'], 4)
        self.assertEqual(cell['operations']['budget_rejections'], 1)
        self.assertAlmostEqual(cell['known_usd'], .012)
        self.assertEqual(cell['unknown_attempts'], 1)
        plan = costs.calculate(dev, [{'model':'jev','variant':'C_match','benchmark':'abt-buy',
            'expected':10,'responses':0,'missing':10,'missing_pair_ids_sha256':'synthetic'}])
        self.assertAlmostEqual(plan['known_proxy_usd'], .06)
        self.assertAlmostEqual(plan['historical_unknown_scenario_usd'], .25)
        with self.assertRaisesRegex(ValueError, 'provenance changed'): costs.review(*self.args())
        event['operational_stop'].update(phase='future', budget_reservation_rejected=False, outcome_unknown=True)
        event['error'] = 'worker outcome unknown'; op.write_text(json.dumps(event)+'\n')
        with self.assertRaisesRegex(ValueError, 'Unknown schema outcomes'):
            costs.development.inventory(self.target, self.rid, self.schema)


if __name__ == '__main__': unittest.main()
