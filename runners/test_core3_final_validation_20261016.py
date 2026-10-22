"""Independent final proof against full frozen reports and real mocked requests."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import protocol
import core3_final_validation_20261016 as final
import test_core3_timing_wrapper_20261016 as fixture_module


class FinalValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixture_module.TimingWrapperTests('test_full_frozen_analysis_in_private_workspace')
        f = cls.fixture; f.setUp()
        original = f.execute
        def execute(*args,**kwargs):
            selected = {m:{'variant':'P1','development_macro_match_f1':{v:2/3 for v in ('P1','P2','S1')}}
                        for m in protocol.ABLATION_MODELS}
            (f.target/'configs/selected_variants.json').write_text(json.dumps(selected))
            stop = f.root/'artifacts/verification/grouped_stop_reconciled_20261015.json'
            stop.parent.mkdir(); stop.write_text('{"resolved_reserve_events":{}}')
            f.d['controlled_stop_proof_sha256'] = final.sha(stop)
            f.d['cells'] = []
            for p in sorted((f.target/'runs/quality').rglob('*.jsonl')):
                model,variant,benchmark = p.parts[-4:-1]; split = p.stem
                f.d['cells'].append(dict(namespace='quality',model=model,variant=variant,benchmark=benchmark,split=split,
                    responses=20,attempts=20,unknown_attempts=0,failure_rows=0,known_usd=.02))
            f.d['protected_sha256_or_absent'] = {str(p.relative_to(f.target)):final.sha(p)
                for p in f.target.rglob('*') if p.is_file() and p.name != 'budget.json'}
            return original(*args,**kwargs)
        f.execute = execute
        f.test_full_frozen_analysis_in_private_workspace()
        cls.target = f.target; cls.attempt = f.target/final.ATTEMPT
        cls.addClassCleanup(f.doCleanups)

    def verify(self):
        f = self.fixture
        return final.verify_evidence(f.target,protocol,f.saved,f.d,self.attempt)

    @contextmanager
    def changed(self,path,mutate):
        old = path.read_bytes(); value = json.loads(old)
        path.write_text(json.dumps(mutate(value)))
        try:
            yield
        finally:
            path.write_bytes(old)

    @contextmanager
    def changed_jsonl(self,path,mutate):
        old = path.read_bytes(); records = [json.loads(x) for x in old.splitlines()]
        path.write_text(''.join(json.dumps(r)+'\n' for r in mutate(records)))
        try:
            yield
        finally:
            path.write_bytes(old)

    def test_whole_independent_raw_report_and_cost_proof(self):
        proof = self.verify()
        self.assertEqual(proof['quality_responses'],6560)
        self.assertEqual(proof['timing_responses'],2240)
        self.assertEqual(proof['new_per_call_account']['attempts'],420)
        self.assertAlmostEqual(proof['new_per_call_account']['known_usd'],.420)
        self.assertEqual(proof['all_timing_costs']['all_retained_timing_attempts'],2270)
        self.assertEqual(proof['all_timing_costs']['all_retained_unknown_attempts'],21)
        self.assertAlmostEqual(proof['all_timing_costs']['all_retained_known_usd'],2.249)
        self.assertEqual(proof['report']['cascade'],144)
        self.assertEqual(proof['figure_files'],15)
        self.assertFalse(proof['figures_visually_reviewed'])
        self.assertFalse((self.target/'artifacts/results').exists())

    def test_independent_parser_request_runid_and_operations(self):
        p = self.target/'runs'/final.REPLACEMENT/'kev/B0/billiger-de/prompt.jsonl'
        for field,value in [('study_run_id',True),('request_hash','wrong'),('prediction',0)]:
            with self.changed_jsonl(p,lambda rr: [{**rr[0],field:value}]+rr[1:]):
                with self.assertRaises(ValueError): self.verify()
        op = p.with_name('prompt.operations.jsonl'); op.write_text('{}\n')
        try:
            with self.assertRaises((ValueError,KeyError)):self.verify()
        finally:op.unlink()

    def test_independent_walltime_and_account_and_cell_end_guards(self):
        p = self.target/'runs'/final.REPLACEMENT/'kev/B0/billiger-de/prompt.executions.jsonl'
        for field,value in [('new_pairs',11),('workers',1),('wall_seconds',-1),('throughput_pairs_per_second',-1)]:
            with self.changed_jsonl(p,lambda rr: [{**rr[0],field:value}]):
                with self.assertRaises(ValueError):self.verify()
        p = self.attempt/'timing_measurements.json'
        def mutate(value):
            value['account']['known_usd'] += .1
            return value
        with self.changed(p,mutate):
            with self.assertRaisesRegex(ValueError,'account'):self.verify()
        p = self.attempt/'cells.jsonl'
        with self.changed_jsonl(p,lambda rr:rr[:-1]):
            with self.assertRaisesRegex(ValueError,'starts/ends'):self.verify()

    def test_history_protection_no_omissions_or_extra_journals(self):
        p = self.attempt/'prior.json'
        def mutate(value):
            value['protected_sha256_or_absent'].pop(next(iter(value['protected_sha256_or_absent'])))
            return value
        with self.changed(p,mutate):
            with self.assertRaisesRegex(ValueError,'protection scope'):self.verify()
        p = self.target/'runs/unplanned.jsonl';p.write_text('{}\n')
        try:
            with self.assertRaisesRegex(ValueError,'scope'):self.verify()
        finally:p.unlink()

    def test_report_tampering_even_with_updated_staging_hash(self):
        p = self.attempt/'analysis_staging/artifacts/results/main_metrics.json'
        original = p.read_bytes(); staged = self.attempt/'staged_result.json'; staged_bytes = staged.read_bytes()
        value = final.read(p);value[0]['f1'] = .123;p.write_text(json.dumps(value))
        data = final.read(staged);data['reports_sha256']['artifacts/results/main_metrics.json'] = final.sha(p)
        staged.write_text(json.dumps(data))
        try:
            with self.assertRaisesRegex(ValueError,'main_metrics/0/f1'):self.verify()
        finally:
            p.write_bytes(original);staged.write_bytes(staged_bytes)

    def test_nontrivial_confusion_calibration_bins_ties_and_risks(self):
        import analyze
        y = final.np.array([1,1,1,0,0,0]);pred = final.np.array([1,1,0,1,1,0])
        self.assertEqual(final.metrics(y,pred),dict(n=6,tp=2,fp=2,fn=1,tn=1,precision=.5,recall=2/3,f1=4/7,accuracy=.5))
        prob = final.np.array([.99,.5,.1,.9,.5,0.])
        expected = analyze.calibration(y,prob)
        expected['risk_coverage'] = analyze.risk_curve(y,pred,prob)
        final.equal(final.reliability(dict(y=y,p=pred,prob=prob)),expected)

    def test_real_entry_rejects_unadmitted_fixture_without_writes(self):
        before = {p:final.sha(p) for p in self.target.rglob('*') if p.is_file()}
        with self.assertRaises((ValueError,FileNotFoundError)):
            final.verify(self.fixture.root,self.fixture.admission,protocol)
        self.assertEqual(before,{p:final.sha(p) for p in before})

    def test_independent_graph_bootstrap_matches_nontrivial_frozen_reference(self):
        import analyze
        cases = [{'entity_ids':{'left':str(i//2),'right':str(i)}} for i in range(60)]
        y = final.np.array([i%2 for i in range(60)])
        pred = y.copy();pred[::7] = 1-pred[::7]
        control = y.copy();control[::5] = 1-control[::5]
        data = dict(cases=cases,y=y)
        final.equal(final.interval(data,pred,control),analyze.bootstrap_delta(cases,y,pred,control))
        # Equal-size groups preserve order; merged components exercise actual graph.
        cases[2]['entity_ids']['right'] = cases[0]['entity_ids']['right']
        final.equal(final.interval(data,pred,control),analyze.bootstrap_delta(cases,y,pred,control))


if __name__ == '__main__': unittest.main()
