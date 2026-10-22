import copy
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import expanded_cascades as cascade


def data(probabilities, labels):
    cases = [{'benchmark': 'abt-buy', 'pair_id': str(i), 'label': y, 'input_hash': str(i),
              'records': {'left': {'name': 'a'}, 'right': {'name': 'b'}},
              'entity_ids': {'left': 'shared', 'right': str(i)}} for i, y in enumerate(labels)]
    p = np.array(probabilities, dtype=float)
    y = np.array(labels)
    return {'cases': cases, 'y': y, 'prob': p, 'pred': (p >= .5).astype(int),
            'costs': np.full(len(cases), .01), 'unknown_cost_by_pair': np.zeros(len(cases), dtype=int),
            'unknown_cost_attempts': 0}


class CascadeTests(unittest.TestCase):
    def test_asymmetric_quota_and_label_free_routing(self):
        d = data([.1, .3, .45, .55, .7, .9], [0, 1, 0, 1, 0, 1])
        fallback = copy.deepcopy(d); fallback['pred'] = d['y']
        entry = cascade.asymmetric_selection(d, fallback, 3)
        mask = cascade.route(entry, d['cases'], d['prob'])
        self.assertEqual(int(mask.sum()), 3)
        changed = copy.deepcopy(d['cases'])
        for c in changed:
            c['label'] = 1-c['label']
        np.testing.assert_array_equal(cascade.route(entry, changed, d['prob']), mask)

    def test_disagreement_never_fills_agreements(self):
        d = data([.1, .2, .7, .9], [0, 0, 1, 1])
        peer = [.2, .3, .3, .8]
        entry = {'strategy': 'disagreement'}
        scores = cascade.score(entry, d['cases'], d['prob'], peer)
        entry['cutoff'] = cascade.cutoff(d['cases'], scores, 3, scores > 0)
        self.assertEqual(cascade.route(entry, d['cases'], d['prob'], peer).tolist(), [False, False, True, False])

    def test_oracle_matches_exhaustive_f1_optimum(self):
        d = data([.9, .9, .2, .1, .9, .1], [0, 1, 1, 0, 0, 1])
        fallback = np.array([0, 0, 1, 0, 0, 1])
        for quota in range(7):
            mask = cascade.oracle_mask(d['cases'], d['y'], d['pred'], fallback, quota)
            observed = cascade.analyze.metric(d['y'], np.where(mask, fallback, d['pred']))['f1']
            optimum = max(cascade.analyze.metric(d['y'], np.where(np.array(bits), fallback, d['pred']))['f1']
                          for bits in itertools.product([False, True], repeat=6) if sum(bits) <= quota)
            self.assertAlmostEqual(observed, optimum)

    def test_budget_refusal_retains_paid_primary(self):
        d = data([.49], [1])
        entry = {'model': 'jev', 'peer': None, 'strategy': 'symmetric', 'cutoff': {'score': 0., 'tie': '', 'selected_dev': 1}}
        primary = {'probability_match': .49, 'prediction': 0, 'attempts': [{'cost': .02, 'seconds': .2}]}
        with patch.object(cascade.run, 'request', side_effect=[primary, RuntimeError('Project spending ceiling reached')]):
            result = cascade.execute_chain(entry, d['cases'][0], 'not-a-real-key')
        self.assertTrue(result['failed'])
        self.assertEqual(result['calls'][0]['attempts'][0]['cost'], .02)
        self.assertIsNone(result['prediction'])

    def test_live_resume_and_interrupted_call_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = data([.49, .49], [1, 1])['cases']
            path = root/'data/abt-buy'; path.mkdir(parents=True)
            (path/'cascade_live.json').write_text(cascade.canonical(cases))
            directory = root/'runs/cascade_live/jev/symmetric/abt-buy'; directory.mkdir(parents=True)
            (directory/'call_journal.jsonl').write_text(cascade.canonical({'chain_id': 'interrupted', 'call_index': 0,
                'call': {'attempts': [{'cost': None, 'seconds': .1}]}})+'\n')
            (root/'artifacts/results/expanded_cascades').mkdir(parents=True)
            entry = {'model': 'jev', 'benchmark': 'abt-buy', 'peer': None, 'strategy': 'symmetric', 'status': 'frozen',
                     'fraction': .25, 'cutoff': {'score': 0., 'tie': '', 'selected_dev': 1}}
            def request(case, model, variant, credential):
                return {'probability_match': .49 if model == 'jev' else None, 'prediction': int(model == 'luna'),
                        'attempts': [{'cost': .01 if model == 'jev' else .02, 'seconds': .1}]}
            with patch.object(cascade, 'PROJECT', root), patch.object(cascade.run, 'key', return_value='fake'), \
                    patch.object(cascade.run, 'request', side_effect=request) as mocked:
                cascade.live_task('jev', [entry])
                self.assertEqual(mocked.call_count, 4)
                cascade.live_task('jev', [entry])
                self.assertEqual(mocked.call_count, 4)
                cascade.live_report([entry])
                row = cascade.read(root/'artifacts/results/expanded_cascades/live_latency.json')[0]
                self.assertAlmostEqual(row['known_cost_usd_including_failures'], .06)
                self.assertEqual(row['unknown_cost_attempts'], 1)
                self.assertEqual(row['interrupted_chains_with_journaled_calls'], 1)

    def test_full_offline_path_freezes_before_test_and_accounts_peer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = data([.1, .7, .4, .8, .2, .6, .3, .9], [0, 0, 1, 1, 0, 1, 1, 0])
            def load(m, variant, b, split):
                if split == 'test':
                    self.assertTrue((root/'configs/expanded_cascade_selection.json').exists())
                d = copy.deepcopy(original)
                if m == 'luna':
                    d['pred'] = d['y'].copy(); d['prob'] = None
                if m == 'clef':
                    d['prob'] = 1-d['prob']; d['pred'] = 1-d['pred']
                return d
            with patch.object(cascade, 'PROJECT', root), patch.object(cascade, 'BENCHMARKS', {'billiger-de': {}}), \
                    patch.object(cascade, 'MODELS', {'jev': {}, 'clef': {}, 'luna': {}}), \
                    patch.object(cascade, 'DECISIONS', ['jev']), patch.object(cascade, 'PEERS', {'jev': 'clef'}), \
                    patch.object(cascade.analyze, 'load', load):
                selected = cascade.select()
                self.assertEqual(len(selected), 15)
                cascade.report(selected)
                rows = cascade.read(root/'artifacts/results/expanded_cascades/metrics.json')
                self.assertEqual(len(rows), 30)
                row = next(r for r in rows if r['strategy'] == 'disagreement')
                self.assertAlmostEqual(row['known_cost_usd'], .16+.01*row['test_routes'])
                self.assertEqual(cascade.select(), selected)


if __name__ == '__main__':
    unittest.main()
