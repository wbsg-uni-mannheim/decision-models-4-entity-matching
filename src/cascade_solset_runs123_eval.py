"""Tables IX, X and XI over three runs with the cascade threshold t tuned on our Sol labels (Aaron, 5 Nov: "Ja, Sol-Labels").
Read-only, no inference. Copy of later_tables_runs123_eval.py; everything else unchanged.

--tuning valid: t as before (valsel_eval.tune_risk on the pooled official validation sets without rules_train, B0
promptval_run01_20261027); reproduces later_tables_runs123_20261104.json (asserted, as there).
--tuning sol200 / sol2000: t = valsel_eval.tune_risk on the pooled Sol-labelled 200 or 2,000 set (B0 runs/solset_b0_run01_20261105,
one run, configs/solset_b0_cases_20261105.json, without the pairs whose records equal a test pair), i.e. the selective
risk of the kept pairs against the Sol labels. Applied unchanged to every test run as before.
The RULESFS10_2000 cascade of Table XI: --tuning valid keeps its validation t; sol200 / sol2000 tune its t on its own
outputs on the validation split of the Sol 2,000 set (runs/solvalid_rulesfs_run01_20261105, 3,200 pairs minus those whose
records equal a test pair; the only Sol-labelled outputs of that configuration, so both sol modes use them; the Astra rule
agents saw this split as their held-out check). lab_tuning: its pairs and t, and for comparison the B0 t of PPLX v1.2 on
the same pairs (b0_t_same_pairs).
targets: Jev 1.13, PPLX v1.2, Clef and Luna Dec. at the target risks 0.25, 0.5 and 1 % (t from the same tuning source),
per run, over the runs, with the run-mean and multi-bootstrap of each cascade against the model alone (significance,
table 'targets'). tuning: pooled tuning pairs, t and the tuning-set deferral rate and selective risk per model and risk.
Outputs: --out (layout of later_tables_runs123_eval.py plus targets and tuning), and next to it the extracts in the layout
of cascade_runs123_20261104.json (table9, its significance, targets) and of cascade_extras_runs123_20261104.json
(majority_alone, deferred, luna_cascade; cascade_extras_runs123_eval.py with the same t).

Original description:
Tables IX (cascade), X (Sol-labeled pairs) and XI (literature) of main.pdf over three runs. Read-only, no inference.

Decision-model runs: B0 test Runs 1-3 as thresholds_runs123_eval.py (Run 1 runs/quality, PPLX v1.2
prompt12_run01_20261019; Runs 2/3 replications/run0{2,3}/runs/quality, PPLX v1.2 pplx12_b0_run0{2,3}_20261021).
Partner runs: Sol, Opus, Argon Run 1 = ensemble_full (cascade_b0_valsel_eval.ensemble), Runs 2/3 =
runs/sota_run0{2,3}_20261103 (as sota_runs123_eval.py; F1 checked against sota_runs123_20261104.json). Ensemble: majority
of the three members of the same run. Cascade rule as cascade_b0_valsel_eval.py: t = valsel_eval.tune_risk at a target
risk of 0.5 % on the pooled official validation sets without rules_train (one validation run: B0 promptval_run01_20261027,
RULESFS10_2000 labeledch_valid_run01_20261101), applied unchanged to every test run; a pair is deferred when
min(p, 1-p) > t, kept pairs are a match when p >= 0.5. Pairing: decision-model run r with partner run r (r = 1, 2, 3).
all_pairings: mean F1 over all nine combinations of decision-model run and partner run, as a check of that pairing.
Cost at list prices: decision model on all test pairs (journal attempt costs, unpriced attempts count zero and are
counted) plus the partner on the deferred pairs. Partner list cost per pair: Run 1 as ensemble_full; Sol Runs 2/3 twice
the row cost (Batch API); Argon Runs 2/3 twice the attempt cost (flex); Opus Runs 2/3 twice the batch cost of its result
file, spread over the rows of that file in proportion to prompt tokens + 5 x completion tokens (the rows carry no cost).
Table IX: Jev 1.13, PPLX v1.2, Clef, Luna Dec. at 0.5 %: alone, + Sol, + ensemble per run (t, deferred share, selective
risk, F1 per benchmark, mean F1, errors, cost), mean and sample SD over the runs. Run 1 must reproduce paper
sources/cascade_b0.json (A_b0, 0.005; asserted).
Table X: B0 (journal predictions) of PPLX v1.2 and Jev 1.13, handwritten rules (Run 1 prompt12_run01_20261019; Runs 2/3
runs/prompt12hw_run0{2,3}_20261104 where complete, else 1 run), Sol, the PPLX B0 cascades (rows of
Table IX; Run 1 = sources/labeled_cascade.json B0 0.005, asserted), Sol-labeled LR (deterministic) and Ditto (3 seeds), Astra
rules (runs 1-3 of the three Astra repetitions), 10 examples (FS10, 3 runs), rules + 10 examples with run-1 rules
(RULESFS10 fixed, 3 runs) and with the rules of Astra repetition r in run r (RULESFS10 varying). Per row: F1 per
benchmark per run, mean and SD; mean F1 per run, mean and SD; difference to the 3-run mean of B0 of the same decision
model (matchers, Sol and cascades: PPLX v1.2 B0).
Table XI: PPLX v1.2 with B0 and with RULESFS10_2000 (run-1 rules, 3 test runs), alone and with the ensemble at 0.5 %; per
benchmark per run, mean and SD; how often each column exceeds both published values (per run and on the 3-run means).
Run 1 reproduces sources/cascade_b0.json literature_pplx_v12_at_0.005 and sources/labeled_cascade.json global (asserted).
Significance: paired bootstrap on the run mean (10,000 replicates, seed 0, pairs drawn per benchmark in protocol order,
the same pairs for every arm and run; fixed_ci95) and the multi-bootstrap of multibootstrap_eval.py (multi_ci95; runs drawn
with replacement, rng seed 1). Run draws joint for a cascade against the same model alone (same probabilities of run r),
independent against B0 of other API calls; a 1-run side stays fixed. stored_ci95: the interval the paper shows today.
one_run_costs: list cost of two more runs for the configurations with one run.

    .venv/bin/python src/later_tables_runs123_eval.py --paper-sources <paper sources/> --out artifacts/results/later_tables_runs123_20261104.json
"""
import argparse
import csv
import glob
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cascade_b0_valsel_eval as cv
import labeled_valsel_cascade_eval as lv
import protocol
import valsel_eval
from inputs_b0_eval import journal
from multibootstrap_eval import Data as BootData, RunDraws, compare
from prompt12_runs123_eval import path as prompt_path
from thresholds_runs123_eval import test_path as b0_path
import solset_b0

ROOT = cv.ROOT
RES = ROOT/'artifacts/results'
LC = ROOT/'runs/labeled_chapter_20261028'
B = list(protocol.BENCHMARKS)
RUNS = (1, 2, 3)
X = .005
TARGETS = (.0025, .005, .01)
TUNING = ['valid']
SOLSET_NS = 'runs/solset_b0_run01_20261105'
SOLVALID_NS = 'runs/solvalid_rulesfs_run01_20261105'
DM4 = ['jev', 'pplx_v12', 'clef', 'luna_decisions']
PARTNERS = ('sol61', 'majority')
SIZES, SEEDS = ['200', '2000'], [42, 52, 62]
SERIES = {'RULES': ('RULES', {r: f'runs/labeledch_rules_run0{r}_20261028' for r in RUNS}),
          'FS10': ('FS10', {r: f'runs/labeledch_fewshot2_run0{r}_20261031' for r in RUNS}),
          'RULESFS10_fixed': ('RULESFS10', {r: f'runs/labeledch_rulesfs_run0{r}_20261031' for r in RUNS}),
          'RULESFS10_varying': ('RULESFS10', {1: 'runs/labeledch_rulesfs_run01_20261031', 2: 'runs/labeledch_rulesfsrep_run02_20261104',
                                              3: 'runs/labeledch_rulesfsrep_run03_20261104'})}


def hw_path(m, b, r):
    return prompt_path(m, 'R_handwritten', b, 1) if r == 1 else ROOT/f'runs/prompt12hw_run0{r}_20261104'/m/'R_handwritten'/b/'test.jsonl'


def hw_complete(m, r):
    """A later handwritten run counts only when every benchmark journal holds all test pairs."""
    for b in B:
        p = hw_path(m, b, r)
        if not p.exists() or len({x['pair_id'] for x in cv.lines(p)}) != len(json.loads((ROOT/'data'/b/'test.json').read_text())):
            return False
    return True


HW_RUNS = {}


def tuning_pairs(m):
    """(p, y, b) of the tuning source: official validation (gold) or the pooled Sol set (Sol labels)."""
    if TUNING[0] == 'valid':
        return cv.validation(m, lambda b: 'B0', cv.VALID_NS)
    sets, out = solset_b0.tuning_sets(ROOT)[TUNING[0][3:]], []
    for b in B:
        rows = json.loads((ROOT/'data'/b/'solset.json').read_text())
        label = {c['pair_id']: c['label'] for c in rows}
        test_inputs = {c['input_hash'] for c in json.loads((ROOT/'data'/b/'test.json').read_text())}
        skip = {c['pair_id'] for c in rows if c['input_hash'] in test_inputs}
        got = {x['pair_id']: x['probability_match'] for x in cv.lines(ROOT/SOLSET_NS/m/'B0'/b/'solset.jsonl')}
        assert got.keys() == label.keys() and all(p is not None for p in got.values()), (m, b)
        out += [(got[i], label[i], b) for i in sets[b] if i not in skip]
    return out


def solvalid_pairs(m, ns, prompt):
    """(p, Sol label, b) on the validation split of the Sol 2,000 set (data/<b>/solvalid2000.json), test-equal records left out."""
    out = []
    for b in B:
        rows = json.loads((ROOT/'data'/b/'solvalid2000.json').read_text())
        test_inputs = {c['input_hash'] for c in json.loads((ROOT/'data'/b/'test.json').read_text())}
        got = {x['pair_id']: x['probability_match'] for x in cv.lines(ROOT/ns/m/prompt/b/(ns_split(ns)+'.jsonl'))}
        assert {c['pair_id'] for c in rows} <= got.keys() and all(got[c['pair_id']] is not None for c in rows), (m, b)
        out += [(got[c['pair_id']], c['label'], b) for c in rows if c['input_hash'] not in test_inputs]
    return out


def ns_split(ns):
    return 'solvalid2000' if ns == SOLVALID_NS else 'solset'


def ms(xs):
    xs = list(xs)
    return {'mean': statistics.mean(xs), 'sd': statistics.stdev(xs) if len(xs) > 1 else None, 'runs': len(xs)}


def opus_rows(path, summary):
    """Valid rows of one Opus batch result file with their list cost (twice the file's batch cost, spread by tokens)."""
    rows = cv.lines(path)
    w = [((x.get('response') or {}).get('usage') or {}) for x in rows]
    w = [u.get('prompt_tokens', 0)+5*u.get('completion_tokens', 0) for u in w]
    paid = json.loads(Path(summary).read_text())['cost_usd']
    return [(x, 2*paid*wi/sum(w)) for x, wi in zip(rows, w)]


def partner_run(r, ids, dec1):
    """{member: {b: (predictions, list USD per pair)}} for one run."""
    out = {}
    for m in cv.MEMBERS:
        out[m] = {}
        if r == 1:
            for b in B:
                out[m][b] = (np.array([dec1[m][(b, i)][0] for i in ids[b]], dtype=np.int8), np.array([dec1[m][(b, i)][1] for i in ids[b]]))
            continue
        ns = ROOT/f'runs/sota_run0{r}_20261103'
        if m == 'argon4':
            for b in B:
                rows = {x['pair_id']: x for x in cv.lines(ns/'argon4/E0'/b/'test.jsonl')}
                assert rows.keys() == set(ids[b]), (m, r, b)
                out[m][b] = (np.array([rows[i]['prediction'] for i in ids[b]], dtype=np.int8),
                             np.array([2*cv.attempt_cost(rows[i]) for i in ids[b]]))
            continue
        got = {}
        for path in sorted(glob.glob(str(ns/m/'results*.jsonl'))):
            if m == 'opus55':
                summary = path.replace('results', 'summary').replace('.jsonl', '.json')
                rows = opus_rows(path, summary)
            else:
                rows = [(x, 2*(x.get('cost') or 0.)) for x in cv.lines(path)]
            for x, usd in rows:
                if 'prediction' in x and x['prediction'] in (0, 1):
                    got.setdefault((x['benchmark'], x['pair_id']), (x['prediction'], usd))
        for b in B:
            assert all((b, i) in got for i in ids[b]), (m, r, b)
            out[m][b] = (np.array([got[(b, i)][0] for i in ids[b]], dtype=np.int8), np.array([got[(b, i)][1] for i in ids[b]]))
    ens = {b: ((sum(out[m][b][0] for m in cv.MEMBERS) >= 2).astype(np.int8), sum(out[m][b][1] for m in cv.MEMBERS)) for b in B}
    return {'sol61': out['sol61'], 'majority': ens}, {m: sum(float(out[m][b][1].sum()) for b in B) for m in cv.MEMBERS}


def test_rows(path_of, ids):
    """Per benchmark probabilities, journal predictions, decision-model cost and unpriced pairs."""
    out = {}
    for b in B:
        rows = {r['pair_id']: r for r in cv.lines(path_of(b))}
        assert rows.keys() == set(ids[b]) and all(r['probability_match'] is not None for r in rows.values()), (path_of(b))
        rs = [rows[i] for i in ids[b]]
        out[b] = {'p': np.array([r['probability_match'] for r in rs]), 'pred': np.array([r['prediction'] for r in rs], dtype=np.int8),
                  'usd': sum(cv.attempt_cost(r) for r in rs), 'unpriced': sum(cv.unpriced(r) for r in rs)}
    return out


def cascade(y, test, partner, t):
    """One test run of one decision model with one partner run: predictions per configuration and the level's numbers."""
    preds = {c: {} for c in ('alone',)+PARTNERS}
    n = kept = kept_wrong = 0
    usd = {c: sum(test[b]['usd'] for b in B) for c in preds}
    for b in B:
        p = test[b]['p']
        alone = (p >= .5).astype(np.int8)
        d = np.minimum(p, 1-p) > t
        preds['alone'][b] = alone
        for c in PARTNERS:
            preds[c][b] = np.where(d, partner[c][b][0], alone).astype(np.int8)
            usd[c] += float(partner[c][b][1][d].sum())
        n += len(p); kept += int((~d).sum()); kept_wrong += int((alone != y[b])[~d].sum())
    res = {'t': t, 'deferred': n-kept, 'deferred_share': (n-kept)/n, 'selective_risk': kept_wrong/kept,
           'decision_usd': sum(test[b]['usd'] for b in B), 'decision_pairs_unpriced': sum(test[b]['unpriced'] for b in B), 'configs': {}}
    for c, per in preds.items():
        pb = {b: cv.f1(y[b], v) for b, v in per.items()}
        res['configs'][c] = {'per_benchmark': pb, 'mean_f1': statistics.mean(pb.values()),
                             'errors': sum(int((v != y[b]).sum()) for b, v in per.items()), 'usd_list': usd[c],
                             'usd_per_1m_pairs': usd[c]/n*1e6}
    return res, preds


def over_runs(per_run):
    """Mean and SD over runs of a cascade level (per configuration: mean F1, per benchmark F1, errors, cost)."""
    rs = list(per_run.values())
    return {'deferred_share': ms(x['deferred_share'] for x in rs), 'selective_risk': ms(x['selective_risk'] for x in rs),
            'configs': {c: {'mean_f1': ms(100*x['configs'][c]['mean_f1'] for x in rs),
                            'per_benchmark': {b: ms(100*x['configs'][c]['per_benchmark'][b] for x in rs) for b in B},
                            'errors': ms(x['configs'][c]['errors'] for x in rs), 'usd_list': ms(x['configs'][c]['usd_list'] for x in rs)}
                        for c in rs[0]['configs']}}


def row(D, key_of, runs, bs=B):
    """F1 per benchmark per run (fractions), mean and SD; mean F1 per run (points), mean and SD."""
    per = {b: {str(r): D.point[(key_of(r), b)] for r in runs} for b in bs}
    means = {str(r): 100*statistics.mean(per[b][str(r)] for b in bs) for r in runs}
    return {'runs': len(runs), 'f1': per, 'per_benchmark': {b: ms(100*v for v in per[b].values()) for b in bs},
            'mean_f1_per_run': means, 'mean_f1': ms(means.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tuning', choices=['valid', 'sol200', 'sol2000'], required=True)
    ap.add_argument('--skip-series', nargs='*', default=[], help='test only: leave out Table X series whose runs are not complete')
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    for k in args.skip_series:
        SERIES.pop(k)
    TUNING[0] = args.tuning
    ref_tables = json.loads((RES/'later_tables_runs123_20261104.json').read_text())
    for m in ('pplx_v12', 'jev'):
        HW_RUNS[m] = tuple(r for r in RUNS if r == 1 or hw_complete(m, r))
    print('handwritten runs', HW_RUNS, flush=True)
    src = {k: json.loads((args.paper_sources/f'{k}.json').read_text()) for k in
           ('cascade_b0', 'labeled_cascade', 'labeled_v2', 'labeled_facts', 'literature', 'prompt_selection')}
    sota = json.loads((RES/'sota_runs123_20261104.json').read_text())
    D = BootData()
    ids, y = D.ids, D.y
    dec1 = cv.ensemble()
    partner, partner_totals = {}, {}
    for r in RUNS:
        partner[r], partner_totals[str(r)] = partner_run(r, ids, dec1)
        for b in B:
            assert abs(cv.f1(y[b], partner[r]['sol61'][b][0])-sota['f1']['sol61'][b][str(r)]) < 1e-12, (r, b)
    print('partners loaded', partner_totals, flush=True)
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': 10_000,
           'units': 'f1 and per_run F1 as fractions; mean/sd blocks of F1 and differences in F1 points; USD at list prices',
           'partner_list_usd_all_pairs': partner_totals, 'table9': {}, 'table10': {}, 'table11': {}, 'significance': [],
           'one_run_costs': {}, 'tuning_source': args.tuning, 'targets': {}, 'tuning': {},
           'lab_t_source': 'official validation outputs' if args.tuning == 'valid' else 'its outputs on the validation split of the Sol 2,000 set'}
    specs = []

    # Table IX
    tests = {}
    for m in DM4:
        val = tuning_pairs(m)
        t = valsel_eval.tune_risk(val, X)
        out['tuning'][m] = {'pairs': len(val), 'positive': sum(v[1] for v in val), 'levels': {}}
        for x in TARGETS:
            tx = valsel_eval.tune_risk(val, x)
            va = valsel_eval.accept(val, tx)
            out['tuning'][m]['levels'][str(x)] = {'t': tx, 'deferral_rate': 1-va['coverage'], 'selective_risk': va['risk'],
                                                  'allowed_errors_at_full_coverage': x*len(val)}
        per_run, combos = {}, {}
        for r in RUNS:
            tests[(m, r)] = test_rows(lambda b: b0_path(m, b, r), ids)
            res, preds = cascade(y, tests[(m, r)], partner[r], t)
            per_run[str(r)] = res
            for c, per in preds.items():
                for b in B:
                    D.add(('casc', m, c, r), b, per[b])
            for s in RUNS:  # all pairings of decision-model run r and partner run s
                res2, _ = cascade(y, tests[(m, r)], partner[s], t)
                for c in PARTNERS:
                    combos.setdefault(c, []).append(100*res2['configs'][c]['mean_f1'])
        old = src['cascade_b0']['variants']['A_b0'][m]['levels'][str(X)]
        one = per_run['1']
        if args.tuning == 'valid':
            assert abs(one['t']-old['t']) < 1e-12 and one['deferred'] == old['test']['deferred'], m
            assert abs(one['selective_risk']-old['test']['selective_risk']) < 1e-12, m
            for c in ('alone',)+PARTNERS:
                o = old['configs'][c]
                assert abs(one['configs'][c]['mean_f1']-o['mean_f1']) < 1e-12 and one['configs'][c]['errors'] == o['errors'], (m, c)
                assert abs(one['configs'][c]['usd_list']-o['usd_list']) < 1e-9, (m, c)
        out['targets'][m] = {}
        for x in TARGETS:
            tx = valsel_eval.tune_risk(val, x)
            pr = {}
            for r in RUNS:
                pr[str(r)], px = cascade(y, tests[(m, r)], partner[r], tx)
                for c, per in px.items():
                    for b in B:
                        D.add(('tcasc', m, x, c, r), b, per[b])
            out['targets'][m][str(x)] = {'t': tx, 'per_run': pr, 'over_runs': over_runs(pr)}
            if x == X:
                assert out['targets'][m][str(x)]['per_run'] == per_run, m
            if args.tuning == 'valid':
                lv0 = src['cascade_b0']['variants']['A_b0'][m]['levels'].get(str(x))
                if lv0:
                    assert abs(tx-lv0['t']) < 1e-12 and pr['1']['deferred'] == lv0['test']['deferred'], (m, x)
            for c in PARTNERS:
                spec = {b: ((('dm', m), [('tcasc', m, x, c, r) for r in RUNS]), (('dm', m), [('tcasc', m, x, 'alone', r) for r in RUNS])) for b in B}
                specs.append(('targets', {'model': m, 'target_risk': x, 'comparison': f'{c} vs alone'}, spec, None))
        out['table9'][m] = {'t': t, 'per_run': per_run, 'over_runs': over_runs(per_run),
                            'all_pairings_mean_f1': {c: ms(v) for c, v in combos.items()}}
        print('IX', m, round(t, 4), {c: (round(v['mean_f1']['mean'], 2), round(v['mean_f1']['sd'], 2)) for c, v in out['table9'][m]['over_runs']['configs'].items()},
              'deferred', [round(100*x['deferred_share'], 2) for x in per_run.values()], flush=True)
        cas = src['cascade_b0']['bootstrap']['cascade_vs_alone']['A_b0'][m][str(X)]
        for c in PARTNERS:
            spec = {b: ((('dm', m), [('casc', m, c, r) for r in RUNS]), (('dm', m), [('casc', m, 'alone', r) for r in RUNS])) for b in B}
            specs.append(('table9', {'model': m, 'comparison': f'{c} vs alone'}, spec, cas[c]['ci95'] if args.tuning == 'valid' else None))

    # Table X
    t10 = out['table10']
    for m in ('pplx_v12', 'jev'):
        for r in RUNS:
            for b in B:
                D.add(('b0j', m, r), b, journal(b0_path(m, b, r), ids[b]))
        for b in B:
            for r in HW_RUNS[m]:
                D.add(('hw', m, r), b, journal(hw_path(m, b, r), ids[b]))
        out['one_run_costs'][f'{m} handwritten rules'] = {
            'run1_list_usd': sum(cv.attempt_cost(x) for b in B for x in cv.lines(prompt_path(m, 'R_handwritten', b, 1)))}
        out['one_run_costs'][f'{m} handwritten rules']['two_more_runs_usd'] = 2*out['one_run_costs'][f'{m} handwritten rules']['run1_list_usd']
        for name, (prefix, runs) in SERIES.items():
            for s in SIZES:
                for r, ns in runs.items():
                    for b in B:
                        D.add(('lab', name, m, s, r), b, journal(ROOT/ns/m/f'{prefix}_{s}'/b/'test.jsonl', ids[b]))
    for r in RUNS:
        for b in B:
            D.add(('sol', r), b, partner[r]['sol61'][b][0])
    lr = json.loads((LC/'lr/test_predictions.json').read_text())
    for s in SIZES:
        for b in B:
            D.add(('lr', s), b, np.array([lr[b][s]['predictions'][i] for i in ids[b]], dtype=np.int8))
            for seed in SEEDS:
                summ = json.loads((LC/f'ditto/output/{b}_set{s}_seed{seed}/summary.json').read_text())['results'][0]
                with open(Path(summ['run_dir'])/'predictions.csv') as f:
                    pred = {x['pair_id']: int(x['pred']) for x in csv.DictReader(f)}
                D.add(('ditto', s, seed), b, np.array([pred[i] for i in ids[b]], dtype=np.int8))

    # Table XI: RULESFS10_2000 of PPLX v1.2 with its own threshold (validation, or its Sol-valid outputs in the sol modes)
    pooled = [(float(rw['probability_match']), int(l), b) for b in B for rw, l in lv.valid_rows('RULESFS10_2000', b)]
    t_lab = valsel_eval.tune_risk(pooled, X)
    if args.tuning != 'valid':
        sv = solvalid_pairs('pplx_v12', SOLVALID_NS, 'RULESFS10_2000')
        b0sv = solvalid_pairs('pplx_v12', SOLSET_NS, 'B0')
        t_lab = valsel_eval.tune_risk(sv, X)
        out['lab_tuning'] = {'pairs': len(sv), 'positive': sum(v[1] for v in sv), 't': t_lab,
                             'deferral_rate': 1-valsel_eval.accept(sv, t_lab)['coverage'], 'selective_risk': valsel_eval.accept(sv, t_lab)['risk'],
                             'b0_t_same_pairs': valsel_eval.tune_risk(b0sv, X),
                             'levels': {str(x): valsel_eval.tune_risk(sv, x) for x in TARGETS}}
    lab_runs = {}
    for r in RUNS:
        test = test_rows(lambda b: ROOT/SERIES['RULESFS10_fixed'][1][r]/'pplx_v12/RULESFS10_2000'/b/'test.jsonl', ids)
        lab_runs[str(r)], preds = cascade(y, test, partner[r], t_lab)
        for c, per in preds.items():
            for b in B:
                D.add(('labcasc', c, r), b, per[b])
    g = src['labeled_cascade']['levels']['global'][str(X)]
    assert args.tuning != 'valid' or abs(t_lab-g['t']) < 1e-12, t_lab
    for c in (('alone', 'majority', 'sol61') if args.tuning == 'valid' else ('alone',)):
        assert all(abs(lab_runs['1']['configs'][c]['per_benchmark'][b]-g[c]['per_benchmark'][b]) < 1e-12 for b in B), c
    lb0 = src['labeled_cascade']['levels']['B0'][str(X)]
    lit0 = src['cascade_b0']['literature_pplx_v12_at_0.005']['A_b0']
    for c in (('alone', 'majority', 'sol61') if args.tuning == 'valid' else ('alone',)):
        assert all(abs(out['table9']['pplx_v12']['per_run']['1']['configs'][c]['per_benchmark'][b]-lb0[c]['per_benchmark'][b]) < 1e-12 for b in B), c
    for c in (('alone', 'majority') if args.tuning == 'valid' else ('alone',)):
        assert all(abs(out['table9']['pplx_v12']['per_run']['1']['configs'][c]['per_benchmark'][b]-lit0[c][b]) < 1e-12 for b in B), c

    D.compute()
    draws = RunDraws()

    # Table X rows
    for m in ('pplx_v12', 'jev'):
        t10[f'{m} B0'] = row(D, lambda r: ('b0j', m, r), RUNS)
        assert all(abs(t10[f'{m} B0']['f1'][b]['1']-src['labeled_v2']['rules'][m]['200']['cells'][b]['f1_b0']) < 1e-12 for b in B), m
        t10[f'{m} handwritten rules'] = row(D, lambda r: ('hw', m, r), HW_RUNS[m])
        assert all(abs(t10[f'{m} handwritten rules']['f1'][b]['1']-src['prompt_selection']['f1'][m]['R_handwritten'][b]) < 1e-12 for b in B), m
    t10['Sol'] = row(D, lambda r: ('sol', r), RUNS)
    for c in PARTNERS:
        t10[f'pplx_v12 + cascade {c}'] = row(D, lambda r: ('casc', 'pplx_v12', c, r), RUNS)
    for s in SIZES:
        t10[f'LR {s}'] = row(D, lambda r: ('lr', s), (1,))
        t10[f'Ditto {s}'] = row(D, lambda r: ('ditto', s, SEEDS[r-1]), RUNS)
    for m in ('pplx_v12', 'jev'):
        for name in SERIES:
            for s in SIZES:
                t10[f'{m} {name} {s}'] = row(D, lambda r: ('lab', name, m, s, r), RUNS)
    for k, v in t10.items():
        ref = 'jev B0' if k.startswith('jev') else 'pplx_v12 B0'
        v['diff_to_b0_3run_mean'] = v['mean_f1']['mean']-t10[ref]['mean_f1']['mean']
        v['reference'] = ref
    b0ref = {m: (('b0j', m), [('b0j', m, r) for r in RUNS]) for m in ('pplx_v12', 'jev')}
    facts = src['labeled_facts']['vs_pplx_b0']
    for m in ('pplx_v12', 'jev'):
        for name in SERIES:
            for s in SIZES:
                spec = {b: ((('lab', name, m, s), [('lab', name, m, s, r) for r in RUNS]), b0ref[m]) for b in B}
                if name == 'RULES':
                    stored = src['labeled_v2']['rules'][m][s]['mean']['ci95']
                elif name == 'FS10':
                    stored = [100*x for x in src['labeled_v2']['fewshot2'][m][f'FS10_{s}']['mean']['ci95']]
                else:
                    stored = [100*x for x in src['labeled_v2']['rulesfs'][m][f'RULESFS10_{s}']['mean']['ci95']]
                specs.append(('table10', {'model': m, 'row': f'{name} {s}', 'reference': 'B0 3-run mean'}, spec, stored))
        spec = {b: ((('hw', m), [('hw', m, r) for r in HW_RUNS[m]]), b0ref[m]) for b in B}
        specs.append(('table10', {'model': m, 'row': f'handwritten rules ({len(HW_RUNS[m])} run(s))', 'reference': 'B0 3-run mean'}, spec, None))
    for s in SIZES:
        specs.append(('table10', {'model': 'matcher', 'row': f'LR {s}', 'reference': 'PPLX B0 3-run mean'},
                      {b: ((('lr', s), [('lr', s)]), b0ref['pplx_v12']) for b in B}, facts['lr'][s]['ci95']))
        specs.append(('table10', {'model': 'matcher', 'row': f'Ditto {s}', 'reference': 'PPLX B0 3-run mean'},
                      {b: ((('ditto', s), [('ditto', s, x) for x in SEEDS]), b0ref['pplx_v12']) for b in B}, facts['ditto'][s]['ci95']))
    specs.append(('table10', {'model': 'sol61', 'row': 'Sol', 'reference': 'PPLX B0 3-run mean'},
                  {b: ((('sol',), [('sol', r) for r in RUNS]), b0ref['pplx_v12']) for b in B}, None))

    # Table XI
    lit = src['literature']['benchmarks']
    published = {b: max(round(lit[b]['trained']['f1'], 2), round(lit[b]['zero_shot']['f1'], 2)) for b in B}
    cols = {'b0_alone': lambda r: ('casc', 'pplx_v12', 'alone', r), 'b0_cascade': lambda r: ('casc', 'pplx_v12', 'majority', r),
            'lab_alone': lambda r: ('labcasc', 'alone', r), 'lab_cascade': lambda r: ('labcasc', 'majority', r)}
    for k, key_of in cols.items():
        x = row(D, key_of, RUNS)
        x['above_both_published_per_run'] = {str(r): sum(round(100*x['f1'][b][str(r)], 2) > published[b] for b in B) for r in RUNS}
        x['above_both_published_on_mean'] = sum(round(x['per_benchmark'][b]['mean'], 2) > published[b] for b in B)
        out['table11'][k] = x
    out['table11']['rulesfs10_2000'] = {'t': t_lab, 'per_run': lab_runs, 'over_runs': over_runs(lab_runs)}
    spec = {b: ((('lab', 'RULESFS10_fixed', 'pplx_v12', '2000'), [('labcasc', 'alone', r) for r in RUNS]),
                (('dm', 'pplx_v12'), [('casc', 'pplx_v12', 'alone', r) for r in RUNS])) for b in B}
    specs.append(('table11', {'model': 'pplx_v12', 'comparison': 'RULESFS10_2000 alone vs B0 alone'}, spec, None))
    spec = {b: ((('lab', 'RULESFS10_fixed', 'pplx_v12', '2000'), [('labcasc', 'majority', r) for r in RUNS]),
                (('dm', 'pplx_v12'), [('casc', 'pplx_v12', 'majority', r) for r in RUNS])) for b in B}
    specs.append(('table11', {'model': 'pplx_v12', 'comparison': 'RULESFS10_2000 cascade vs B0 cascade'}, spec, None))

    for table, label, spec, stored in specs:
        out['significance'].append({'table': table, **label, **compare(D, draws, spec, stored)})
    lost = [s for s in out['significance'] if s['stored_significant'] and not s['multi_significant']]
    out['significance_summary'] = {'comparisons': len(out['significance']),
                                   'run_mean_significant': sum(s['fixed_ci95'][0] > 0 or s['fixed_ci95'][1] < 0 for s in out['significance']),
                                   'multi_significant': sum(s['multi_significant'] for s in out['significance']),
                                   'stored_significant': sum(s['stored_significant'] for s in out['significance']),
                                   'stored_dagger_lost_under_multi': [f"{s['table']} {s.get('model')} {s.get('row', s.get('comparison'))}" for s in lost],
                                   'run_mean_vs_multi_differ': [f"{s['table']} {s.get('model')} {s.get('row', s.get('comparison'))}" for s in out['significance']
                                                                if (s['fixed_ci95'][0] > 0 or s['fixed_ci95'][1] < 0) != s['multi_significant']]}
    print(json.dumps(out['significance_summary'], indent=1), flush=True)
    plain = json.loads(json.dumps(out, default=lambda o: o.item()))
    if args.tuning == 'valid':  # the copy reproduces the current tables
        for k in ('table9', 'table10', 'table11', 'partner_list_usd_all_pairs', 'one_run_costs'):
            assert plain[k] == ref_tables[k], k
        assert [x for x in plain['significance'] if x['table'] != 'targets'] == ref_tables['significance']
        print('valid mode reproduces later_tables_runs123_20261104.json', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(plain, f, indent=1)
    print('wrote', args.out)
    extract = {'description': 'Extract of '+args.out.name+' in the layout of cascade_runs123_20261104.json (table9, its significance) '
                              'plus targets, tuning and their significance; t from tuning source '+args.tuning+'.',
               'created_at': plain['created_at'], 'replicates': plain['replicates'], 'units': plain['units'], 'method': __doc__,
               'source': {'run': f'src/cascade_solset_runs123_eval.py --tuning {args.tuning}', 'file': args.out.name},
               'tuning_source': args.tuning, 'partner_list_usd_all_pairs': plain['partner_list_usd_all_pairs'],
               'table9': plain['table9'], 'targets': plain['targets'], 'tuning': plain['tuning'],
               'significance': [x for x in plain['significance'] if x['table'] in ('table9', 'targets')]}
    path = args.out.with_name(args.out.name.replace('later_tables', 'cascade'))
    assert path != args.out
    with path.open('x') as f:
        json.dump(extract, f, indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
