"""Additions to the cascade extract with t from --tuning (valid, sol200, sol2000; cascade_solset_runs123_eval.tuning_pairs).
Copy of cascade_extras_runs123_eval.py; the Run-1 asserts against sources/cascade_b0.json hold in valid mode only.

Original description: Additions to cascade_runs123_20261104.json (Table IX) over three runs. Read-only, no inference.

Same runs, partners, pairing (decision-model run r with partner run r), threshold and cost rules as
later_tables_runs123_eval.py (t at 0.5 % from the pooled official validation sets, B0 Run 1).
majority_alone: the majority of Sol, Opus and Argon of the same run on all 30,225 test pairs: F1 per benchmark, mean F1,
  errors, list_usd (sum of the three members' list costs on all pairs), per run and over_runs (mean, sample SD; F1 in
  points), format of table9.<m>.over_runs.configs.majority. Run 1 reproduces paper sources/ensemble_full.json majority.
deferred: per decision model (Jev 1.13, PPLX v1.2, Clef, Luna Dec.) and run: deferred pairs, the share of the model's own
  errors (p >= 0.5 alone) that lie in the deferred pairs, and the share of deferred pairs decided correctly by the ensemble
  majority, Sol alone, the decision model alone and (Jev, PPLX) Luna Gen.; over_runs mean and SD.
luna_cascade: Jev 1.13 and PPLX v1.2 with Luna Gen. B0 of the same run as the only partner (Run 1 runs/quality/luna/B0,
  Runs 2/3 replications/run0{2,3}/runs/quality/luna/B0; partner cost = its attempt costs): mean F1, F1 per benchmark,
  errors, list cost, accuracy on the deferred pairs, difference to the model alone, per run and over runs.
Run 1 reproduces sources/cascade_b0.json A_b0 0.005 (error share, accuracies, Luna cascade; asserted).

    .venv/bin/python src/cascade_extras_runs123_eval.py --paper-sources <paper sources/> --out artifacts/results/cascade_extras_runs123_20261104.json
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cascade_b0_valsel_eval as cv
import protocol
import valsel_eval
from inputs_b0_runs123_eval import b0_path as journal_b0_path
import cascade_solset_runs123_eval as cs
from cascade_solset_runs123_eval import DM4, RUNS, X, cascade, ms, partner_run, test_rows
from thresholds_runs123_eval import test_path as b0_path

B = list(protocol.BENCHMARKS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tuning', choices=['valid', 'sol200', 'sol2000'], required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    cs.TUNING[0] = args.tuning
    V = args.tuning == 'valid'
    cas = json.loads((args.paper_sources/'cascade_b0.json').read_text())['variants']['A_b0']
    ens = json.loads((args.paper_sources/'ensemble_full.json').read_text())['majority']
    ids, y = {}, {}
    for b in B:
        cases = json.loads((cv.ROOT/'data'/b/'test.json').read_text())
        ids[b] = [c['pair_id'] for c in cases]
        y[b] = np.array([c['label'] for c in cases], dtype=np.int8)
    dec1 = cv.ensemble()
    partner, totals = {}, {}
    for r in RUNS:
        partner[r], totals[r] = partner_run(r, ids, dec1)
        luna = {}
        for b in B:
            rows = {x['pair_id']: x for x in cv.lines(journal_b0_path('luna', b, r))}
            assert rows.keys() == set(ids[b]) and all(rows[i]['prediction'] in (0, 1) for i in ids[b]), (r, b)
            luna[b] = (np.array([rows[i]['prediction'] for i in ids[b]], dtype=np.int8), np.array([cv.attempt_cost(rows[i]) for i in ids[b]]))
        partner[r]['luna'] = luna
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(),
           'units': 'per_run F1 as fractions; over_runs F1 and differences in points; shares as fractions; USD at list prices',
           'majority_alone': {'per_run': {}}, 'deferred': {}, 'luna_cascade': {}, 'tuning_source': args.tuning}

    per_run = out['majority_alone']['per_run']
    for r in RUNS:
        pred = {b: partner[r]['majority'][b][0] for b in B}
        pb = {b: cv.f1(y[b], pred[b]) for b in B}
        per_run[str(r)] = {'per_benchmark': pb, 'mean_f1': statistics.mean(pb.values()),
                           'errors': sum(int((pred[b] != y[b]).sum()) for b in B), 'usd_list': sum(totals[r].values()),
                           'members_usd_list': totals[r]}
    one = per_run['1']
    assert abs(one['mean_f1']-ens['macro_f1']) < 1e-12 and one['errors'] == ens['errors'] and abs(one['usd_list']-ens['usd_list']) < 1e-9
    rs = list(per_run.values())
    out['majority_alone']['over_runs'] = {'mean_f1': ms(100*x['mean_f1'] for x in rs),
                                          'per_benchmark': {b: ms(100*x['per_benchmark'][b] for x in rs) for b in B},
                                          'errors': ms(x['errors'] for x in rs), 'usd_list': ms(x['usd_list'] for x in rs)}
    print('majority alone', out['majority_alone']['over_runs']['mean_f1'], flush=True)

    for m in DM4:
        t = valsel_eval.tune_risk(cs.tuning_pairs(m), X)
        old = cas[m]['levels'][str(X)]
        assert not V or abs(t-old['t']) < 1e-12, m
        dres, lres = {}, {}
        for r in RUNS:
            test = test_rows(lambda b: b0_path(m, b, r), ids)
            err = err_d = n_d = 0
            correct = {'majority': 0, 'sol61': 0, 'alone': 0, 'luna': 0}
            for b in B:
                p = test[b]['p']
                alone = (p >= .5).astype(np.int8)
                d = np.minimum(p, 1-p) > t
                wrong = alone != y[b]
                err += int(wrong.sum()); err_d += int(wrong[d].sum()); n_d += int(d.sum())
                correct['alone'] += int((~wrong[d]).sum())
                for c in ('majority', 'sol61', 'luna'):
                    correct[c] += int((partner[r][c][b][0][d] == y[b][d]).sum())
            x = {'deferred': n_d, 'deferred_share': n_d/sum(len(v) for v in ids.values()), 'errors_alone': err,
                 'error_share_in_deferred': err_d/err,
                 'accuracy_on_deferred': {c: v/n_d for c, v in correct.items() if c != 'luna' or m in ('jev', 'pplx_v12')}}
            dres[str(r)] = x
            if m in ('jev', 'pplx_v12'):
                res, _ = cascade(y, test, {'sol61': partner[r]['luna'], 'majority': partner[r]['majority']}, t)
                lc = res['configs']['sol61']  # the 'sol61' slot carries Luna Gen. here
                lres[str(r)] = {'mean_f1': lc['mean_f1'], 'per_benchmark': lc['per_benchmark'], 'errors': lc['errors'],
                                'usd_list': lc['usd_list'], 'accuracy_on_deferred': x['accuracy_on_deferred']['luna'],
                                'diff_to_alone': 100*(lc['mean_f1']-res['configs']['alone']['mean_f1'])}
            if r == 1 and V:
                o = old['test']
                assert x['deferred'] == o['deferred'] and abs(x['error_share_in_deferred']-o['error_share_in_deferred']) < 1e-12, m
                assert abs(x['accuracy_on_deferred']['alone']-o['model_accuracy_on_deferred']) < 1e-12, m
                for c in ('majority', 'sol61') + (('luna',) if m in ('jev', 'pplx_v12') else ()):
                    assert abs(x['accuracy_on_deferred'][c]-o['partner_accuracy_on_deferred'][c]) < 1e-12, (m, c)
                if m in ('jev', 'pplx_v12'):
                    assert abs(lres['1']['mean_f1']-old['configs']['luna']['mean_f1']) < 1e-12, m
                    assert abs(lres['1']['usd_list']-old['configs']['luna']['usd_list']) < 1e-9, m
        rs = list(dres.values())
        out['deferred'][m] = {'t': t, 'per_run': dres, 'over_runs': {
            'deferred_share': ms(x['deferred_share'] for x in rs), 'error_share_in_deferred': ms(x['error_share_in_deferred'] for x in rs),
            'accuracy_on_deferred': {c: ms(x['accuracy_on_deferred'][c] for x in rs) for c in rs[0]['accuracy_on_deferred']},
            'ensemble_minus_model_on_deferred_points': ms(100*(x['accuracy_on_deferred']['majority']-x['accuracy_on_deferred']['alone']) for x in rs)}}
        if lres:
            ls = list(lres.values())
            out['luna_cascade'][m] = {'t': t, 'per_run': lres, 'over_runs': {
                'mean_f1': ms(100*x['mean_f1'] for x in ls), 'per_benchmark': {b: ms(100*x['per_benchmark'][b] for x in ls) for b in B},
                'errors': ms(x['errors'] for x in ls), 'usd_list': ms(x['usd_list'] for x in ls),
                'accuracy_on_deferred': ms(x['accuracy_on_deferred'] for x in ls), 'diff_to_alone': ms(x['diff_to_alone'] for x in ls)}}
        print(m, {k: (round(v['mean'], 4), round(v['sd'], 4)) for k, v in out['deferred'][m]['over_runs'].items() if 'mean' in v},
              {c: round(v['mean'], 4) for c, v in out['deferred'][m]['over_runs']['accuracy_on_deferred'].items()},
              out['luna_cascade'].get(m, {}).get('over_runs', {}).get('diff_to_alone'), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
    print('wrote', args.out)


if __name__ == '__main__':
    main()
