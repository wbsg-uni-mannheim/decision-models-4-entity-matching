"""Per-model error classes from the F0 sample plus the census (majority of three GPT-6.1 Sol runs each).

Models: Jev 1.13 and PPLX v1.2 with their selected prompts (F0) and the ensemble members GPT-6.1 Sol,
Claude Opus 5.5 and Gemini 4 Argon (B0, escalation run plus reference run, all 30,225 test pairs).
Cases: artifacts/astra_error_analysis/f0_v1 (479) and census_v1 (886, src/errcat_census_sample.py), results in
runs/error_categorization/sol_flex_{f0,census}_run{1,2,3}. The identifier class G1 is kept in the stability
numbers only; G2 to G8 are reported as C1 to C7. No API calls.

Writes artifacts/results/errcat_census_models_20261022.json (aggregated counts and stability, the paper's
sources/errcat_models.json) and artifacts/results/errcat_final_assignments_census_20261022.json (majority
groups per census case). An existing output must be identical, so a rerun checks reproducibility.

    python3 src/errcat_census_eval.py --root /ceph/aasteine/decision-em-20261013
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

GROUPS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8']
MAP = {'G2': 'C1', 'G3': 'C2', 'G4': 'C3', 'G5': 'C4', 'G6': 'C5', 'G7': 'C6', 'G8': 'C7'}
CL = list(MAP.values())
FOCUS = ['jev', 'pplx_v12', 'sol', 'opus', 'argon']
REPORTED = FOCUS+['clef', 'clef_flash', 'kev']
F0_MODELS = ['jev', 'pplx_v12', 'clef', 'clef_flash', 'kev']
MEMBERS = {'sol': ('sol61', 'sol61_direct/results*.jsonl'), 'opus': ('opus55', 'opus55_batch/results.jsonl'),
           'argon': ('argon4', 'argon4/E0/*/reference_rest.jsonl')}
DESCRIPTION = ('Error classes of all test errors of Jev 1.13 and PPLX v1.2 (selected prompts, F0) and of the ensemble '
               'members GPT-6.1 Sol, Claude Opus 5.5 and Gemini 4 Argon (B0, all 30,225 test pairs). Cases: the 479 F0 '
               'sample cases (artifacts/astra_error_analysis/f0_v1) plus the census of the 886 remaining error pairs of '
               'these five models (artifacts/astra_error_analysis/census_v1, runs/error_categorization/'
               'sol_flex_census_run1-3, approval 2026-10-22, 6.92 USD). Frozen 8-group codebook (revision 2), GPT-6.1 '
               'Sol flex low reasoning, three runs, majority of two. Stability (kappa, same_set_share) over the 1,100 '
               'error pairs of the five models. Counts per model: pairs with the class; clef, clef_flash, kev are '
               'partial (only their errors among the assigned cases). Computed by src/errcat_census_eval.py.')
OUT_MODELS = 'artifacts/results/errcat_census_models_20261022.json'
OUT_CASES = 'artifacts/results/errcat_final_assignments_census_20261022.json'


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def decisions(root):
    """(benchmark, pair_id) -> label and decision per model."""
    sel = json.loads((root/'configs/inputs_prompt_selection_20261019.json').read_text())['selected']
    mem = {}
    for m, (alias, pattern) in MEMBERS.items():
        paths = sorted(root.glob(f'runs/escalation_run01_20261020/{alias}/E0/*/escalation.jsonl'))
        paths += sorted(root.glob(f'runs/reference_rest_run01_20261021/{pattern}'))
        mem[m] = {(r['benchmark'], r['pair_id']): r['prediction'] for p in paths for r in lines(p)
                  if r.get('prediction') in (0, 1)}
    out = {}
    for b in sorted({b for per in sel.values() for b in per}):
        preds = {}
        for m in F0_MODELS:
            v = sel[m][b]
            own = v != 'B0' or m == 'pplx_v12'
            path = root/'runs'/('prompt12_run01_20261019' if own else 'quality')/m/v/b/'test.jsonl'
            preds[m] = {r['pair_id']: r['prediction'] for r in lines(path)}
        for c in json.loads((root/'data'/b/'test.json').read_text()):
            k = (b, c['pair_id'])
            d = {m: preds[m][c['pair_id']] for m in F0_MODELS} | {m: mem[m][k] for m in mem}
            assert all(v in (0, 1) for v in d.values()), k
            out[k] = (c['label'], d)
    assert len(out) == 30225, len(out)
    return out


def runs(root, prefix):
    out = []
    for r in (1, 2, 3):
        rows = {}
        for p in sorted((root/'runs/error_categorization'/f'{prefix}_run{r}').glob('results*.jsonl')):
            for x in lines(p):
                if 'result' in x:
                    assert x['case_id'] not in rows
                    rows[x['case_id']] = {a['group'] for a in x['result']['assignments']}
        out.append(rows)
    return out


def fleiss(votes):
    n, k = len(votes), 3
    p = sum(votes)/(n*k)
    pe = p*p+(1-p)*(1-p)
    po = sum((v*(v-1)+(k-v)*(k-v-1))/(k*(k-1)) for v in votes)/n
    return (po-pe)/(1-pe) if pe < 1 else 1.


def write_or_check(path, text):
    if path.exists():
        if path.read_text() != text:
            sys.exit(f'{path} exists with different content')
        print(f'{path}: identical')
    else:
        path.write_text(text)
        print(f'{path}: written')


def main(root):
    dec = decisions(root)
    err = lambda k, m: None if dec[k][1][m] == dec[k][0] else ('FP' if dec[k][0] == 0 else 'FN')
    f0 = json.loads((root/'artifacts/astra_error_analysis/f0_v1/cases_private.json').read_text())['evaluation']
    census = json.loads((root/'artifacts/astra_error_analysis/census_v1/cases_private.json').read_text())['evaluation']
    R = [{**a, **b} for a, b in zip(runs(root, 'sol_flex_f0'), runs(root, 'sol_flex_census'))]
    cases = f0+census
    assert len({c['case_id'] for c in cases}) == len(cases)
    assert all(c['case_id'] in r for c in cases for r in R), 'case without a valid result in some run'
    maj = {c['case_id']: {g for g in GROUPS if sum(g in r[c['case_id']] for r in R) >= 2} for c in cases}
    final = json.loads((root/'artifacts/results/errcat_final_assignments_f0_20261021.json').read_text())
    assert all(set(final[c['case_id']]) == maj[c['case_id']] for c in f0), 'F0 majority differs from the final file'
    by_pair = {(c['benchmark'], c['pair_id']): {MAP[g] for g in maj[c['case_id']] if g in MAP} for c in cases}

    target = [c for c in cases if any(err((c['benchmark'], c['pair_id']), m) for m in FOCUS)]
    out = {'description': DESCRIPTION, 'cases': len(cases), 'f0_cases': len(f0), 'census_cases': len(census),
           'target_cases': len(target), 'models': {}, 'stability': {}}
    for g in GROUPS:  # stability over the error pairs of the five focus models
        votes = [sum(g in r[c['case_id']] for r in R) for c in target]
        out['stability'][MAP.get(g, g)] = {'fleiss_kappa': fleiss(votes), 'present_majority': sum(v >= 2 for v in votes)}
    out['same_set_share'] = sum(len({frozenset(r[c['case_id']] & set(MAP)) for r in R}) == 1 for c in target)/len(target)

    def profile(pairs):
        cnt = Counter(k for p in pairs for k in by_pair[p])
        return {'n': len(pairs), **{k: cnt[k] for k in CL}}

    missing = Counter()
    for m in REPORTED:
        e = {t: [k for k in dec if err(k, m) == t] for t in ('FP', 'FN')}
        have = {t: [k for k in e[t] if k in by_pair] for t in e}
        if m in FOCUS:
            missing[m] = sum(len(e[t])-len(have[t]) for t in e)
        out['models'][m] = {'all': profile(have['FP']+have['FN']), 'FP': profile(have['FP']), 'FN': profile(have['FN']),
                            'errors': len(e['FP'])+len(e['FN'])}
    out['missing_focus'] = dict(missing)
    assert not any(missing.values()), missing
    write_or_check(root/OUT_MODELS, json.dumps(out, indent=1)+'\n')
    write_or_check(root/OUT_CASES, json.dumps({c['case_id']: sorted(maj[c['case_id']]) for c in census}, indent=1))  # same format as the F0 file


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    main(ap.parse_args().root.resolve())
