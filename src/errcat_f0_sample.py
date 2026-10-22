"""Error sample for the categorization on F0 (selected prompt per model and benchmark, PPLX v1.2).

Replaces the B0 Run-1 sample (artifacts/astra_error_analysis/preparation_v1), whose PPLX v1.1
errors partly came from the provider outage of 13 Oct 2026. Union of the misclassified test
pairs of the seven models; one pair is one case, its models are listed privately. Up to 32 cases
per benchmark and error type, chosen by a salted hash of the pair id. Pairs whose records equal
a discovery case (same or mirrored order) are excluded, so the frozen codebook never saw them.

    PYTHONPATH=src python3 src/errcat_f0_sample.py
"""
import hashlib
import json
from pathlib import Path

import inputs_study
import protocol

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['jev', 'clef', 'clef_flash', 'luna_decisions', 'pplx_v12', 'kev', 'luna']
PER_STRATUM = 32
SALT = 'errcat-f0-v1'
OUT = ROOT/'artifacts/astra_error_analysis/f0_v1/cases_private.json'


def record_key(records, mirrored=False):
    sides = ('right', 'left') if mirrored else ('left', 'right')
    return protocol.digest([records[s] for s in sides])


def predictions(model, b):
    p = inputs_study.SELECTED[model][b]
    own = p != 'B0' or model == 'pplx_v12'
    path = ROOT/'runs'/('prompt12_run01_20261019' if own else 'quality')/model/p/b/'test.jsonl'
    return {json.loads(x)['pair_id']: json.loads(x)['prediction'] for x in path.read_text().splitlines()}, p


def main():
    inputs_study.load_selection(ROOT)
    discovery = json.loads((ROOT/'artifacts/astra_error_analysis/preparation_v1/cases_private.json').read_text())['discovery']
    seen = {record_key(c['records']) for c in discovery} | {record_key(c['records'], True) for c in discovery}
    strata, excluded, totals = {}, 0, {}
    for b in protocol.BENCHMARKS:
        cases = {c['pair_id']: c for c in json.loads((ROOT/'data'/b/'test.json').read_text())}
        preds = {m: predictions(m, b) for m in MODELS}
        for pid, c in cases.items():
            wrong = [m for m in MODELS if preds[m][0][pid] != c['label']]
            if not wrong:
                continue
            if record_key(c['records']) in seen:
                excluded += 1
                continue
            et = 'FN' if c['label'] == 1 else 'FP'
            totals[(b, et)] = totals.get((b, et), 0)+1
            strata.setdefault((b, et), []).append({
                'benchmark': b, 'pair_id': pid, 'records': c['records'], 'record_key': record_key(c['records']),
                'gold_label': c['label'], 'prediction': 1-c['label'], 'error_type': et, 'models': wrong,
                'prompts': {m: preds[m][1] for m in wrong}, 'source_split': 'test',
                'case_id': hashlib.sha256((SALT+'|'+b+'|'+pid).encode()).hexdigest()[:24]})
    out = []
    for k in sorted(strata):
        ranked = sorted(strata[k], key=lambda c: hashlib.sha256((SALT+'|'+c['pair_id']).encode()).hexdigest())
        out += ranked[:PER_STRATUM]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x') as f:
        json.dump({'evaluation': out, 'rule': __doc__.strip().splitlines()[0], 'excluded_discovery_overlap': excluded,
                   'error_pairs_per_stratum': {f'{b}|{t}': n for (b, t), n in sorted(totals.items())}}, f,
                  ensure_ascii=False, indent=1)
    by = {}
    for c in out:
        by[(c['benchmark'], c['error_type'])] = by.get((c['benchmark'], c['error_type']), 0)+1
    print(json.dumps({'cases': len(out), 'excluded_discovery_overlap': excluded,
                      'per_stratum': {f'{b}|{t}': n for (b, t), n in sorted(by.items())},
                      'available_per_stratum': {f'{b}|{t}': n for (b, t), n in sorted(totals.items())},
                      'single_model_cases': sum(len(c['models']) == 1 for c in out),
                      'pplx_v12_only': sum(c['models'] == ['pplx_v12'] for c in out)}, indent=1))


if __name__ == '__main__':
    main()
