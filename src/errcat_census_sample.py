"""Census of the error categorization: all F0 test errors of Jev 1.13 and PPLX v1.2 and all test errors
of the ensemble members GPT-6.1 Sol, Claude Opus 5.5 and Gemini 4 Argon (B0, escalation run plus
reference run) that the F0 sample (artifacts/astra_error_analysis/f0_v1) does not contain yet.
One pair is one case, its erring models are listed privately. Pairs whose records equal a discovery
case (same or mirrored order) are excluded, as in errcat_f0_sample.py.

    PYTHONPATH=src python3 src/errcat_census_sample.py [--dry-run]
"""
import hashlib
import json
import sys
from pathlib import Path

import inputs_study
import protocol
from errcat_f0_sample import MODELS, predictions, record_key

ROOT = Path(__file__).resolve().parents[1]
TARGET = ['jev', 'pplx_v12', 'sol', 'opus', 'argon']
SALT = 'errcat-census-v1'
F0 = ROOT/'artifacts/astra_error_analysis/f0_v1/cases_private.json'
OUT = ROOT/'artifacts/astra_error_analysis/census_v1/cases_private.json'
ESC = ROOT/'runs/escalation_run01_20261020'
REF = ROOT/'runs/reference_rest_run01_20261021'
MEMBERS = {'sol': ('sol61', sorted(REF.glob('sol61_direct/results*.jsonl'))),
           'opus': ('opus55', [REF/'opus55_batch/results.jsonl']),
           'argon': ('argon4', sorted(REF.glob('argon4/E0/*/reference_rest.jsonl')))}


def member(alias, ref_paths):
    """(benchmark, pair_id) -> answered prediction; a later file overrides a failed earlier row."""
    out = {}
    for path in sorted(ESC.glob(f'{alias}/E0/*/escalation.jsonl'))+list(ref_paths):
        for line in path.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get('prediction') in (0, 1):
                out[(r['benchmark'], r['pair_id'])] = r['prediction']
    return out


def main(dry_run):
    inputs_study.load_selection(ROOT)
    discovery = json.loads((ROOT/'artifacts/astra_error_analysis/preparation_v1/cases_private.json').read_text())['discovery']
    seen = {record_key(c['records']) for c in discovery} | {record_key(c['records'], True) for c in discovery}
    sampled = {(c['benchmark'], c['pair_id']) for c in json.loads(F0.read_text())['evaluation']}
    mem = {m: member(*v) for m, v in MEMBERS.items()}
    out, excluded, already, total = [], 0, 0, 0
    for b in protocol.BENCHMARKS:
        cases = json.loads((ROOT/'data'/b/'test.json').read_text())
        preds = {m: predictions(m, b) for m in MODELS}
        for c in cases:
            pid = c['pair_id']
            for m in mem:
                assert dry_run or (b, pid) in mem[m], f'{m} has no answer for {b} {pid}'
            dec = {m: preds[m][0][pid] for m in MODELS} | {m: mem[m].get((b, pid), c['label']) for m in mem}
            wrong = [m for m in dec if dec[m] != c['label']]
            if not set(wrong) & set(TARGET):
                continue
            total += 1
            if (b, pid) in sampled:
                already += 1
                continue
            if record_key(c['records']) in seen:
                excluded += 1
                continue
            out.append({'benchmark': b, 'pair_id': pid, 'records': c['records'], 'record_key': record_key(c['records']),
                        'gold_label': c['label'], 'prediction': 1-c['label'],
                        'error_type': 'FN' if c['label'] == 1 else 'FP', 'models': wrong, 'source_split': 'test',
                        'case_id': hashlib.sha256((SALT+'|'+b+'|'+pid).encode()).hexdigest()[:24]})
    summary = {'target_error_pairs': total, 'in_f0_sample': already, 'excluded_discovery_overlap': excluded,
               'new_cases': len(out)}
    if not dry_run:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open('x') as f:
            json.dump({'evaluation': out, 'rule': __doc__.strip().splitlines()[0]} | summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main('--dry-run' in sys.argv)
