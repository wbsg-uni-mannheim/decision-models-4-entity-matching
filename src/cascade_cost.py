"""Cascade overview: thresholds from validation, accepted accuracy and escalation share on test,
and a cost projection for escalating the deferred test pairs to the three-model ensemble.

Read-only, no inference. Cost per pair and member comes from the 16-pair technical pilot
(artifacts/strong_ensemble/pilot_v1, development pairs): mean completion cost per member plus
prompt cost per character of the Luna B0 messages (the ensemble sends exactly these messages),
applied to the Luna B0 messages of each deferred test pair.

    PYTHONPATH=src python3 src/cascade_cost.py
"""
import json
from pathlib import Path

import inputs_study
import protocol
import valsel_eval

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['jev', 'clef', 'clef_flash', 'luna_decisions', 'pplx_v12', 'kev']
TARGETS = [('risk', .0025), ('risk', .005), ('risk', .01), ('coverage', .85)]
PILOT = ROOT/'artifacts/strong_ensemble/pilot_v1'


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def chars(body):
    return sum(len(m['content']) for m in body['messages'])


def luna_chars():
    out = {}
    for b in protocol.BENCHMARKS:
        for c in json.loads((ROOT/'data'/b/'test.json').read_text()):
            out[(b, c['pair_id'])] = chars(protocol.payload(c, 'luna', 'B0'))
    return out


def member_costs():
    requests = {r['request_id']: r for r in lines(PILOT/'requests.jsonl')}
    stats = {}
    for e in lines(PILOT/'execution.jsonl'):
        if e.get('event') not in ('response', 'prior_response_reused_without_api_call'):
            continue
        req = requests[e['request_id']]
        usage = e['data']['usage']
        details = usage.get('cost_details') or {}
        s = stats.setdefault(req['body']['model'], {'n': 0, 'completion': 0., 'prompt': 0., 'chars': 0, 'cost': 0.})
        s['n'] += 1; s['cost'] += usage['cost']
        s['completion'] += details.get('upstream_inference_completions_cost', 0.)
        s['prompt'] += details.get('upstream_inference_prompt_cost', usage['cost'])
        s['chars'] += chars(req['body'])
    return {m: {'per_request': s['completion']/s['n'], 'per_char': s['prompt']/s['chars'],
                'pilot_requests': s['n'], 'pilot_cost': s['cost']} for m, s in stats.items()}


def test_rows(model):
    out = []
    for b in protocol.BENCHMARKS:
        labels = {c['pair_id']: c['label'] for c in json.loads((ROOT/'data'/b/'test.json').read_text())}
        p = inputs_study.SELECTED[model][b]
        own = p != 'B0' or model == 'pplx_v12'
        for r in lines(ROOT/'runs'/('prompt12_run01_20261019' if own else 'quality')/model/p/b/'test.jsonl'):
            out.append((b, r['pair_id'], r['probability_match'], labels[r['pair_id']]))
    return out


def main():
    inputs_study.load_selection(ROOT)
    tokens = luna_chars()
    members = member_costs()

    def cost(keys):
        return sum(m['per_request']+m['per_char']*tokens[k] for k in keys for m in members.values())
    out = {'members': members, 'models': {}, 'union': {}}
    union = {}
    for model in MODELS:
        val = valsel_eval.pairs(ROOT, model, 'valid', 'valsel_run01_20261019')
        rows = test_rows(model)
        for kind, x in TARGETS:
            t = valsel_eval.tune_risk(val, x) if kind == 'risk' else valsel_eval.tune_coverage(val, x)
            accepted = [r for r in rows if min(r[2], 1-r[2]) <= t]
            deferred = {(r[0], r[1]) for r in rows if min(r[2], 1-r[2]) > t}
            wrong = sum((r[2] >= .5) != r[3] for r in accepted)
            key = f'{kind}_{x}'
            out['models'].setdefault(model, {})[key] = {
                't': t, 'accepted': len(accepted), 'accepted_accuracy': 1-wrong/len(accepted),
                'deferred': len(deferred), 'deferred_share': len(deferred)/len(rows),
                'overall_accuracy_without_escalation': 1-sum((r[2] >= .5) != r[3] for r in rows)/len(rows),
                'ensemble_usd': cost(deferred)}
            union.setdefault(key, set()).update(deferred)
    for key, keys in union.items():
        out['union'][key] = {'deferred': len(keys), 'deferred_share': len(keys)/len(tokens), 'ensemble_usd': cost(keys)}
    out['full_test_ensemble_usd'] = cost(tokens.keys())
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
