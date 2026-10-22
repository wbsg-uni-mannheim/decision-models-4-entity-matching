"""Escalation of uncertain test pairs to the three-model ensemble (one run).

Routing is label-free on test: a decision model defers a pair when
min(p, 1-p) > t, with t tuned on the full validation set for an error target
among accepted validation pairs (src/valsel_eval.py). The escalation set is the
union of the deferred test pairs of
    Jev and PPLX v1.2 at 0.25 % (contains their 0.5 % and 1 % sets),
    Clef and Luna Dec. at 0.5 %,
each model with its selected prompt (F0). The ensemble members receive the frozen
Luna B0 messages with the member settings of the technical pilot
(artifacts/operations/strong_ensemble.py): reasoning effort medium, 2,048 output
tokens, pinned provider without fallback. No labels or model votes reach the members.

Service tiers (Aaron, 20 Oct): Gemini 4 Argon uses the flex tier at half price through
Google AI Studio (OpenRouter offers flex there, not on Vertex). GPT-6.1 Sol first used flex,
but OpenAI reported flex as temporarily unavailable for most requests, so Sol runs on the
standard OpenAI endpoint (Aaron, 20 Oct). Claude Opus 5.5 has no flex tier and stays on the
standard Anthropic endpoint. The
response reports the tier that served the request; billing follows it.
"""
import json
from pathlib import Path

import inputs_study
import protocol
import strong_ensemble
import valsel_eval

PROJECT = Path(__file__).resolve().parents[1]
SPLIT = 'escalation'
RUN_MODELS = list(strong_ensemble.MEMBERS)
DESIGN = [('jev', .0025), ('pplx_v12', .0025), ('clef', .005), ('luna_decisions', .005)]
NESTED = {'jev': (.005, .01), 'pplx_v12': (.005, .01)}
OBSERVED_PROVIDER = {'sol61': 'OpenAI', 'opus55': 'Anthropic', 'argon4': 'Google AI Studio'}
ENDPOINT = {'sol61': 'openai', 'opus55': 'anthropic', 'argon4': 'google-ai-studio/flex'}
SERVICE_TIER = {'argon4': 'flex'}
SELECTION = inputs_study.SELECTION
VALID_NS = 'valsel_run01_20261019'


def register_models():
    inputs_study.register_models()
    for alias, spec in strong_ensemble.MEMBERS.items():
        entry = {'id': spec['id'], 'provider': ENDPOINT[alias], 'kind': 'generative'}
        assert protocol.MODELS.get(alias, entry) == entry
        protocol.MODELS[alias] = entry


def run_variants(alias):
    return ['E0']


def benchmarks(variant):
    return list(protocol.BENCHMARKS)


def load_selection(root=PROJECT):
    return inputs_study.load_selection(root)


def payload(case, alias, variant='E0'):
    assert variant == 'E0', variant
    body = strong_ensemble.payload_from_frozen_b0(protocol.payload(case, 'luna', 'B0'), alias)
    body['provider'] = {**body['provider'], 'only': [ENDPOINT[alias]], 'order': [ENDPOINT[alias]]}
    if alias in SERVICE_TIER:
        body['service_tier'] = SERVICE_TIER[alias]
    return body


def parse_response(alias, data):
    """Exact model id and provider display name, as in the pilot; vote as (0/1, None)."""
    if 'error' in data:
        raise ValueError('API error response')
    spec = strong_ensemble.MEMBERS[alias]
    if data.get('model') != spec['id'] or data.get('provider') != OBSERVED_PROVIDER[alias]:
        raise ValueError('Unexpected model or provider: '+str(data.get('model'))+' / '+str(data.get('provider')))
    return int(strong_ensemble.parse_vote(data)), None


def deferred_sets(root=PROJECT):
    """Test pair ids deferred by each (model, target) of the design and its nested targets."""
    root = Path(root)
    load_selection(root)
    out, thresholds = {}, {}
    for model in {m for m, _ in DESIGN}:
        val = valsel_eval.pairs(root, model, 'valid', VALID_NS)
        targets = [x for m, x in DESIGN if m == model] + list(NESTED.get(model, ()))
        for x in targets:
            t = valsel_eval.tune_risk(val, x)
            thresholds[f'{model}@{x}'] = t
            keys = set()
            for b in protocol.BENCHMARKS:
                p = inputs_study.SELECTED[model][b]
                own = p != 'B0' or model == 'pplx_v12'
                path = root/'runs'/('prompt12_run01_20261019' if own else 'quality')/model/p/b/'test.jsonl'
                for line in path.read_text().splitlines():
                    r = json.loads(line)
                    if min(r['probability_match'], 1-r['probability_match']) > t:
                        keys.add((b, r['pair_id']))
            out[f'{model}@{x}'] = keys
    return out, thresholds


def freeze_set(root=PROJECT):
    """Write data/<b>/escalation.json once (test cases in test order) and return the report."""
    import prepare
    root = Path(root)
    sets, thresholds = deferred_sets(root)
    for model, (a, b) in NESTED.items():
        big = sets[f'{model}@{.0025}']
        assert sets[f'{model}@{a}'] <= big and sets[f'{model}@{b}'] <= big, model
    union = set().union(*(sets[f'{m}@{x}'] for m, x in DESIGN))
    report = {'thresholds': thresholds, 'set_sizes': {k: len(v) for k, v in sets.items()},
              'union': len(union), 'benchmarks': {}}
    for b in protocol.BENCHMARKS:
        cases = [c for c in json.loads((root/'data'/b/'test.json').read_text()) if (b, c['pair_id']) in union]
        prepare.write_frozen(root/'data'/b/(SPLIT+'.json'), cases)
        report['benchmarks'][b] = {'n': len(cases), 'positive': sum(c['label'] for c in cases),
                                   'sha256': protocol.digest(cases)}
    assert sum(x['n'] for x in report['benchmarks'].values()) == len(union)
    return report


def sample_set(root=PROJECT, n=500, salt='20261020:escalation-sample'):
    """Write data/<b>/escalation_sample.json once: n pairs of the escalation set, stratified by
    benchmark (largest remainder), chosen by a salted hash of the pair id."""
    import prepare
    root = Path(root)
    pools = {b: json.loads((root/'data'/b/(SPLIT+'.json')).read_text()) for b in protocol.BENCHMARKS}
    total = sum(len(v) for v in pools.values())
    quota = {b: n*len(v)//total for b, v in pools.items()}
    rest = sorted(pools, key=lambda b: (-(n*len(pools[b]) % total), b))
    for b in rest[:n-sum(quota.values())]:
        quota[b] += 1
    report = {'n': n, 'salt': salt, 'benchmarks': {}}
    for b, cases in pools.items():
        chosen = set(sorted((c['pair_id'] for c in cases), key=lambda i: protocol.digest([salt, i]))[:quota[b]])
        rows = [c for c in cases if c['pair_id'] in chosen]
        prepare.write_frozen(root/'data'/b/(SPLIT+'_sample.json'), rows)
        report['benchmarks'][b] = {'n': len(rows), 'positive': sum(c['label'] for c in rows),
                                   'sha256': protocol.digest(rows)}
    assert sum(x['n'] for x in report['benchmarks'].values()) == n
    return report
