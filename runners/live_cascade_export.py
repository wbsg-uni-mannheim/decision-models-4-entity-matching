"""Verify fresh live chains against raw call journals, then export latency per run."""
import hashlib
import json
import math
import numpy as np


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def check_call(call, case, model, protocol, origin):
    if not call.get('attempts') and call.get('error'):
        return  # locally refused stage; no returned paid request
    origin([call])
    for key, value in [('pair_id', case['pair_id']), ('input_hash', case['input_hash']),
                       ('benchmark', case['benchmark']), ('model_alias', model), ('variant', 'B0'),
                       ('request_hash', protocol.digest(protocol.payload(case, model, 'B0')))]:
        if call.get(key) != value:
            raise ValueError('Live request provenance differs: '+key)
    if not call.get('error'):
        parsed = protocol.parse_response(model, call['attempts'][-1]['response'])
        if parsed != (call['prediction'], call['probability_match']):
            raise ValueError('Live raw response disagrees with parsed decision')
    for attempt in call['attempts']:
        cost = attempt['cost']
        if cost is not None and (not math.isfinite(cost) or cost < 0):
            raise ValueError('Invalid live attempt cost')
        if not math.isfinite(attempt['seconds']) or attempt['seconds'] < 0:
            raise ValueError('Invalid live attempt duration')
        usage = (attempt.get('response') or {}).get('usage', {}).get('cost')
        if isinstance(usage, (int, float)) and usage >= 0 and cost != usage:
            raise ValueError('Live cost differs from returned usage')


def verify_cell(entry, cases, success, failed, journal, module, protocol, origin):
    indexed = {c['pair_id']: c for c in cases}
    if (len(indexed) != len(cases) or len(success) != len(cases)
            or {s['pair_id'] for s in success} != set(indexed)):
        raise ValueError('Live cell missing or duplicate success pairs')
    finished = success+failed
    ids = [s['chain_id'] for s in finished]
    keys = [(j['chain_id'], j['call_index']) for j in journal]
    if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
        raise ValueError('Duplicate live chain/journal entry')
    grouped = {}
    for item in journal:
        case = indexed[item['pair_id']]
        if item['router_hash'] != protocol.digest(entry):
            raise ValueError('Journal router drift')
        grouped.setdefault(item['chain_id'], []).append(item)
        model = entry['model'] if item['call_index'] == 0 else (
            entry['peer'] if entry['peer'] and item['call_index'] == 1 else 'luna')
        if item['call_index'] not in range(3 if entry['peer'] else 2):
            raise ValueError('Unexpected live call index')
        check_call(item['call'], case, model, protocol, origin)
    for chain in finished:
        case = indexed[chain['pair_id']]
        logged = sorted(grouped.get(chain['chain_id'], []), key=lambda j: j['call_index'])
        if ([j['call_index'] for j in logged] != list(range(len(chain['calls'])))
                or any(j['pair_id'] != chain['pair_id'] for j in logged)
                or protocol.digest([j['call'] for j in logged]) != protocol.digest(chain['calls'])):
            raise ValueError('Chain and durable call journal disagree')
        if (chain['router_hash'] != protocol.digest(entry) or chain['input_hash'] != case['input_hash']
                or chain['primary_request_hash'] != protocol.digest(protocol.payload(case, entry['model'], 'B0'))):
            raise ValueError('Live chain provenance differs')
        if not math.isfinite(chain['wall_seconds']) or chain['wall_seconds'] < 0:
            raise ValueError('Invalid live wall duration')
        if chain['failed'] != any(c.get('error') for c in chain['calls']):
            raise ValueError('Live failure flag disagrees')
    for chain in success:
        if chain['failed'] or not chain['calls']:
            raise ValueError('Failed chain in success cache')
        calls = chain['calls']
        peer = [calls[1]['probability_match']] if entry['peer'] else None
        routed = bool(module.route(entry, [indexed[chain['pair_id']]], [calls[0]['probability_match']], peer)[0])
        if chain['routed'] != routed or len(calls) != 1+bool(entry['peer'])+routed:
            raise ValueError('Live route or stage count differs')
        if chain['prediction'] != calls[-1 if routed else 0]['prediction']:
            raise ValueError('Live final prediction differs')
    if any(not s['failed'] for s in failed):
        raise ValueError('Successful chain in failure cache')
    durations = [s['wall_seconds'] for s in success]
    attempts = [a for j in journal for a in j['call']['attempts']]
    return {'model': entry['model'], 'benchmark': entry['benchmark'], 'strategy': entry['strategy'],
            'n': len(success), 'routed': sum(s['routed'] for s in success),
            'end_to_end_p50_seconds': float(np.quantile(durations, .5)),
            'end_to_end_p95_seconds': float(np.quantile(durations, .95)),
            'failed_chains': len(failed), 'failed_chain_wall_seconds': sum(s['wall_seconds'] for s in failed),
            'interrupted_chains_with_journaled_calls': len(set(grouped)-set(ids)),
            'known_cost_usd_including_failures': sum(a['cost'] for a in attempts if a['cost'] is not None),
            'unknown_cost_attempts': sum(a['cost'] is None for a in attempts),
            'journaled_calls': len(journal), 'attempts': len(attempts)}


def export_live(workspace, module, protocol, origin, provenance, run_id):
    marker = read(workspace/'artifacts/expanded_cascades_live_complete.json')
    offline = read(workspace/'artifacts/expanded_cascades_offline_complete.json')
    selection = read(workspace/'configs/expanded_cascade_selection.json')
    if marker.get('status') != 'complete' or offline.get('selection_sha256') != protocol.digest(selection):
        raise ValueError('Live completion or frozen routing selection missing')
    module.verify()
    entries = [e for e in selection if e['status'] == 'frozen' and e['fraction'] == .25]
    if marker['cells'] != len(entries):
        raise ValueError('Live completion matrix differs')
    saved = read(workspace/'artifacts/results/expanded_cascades/live_latency.json')
    key = lambda x: (x['model'], x['strategy'], x['benchmark'])
    if len(saved) != len(entries) or len({key(e) for e in entries}) != len(entries) or {key(r) for r in saved} != {key(e) for e in entries}:
        raise ValueError('Live result matrix differs')
    reports = {key(r): r for r in saved}
    output, summaries, all_ids = [], [], set()
    for entry in entries:
        directory = workspace/'runs/cascade_live'/entry['model']/entry['strategy']/entry['benchmark']
        cases = read(workspace/'data'/entry['benchmark']/'cascade_live.json')
        success, failed, journal = [lines(directory/name) for name in ['chains.jsonl', 'failed_chains.jsonl', 'call_journal.jsonl']]
        ids = {j['chain_id'] for j in journal}
        if all_ids & ids:
            raise ValueError('Live chain reused across cells')
        all_ids |= ids
        result = verify_cell(entry, cases, success, failed, journal, module, protocol, origin)
        for k, v in result.items():
            if k in reports[key(entry)] and reports[key(entry)][k] != v:
                raise ValueError('Archived live result disagrees: '+k)
        summaries.append(result)
        provenance.append({'namespace': 'cascade_live', 'model': entry['model'], 'variant': entry['strategy'],
                           'benchmark': entry['benchmark'], 'split': 'cascade_live', 'response_count': len(success),
                           'request_hashes_verified': True,
                           'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.jsonl')}})
        values = {k: result[k] for k in ['end_to_end_p50_seconds', 'end_to_end_p95_seconds', 'failed_chains',
                  'interrupted_chains_with_journaled_calls']}
        values.update(routed_fraction=result['routed']/result['n'],
                      known_cost_usd_per_1000=1000*result['known_cost_usd_including_failures']/result['n'],
                      unknown_attempts_per_1000=1000*result['unknown_cost_attempts']/result['n'])
        output.extend({'run_id': run_id, 'model': entry['model'], 'benchmark': entry['benchmark'],
                       'condition': 'live/'+entry['strategy']+'/0.25', 'split': 'cascade_live',
                       'n': result['n'], 'metric': k, 'value': v} for k, v in values.items())
    return output, summaries
