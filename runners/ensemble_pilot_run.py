"""One-shot technical pilot. No retry/resume; unknown outcomes retain USD10 hold."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from prepare_ensemble_pilot import canonical, digest
from strong_ensemble import MEMBERS, parse_vote


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def money(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Missing/invalid provider-reported cost')
    return float(value)


def ledger(root, package_hash, action, actual=None, prior_cost=None):
    path = root/'artifacts/budget.json'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        obj = json.loads(path.read_text())
        entries = obj.setdefault('ensemble_pilot_commitments', {})
        if action == 'reserve':
            hold = 10.
            if prior_cost is not None:
                prior_cost = money(prior_cost)
                entry = entries[package_hash]
                if (entry['status'] != 'reconciled' or entry.get('resume_count', 0) != 0
                        or abs(entry['actual_usd']-prior_cost) > 1e-10 or prior_cost >= 10):
                    raise ValueError('Prior known cost not settled or already resumed')
                hold -= prior_cost
            elif entries:
                raise ValueError('Existing pilot commitment')
            if obj['spent']+obj['reserved']+hold > 100:
                raise ValueError('Existing commitment or insufficient shared USD100 budget')
            entries[package_hash] = {'reserved_usd': hold, 'status': 'reserved',
                'prior_actual_usd': prior_cost or 0., 'resume_count': int(prior_cost is not None),
                'authorization': 'docs/APPROVED_TODO_20261014.md; operator admission verifies broad ensemble authorization'}
            obj['reserved'] += hold
        elif action == 'reconcile_known':
            actual = money(actual)
            entry = entries[package_hash]
            hold = entry['reserved_usd']
            previous = entry.get('prior_actual_usd', 0.)
            if entry['status'] != 'reserved' or obj['reserved']+1e-8 < hold or actual < previous:
                raise ValueError('Invalid or already reconciled hold')
            obj['reserved'] = max(0., obj['reserved']-hold)
            obj['spent'] += actual-previous
            entry.update(status='reconciled', actual_usd=actual,
                         ceiling_exceeded=actual > 10.)
        else:
            raise ValueError('Unsupported ledger action')
        tmp = path.with_suffix('.tmp')
        tmp.write_text(canonical(obj)+'\n'); tmp.replace(path)
        return obj


def validate(root, package, admission, active):
    manifest = json.loads((package/'manifest.json').read_text())
    package_hash = sha(package/'requests.jsonl')
    if manifest['requests_sha256'] != package_hash or manifest['pilot_ceiling_usd'] != 10:
        raise ValueError('Pilot manifest changed')
    if active or not admission.get('all_project_inference_jobs_reviewed'):
        raise ValueError('Existing inference chain must finish first')
    if admission.get('package_sha256') != package_hash or admission.get('broad_ensemble_authorization_verified') is not True:
        raise ValueError('Operator must verify actual broad user authorization and exact package')
    if admission.get('todo_sha256') != sha(root/'docs/APPROVED_TODO_20261014.md'):
        raise ValueError('Wrong authorization document')
    if admission.get('runner_sha256') != sha(Path(__file__)):
        raise ValueError('Admission must freeze tested runner')
    if manifest['ensemble_implementation_sha256'] != sha(Path(__file__).parent/'strong_ensemble.py'):
        raise ValueError('Ensemble primitive changed')
    rows = [json.loads(s) for s in (package/'requests.jsonl').read_text().splitlines()]
    if len(rows) != 48 or len({r['request_id'] for r in rows}) != 48:
        raise ValueError('Expected exactly 48 unique requests')
    if sum(money(r['planning_allowance_usd']) for r in rows) > 10:
        raise ValueError('Pilot planning allowance exceeds ceiling')
    for r in rows:
        body, spec = r['body'], MEMBERS[r['member']]
        if digest(body) != r['request_sha256'] or body['model'] != spec['id']:
            raise ValueError('Request hash/model mismatch')
        if body['provider'] != {'only': [spec['provider']], 'order': [spec['provider']],
                'allow_fallbacks': False, 'require_parameters': True}:
            raise ValueError('Provider pin changed')
        if body['max_tokens'] != 2048 or body['reasoning'] != {'effort': 'medium'} or 'tools' in body:
            raise ValueError('Output/reasoning/tool settings changed')
    return package_hash, rows


def interpret(member, data):
    cost = money((data.get('usage') or {}).get('cost'))
    spec = MEMBERS[member]
    expected_provider = {'sol61': 'OpenAI', 'opus55': 'Anthropic', 'argon4': 'Google'}[member]
    if data.get('model') != spec['id'] or data.get('provider') != expected_provider:
        raise ValueError('Observed model/provider identity differs; inspect without fallback')
    # Google Vertex region is a requested pin, not independently proven by this field.
    return cost, parse_vote(data)


def post(key, body):
    request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
        data=canonical(body).encode(), headers={'Authorization': 'Bearer '+key,
        'Content-Type': 'application/json', 'X-OpenRouter-Title': 'Decision EM ensemble technical pilot'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors='replace').replace(key, '[REDACTED]')
        try:
            data = json.loads(raw)
        except ValueError:
            data = {'error_body': raw}
        return error.code, data


def known_prefix(path, rows):
    events = [json.loads(s) for s in path.read_text().splitlines()]
    if not events or events[-1].get('event') != 'stopped_known_cost':
        raise ValueError('Only diagnosed fully known outcomes can be resumed')
    if events[-1].get('reason') != 'Observed model/provider identity differs; inspect without fallback':
        raise ValueError('Continuation restricted to the diagnosed provider display-name mismatch')
    responses = [e for e in events if e['event'] == 'response']
    starts = [e for e in events if e['event'] == 'request_start_unknown_until_response']
    if not responses or len(responses) >= len(rows) or len(starts) != len(responses):
        raise ValueError('Unknown, empty or complete prior request prefix')
    total = 0.
    for expected, response, start in zip(rows, responses, starts):
        if (response['request_id'] != expected['request_id'] or start['request_id'] != expected['request_id']
                or start['request_sha256'] != expected['request_sha256'] or response['status'] != 200):
            raise ValueError('Prior requests differ from frozen ordered prefix')
        cost, _ = interpret(expected['member'], response['data'])
        total += cost
    if abs(total-events[-1]['known_cost_usd']) > 1e-10:
        raise ValueError('Prior cost does not match responses')
    return responses, total


def run(root, package, admission, key, active, transport=post, prior_journal=None):
    package_hash, rows = validate(root, package, admission, active)
    prefix, total = ([], 0.) if prior_journal is None else known_prefix(prior_journal, rows)
    if prior_journal is not None and admission.get('prior_journal_sha256') != sha(prior_journal):
        raise ValueError('Prior journal must be frozen in continuation admission')
    # Exclusive durable journal means an uncertain submission can never be retried blindly.
    with (package/'execution.jsonl').open('x') as f:
        def event(kind, **fields):
            f.write(canonical({'event': kind, **fields})+'\n'); f.flush(); os.fsync(f.fileno())
        event('validated', package_sha256=package_hash)
        ledger(root, package_hash, 'reserve', prior_cost=total if prefix else None)
        event('reserved', usd=10-total)
        if prefix:
            event('validated_prior_prefix', journal_sha256=sha(prior_journal), requests=len(prefix), already_booked_usd=total)
            for r, response in zip(rows, prefix):
                cost, vote = interpret(r['member'], response['data'])
                event('prior_response_reused_without_api_call', **{k:v for k,v in response.items() if k!='event'})
                event('valid_vote', request_id=r['request_id'], vote=vote, cost_usd=cost, source='prior_known_response')
        for r in rows[len(prefix):]:
            if total+r['planning_allowance_usd'] > 10:
                ledger(root, package_hash, 'reconcile_known', total)
                event('stopped_before_request', reason='pilot ceiling', known_cost_usd=total)
                return
            event('request_start_unknown_until_response', request_id=r['request_id'], request_sha256=r['request_sha256'])
            start = time.monotonic()
            try:
                status, data = transport(key, r['body'])
            except Exception as exc:
                event('transport_unknown_hold_retained', request_id=r['request_id'], error_type=type(exc).__name__)
                return
            # Persist complete raw result before parsing or reconciling any costs.
            data = json.loads(canonical(data).replace(key, '[REDACTED]')) if key else data
            event('response', request_id=r['request_id'], status=status, data=data, seconds=time.monotonic()-start)
            try:
                cost = money((data.get('usage') or {}).get('cost'))
            except (ValueError, AttributeError):
                event('unknown_cost_hold_retained', known_cost_usd=total)
                return
            total += cost
            try:
                if status != 200:
                    raise ValueError('Non-200 response')
                _, vote = interpret(r['member'], data)
                if total > 10:
                    raise ValueError('Provider charge exceeded reserved ceiling')
                event('valid_vote', request_id=r['request_id'], vote=vote, cost_usd=cost)
            except (ValueError, KeyError, TypeError) as exc:
                ledger(root, package_hash, 'reconcile_known', total)
                event('stopped_known_cost', reason=str(exc), known_cost_usd=total)
                return
        ledger(root, package_hash, 'reconcile_known', total)
        event('complete_technical_pilot_only', requests=len(rows), known_cost_usd=total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    admission = json.loads(args.admission.read_text())
    queue = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i %j'], text=True)
    excluded = set(admission.get('reviewed_nonstudy_job_ids', []))
    known = {'365636','365221','365268','365279','365288','365300'}
    active = [line for line in queue.splitlines() if line.split()[0] in known or line.split()[0] not in excluded]
    import sys
    sys.path.insert(0, str(args.root/'src'))
    import run as frozen_run
    run(args.root, args.package, admission, frozen_run.key(), active)
