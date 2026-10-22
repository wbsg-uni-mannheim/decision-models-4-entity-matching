"""Resumable, provider-pinned OpenRouter inference with a shared spending ceiling."""
import argparse
import concurrent.futures
import fcntl
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from protocol import MODELS, BENCHMARKS, canonical, digest, payload, parse_response

PROJECT = Path(__file__).resolve().parents[1]
ENDPOINTS = {'decision': 'https://openrouter.ai/api/alpha/decisions',
             'generative': 'https://openrouter.ai/api/v1/chat/completions'}

def key():
    for name in ['OPENROUTER_API_KEY', 'OPEN_ROUTER_API_KEY']:
        if os.environ.get(name):
            return os.environ[name]
    for line in (PROJECT/'.secrets/source.env').read_text().splitlines():
        if line.strip().startswith('#') or '=' not in line:
            continue
        name, value = line.split('=', 1)
        if name.strip() in ['OPENROUTER_API_KEY', 'OPEN_ROUTER_API_KEY']:
            return value.strip().strip('\"\'')
    raise RuntimeError('OpenRouter credential unavailable')

def account(action, amount=0, cap=100):
    """POSIX locking across workers; no secrets, no SQLite WAL on shared storage.

    Reserve $0.05 per request: deliberately exceeds expected bounded-request costs.
    A crash retains the reservation and forces an explicit accounting reconciliation.
    """
    path = PROJECT/'artifacts/budget.json'
    with (PROJECT/'artifacts/budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        obj = json.loads(path.read_text()) if path.exists() else {'spent': 0., 'reserved': 0., 'attempts': 0, 'unknown_cost_attempts': 0}
        if action == 'reserve':
            if obj['spent'] + obj['reserved'] + .05 > cap:
                raise RuntimeError('Project spending ceiling reached')
            obj['reserved'] += .05
            obj['attempts'] += 1
        elif action == 'settle':
            obj['reserved'] = max(0, obj['reserved'] - .05)
            if amount is None:
                obj['unknown_cost_attempts'] += 1
                obj['spent'] += .05  # conservative accounting, not reported as measured cost
            else:
                obj['spent'] += amount
        tmp = path.with_suffix('.tmp')
        tmp.write_text(canonical(obj)+'\n')
        tmp.replace(path)
        return obj

def request(case, alias, variant, credential):
    body = payload(case, alias, variant)
    request_hash = digest(body)
    attempts = []
    for attempt in range(3):
        account('reserve')
        start = time.perf_counter()
        data = None
        error = None
        status = None
        cost = None
        retry = False
        try:
            req = urllib.request.Request(ENDPOINTS[MODELS[alias]['kind']], data=canonical(body).encode(), headers={
                'Authorization': 'Bearer '+credential, 'Content-Type': 'application/json',
                'X-OpenRouter-Title': 'Decision EM controlled study'})
            with urllib.request.urlopen(req, timeout=120) as response:
                status = response.status
                data = json.load(response)
            cost = data.get('usage', {}).get('cost')
            if type(cost) not in (float, int) or cost < 0:
                cost = None
                raise ValueError('Missing actual usage cost')
            prediction, probability = parse_response(alias, data)
        except urllib.error.HTTPError as exc:
            status = exc.code
            error = 'HTTP '+str(status)
            # Store provider diagnostics with credential redaction; never headers.
            raw = exc.read().decode(errors='replace').replace(credential, '[REDACTED]')
            try:
                data = json.loads(raw)
            except ValueError:
                data = {'error_body': raw[:4000]}
            retry = status in [408, 429, 500, 502, 503, 504]
            # Rejected requests have no generation cost; timeouts/5xx remain unknown.
            if status in [400, 401, 402, 403, 404, 422, 429]:
                cost = 0.
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = type(exc).__name__
            retry = True
        except (ValueError, KeyError, TypeError) as exc:
            error = (type(exc).__name__+': '+str(exc)).replace(credential, '[REDACTED]')[:500]
        finally:
            account('settle', cost)
        attempts.append({'status': status, 'seconds': time.perf_counter()-start,
                         'cost': cost, 'error': error, 'response': data})
        if error is None:
            return {'benchmark': case['benchmark'], 'pair_id': case['pair_id'],
                    'input_hash': case['input_hash'], 'request_hash': request_hash,
                    'model_alias': alias, 'variant': variant, 'prediction': prediction,
                    'probability_match': probability, 'attempts': attempts,
                    'completed_at': datetime.now(timezone.utc).isoformat()}
        if not retry or attempt == 2:
            break
        time.sleep(min(30, 2**(attempt+1)))
    return {'benchmark': case['benchmark'], 'pair_id': case['pair_id'], 'input_hash': case['input_hash'],
            'request_hash': request_hash, 'model_alias': alias, 'variant': variant,
            'error': error, 'attempts': attempts, 'completed_at': datetime.now(timezone.utc).isoformat()}

def run_cell(alias, variant, benchmark, split, workers=8, limit=None, namespace='quality'):
    cases = json.loads((PROJECT/'data'/benchmark/(split+'.json')).read_text())
    if limit:
        cases = cases[:limit]
    directory = PROJECT/'runs'/namespace/alias/variant/benchmark
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(split+'.jsonl')
    with (directory/(split+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = {}
        if path.exists():
            for line in path.read_text().splitlines():
                result = json.loads(line)
                assert result['pair_id'] not in existing, 'Duplicate cached prediction'
                existing[result['pair_id']] = result
        wanted = {r['pair_id']: r for r in cases}
        for pid, result in existing.items():
            if pid in wanted:
                assert result['request_hash'] == digest(payload(wanted[pid], alias, variant)), 'Cache configuration mismatch'
        pending = [r for r in cases if r['pair_id'] not in existing]
        credential = key() if pending else None
        errors = 0
        start = time.perf_counter()
        # Bounded batches make outages fail early rather than enqueueing thousands of paid requests.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for offset in range(0, len(pending), workers):
                batch = pending[offset:offset+workers]
                futures = [executor.submit(request, c, alias, variant, credential) for c in batch]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result.get('error'):
                        errors += 1
                        with (directory/(split+'.failures.jsonl')).open('a') as out:
                            out.write(canonical(result)+'\n')
                        print('FAIL', alias, variant, benchmark, split, result['error'], flush=True)
                    else:
                        with path.open('a') as out:
                            out.write(canonical(result)+'\n')
                        existing[result['pair_id']] = result
                if errors:
                    raise RuntimeError(f'{alias}/{variant}/{benchmark}/{split}: failed batch; inspect diagnostics before resume')
                if offset % (workers*25) == 0:
                    print('PROGRESS', alias, variant, benchmark, split, len(existing), '/', len(cases), flush=True)
        elapsed = time.perf_counter()-start
        if pending:
            execution = {'new_pairs':len(pending), 'wall_seconds':elapsed, 'workers':workers,
                         'throughput_pairs_per_second':len(pending)/elapsed,
                         'completed_at':datetime.now(timezone.utc).isoformat()}
            with (directory/(split+'.executions.jsonl')).open('a') as out:
                out.write(canonical(execution)+'\n')
        print('CELL', alias, variant, benchmark, split, len(cases), round(elapsed, 2), flush=True)
        assert wanted.keys() <= existing.keys()
    return len(cases)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--variant', default='B0')
    ap.add_argument('--splits', nargs='+', default=['prompt', 'route', 'test'])
    ap.add_argument('--benchmarks', nargs='+', default=list(BENCHMARKS))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--namespace', default='quality')
    args = ap.parse_args()
    for split in args.splits:
        for b in args.benchmarks:
            run_cell(args.model, args.variant, b, split, args.workers, args.limit, args.namespace)

if __name__ == '__main__':
    main()
