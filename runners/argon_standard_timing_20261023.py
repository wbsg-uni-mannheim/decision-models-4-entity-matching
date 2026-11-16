"""Throughput of Gemini 4 Argon on the standard tier (Aaron, 23 Oct: "Ja, miss Argon im Standard-Tarif auf
1.000 eskalierten Paaren mit 32 Workern (bis 3 USD)").

The escalation run measured Argon on the flex tier of Google AI Studio, which is served with lower priority.
This run sends the escalation body of 1,000 escalated pairs (uniform, fixed: ordered by
sha256(benchmark|pair_id)) to the standard Google AI Studio endpoint, without service_tier, with 32 parallel
requests, and records per-request latency and the wall-clock time of the whole run.

    python3 artifacts/operations/argon_standard_timing_20261023.py --root .
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

SPLIT = 'escalation'
N = 1000
WORKERS = 32
OUT = 'runs/argon_standard_timing_20261023'
APPROVAL = 'artifacts/operations/approval_argon_standard_timing_aaron_20261023.json'
TIMEOUT = 120
TRIES = 6
HOLD = .01
RETRY = (408, 409, 429, 500, 502, 503, 504)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def modules(root):
    sys.path[:0] = [str(root/'src'), str(root/'artifacts/operations')]
    import protocol
    import escalation
    import run
    escalation.register_models()
    return protocol, escalation, run


def sample(root, protocol):
    cases = [c for b in protocol.BENCHMARKS for c in read(root/'data'/b/(SPLIT+'.json'))]
    cases.sort(key=lambda c: hashlib.sha256(f"{c['benchmark']}|{c['pair_id']}".encode()).hexdigest())
    return cases[:N]


def body(escalation, case):
    b = escalation.payload(case, 'argon4')
    b.pop('service_tier', None)
    b['provider'] = {**b['provider'], 'only': ['google-ai-studio'], 'order': ['google-ai-studio']}
    return b


def post(key, data):
    req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', data=json.dumps(data).encode(),
                                 headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def ledger(root, commitment, action, cap=0., quote=None, spent=0.):
    path = root/'artifacts/budget.json'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        obj = json.loads(path.read_text())
        rows = obj.setdefault('openrouter_live_commitments', {})
        if action == 'reserve':
            require(commitment not in rows, 'Commitment exists; never reserve twice')
            require(obj['spent']+obj['reserved']+cap <= obj['project_cap_usd']+1e-9, 'Project cap reached')
            obj['reserved'] += cap
            rows[commitment] = {'reserved_usd': cap, 'status': 'reserved', 'actual_usd': None,
                                'approval_reference': quote, 'model': 'google/gemini-4-argon', 'tier': 'standard'}
        else:
            row = rows[commitment]
            require(row['status'] == 'reserved', 'Commitment already closed')
            obj['reserved'] -= row['reserved_usd']; obj['spent'] += spent
            row.update(status='reconciled', actual_usd=spent)
        tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, sort_keys=True)+'\n'); tmp.replace(path)


def main(root):
    protocol, escalation, run_mod = modules(root)
    approval = read(root/APPROVAL)
    require(approval.get('approved_by') == 'Aaron' and approval.get('quote') and approval.get('member') == 'argon4',
            'Approval required')
    cap = float(approval['run_cap_usd'])
    cases = sample(root, protocol)
    out = root/OUT
    out.mkdir(parents=True, exist_ok=False)
    (out/'sample.json').write_text(json.dumps([[c['benchmark'], c['pair_id']] for c in cases])+'\n')
    key = run_mod.key()
    commitment = 'argon-standard-timing-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    ledger(root, commitment, 'reserve', cap, approval['quote'])
    state = {'spent': 0., 'open': 0}
    lock, jlock = threading.Lock(), threading.Lock()
    journal = (out/'results.jsonl').open('x')

    def one(c):
        with lock:
            if state['spent']+HOLD*(state['open']+1) > cap:
                return {'benchmark': c['benchmark'], 'pair_id': c['pair_id'], 'error': 'run cap reached', 'attempts': []}
            state['open'] += 1
        data, attempts, spent = body(escalation, c), [], 0.
        try:
            for attempt in range(TRIES):
                t = time.perf_counter()
                try:
                    resp = post(key, data)
                    c_usd = float((resp.get('usage') or {}).get('cost') or 0.); spent += c_usd
                    attempts.append({'seconds': time.perf_counter()-t, 'status': 200, 'cost': c_usd,
                                     'service_tier': resp.get('service_tier'), 'response': resp})
                    prediction, _ = escalation.parse_response('argon4', resp)
                    return {'benchmark': c['benchmark'], 'pair_id': c['pair_id'], 'prediction': prediction, 'attempts': attempts}
                except urllib.error.HTTPError as e:
                    attempts.append({'seconds': time.perf_counter()-t, 'status': e.code, 'cost': 0.,
                                     'detail': e.read().decode(errors='replace')[:500]})
                    if e.code not in RETRY:
                        break
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    attempts.append({'seconds': time.perf_counter()-t, 'error': type(e).__name__, 'cost': None})
                except (ValueError, KeyError, TypeError) as e:
                    attempts.append({'seconds': time.perf_counter()-t, 'error': (type(e).__name__+': '+str(e))[:300], 'cost': None})
                    break
                time.sleep(min(60, 2*2**attempt))
            return {'benchmark': c['benchmark'], 'pair_id': c['pair_id'], 'error': 'failed', 'attempts': attempts}
        finally:
            with lock:
                state['open'] -= 1; state['spent'] += spent

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        for r in pool.map(one, cases):
            with jlock:
                journal.write(json.dumps(r, ensure_ascii=False)+'\n'); journal.flush(); os.fsync(journal.fileno())
    wall = time.perf_counter()-start
    journal.close()
    ledger(root, commitment, 'reconcile', spent=state['spent'])
    rows = [json.loads(x) for x in (out/'results.jsonl').read_text().splitlines()]
    ok = [r for r in rows if 'prediction' in r]
    lat = sorted(a['seconds'] for r in ok for a in r['attempts'] if a.get('status') == 200)
    tiers = {}
    for r in ok:
        t = r['attempts'][-1].get('service_tier'); tiers[t] = tiers.get(t, 0)+1
    summary = {'pairs': len(rows), 'valid': len(ok), 'workers': WORKERS, 'wall_seconds': wall,
               'pairs_per_second': len(ok)/wall, 'seconds_per_1k': 1000*wall/max(1, len(ok)),
               'latency_median_s': lat[len(lat)//2] if lat else None, 'latency_p90_s': lat[int(.9*len(lat))] if lat else None,
               'served_tiers': tiers, 'cost_usd': state['spent'], 'commitment': commitment}
    (out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    print(json.dumps(main(ap.parse_args().root.resolve()), indent=1))
