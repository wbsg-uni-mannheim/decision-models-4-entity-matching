"""Claude Opus 5.5 with normal OpenRouter requests on 1,000 of the remaining test pairs (Aaron, 22 Oct:
"Ne reichen doch 1000 stück" instead of rerunning all 23,543 pairs).

The sample is uniform and fixed: the reference_rest pairs ordered by sha256(benchmark|pair_id), first N.
The request body is exactly the escalation body (src/escalation.py, Anthropic only, no fallback), so the
answers can be compared with the batch answers in runs/reference_rest_run01_20261021/opus55_batch.
Cost per request is OpenRouter's usage.cost; the run reserves the approved cap in the shared ledger.

    python3 artifacts/operations/reference_opus_live_sample_20261022.py --root . --action run [--fill]
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

SPLIT = 'reference_rest'
N = 1000
OUT = 'runs/reference_rest_run01_20261021/opus55_live_sample1000'
APPROVAL = 'artifacts/operations/approval_reference_opus_live_sample_aaron_20261022.json'
WORKERS = 12
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


def post(key, body):
    req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


class Budget:
    def __init__(self, root, commitment, cap, quote):
        self.root, self.commitment, self.cap, self.spent, self.open = root, commitment, cap, 0., 0
        self.lock = threading.Lock()
        self.ledger('reserve', quote)

    def ledger(self, action, quote=None):
        path = self.root/'artifacts/budget.json'
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            obj = json.loads(path.read_text())
            rows = obj.setdefault('openrouter_live_commitments', {})
            if action == 'reserve':
                require(self.commitment not in rows, 'Commitment exists; never reserve twice')
                require(obj['spent']+obj['reserved']+self.cap <= obj['project_cap_usd']+1e-9, 'Project cap reached')
                obj['reserved'] += self.cap
                rows[self.commitment] = {'reserved_usd': self.cap, 'status': 'reserved', 'actual_usd': None,
                                         'approval_reference': quote, 'model': 'anthropic/claude-opus-5.5'}
            else:
                row = rows[self.commitment]
                require(row['status'] == 'reserved', 'Commitment already closed')
                obj['reserved'] -= row['reserved_usd']; obj['spent'] += self.spent
                row.update(status='reconciled', actual_usd=self.spent)
            tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, sort_keys=True)+'\n'); tmp.replace(path)

    def admit(self):
        with self.lock:
            if self.spent+HOLD*(self.open+1) > self.cap:
                return False
            self.open += 1
            return True

    def settle(self, amount):
        with self.lock:
            self.open -= 1; self.spent += amount


def valid_keys(out):
    done = set()
    for path in sorted(out.glob('results*.jsonl')):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if 'prediction' in r:
                done.add((r['benchmark'], r['pair_id']))
    return done


def run(root, fill=False):
    protocol, escalation, run_mod = modules(root)
    approval = read(root/APPROVAL)
    require(approval.get('approved_by') == 'Aaron' and approval.get('quote') and approval.get('member') == 'opus55',
            'Approval required')
    cases = sample(root, protocol)
    out = root/OUT
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    if fill:
        require(out.exists(), 'No earlier run to fill')
        done = valid_keys(out)
        cases = [c for c in cases if (c['benchmark'], c['pair_id']) not in done]
        name = f'results.fill-{stamp}.jsonl'
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out/'sample.json').write_text(json.dumps([[c['benchmark'], c['pair_id']] for c in cases])+'\n')
        name = 'results.jsonl'
    key = run_mod.key()
    budget = Budget(root, f'reference-opus-live-{stamp}', float(approval['run_cap_usd']), approval['quote'])
    journal, jlock = (out/name).open('x'), threading.Lock()

    def one(c):
        base = {'benchmark': c['benchmark'], 'pair_id': c['pair_id'], 'input_hash': c['input_hash'],
                'model_alias': 'opus55', 'variant': 'E0', 'transport': 'openrouter-live'}
        if not budget.admit():
            return {**base, 'error': 'run cap reached', 'attempts': []}
        body = escalation.payload(c, 'opus55')
        base['request_hash'] = protocol.digest(body)
        attempts, spent = [], 0.
        try:
            for attempt in range(TRIES):
                t = time.perf_counter()
                try:
                    resp = post(key, body)
                    c_usd = float((resp.get('usage') or {}).get('cost') or 0.); spent += c_usd
                    attempts.append({'seconds': time.perf_counter()-t, 'status': 200, 'cost': c_usd, 'response': resp})
                    prediction, _ = escalation.parse_response('opus55', resp)
                    return {**base, 'prediction': prediction, 'probability_match': None, 'attempts': attempts,
                            'completed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                except urllib.error.HTTPError as e:
                    attempts.append({'seconds': time.perf_counter()-t, 'status': e.code, 'cost': 0.,
                                     'detail': e.read().decode(errors='replace')[:500]})
                    if e.code not in RETRY:
                        break
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    attempts.append({'seconds': time.perf_counter()-t, 'error': type(e).__name__, 'cost': None})
                except (ValueError, KeyError, TypeError) as e:
                    note = (type(e).__name__+': '+str(e))[:300]
                    if attempts and attempts[-1].get('status') == 200 and 'error' not in attempts[-1]:
                        attempts[-1]['error'] = note
                    else:
                        attempts.append({'seconds': time.perf_counter()-t, 'error': note, 'cost': None})
                    break
                time.sleep(min(120, 5*2**attempt))
            return {**base, 'error': 'failed', 'attempts': attempts}
        finally:
            budget.settle(spent)

    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        for r in pool.map(one, cases):
            with jlock:
                journal.write(json.dumps(r, ensure_ascii=False)+'\n'); journal.flush(); os.fsync(journal.fileno())
    journal.close()
    budget.ledger('reconcile')
    summary = {'requested': len(cases), 'cost_usd': budget.spent,
               'missing_after_this_pass': len({(c['benchmark'], c['pair_id']) for c in sample(root, protocol)} - valid_keys(out))}
    (out/name.replace('results', 'summary').replace('.jsonl', '.json')).write_text(json.dumps(summary, indent=1)+'\n')
    return summary


def compare(root):
    """Agreement of the live answers with the batch answers on the sample, per benchmark and overall."""
    protocol, _, _ = modules(root)
    live = {}
    for path in sorted((root/OUT).glob('results*.jsonl')):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if 'prediction' in r:
                live[(r['benchmark'], r['pair_id'])] = r['prediction']
    batch = {(r['benchmark'], r['pair_id']): r.get('prediction')
             for r in map(json.loads, (root/'runs/reference_rest_run01_20261021/opus55_batch/results.jsonl').read_text().splitlines())}
    labels = {(c['benchmark'], c['pair_id']): c['label'] for c in sample(root, protocol)}
    per = {}
    for k, p in live.items():
        s = per.setdefault(k[0], {'n': 0, 'agree': 0, 'live_correct': 0, 'batch_correct': 0})
        s['n'] += 1; s['agree'] += p == batch[k]
        s['live_correct'] += p == labels[k]; s['batch_correct'] += batch[k] == labels[k]
    total = {f: sum(s[f] for s in per.values()) for f in ('n', 'agree', 'live_correct', 'batch_correct')}
    result = {'overall': total, 'per_benchmark': per}
    (root/OUT/'comparison.json').write_text(json.dumps(result, indent=1)+'\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--action', choices=['run', 'compare'], required=True)
    ap.add_argument('--fill', action='store_true')
    a = ap.parse_args()
    print(json.dumps((run(a.root.resolve(), a.fill) if a.action == 'run' else compare(a.root.resolve())), indent=1))
