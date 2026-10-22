"""Reference run, GPT-6.1 Sol on the remaining test pairs, direct OpenAI API, flex tier.

Tier switch (Aaron, 22 Oct): flex was mostly unavailable (429 flex_unavailable), so the remaining
pairs run on the standard tier (--tier default, approval_reference_sol_standard_aaron_20261022.json).
The killed flex job's commitment is closed with --action close-stale from its journal (lower bound:
responses in flight at the kill are not journaled).

Same messages, JSON schema, reasoning effort and output limit as the escalation requests
(src/reference_rest.py). Flex outages ("Flex processing is temporarily unavailable") are
retried patiently; cases without a valid answer are re-requested with --fill. Billing follows
the tier that served the request (response field service_tier).

    python3 artifacts/operations/reference_sol_direct_20261021.py --root . --action run [--fill] [--tier default --approval PATH]
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
OUT = 'runs/reference_rest_run01_20261021/sol61_direct'
APPROVAL = 'artifacts/operations/approval_reference_sol_aaron_20261021.json'
WORKERS = 16
TIMEOUT = 120
TRIES = 6
# USD per token, GPT-6.1 Sol: standard 2 / 0.10 cached / 10 per 1M, flex 50 %.
PRICE = {'default': (2e-6, 1e-7, 1e-5), 'flex': (1e-6, 5e-8, 5e-6)}
RETRY = (408, 409, 429, 500, 502, 503, 504)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def modules(root):
    sys.path[:0] = [str(root/'src'), str(root/'artifacts/operations')]
    import protocol
    import reference_rest
    import escalation
    escalation.register_models()
    return protocol, reference_rest


def cost(resp):
    u = resp['usage']
    p_in, p_cached, p_out = PRICE['flex' if resp.get('service_tier') == 'flex' else 'default']
    cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
    return (u['prompt_tokens']-cached)*p_in + cached*p_cached + u['completion_tokens']*p_out


def parse(resp, reference_rest):
    require(str(resp.get('model', '')).startswith(reference_rest.SOL_MODEL), 'Unexpected model '+str(resp.get('model')))
    choice, = resp['choices']
    require(choice.get('finish_reason') == 'stop' and not choice['message'].get('refusal'), 'Incomplete or refused answer')
    return reference_rest.vote(choice['message']['content'])


def post(key, body):
    req = urllib.request.Request('https://api.openai.com/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


class Budget:
    def __init__(self, root, commitment, cap, quote, tier='flex'):
        self.root, self.commitment, self.cap, self.spent, self.open, self.tier = root, commitment, cap, 0., 0, tier
        self.lock = threading.Lock()
        self.ledger('reserve', quote)

    def ledger(self, action, quote=None):
        path = self.root/'artifacts/budget.json'
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            obj = json.loads(path.read_text())
            rows = obj.setdefault('direct_openai_commitments', {})
            if action == 'reserve':
                require(self.commitment not in rows, 'Commitment exists; never reserve twice')
                require(obj['spent']+obj['reserved']+self.cap <= obj['project_cap_usd']+1e-9, 'Project cap reached')
                obj['reserved'] += self.cap
                rows[self.commitment] = {'reserved_usd': self.cap, 'status': 'reserved', 'actual_usd': None,
                                         'approval_reference': quote, 'model': 'gpt-6.1-sol', 'tier': self.tier}
            else:
                row = rows[self.commitment]
                require(row['status'] == 'reserved', 'Commitment already closed')
                obj['reserved'] -= row['reserved_usd']; obj['spent'] += self.spent
                row.update(status='reconciled', actual_usd=self.spent)
            tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, sort_keys=True)+'\n'); tmp.replace(path)

    def admit(self, hold=.02):
        with self.lock:
            if self.spent+hold*(self.open+1) > self.cap:
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


def run(root, fill=False, tier='flex', approval_path=APPROVAL, workers=WORKERS):
    protocol, reference_rest = modules(root)
    approval = read(root/approval_path)
    require(approval.get('approved_by') == 'Aaron' and approval.get('quote') and approval.get('member') == 'sol61',
            'Approval required')
    cases = [c for b in protocol.BENCHMARKS for c in read(root/'data'/b/(SPLIT+'.json'))]
    out = root/OUT
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    if fill:
        require(out.exists(), 'No earlier run to fill')
        done = valid_keys(out)
        cases = [c for c in cases if (c['benchmark'], c['pair_id']) not in done]
        name = f'results.fill-{stamp}.jsonl'
    else:
        out.mkdir(parents=True, exist_ok=False)
        name = 'results.jsonl'
    import astra_discovery_submit
    key = astra_discovery_submit.credential(root)
    import uuid
    budget = Budget(root, f'reference-sol-{stamp}-{uuid.uuid4().hex[:8]}', float(approval['run_cap_usd']), approval['quote'], tier)
    journal, jlock = (out/name).open('x'), threading.Lock()

    def one(c):
        base = {'benchmark': c['benchmark'], 'pair_id': c['pair_id'], 'input_hash': c['input_hash'],
                'model_alias': 'sol61', 'variant': 'E0', 'transport': 'openai-direct-'+tier}
        if not budget.admit():
            return {**base, 'error': 'run cap reached', 'attempts': []}
        body = reference_rest.sol_direct_body(c)
        body['service_tier'] = tier
        base['request_hash'] = protocol.digest(body)
        attempts, spent = [], 0.
        try:
            for attempt in range(TRIES):
                t = time.perf_counter()
                try:
                    resp = post(key, body)
                    c_usd = cost(resp); spent += c_usd
                    attempts.append({'seconds': time.perf_counter()-t, 'status': 200, 'cost': c_usd,
                                     'service_tier': resp.get('service_tier'), 'response': resp})
                    return {**base, 'prediction': parse(resp, reference_rest), 'probability_match': None,
                            'attempts': attempts, 'completed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
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

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for r in pool.map(one, cases):
            with jlock:
                journal.write(json.dumps(r, ensure_ascii=False)+'\n'); journal.flush(); os.fsync(journal.fileno())
    journal.close()
    budget.ledger('reconcile')
    rows = [json.loads(x) for x in (out/name).read_text().splitlines()]
    tiers = {}
    for r in rows:
        for a in r['attempts']:
            if a.get('status') == 200:
                tiers[a.get('service_tier')] = tiers.get(a.get('service_tier'), 0)+1
    summary = {'requested': len(rows), 'valid': sum('prediction' in r for r in rows), 'cost_usd': budget.spent,
               'served_tiers': tiers, 'missing_after_this_pass': len({(c['benchmark'], c['pair_id']) for b in protocol.BENCHMARKS
                                                                      for c in read(root/'data'/b/(SPLIT+'.json'))} - valid_keys(out))}
    (out/name.replace('results', 'summary').replace('.jsonl', '.json')).write_text(json.dumps(summary, indent=1)+'\n')
    return summary


def close_stale(root, commitment, journal):
    """Reconcile the commitment of a killed job with the summed attempt costs of its journal."""
    rows = [json.loads(x) for x in (root/OUT/journal).read_text().splitlines()]
    spent = sum(a.get('cost') or 0. for r in rows for a in r['attempts'])
    path = root/'artifacts/budget.json'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        obj = json.loads(path.read_text())
        row = obj['direct_openai_commitments'][commitment]
        require(row['status'] == 'reserved' and spent <= row['reserved_usd']+1e-9, 'Commitment not open or cost exceeds reservation')
        obj['reserved'] -= row['reserved_usd']; obj['spent'] += spent
        row.update(status='reconciled', actual_usd=spent, closed_from_journal=journal, lower_bound=True)
        tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, sort_keys=True)+'\n'); tmp.replace(path)
    return {'commitment': commitment, 'journal': journal, 'rows': len(rows), 'actual_usd': spent}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--action', choices=['run', 'close-stale'], required=True)
    ap.add_argument('--fill', action='store_true')
    ap.add_argument('--tier', choices=['flex', 'default'], default='flex')
    ap.add_argument('--approval', default=APPROVAL)
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--commitment')
    ap.add_argument('--journal', default='results.jsonl')
    a = ap.parse_args()
    if a.action == 'close-stale':
        print(json.dumps(close_stale(a.root.resolve(), a.commitment, a.journal), indent=1))
    else:
        print(json.dumps(run(a.root.resolve(), a.fill, a.tier, a.approval, a.workers), indent=1))
