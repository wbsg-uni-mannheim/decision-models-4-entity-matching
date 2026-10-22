"""Reference run, Claude Opus 5.5 on the remaining test pairs via the OpenRouter Batch API.

Batch API (openrouter.ai/docs/batch-quickstart): POST /api/v1/batches with endpoint, model,
provider and an inline `requests` array (must be the last key), 24 h window, results inline in
the batch object, usage and cost only at batch level. Model anthropic/claude-opus-5.5, provider
pinned to Anthropic without fallback. Request bodies equal the escalation requests minus model
and provider (src/reference_rest.py).

Actions:
  build    write chunks of at most CHUNK requests (artifacts/reference_rest/opus_batch/chunk_NN.json)
  submit   one ledger hold for the approved cap, then submit every chunk once (receipts kept)
  collect  GET each batch; when all are terminal, write runs/.../opus55/E0/results.jsonl and
           reconcile the hold with the summed batch cost
No resubmission of a chunk that has a receipt.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

SPLIT = 'reference_rest'
DIR = 'artifacts/reference_rest/opus_batch'
OUT = 'runs/reference_rest_run01_20261021/opus55_batch'
APPROVAL = 'artifacts/operations/approval_reference_opus_aaron_20261021.json'
CHUNK = 2000
API = 'https://openrouter.ai/api/v1/batches'
TERMINAL = ('completed', 'failed', 'expired', 'cancelled')
COMMITMENT = 'reference-opus-batch-20261021'


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
    import run
    escalation.register_models()
    return protocol, reference_rest, run


def build(root):
    protocol, reference_rest, _ = modules(root)
    d = root/DIR
    d.mkdir(parents=True, exist_ok=False)
    cases = [c for b in protocol.BENCHMARKS for c in read(root/'data'/b/(SPLIT+'.json'))]
    # Anthropic batch custom_id allows only [A-Za-z0-9_-]; keep a map back to benchmark and pair id.
    ids = {f'r{i:05d}': [c['benchmark'], c['pair_id']] for i, c in enumerate(cases)}
    reqs = [{'custom_id': cid, 'body': reference_rest.opus_batch_body(c)} for cid, c in zip(ids, cases)]
    (d/'custom_ids.json').write_text(json.dumps(ids)+'\n')
    chunks = [reqs[i:i+CHUNK] for i in range(0, len(reqs), CHUNK)]
    for i, ch in enumerate(chunks):
        batch = {'endpoint': '/v1/chat/completions', 'model': reference_rest.OPUS_MODEL,
                 'provider': {'only': ['anthropic'], 'allow_fallbacks': False}, 'completion_window': '24h',
                 'requests': ch}  # requests last: the API stream-parses the body
        (d/f'chunk_{i:02d}.json').write_text(json.dumps(batch, ensure_ascii=False))
    manifest = {'requests': len(reqs), 'chunks': len(chunks), 'custom_ids_sha256': hashlib.sha256((d/'custom_ids.json').read_bytes()).hexdigest(),
                'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob('chunk_*.json'))}}
    (d/'manifest.json').write_text(json.dumps(manifest, indent=1)+'\n')
    return manifest


def ledger(root, action, cap=None, quote=None, actual=None):
    path = root/'artifacts/budget.json'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        obj = json.loads(path.read_text())
        rows = obj.setdefault('openrouter_batch_commitments', {})
        if action == 'reserve':
            require(COMMITMENT not in rows, 'Commitment exists; never reserve twice')
            require(obj['spent']+obj['reserved']+cap <= obj['project_cap_usd']+1e-9, 'Project cap reached')
            obj['reserved'] += cap
            rows[COMMITMENT] = {'reserved_usd': cap, 'status': 'reserved', 'actual_usd': None, 'approval_reference': quote}
        else:
            row = rows[COMMITMENT]
            require(row['status'] == 'reserved', 'Commitment already closed')
            require(actual <= row['reserved_usd']+1e-9, 'Cost exceeds reservation; inspect before reconciling')
            obj['reserved'] -= row['reserved_usd']; obj['spent'] += actual
            row.update(status='reconciled', actual_usd=actual)
        tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, sort_keys=True)+'\n'); tmp.replace(path)


def http(key, method, url, data=None):
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def submit(root):
    _, _, run = modules(root)
    approval = read(root/APPROVAL)
    require(approval.get('approved_by') == 'Aaron' and approval.get('quote') and approval.get('member') == 'opus55',
            'Approval required')
    d = root/DIR
    manifest = read(d/'manifest.json')
    for name, digest in manifest['sha256'].items():
        require(hashlib.sha256((d/name).read_bytes()).hexdigest() == digest, 'Chunk changed: '+name)
    key = run.key()
    if not (d/'reserved.json').exists():
        ledger(root, 'reserve', float(approval['run_cap_usd']), approval['quote'])
        (d/'reserved.json').write_text(json.dumps({'usd': approval['run_cap_usd'], 'utc': time.time()})+'\n')
    receipts = []
    for name in sorted(manifest['sha256']):
        receipt = d/name.replace('chunk_', 'receipt_')
        if receipt.exists():
            receipts.append(read(receipt)['id']); continue
        with (d/(name+'.submit.journal')).open('x') as j:  # an unknown outcome blocks a second submit
            j.write('submit_start_unknown_until_receipt\n'); j.flush()
            resp = http(key, 'POST', API, (d/name).read_bytes())
            receipt.write_text(json.dumps(resp, indent=1)+'\n')
            j.write('receipt '+str(resp.get('id'))+'\n')
        receipts.append(resp['id'])
    return {'batches': receipts}


def collect(root, final=False):
    protocol, reference_rest, run = modules(root)
    d = root/DIR
    key = run.key()
    status, batches = {}, {}
    for receipt in sorted(d.glob('receipt_*.json')):
        bid = read(receipt)['id']
        b = http(key, 'GET', f'{API}/{bid}')
        batches[receipt.name] = b
        status[bid] = {'status': b['status'], 'counts': b.get('request_counts'), 'cost': (b.get('usage') or {}).get('cost')}
    if not all(s['status'] in TERMINAL for s in status.values()):
        return {'done': False, 'status': status}
    cases = {(c['benchmark'], c['pair_id']): c for b in protocol.BENCHMARKS for c in read(root/'data'/b/(SPLIT+'.json'))}
    ids = read(d/'custom_ids.json')
    out = root/OUT
    out.mkdir(parents=True, exist_ok=False)
    raw = out/'raw'; raw.mkdir()
    rows, total = [], 0.
    for name, b in batches.items():
        (raw/name.replace('receipt_', 'batch_')).write_text(json.dumps(b)+'\n')
        total += float((b.get('usage') or {}).get('cost') or 0.)
        for res in b.get('results') or []:
            bench, pid = ids[res['custom_id']]
            c = cases[(bench, pid)]
            row = {'benchmark': bench, 'pair_id': pid, 'input_hash': c['input_hash'], 'model_alias': 'opus55',
                   'variant': 'E0', 'transport': 'openrouter-batch', 'batch_id': b['id'],
                   'request_hash': protocol.digest(reference_rest.opus_batch_body(c))}
            try:
                require(res.get('response') and res['response'].get('status_code') == 200, 'Request error: '+str(res.get('error'))[:200])
                body = res['response']['body']
                require(str(body.get('model', '')).startswith(reference_rest.OPUS_MODEL), 'Unexpected model '+str(body.get('model')))
                require(body.get('provider') in (None, 'Anthropic'), 'Unexpected provider '+str(body.get('provider')))
                choice, = body['choices']
                require(choice.get('finish_reason') == 'stop' and not choice['message'].get('refusal'), 'Incomplete answer')
                row.update(prediction=reference_rest.vote(choice['message']['content']), probability_match=None, response=body)
            except (ValueError, KeyError, TypeError) as e:
                row.update(error=(type(e).__name__+': '+str(e))[:300], result=res)
            rows.append(row)
    with (out/'results.jsonl').open('x') as f:
        f.writelines(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    summary = {'requested': len(cases), 'returned': len(rows), 'valid': sum('prediction' in r for r in rows),
               'cost_usd': total, 'status': status}
    (out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    ledger(root, 'reconcile', actual=total)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--action', choices=['build', 'submit', 'collect'], required=True)
    a = ap.parse_args(); root = a.root.resolve()
    print(json.dumps({'build': build, 'submit': submit, 'collect': collect}[a.action](root), indent=1))
