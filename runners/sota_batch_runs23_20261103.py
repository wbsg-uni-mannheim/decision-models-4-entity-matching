"""Runs 2 and 3 of GPT-6.1 Sol and Claude Opus 5.5 on all test pairs via batch APIs (Aaron, 3 Nov: "Can you please
run Sol, Opus, and Argon each twice ideally using the edge batch API so it doesn't cost an arm and a leg?"; cap
1,135 USD confirmed directly in the experiment session: "Yes, 1,135 USD").

Run 1 (ensemble_full) split the test pairs: the 6,682 escalation pairs ran live over OpenRouter
(runs/escalation_run01_20261020), the rest per member (runs/reference_rest_run01_20261021). Runs 2 and 3 send every
test pair (data/<b>/test.json, 30,225) through one batch route per member, with the request content of Run 1
(src/reference_rest.py, from escalation.payload: frozen Luna B0 messages, JSON schema, reasoning effort, output limit):
- sol61: OpenAI Batch API (/v1/batches, 24 h window, 50 % of the standard price), body = reference_rest.sol_direct_body
  without service_tier (the batch is its own tier); one JSONL file per run.
- opus55: OpenRouter Batch API exactly as reference_opus_batch_20261021.py (Anthropic pinned, no fallback; chunks of
  2,000; body = reference_rest.opus_batch_body).
Actions per --model and --run: build (frozen request files + manifest), submit (one ledger hold of the approved run
cap per model and run, every file once; a submit journal blocks a second submit after an unknown outcome),
collect (when terminal: runs/sota_run0<r>_20261103/<model>/results.jsonl in the Run-1 row format, reconcile the hold
with the batch cost; Sol cost from token usage at batch prices). Failed requests are re-sent with build --fill.

    python3 artifacts/operations/sota_batch_runs23_20261103.py --root . --model sol61|opus55 --run 2|3 --action build|submit|collect [--fill N]
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import uuid

APPROVAL = 'artifacts/operations/approval_sota_runs23_aaron_20261103.json'
OPUS_API = 'https://openrouter.ai/api/v1/batches'
OPENAI = 'https://api.openai.com/v1'
CHUNK = {'opus55': 2000, 'sol61': 50000}
TERMINAL = {'opus55': ('completed', 'failed', 'expired', 'cancelled'), 'sol61': ('completed', 'failed', 'expired', 'cancelled')}
SOL_BATCH_PRICE = (1e-6, 5e-8, 5e-6)  # USD per token: input, cached input, output (50 % of 2 / 0.10 / 10 per 1M)


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


def paths(root, model, run_id, fill):
    d = root/f'artifacts/sota_runs23/{model}_run0{run_id}' / (f'fill{fill}' if fill else 'main')
    return d, root/f'runs/sota_run0{run_id}_20261103/{model}', f'sota-{model}-run{run_id}-20261103' + (f'-fill{fill}' if fill else '')


def body(reference_rest, model, case):
    if model == 'sol61':
        b = reference_rest.sol_direct_body(case)
        del b['service_tier']
        return b
    return reference_rest.opus_batch_body(case)


def build(root, model, run_id, fill=0):
    protocol, reference_rest, _ = modules(root)
    d, out, _ = paths(root, model, run_id, fill)
    cases = [c for b in protocol.BENCHMARKS for c in read(root/'data'/b/'test.json')]
    if fill:
        done = {(r['benchmark'], r['pair_id']) for p in sorted(out.glob('results*.jsonl')) for r in map(json.loads, p.read_text().splitlines()) if 'prediction' in r}
        cases = [c for c in cases if (c['benchmark'], c['pair_id']) not in done]
        require(cases, 'Nothing to fill')
    d.mkdir(parents=True, exist_ok=False)
    ids = {f'r{i:05d}': [c['benchmark'], c['pair_id']] for i, c in enumerate(cases)}  # [A-Za-z0-9_-] only
    (d/'custom_ids.json').write_text(json.dumps(ids)+'\n')
    reqs = [(cid, body(reference_rest, model, c)) for cid, c in zip(ids, cases)]
    for i in range(0, len(reqs), CHUNK[model]):
        ch = reqs[i:i+CHUNK[model]]
        if model == 'sol61':
            (d/f'chunk_{i//CHUNK[model]:02d}.jsonl').write_text(''.join(json.dumps({'custom_id': cid, 'method': 'POST', 'url': '/v1/chat/completions',
                                                                                 'body': b}, ensure_ascii=False)+'\n' for cid, b in ch))
        else:
            batch = {'endpoint': '/v1/chat/completions', 'model': reference_rest.OPUS_MODEL,
                     'provider': {'only': ['anthropic'], 'allow_fallbacks': False}, 'completion_window': '24h',
                     'requests': [{'custom_id': cid, 'body': b} for cid, b in ch]}  # requests last: the API stream-parses the body
            (d/f'chunk_{i//CHUNK[model]:02d}.json').write_text(json.dumps(batch, ensure_ascii=False))
    manifest = {'model': model, 'run_id': run_id, 'fill': fill, 'requests': len(reqs),
                'request_sha256': protocol.digest([b for _, b in reqs]),
                'custom_ids_sha256': hashlib.sha256((d/'custom_ids.json').read_bytes()).hexdigest(),
                'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob('chunk_*'))}}
    (d/'manifest.json').write_text(json.dumps(manifest, indent=1)+'\n')
    return manifest


def ledger(root, commitment, action, cap=None, quote=None, actual=None, model=None):
    path = root/'artifacts/budget.json'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        obj = json.loads(path.read_text())
        rows = obj.setdefault('direct_openai_commitments' if model == 'sol61' else 'openrouter_batch_commitments', {})
        if action == 'reserve':
            require(commitment not in rows, 'Commitment exists; never reserve twice')
            require(obj['spent']+obj['reserved']+cap <= obj['project_cap_usd']+1e-9, 'Project cap reached')
            obj['reserved'] += cap
            rows[commitment] = {'reserved_usd': cap, 'status': 'reserved', 'actual_usd': None, 'approval_reference': quote,
                                'model': model, 'tier': 'batch'}
        else:
            row = rows[commitment]
            require(row['status'] == 'reserved', 'Commitment already closed')
            require(actual <= row['reserved_usd']+1e-9, 'Cost exceeds reservation; inspect before reconciling')
            obj['reserved'] -= row['reserved_usd']; obj['spent'] += actual
            row.update(status='reconciled', actual_usd=actual)
        tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, sort_keys=True)+'\n'); tmp.replace(path)


def http(key, method, url, data=None, ctype='application/json', raw=False):
    req = urllib.request.Request(url, data=data, method=method, headers={'Authorization': 'Bearer '+key, 'Content-Type': ctype})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read() if raw else json.load(r)


def openai_upload(key, path):
    boundary = uuid.uuid4().hex
    data = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            'Content-Type: application/jsonl\r\n\r\n').encode() + path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    return http(key, 'POST', OPENAI+'/files', data, 'multipart/form-data; boundary='+boundary)


def credentials(root, model, run):
    if model == 'sol61':
        import astra_discovery_submit
        return astra_discovery_submit.credential(root)
    return run.key()


def submit(root, model, run_id, fill=0, max_chunks=None):
    _, _, run = modules(root)
    a = read(root/APPROVAL)
    require(a.get('approved_by') == 'Aaron' and a.get('quote') and run_id in a.get('released_runs', []), 'Approval required')
    cap = float(a['run_cap_usd'][model]) if not fill else float(a['fill_cap_usd'][model])
    d, _, commitment = paths(root, model, run_id, fill)
    manifest = read(d/'manifest.json')
    for name, digest in manifest['sha256'].items():
        require(hashlib.sha256((d/name).read_bytes()).hexdigest() == digest, 'Chunk changed: '+name)
    key = credentials(root, model, run)
    if not (d/'reserved.json').exists():
        ledger(root, commitment, 'reserve', cap, a['quote'], model=model)
        (d/'reserved.json').write_text(json.dumps({'usd': cap, 'commitment': commitment, 'utc': time.time()})+'\n')
    receipts, sent = [], 0
    for name in sorted(manifest['sha256']):
        receipt = d/('receipt_'+name.split('_', 1)[1].split('.')[0]+'.json')
        if receipt.exists():
            receipts.append(read(receipt)['id']); continue
        if max_chunks is not None and sent >= max_chunks:  # waves: OpenRouter books the maximum batch cost up front
            break
        sent += 1
        with (d/(name+'.submit.journal')).open('x') as j:  # an unknown outcome blocks a second submit
            j.write('submit_start_unknown_until_receipt\n'); j.flush()
            if model == 'sol61':
                f = openai_upload(key, d/name)
                j.write('file '+str(f.get('id'))+'\n'); j.flush()
                resp = http(key, 'POST', OPENAI+'/batches', json.dumps({'input_file_id': f['id'], 'endpoint': '/v1/chat/completions',
                                                                         'completion_window': '24h'}).encode())
            else:
                resp = http(key, 'POST', OPUS_API, (d/name).read_bytes())
            receipt.write_text(json.dumps(resp, indent=1)+'\n')
            j.write('receipt '+str(resp.get('id'))+'\n')
        receipts.append(resp['id'])
    return {'batches': receipts, 'reserved_usd': cap}


def sol_cost(usage):
    p_in, p_cached, p_out = SOL_BATCH_PRICE
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
    return (usage['prompt_tokens']-cached)*p_in + cached*p_cached + usage['completion_tokens']*p_out


def collect(root, model, run_id, fill=0):
    protocol, reference_rest, run = modules(root)
    d, out, commitment = paths(root, model, run_id, fill)
    key = credentials(root, model, run)
    status, batches = {}, {}
    for receipt in sorted(d.glob('receipt_*.json')):
        bid = read(receipt)['id']
        b = http(key, 'GET', (OPENAI+'/batches/' if model == 'sol61' else OPUS_API+'/')+bid)
        batches[receipt.name] = b
        status[bid] = {'status': b['status'], 'counts': b.get('request_counts'), 'cost': (b.get('usage') or {}).get('cost')}
    if not batches or not all(s['status'] in TERMINAL[model] for s in status.values()):
        return {'done': False, 'status': status}
    cases = {(c['benchmark'], c['pair_id']): c for b in protocol.BENCHMARKS for c in read(root/'data'/b/'test.json')}
    ids = read(d/'custom_ids.json')
    out.mkdir(parents=True, exist_ok=True)
    name = f'results.fill{fill}.jsonl' if fill else 'results.jsonl'
    raw = out/('raw' if not fill else f'raw_fill{fill}'); raw.mkdir()
    results, total = [], 0.
    for rname, b in batches.items():
        (raw/rname.replace('receipt_', 'batch_')).write_text(json.dumps(b)+'\n')
        if model == 'sol61':
            for fid in (b.get('output_file_id'), b.get('error_file_id')):
                if fid:
                    text = http(key, 'GET', f'{OPENAI}/files/{fid}/content', raw=True)
                    (raw/f'{fid}.jsonl').write_bytes(text)
                    results += [json.loads(x) for x in text.decode().splitlines() if x.strip()]
        else:
            total += float((b.get('usage') or {}).get('cost') or 0.)
            results += b.get('results') or []
    rows = []
    for res in results:
        bench, pid = ids[res['custom_id']]
        c = cases[(bench, pid)]
        row = {'benchmark': bench, 'pair_id': pid, 'input_hash': c['input_hash'], 'model_alias': model, 'variant': 'E0',
               'study_run_id': run_id, 'transport': 'openai-batch' if model == 'sol61' else 'openrouter-batch',
               'request_hash': protocol.digest(body(reference_rest, model, c))}
        try:
            resp = res.get('response') or {}
            require(resp.get('status_code') == 200, 'Request error: '+str(res.get('error'))[:200])
            rb = resp['body']
            if model == 'sol61':
                row['cost'] = sol_cost(rb['usage']); total += row['cost']
                require(str(rb.get('model', '')).startswith(reference_rest.SOL_MODEL), 'Unexpected model '+str(rb.get('model')))
            else:
                require(str(rb.get('model', '')).startswith(reference_rest.OPUS_MODEL), 'Unexpected model '+str(rb.get('model')))
                require(rb.get('provider') in (None, 'Anthropic'), 'Unexpected provider '+str(rb.get('provider')))
            choice, = rb['choices']
            require(choice.get('finish_reason') == 'stop' and not choice['message'].get('refusal'), 'Incomplete answer')
            row.update(prediction=reference_rest.vote(choice['message']['content']), probability_match=None, response=rb)
        except (ValueError, KeyError, TypeError) as e:
            row.update(error=(type(e).__name__+': '+str(e))[:300], result=res)
        rows.append(row)
    with (out/name).open('x') as f:
        f.writelines(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    summary = {'requested': read(d/'manifest.json')['requests'], 'returned': len(rows), 'valid': sum('prediction' in r for r in rows),
               'cost_usd': total, 'status': status}
    (out/name.replace('results', 'summary').replace('.jsonl', '.json')).write_text(json.dumps(summary, indent=1)+'\n')
    ledger(root, commitment, 'reconcile', actual=total, model=model)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--model', choices=['sol61', 'opus55'], required=True)
    ap.add_argument('--run', type=int, choices=[2, 3], required=True)
    ap.add_argument('--action', choices=['build', 'submit', 'collect'], required=True)
    ap.add_argument('--fill', type=int, default=0, help='fill round number (re-send pairs without a valid answer)')
    ap.add_argument('--max-chunks', type=int, help='submit: at most this many new chunks (waves)')
    a = ap.parse_args(); root = a.root.resolve()
    kw = {'max_chunks': a.max_chunks} if a.action == 'submit' else {}
    print(json.dumps({'build': build, 'submit': submit, 'collect': collect}[a.action](root, a.model, a.run, a.fill, **kw), indent=1))
