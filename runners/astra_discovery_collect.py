"""Read-only Batch retrieval plus auditable discovery import; never submits/retries.

Does not release the shared commitment: usage estimates are not invoice costs.
Call only with an existing submission receipt. Every retrieval gets a new directory.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from astra_discovery_submit import credential, sha
from astra_observation_import import import_rows

TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}


def identifier(value, prefix):
    if not isinstance(value, str) or not re.fullmatch(prefix+r'[A-Za-z0-9_-]+', value):
        raise ValueError('Invalid provider object identifier')
    return value


def get(key, endpoint):
    req = urllib.request.Request('https://api.openai.com/v1/'+endpoint,
                                 headers={'Authorization': 'Bearer '+key})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def verify_batch(batch, receipt, package_hash):
    if batch['id'] != receipt['id'] or batch['input_file_id'] != receipt['input_file_id']:
        raise ValueError('Batch identity differs from submission receipt')
    metadata = batch.get('metadata', {})
    if batch.get('endpoint') != '/v1/responses' or metadata.get('package_sha256') != package_hash:
        raise ValueError('Wrong endpoint/package')
    if metadata.get('study') != 'decision-em-20261013' or metadata.get('stage') != 'discovery_only':
        raise ValueError('Wrong study/stage')


def usage_inventory(rows, expected_ids):
    totals = {'input_tokens': 0, 'output_tokens': 0}
    known, unknown = [], []
    for row in rows:
        body = (row.get('response') or {}).get('body') or {}
        usage = body.get('usage') or {}
        cid = row['custom_id']
        if all(type(usage.get(k)) is int and usage[k] >= 0 for k in totals):
            for k in totals:
                totals[k] += usage[k]
            known.append(cid)
        else:
            unknown.append(cid)
    missing = sorted(set(expected_ids)-{r['custom_id'] for r in rows})
    return {**totals, 'with_usage': known, 'without_usage': unknown, 'missing_rows': missing,
            'all_request_usage_present': not unknown and not missing,
            'invoice_cost_reconciled': False, 'release_commitment': False}


def collect(root, package, output, fetch=get):
    receipt = json.loads((package/'batch_receipt.json').read_text())
    batch_id = identifier(receipt['id'], 'batch_')
    manifest = json.loads((package/'manifest.json').read_text())
    for filename in ['discovery.jsonl', 'cases_private.json']:
        if sha(package/filename) != manifest['files_sha256'][filename]:
            raise ValueError('Frozen package file changed: '+filename)
    package_hash = sha(package/'discovery.jsonl')
    key = credential(root)
    raw = fetch(key, 'batches/'+batch_id)
    batch = json.loads(raw)
    verify_batch(batch, receipt, package_hash)
    # Existing retrievals remain immutable. Network GETs create no inference.
    output.mkdir(parents=True, exist_ok=False)
    (output/'batch_snapshot.json').write_bytes(raw)
    report = {'retrieved_at': datetime.now(timezone.utc).isoformat(), 'batch_id': batch_id,
              'package_sha256': package_hash, 'status': batch['status'],
              'commitment_action': 'retain; no ledger changes', 'human_validation': 'pending'}
    if batch['status'] not in TERMINAL:
        (output/'retrieval_report.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
    rows, hashes = [], {}
    for name in ['output_file_id', 'error_file_id']:
        fid = batch.get(name)
        if fid:
            identifier(fid, 'file-')
            data = fetch(key, 'files/'+fid+'/content')
            (output/(name+'.jsonl')).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
            rows.extend(json.loads(line) for line in data.decode().splitlines() if line.strip())
    cases = json.loads((package/'cases_private.json').read_text())['discovery']
    expected_ids = {'discover-'+c['case_id'] for c in cases}
    request_ids = {json.loads(line)['custom_id'] for line in (package/'discovery.jsonl').read_text().splitlines()}
    if request_ids != expected_ids:
        raise ValueError('Source cases differ from submitted request IDs')
    # Preserve raw files before rejecting duplicate/unrecognized rows.
    normalized = import_rows(rows, cases)
    for row in rows:
        body = (row.get('response') or {}).get('body') or {}
        model = body.get('model')
        if normalized[row['custom_id']]['available'] and not (
                model == 'gpt-6-astra' or isinstance(model, str) and model.startswith('gpt-6-astra-')):
            normalized[row['custom_id']] = {'available': False, 'reason': 'unexpected_or_missing_model'}
    available = sum(v['available'] for v in normalized.values())
    counts = batch.get('request_counts', {})
    report.update(available=available, unavailable=len(cases)-available,
                  reasons=dict(Counter(v['reason'] for v in normalized.values() if not v['available'])),
                  files_sha256=hashes, provider_request_counts=counts,
                  provider_total_matches=counts.get('total') == len(cases),
                  all_observations_available=available == len(cases) and counts.get('total') == len(cases),
                  usage=usage_inventory(rows, expected_ids),
                  interpretation='Literal evidence/schema checks only; no human judgments or internal-cause claims')
    (output/'observations_imported.json').write_text(json.dumps(normalized, indent=2, ensure_ascii=False)+'\n')
    (output/'retrieval_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(collect(a.root, a.package, a.output), indent=2))
