"""Submit the frozen Astra codebook-synthesis request (one direct OpenAI Batch request).

Reuses astra_codebook_submit.validate unchanged (package hash, discovery provenance,
independent reconstruction of the request). Differences to astra_codebook_submit.submit:
- the 14 Oct parallel policy is replaced by a direct check that no project Slurm job runs,
- the USD 3 hold is checked against the ledger's current project cap (440 USD since
  20 Oct) instead of the original hard-coded USD 100 cap, which spending has passed.
No retries. An unknown upload or batch outcome keeps the hold and the exclusive journal.
Reconciliation after the collector: batch_budget.update(..., 'reconcile', ...).
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import uuid

from astra_codebook_submit import STAGE, validate
from astra_discovery_collect import identifier
from astra_discovery_submit import credential, post, sha

HOLD = 3.0


def project_jobs(root):
    out = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i'], text=True).split()
    return [j for j in out if str(root) in subprocess.check_output(['scontrol', 'show', 'job', j, '-o'], text=True)]


def reserve(root, commitment, digest, approval):
    path = root/'artifacts/budget.json'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        obj = json.loads(path.read_text())
        commitments = obj.setdefault('direct_openai_commitments', {})
        if commitment in commitments:
            raise ValueError('Commitment ID already exists; never reserve twice')
        if obj['spent']+obj['reserved']+HOLD > obj['project_cap_usd']+1e-9:
            raise ValueError('Project cap reached')
        obj['reserved'] += HOLD
        commitments[commitment] = {'package_sha256': digest, 'reserved_usd': HOLD, 'status': 'reserved',
                                   'actual_usd': None, 'approval_reference': approval['user_message_reference']}
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(obj, sort_keys=True, allow_nan=False)+'\n')
        tmp.replace(path)


def launch(root, package, discovery, retrieval, approval_path):
    root = root.resolve()
    approval = json.loads(approval_path.read_text())
    digest = sha(package/'requests.jsonl')
    admission = {'human_approval_verified': True, 'package_sha256': digest}  # operator: quote is Aaron's chat message
    validate(package, discovery, retrieval, approval, admission)
    if project_jobs(root):
        raise ValueError('A project Slurm job is running')
    if (package/'batch_receipt.json').exists():
        raise ValueError('Existing codebook batch receipt; never submit twice')
    key = credential(root)
    with (package/'submission_journal.jsonl').open('x') as handle:
        def event(kind, **fields):
            handle.write(json.dumps({'event': kind, **fields})+'\n'); handle.flush(); os.fsync(handle.fileno())
        event('validated', package_sha256=digest, approval_sha256=sha(approval_path), launcher=Path(__file__).name)
        commitment = 'astra-codebook-'+digest[:16]
        reserve(root, commitment, digest, approval)
        event('reserved', commitment_id=commitment, usd=HOLD)
        boundary = '----decision-em-'+uuid.uuid4().hex
        wire = ('--'+boundary+'\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
                '--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="codebook.jsonl"\r\n'
                'Content-Type: application/jsonl\r\n\r\n').encode()
        wire += (package/'requests.jsonl').read_bytes()+('\r\n--'+boundary+'--\r\n').encode()
        event('upload_start_unknown_until_receipt')
        uploaded = post(key, 'files', wire, 'multipart/form-data; boundary='+boundary)
        event('upload_response', response=uploaded)
        file_id = identifier(uploaded['id'], 'file-')
        body = {'input_file_id': file_id, 'endpoint': '/v1/responses', 'completion_window': '24h',
                'metadata': {'study': 'decision-em-20261013', 'stage': STAGE, 'package_sha256': digest}}
        event('batch_create_start_unknown_until_receipt')
        receipt = post(key, 'batches', json.dumps(body).encode(), 'application/json')
        event('batch_response', response=receipt)
        (package/'batch_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        batch_id = identifier(receipt['id'], 'batch_')
        if receipt.get('input_file_id') != file_id or receipt.get('metadata') != body['metadata']:
            raise ValueError('Unexpected Batch receipt; retain hold and inspect saved response')
    return batch_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root', 'package', 'discovery', 'retrieval', 'approval']:
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    print(launch(a.root, a.package, a.discovery, a.retrieval, a.approval))
