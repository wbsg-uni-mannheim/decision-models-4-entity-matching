"""Single codebook Batch submission, separately approved and jointly budgeted.

No retries. Unknown upload/create outcomes retain the hold and exclusive journal.
The completed discovery submission and its files are never modified.
"""
import argparse
import json
import os
from pathlib import Path
import uuid

from astra_discovery_collect import identifier
from astra_discovery_submit import credential, post, sha
from astra_followup_package import synthesis_request
from astra_followup_provenance import verified_discovery
from batch_budget import update
from parallel_admission import review

STAGE = 'codebook_synthesis_only'


def validate(package, discovery, retrieval, approval, admission):
    manifest = json.loads((package / 'manifest.json').read_text())
    digest = sha(package / 'requests.jsonl')
    if digest != manifest['requests_sha256'] or manifest['requests'] != 1:
        raise ValueError('Frozen single codebook request changed')
    cases, observations, provenance = verified_discovery(discovery, retrieval, record_evidence=True)
    if any(manifest['provenance'].get(k) != v for k, v in provenance.items()):
        raise ValueError('Discovery provenance changed')
    for name, expected in manifest['provenance']['validator_files_sha256'].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError('Frozen codebook preparation code changed')
    if sha(package / 'discovery_observations_verified.json') != manifest['provenance']['verified_observations_sha256']:
        raise ValueError('Verified observations changed')
    rows = [json.loads(line) for line in (package / 'requests.jsonl').read_text().splitlines() if line.strip()]
    if rows != [synthesis_request(cases, observations)]:
        raise ValueError('Codebook request differs from independently reconstructed discovery-only input')
    if approval.get('stage') != STAGE or approval.get('approved_by') != 'Aaron':
        raise ValueError('Separate codebook approval required')
    if approval.get('package_sha256') != digest or approval.get('max_cost_usd') != 3:
        raise ValueError('Approval must name exact codebook JSONL and USD3 ceiling')
    if not approval.get('user_message_reference') or not approval.get('quoted_user_authorization'):
        raise ValueError('Actual human-message evidence required')
    if admission.get('human_approval_verified') is not True or admission.get('package_sha256') != digest:
        raise ValueError('Operator must verify approval provenance')
    return digest


def submit(root, package, discovery, retrieval, approval_path, admission_path):
    root = root.resolve()
    approval = json.loads(approval_path.read_text())
    admission = json.loads(admission_path.read_text())
    digest = validate(package, discovery, retrieval, approval, admission)
    policy = root / admission['parallel_policy']
    if sha(policy) != admission['parallel_policy_sha256']:
        raise ValueError('Parallel policy changed since admission')
    if admission.get('submitter_sha256') != sha(Path(__file__)):
        raise ValueError('Codebook submitter changed since admission')
    reviewed = review(root, policy)
    key = credential(root)
    if (package / 'batch_receipt.json').exists():
        raise ValueError('Existing codebook batch receipt; never submit twice')
    with (package / 'submission_journal.jsonl').open('x') as handle:
        def event(kind, **fields):
            handle.write(json.dumps({'event': kind, **fields})+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        event('validated', package_sha256=digest, parallel_review=reviewed)
        commitment = 'astra-codebook-'+digest[:16]
        update(root, 'reserve', commitment, digest, 3, approval)
        event('reserved', commitment_id=commitment, usd=3)
        boundary = '----decision-em-'+uuid.uuid4().hex
        wire = ('--'+boundary+'\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
                '--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="codebook.jsonl"\r\n'
                'Content-Type: application/jsonl\r\n\r\n').encode()
        wire += (package / 'requests.jsonl').read_bytes()+('\r\n--'+boundary+'--\r\n').encode()
        event('upload_start_unknown_until_receipt')
        uploaded = post(key, 'files', wire, 'multipart/form-data; boundary='+boundary)
        event('upload_response', response=uploaded)
        file_id = identifier(uploaded['id'], 'file-')
        body = {'input_file_id': file_id, 'endpoint': '/v1/responses', 'completion_window': '24h',
                'metadata': {'study': 'decision-em-20261013', 'stage': STAGE, 'package_sha256': digest}}
        event('batch_create_start_unknown_until_receipt')
        receipt = post(key, 'batches', json.dumps(body).encode(), 'application/json')
        event('batch_response', response=receipt)
        (package / 'batch_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        batch_id = identifier(receipt['id'], 'batch_')
        if receipt.get('input_file_id') != file_id or receipt.get('metadata') != body['metadata']:
            raise ValueError('Unexpected Batch receipt; retain hold and inspect saved response')
    return batch_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root', 'package', 'discovery', 'retrieval', 'approval', 'admission']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(submit(args.root, args.package, args.discovery, args.retrieval, args.approval, args.admission))
