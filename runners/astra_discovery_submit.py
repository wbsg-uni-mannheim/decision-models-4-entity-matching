"""One-shot discovery submission. Explicit approval and idle inference chain required.

No automatic network retries. On any uncertain outcome retain the commitment and
journal; inspect the provider before another attempt. This file does nothing on import.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import uuid

from batch_budget import update

JOBS = ['365636', '365221', '365268', '365279', '365288', '365300']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(package, approval, admission, active_jobs):
    payload = package/'discovery.jsonl'
    manifest = json.loads((package/'manifest.json').read_text())
    if sha(payload) != manifest['files_sha256']['discovery.jsonl']:
        raise ValueError('Discovery bytes changed')
    if approval.get('stage') != 'discovery_observations_only' or approval.get('approved_by') != 'Aaron':
        raise ValueError('Separate discovery approval required')
    if approval.get('package_sha256') != sha(payload) or approval.get('max_cost_usd') != 11:
        raise ValueError('Approval must identify exact JSONL and USD11 ceiling')
    if not approval.get('user_message_reference') or not approval.get('quoted_user_authorization'):
        raise ValueError('Actual user-message evidence required; preparer must verify provenance')
    if admission.get('human_approval_verified') is not True or admission.get('package_sha256') != sha(payload):
        raise ValueError('Operator must verify actual human approval')
    if active_jobs or not admission.get('all_project_inference_jobs_reviewed'):
        raise ValueError('Wait until previous inference writers and queued chain end')
    rows = [json.loads(line) for line in payload.read_text().splitlines()]
    if len(rows) != 130 or len({r['custom_id'] for r in rows}) != 130:
        raise ValueError('Wrong discovery request set')
    for row in rows:
        b = row['body']
        if row['url'] != '/v1/responses' or row['method'] != 'POST' or b['model'] != 'gpt-6-astra':
            raise ValueError('Unexpected endpoint or model')
        if b['max_output_tokens'] != 2048 or 'tools' in b or b['reasoning'] != {'effort': 'low'}:
            raise ValueError('Unexpected output/reasoning/tool budget')
    return sha(payload)


def credential(root):
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    for line in (root/'.secrets/source.env').read_text().splitlines():
        if '=' in line and line.split('=', 1)[0].strip() == 'OPENAI_API_KEY':
            return line.split('=', 1)[1].strip().strip('\"\'')
    raise ValueError('Direct OpenAI credential missing')


def post(key, endpoint, data, content_type):
    req = urllib.request.Request('https://api.openai.com/v1/'+endpoint, data=data,
        headers={'Authorization': 'Bearer '+key, 'Content-Type': content_type})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def submit(root, package, approval_path, admission_path):
    approval = json.loads(approval_path.read_text())
    admission = json.loads(admission_path.read_text())
    # Review every user job as well as the known chain; new successors cannot be omitted.
    if admission.get('parallel_policy'):
        from parallel_admission import review
        policy_path = root/admission['parallel_policy']
        if sha(policy_path) != admission['parallel_policy_sha256']:
            raise ValueError('Parallel policy changed after admission')
        review(root, policy_path)
        unreviewed = []
    else:
        active = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i %j'], text=True)
        reviewed_nonstudy = set(admission.get('reviewed_nonstudy_job_ids', []))
        unreviewed = [line for line in active.splitlines() if line.split()[0] in JOBS or line.split()[0] not in reviewed_nonstudy]
    package_hash = validate(package, approval, admission, unreviewed)
    key = credential(root)
    journal = package/'submission_journal.jsonl'
    # Exclusive creation also prevents accidental rerun after unknown network outcome.
    with journal.open('x') as f:
        def event(kind, **fields):
            f.write(json.dumps({'event': kind, **fields})+'\n'); f.flush(); os.fsync(f.fileno())
        event('validated', package_sha256=package_hash)
        commitment_id = 'astra-discovery-'+package_hash[:16]
        update(root, 'reserve', commitment_id, package_hash, 11, approval)
        event('reserved', commitment_id=commitment_id, usd=11)
        boundary = '----decision-em-'+uuid.uuid4().hex
        data = ('--'+boundary+'\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
                '--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="discovery.jsonl"\r\n'
                'Content-Type: application/jsonl\r\n\r\n').encode()+(package/'discovery.jsonl').read_bytes()+('\r\n--'+boundary+'--\r\n').encode()
        event('upload_start_unknown_until_receipt')
        uploaded = post(key, 'files', data, 'multipart/form-data; boundary='+boundary)
        event('uploaded', file_id=uploaded['id'])
        body = {'input_file_id': uploaded['id'], 'endpoint': '/v1/responses', 'completion_window': '24h',
                'metadata': {'study': 'decision-em-20261013', 'stage': 'discovery_only', 'package_sha256': package_hash}}
        event('batch_create_start_unknown_until_receipt')
        result = post(key, 'batches', json.dumps(body).encode(), 'application/json')
        event('batch_created', batch_id=result['id'], status=result['status'])
        (package/'batch_receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    return result['id']


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--approval', type=Path, required=True)
    p.add_argument('--admission', type=Path, required=True)
    a = p.parse_args()
    print(submit(a.root, a.package, a.approval, a.admission))
