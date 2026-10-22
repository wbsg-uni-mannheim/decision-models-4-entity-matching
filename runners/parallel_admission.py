"""Admit explicitly reviewed independent jobs under the shared budget lock."""
import hashlib
import json
import os
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(root, policy_path, own_job=None):
    policy = json.loads(policy_path.read_text())
    approval_path = root/policy['approval_file']
    if sha(approval_path) != policy['approval_sha256']:
        raise ValueError('Parallel authorization changed')
    approval = json.loads(approval_path.read_text())
    if approval.get('approved_by') != 'Aaron' or not approval.get('parallel_execution_authorized'):
        raise ValueError('Explicit parallel execution authority required')
    if approval.get('project_cap_usd') != 100 or not approval.get('quoted_user_authorization'):
        raise ValueError('Parallel authority does not preserve the project cap')
    for rel, expected in policy['files_sha256'].items():
        if sha(root/rel) != expected:
            raise ValueError('Parallel execution source changed: '+rel)
    queue = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%j'], text=True)
    reviewed = []
    for line in queue.splitlines():
        job, name = line.split('|', 1)
        meta = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        if job in policy['reviewed_project_jobs']:
            expected = policy['reviewed_project_jobs'][job]
            script = subprocess.check_output(['scontrol', 'write', 'batch_script', job, '-'], text=True)
            if name != expected['name'] or ('WorkDir='+str(root)+' ') not in meta+' ':
                raise ValueError('Project job identity changed: '+job)
            if hashlib.sha256(script.encode()).hexdigest() != expected['batch_script_sha256']:
                raise ValueError('Project job command changed: '+job)
            reviewed.append({'id': job, 'name': name, 'scope': expected['scope'], 'self': job == own_job})
        elif str(root) in meta or name.startswith('decision-em'):
            raise ValueError('Unreviewed project job: '+job)
        else:
            reviewed.append({'id': job, 'scope': 'different project',
                             'metadata_sha256': hashlib.sha256(meta.encode()).hexdigest()})
    return {'policy_sha256': sha(policy_path), 'approval_sha256': sha(approval_path),
            'reviewed_jobs': reviewed, 'all_project_inference_jobs_reviewed': True}
