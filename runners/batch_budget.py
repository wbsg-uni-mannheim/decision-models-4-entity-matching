"""Shared commitment accounting; no network or batch submission capability.

Uses the same lock and reserved/spent fields as the frozen OpenRouter client.
A commitment stays reserved through upload, pending, partial and unknown outcomes.
Only a reconciled terminal batch may release unused funds.
"""
import fcntl
import json
import math
from pathlib import Path


def _money(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid monetary amount')
    return float(value)


def update(root, action, commitment_id, package_sha256, amount, approval=None, terminal=False, usage_complete=False):
    amount = _money(amount)
    path = Path(root)/'artifacts/budget.json'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        obj = json.loads(path.read_text())
        commitments = obj.setdefault('direct_openai_commitments', {})
        if action == 'reserve':
            # A caller must separately authenticate the approval's human provenance.
            if not approval or approval.get('package_sha256') != package_sha256 or approval.get('approved_by') != 'Aaron':
                raise ValueError('Explicit package-specific Aaron approval required')
            if not approval.get('user_message_reference') or amount > _money(approval.get('max_cost_usd')):
                raise ValueError('Approval evidence/limit missing or insufficient')
            if commitment_id in commitments:
                raise ValueError('Commitment ID already exists; never reserve twice')
            if obj['spent']+obj['reserved']+amount > 100:
                raise ValueError('Shared USD100 ceiling reached')
            obj['reserved'] += amount
            commitments[commitment_id] = {'package_sha256': package_sha256, 'reserved_usd': amount,
                'status': 'reserved', 'actual_usd': None, 'approval_reference': approval['user_message_reference']}
        elif action == 'reconcile':
            entry = commitments[commitment_id]
            if entry['package_sha256'] != package_sha256 or entry['status'] != 'reserved':
                raise ValueError('Wrong package or already reconciled commitment')
            if not terminal or not usage_complete:
                raise ValueError('Keep all funds reserved until terminal output/error usage reconciliation')
            if amount > entry['reserved_usd']:
                raise ValueError('Cost exceeds reservation: retain hold and investigate before reconciliation')
            if obj['reserved']+1e-8 < entry['reserved_usd']:
                raise ValueError('Shared ledger inconsistent: retain hold')
            obj['reserved'] = max(0., obj['reserved']-entry['reserved_usd'])
            obj['spent'] += amount
            entry.update(status='reconciled', actual_usd=amount)
        else:
            raise ValueError('Unknown action')
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(obj, sort_keys=True, allow_nan=False)+'\n')
        tmp.replace(path)
        return obj
