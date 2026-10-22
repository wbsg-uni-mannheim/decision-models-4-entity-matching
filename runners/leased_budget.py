"""Bounded process credits; identical .05 reserve/unknown policy, fewer Ceph locks.

Unused credits remain globally reserved until explicit drained close. A crashed
process never expires or automatically releases its credit. Ordinary settlements
are collected in memory and checkpointed every 512 reservations.
Release additionally requires exact reconciliation against durable raw journals. This changes accounting transport, not inference or retry policy.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import threading

CAP = 200.0


def utc():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def ledger(root):
    path = Path(root)/'artifacts/budget.json'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        obj = json.loads(path.read_text())
        yield obj
        tmp = path.with_suffix('.leased.tmp')
        with tmp.open('w') as out:
            json.dump(obj, out, sort_keys=True, separators=(',', ':'))
            out.flush(); os.fsync(out.fileno())
        os.replace(tmp, path)
        fd = os.open(str(path.parent), os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


class Credit:
    def __init__(self, root, identity, provenance, chunk=8.0):
        if not identity or not 0.05 <= chunk <= 8:
            raise ValueError('Bounded named credit required')
        self.root, self.identity, self.chunk = Path(root), identity, chunk
        self.lock = threading.RLock()
        self.funded = self.charged = self.committed = 0.0
        self.attempts = self.settled = self.unknown = self.outstanding = 0
        self.closed = False
        self.global_transactions = 0
        with ledger(root) as obj:
            if obj.get('maintenance_barriers') and any(v['active'] for v in obj['maintenance_barriers'].values()):
                raise ValueError('Drain barrier still active')
            leases = obj.setdefault('inference_credits', {})
            if identity in leases:
                raise ValueError('Credit cannot be reused, including after a crash')
            available = CAP-obj['spent']-obj['reserved']
            if available < .05-1e-9:
                raise RuntimeError('Project spending ceiling reached')
            self.funded = min(chunk, available)
            obj['reserved'] += self.funded
            leases[identity] = {'status': 'active', 'created_at': utc(), 'provenance': provenance,
                'funded': self.funded, 'charged': 0., 'attempts': 0, 'settled': 0,
                'unknown': 0, 'outstanding_at_checkpoint': 0, 'checkpoint_at': utc()}
        self.global_transactions += 1

    def account(self, action, amount=0, cap=200):
        if cap != CAP:
            raise ValueError('Wrong ceiling')
        with self.lock:
            if self.closed:
                raise ValueError('Closed credit')
            if action == 'reserve':
                if self.attempts and self.attempts % 512 == 0:
                    self.checkpoint()
                if self.funded-self.charged-.05*self.outstanding < .05-1e-9:
                    self.checkpoint(refill=True)
                if self.funded-self.charged-.05*self.outstanding < .05-1e-9:
                    raise RuntimeError('Project spending ceiling reached')
                self.attempts += 1; self.outstanding += 1
            elif action == 'settle':
                if not self.outstanding:
                    raise ValueError('Settlement without reservation')
                if amount is not None and (type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0):
                    raise ValueError('Invalid actual charge')
                self.outstanding -= 1; self.settled += 1
                self.unknown += amount is None
                self.charged += .05 if amount is None else amount
            else:
                raise ValueError('Unknown account operation')

    def checkpoint(self, refill=False, close=False):
        with self.lock:
            if self.closed or (close and self.outstanding):
                raise ValueError('Cannot release active or already closed credit')
            with ledger(self.root) as obj:
                row = obj['inference_credits'][self.identity]
                if row['status'] != 'active' or row['charged'] != self.committed:
                    raise ValueError('Concurrent or altered credit state')
                delta = self.charged-self.committed
                obj['spent'] += delta; obj['reserved'] -= delta
                obj['attempts'] += self.attempts-row['attempts']
                obj['unknown_cost_attempts'] += self.unknown-row['unknown']
                if refill:
                    extra = max(0., min(self.chunk, CAP-obj['spent']-obj['reserved']))
                    self.funded += extra; obj['reserved'] += extra
                if close:
                    obj['reserved'] -= self.funded-self.charged
                row.update(funded=self.funded, charged=self.charged, attempts=self.attempts,
                    settled=self.settled, unknown=self.unknown,
                    outstanding_at_checkpoint=self.outstanding, checkpoint_at=utc(),
                    status='closed' if close else 'active')
            self.committed = self.charged
            self.closed = close
            self.global_transactions += 1
        return self.summary()

    def summary(self):
        return {'identity': self.identity, 'funded': self.funded, 'charged': self.charged,
            'attempts': self.attempts, 'settled': self.settled, 'unknown': self.unknown,
            'outstanding': self.outstanding, 'global_transactions': self.global_transactions,
            'closed': self.closed}


def install_barrier(root, identity, provenance):
    """Conservative temporary admission barrier, explicitly NOT a paid commitment.

    Old loaded accounts do not understand flags. The artificial reserve blocks
    their next .05 reservation even after every current call settles. Settlements
    remain writable so the existing drain adapter can save every batch outcome.
    Never change spent or release other reservations. No expiry/retry/reset.
    """
    with ledger(root) as obj:
        rows = obj.setdefault('maintenance_barriers', {})
        if identity in rows or any(r['active'] for r in rows.values()):
            raise ValueError('Existing barrier requires explicit reconciliation')
        amount = max(0., CAP-obj['spent'])+.05
        rows[identity] = {'active': True, 'nonfinancial_admission_barrier': True,
            'amount': amount, 'installed_at': utc(), 'provenance': provenance,
            'real_reserved_at_install': obj['reserved'], 'spent_at_install': obj['spent']}
        obj['reserved'] += amount
    return rows[identity]


def remove_barrier(root, identity, terminal_proof):
    if not terminal_proof or not terminal_proof.get('all_steps_terminal'):
        raise ValueError('Terminal proof required before removing barrier')
    with ledger(root) as obj:
        row = obj['maintenance_barriers'][identity]
        if not row['active']:
            raise ValueError('Barrier already removed')
        remaining = obj['reserved']-row['amount']
        if remaining < -1e-7:
            raise ValueError('Barrier or foreign reserve corrupted')
        obj['reserved'] = max(0., remaining)
        row.update(active=False, removed_at=utc(), terminal_proof=terminal_proof)
    return row
