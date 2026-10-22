"""Opt-in group commit at existing batch boundaries; no scientific changes.

Ordinary failures drain and commit successful siblings before propagating.
Process/storage failure is not a losslessness guarantee: callers must retain
credit and diagnose any partial write. Never install into a running process.
"""
import ast
import builtins
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import threading
import types

import drained_inference

DRAIN_SHA = '28e7ce2b2020a1d2b8a9def38c1b3e70e2b202938e5564a6a45cd981330a85b2'


def write_group(path, lines):
    new = not path.exists()
    with path.open('a') as out:
        out.writelines(lines)
        out.flush()
        os.fsync(out.fileno())
    if new:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class Groups:
    def __init__(self, on_failure=lambda: None):
        self.local = threading.local()
        self.on_failure = on_failure

    def append(self, path, value, canonical):
        line = canonical(value)+'\n'
        group = getattr(self.local, 'group', None)
        if group is None:
            try:
                write_group(path, [line])
            except Exception:
                self.on_failure()
                raise
        else:
            group[path].append(line)

    @contextmanager
    def batch(self):
        if getattr(self.local, 'group', None) is not None:
            raise ValueError('Nested journal batch')
        group = self.local.group = defaultdict(list)
        try:
            yield
        finally:
            self.local.group = None
            errors = []
            # Try every journal even if one write fails. No automatic retry.
            for path, lines in group.items():
                try:
                    write_group(path, lines)
                except Exception as exc:
                    errors.append(exc)
            if errors:
                self.on_failure()
                raise OSError('Incomplete group commit; audit before resume') from errors[0]


def compile_grouped(source, filename, mode):
    tree = ast.parse(source)
    cells = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_cell']
    if len(cells) != 1:
        raise ValueError('Expected exactly one frozen cell')
    batches = [n for n in ast.walk(cells[0]) if isinstance(n, ast.For)
               and isinstance(n.target, ast.Name) and n.target.id == 'offset']
    if len(batches) != 1:
        raise ValueError('Frozen batch boundary changed')
    loop = batches[0]
    # Same submission/collection order and same worker count. Commit happens
    # before the next batch, also when the existing failed-batch guard raises.
    loop.body = [ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Name(id='_journal_batch', ctx=ast.Load()), args=[], keywords=[]))],
        body=loop.body)]
    ast.fix_missing_locations(tree)
    return builtins.compile(tree, filename, mode)


def build(inference, run_id):
    if hashlib.sha256(Path(drained_inference.__file__).read_bytes()).hexdigest() != DRAIN_SHA:
        raise ValueError('Drained adapter changed')
    groups = Groups(getattr(inference, '_on_journal_failure', lambda: None))
    isolated = types.SimpleNamespace(**{**vars(inference), '_journal_batch': groups.batch})
    builder = types.FunctionType(drained_inference.build.__code__,
        {**drained_inference.build.__globals__, 'compile': compile_grouped,
         'durable_append': groups.append}, argdefs=drained_inference.build.__defaults__)
    request, cell, proof = builder(isolated, run_id)
    return request, cell, {**proof, 'journal_transport': 'one fsync per affected file per existing batch',
        'grouped_adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'process_or_storage_failure_losslessness_guaranteed': False}
