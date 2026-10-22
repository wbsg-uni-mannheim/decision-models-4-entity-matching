"""Future confirmation successor under reviewed independent parallel writers.

Never launch until the preceding confirmation supervisor and all steps ended.
Reuse its bounded retry classifier, lock, journal and unchanged inference command;
scope failure deltas to confirmation cells and permit other jobs' budget holds.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import resume_confirmation as base
from parallel_admission import review

ROOT = Path(__file__).resolve().parents[2]


def scoped_failures(root, allowed):
    result = {}
    for model, variant, benchmark, split in sorted(allowed):
        path = root/'runs/quality'/model/variant/benchmark/(split+'.failures.jsonl')
        if path.exists():
            result[str(path)] = [json.loads(line) for line in path.read_text().splitlines()]
    return result


def budget_ready(budget):
    # Other jobs and direct Batch commitments legitimately retain reservations.
    if budget['spent']+budget['reserved']+.05 > 100:
        raise ValueError('Shared project spending ceiling reached')
    return budget


def verify_ready():
    config = json.loads((ROOT/'configs/expanded_execution.json').read_text())
    own = os.environ['SLURM_JOB_ID']
    if int(own) != config['e2_confirmation_job']:
        raise ValueError('Not the registered confirmation successor')
    for rel, expected in config['source_sha256'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source changed: '+rel)
    review(ROOT, ROOT/'configs/parallel_execution_20261014.json', own)
    queue = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%T'], text=True)
    for line in queue.splitlines():
        job, state = line.split('|')
        if job == own or state == 'PENDING':
            continue
        script = subprocess.check_output(['scontrol', 'write', 'batch_script', job, '-'], text=True)
        if any(s in script for s in ['expanded_confirmation.py', 'resume_confirmation.py', 'resume_confirmation_parallel.py', 'expanded_rules.py']):
            raise ValueError('Potential overlapping confirmation/control writer: '+job)
    return budget_ready(json.loads((ROOT/'artifacts/budget.json').read_text()))


def main():
    sys.path.insert(0, str(ROOT/'src'))
    import expanded_confirmation as ec
    ec.verify()
    from protocol import MODELS, BENCHMARKS
    allowed = {(m, v, b, 'prompt') for m in MODELS for b in BENCHMARKS
               for v in ec.PROMPT_VARIANTS+ec.SERIALIZATIONS}
    allowed |= {(m, v, b, 'test') for m, v, b in json.loads(
        (ROOT/'configs/expanded_confirmation_cells.json').read_text())}
    base.verify_ready = verify_ready
    base.failures = lambda: scoped_failures(ROOT, allowed)
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
