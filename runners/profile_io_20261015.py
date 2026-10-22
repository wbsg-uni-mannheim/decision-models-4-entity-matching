"""Bounded read-only live sampling and isolated filesystem microbenchmarks."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import statistics
import tempfile
import time


def describe(values):
    return dict(n=len(values), median_seconds=statistics.median(values),
                p95_seconds=sorted(values)[int(.95*(len(values)-1))], maximum_seconds=max(values))


def profile(root, pid):
    waits=Counter();calls=Counter();samples=0
    for _ in range(30):
        for task in (Path('/proc')/str(pid)/'task').iterdir():
            try:
                waits[(task/'wchan').read_text().strip()]+=1
                calls[(task/'syscall').read_text().split()[0]]+=1
                samples+=1
            except (FileNotFoundError,ProcessLookupError,PermissionError):
                pass
        time.sleep(.1)
    lock_times=[];reads=[]
    # Same existing lock, read only, 12 acquisitions with no ledger mutation.
    for _ in range(12):
        with (root/'artifacts/budget.lock').open('r') as f:
            t=time.perf_counter();fcntl.flock(f,fcntl.LOCK_SH);lock_times.append(time.perf_counter()-t)
            t=time.perf_counter();json.loads((root/'artifacts/budget.json').read_text());reads.append(time.perf_counter()-t)
        time.sleep(.02)
    io={}
    for label,parent in [('shared',root/'artifacts/operations'),('node_local',Path('/tmp'))]:
        with tempfile.TemporaryDirectory(prefix='decision-em-io-profile-',dir=parent) as name:
            p=Path(name);sync=[];replaces=[]
            for _ in range(24):
                t=time.perf_counter()
                with (p/'journal').open('a') as f:
                    f.write('x'*2048+'\n');f.flush();os.fsync(f.fileno())
                sync.append(time.perf_counter()-t)
                t=time.perf_counter();(p/'budget.tmp').write_text('{"spent":1,"reserved":2}\n');(p/'budget.tmp').replace(p/'budget.json')
                replaces.append(time.perf_counter()-t)
            io[label]={'durable_append':describe(sync),'atomic_replace':describe(replaces)}
    return dict(pid=pid,hostname=os.uname().nodename,thread_samples=samples,wait_channels=dict(waits),
                syscall_numbers=dict(calls),live_budget_shared_lock_wait=describe(lock_times),
                live_budget_read=describe(reads),isolated_io=io,
                limitations=['Read-only live sampling; isolated scratch files removed. No HTTP, ledger writes, process interruption or protocol changes.',
                             'Linux syscall numbers are architecture-specific; attempt durations include settlement and are not pure provider latency.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pid',type=int,required=True)
    a=p.parse_args();print(json.dumps(profile(a.root,a.pid)))
