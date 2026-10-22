"""Ten seconds of read-only Linux thread/wait/FD sampling, no process attach."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def sample(pid):
    base = Path('/proc')/str(pid)
    waits, calls, destinations = Counter(), Counter(), Counter()
    start = datetime.now(timezone.utc).isoformat()
    before = (base/'io').read_text()
    for _ in range(100):
        for task in (base/'task').iterdir():
            try:
                wait = (task/'wchan').read_text().strip()
                syscall = (task/'syscall').read_text().split()
                waits[wait] += 1
                calls[syscall[0]] += 1
                if syscall[0] in ('0', '1', '74', '75') and len(syscall) > 1:
                    fd = int(syscall[1], 0)
                    dest = os.readlink(base/'fd'/str(fd))
                    category = ('raw_journal' if '/runs/' in dest else
                                'stdout_log' if '/logs/' in dest else
                                'input_data' if '/data/' in dest else
                                'budget' if '/budget.' in dest else dest)
                    destinations[(syscall[0], wait, category)] += 1
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
                pass
        time.sleep(.1)
    return {'pid': pid, 'hostname': os.uname().nodename, 'start_utc': start,
            'end_utc': datetime.now(timezone.utc).isoformat(), 'wait_channels': dict(waits),
            'syscalls': dict(calls), 'file_waits': [{'syscall': k[0], 'wait': k[1], 'destination': k[2], 'samples': v}
                                                  for k, v in destinations.most_common()],
            'io_before': before, 'io_after': (base/'io').read_text(),
            'limitations': 'Thread-state samples, not additive wall-time shares; futex often idle pool workers. No attach, signal, inference or ledger writes.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--pid', type=int, required=True)
    print(json.dumps(sample(p.parse_args().pid)))
