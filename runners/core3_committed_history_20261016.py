"""Exact Run3/367108 core anchor and narrow first-supplement history adapter.

No inference, funding, scheduler admission, or replacement final validator.
Capture must precede any extension. Consumers must pin manifest.json in their
own separately reviewed admission. This module never makes a failed core pass.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

SELF = 'artifacts/operations/core3_committed_history_20261016.py'
ANCHOR = 'artifacts/verification/core3_commit_anchor_367108_20261016'
TARGET = 'replications/run03'
ATTEMPT = 'artifacts/operations/core3_timing_recovery_366980_v1'
MARKER = 'artifacts/core3_timing_verified_complete.json'
PROGRESS = 'artifacts/progress.json'
STAGE = 'artifacts/stage_execution.jsonl'
ADMISSION = 'configs/core3_timing_admission_20261016.json'
READBACK = 'artifacts/verification/core3_timing_commit_367108_20261016.json'
MARKER_SHA = '93773b6c49a9cc3ebf563bc76a23bced32ba3715d654f88d8ac431714e8d5550'
PROGRESS_SHA = '6c00dc9cf528d73bebab1f4120888cfa456adf984c9060aa6bdac72804f8dd08'
STAGE_SHA = '9627c7558d484c25b642450d57e3114d31fd25f7d9b90c6f9462e25b0e023a71'
PROOF_SHA = '06408a4cd533038ee38b96fc204877592ffffe4ba71ab52ea9a1d94394799513'
END_SHA = '1b9b907a02cf857de1827f85137fd30fd0cbb1e1ece00e06a6939fe1290e0224'
ADMISSION_SHA = 'a15b0ad00332131e9afad06300207f8e1630b0660fe7118bf1eadbba71c1eca6'
READBACK_SHA = '7def6a1d32b951e91257d18c99c8dae6e29dd044131461ce83557458b9965c1c'
HISTORY = ('365894','365895','366118','366520','366644','366681',
           '366697','366745','366955','366980','367108')
MODELS = ('jev','clef','clef_flash','luna_decisions','pplx','kev','luna')
ADAPTER = 'core3_367108_exact_commit_first_supplement_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def own(root, rel):
    p = root/rel
    require(not Path(rel).is_absolute() and '..' not in Path(rel).parts
            and p.resolve().is_relative_to(root.resolve()), 'Path escapes evidence scope')
    require(not any(q.is_symlink() for q in [p, *p.parents] if q != root.parent), 'Symlink in evidence path')
    return p


def protected_path(root, target, rel):
    # Replicas intentionally share frozen inputs through their data symlink.
    # Only the exact same root/data relative path is allowed, never a raw cache.
    if rel.startswith('data/'):
        source = own(root, rel)
        require((target/rel).resolve() == source.resolve(), 'Frozen replica input link changed')
        return source
    return own(target, rel)


def rows(raw):
    require(bool(raw) and raw.endswith(b'\n'), 'Incomplete journal bytes')
    return [json.loads(line) for line in raw.splitlines()]


def history_steps():
    """Every historical step, including smoke and failed recoveries, read live."""
    results = {}
    for job in HISTORY:
        raw = subprocess.check_output(['sacct','-n','-j',job,
            '--format=JobID,State,ExitCode,End','-P'], text=True, env={**os.environ,'TZ':'UTC'})
        parsed = [line.split('|') for line in raw.splitlines() if line.strip()]
        require(len(parsed) >= 3 and len({r[0] for r in parsed}) == len(parsed)
                and {job,job+'.batch',job+'.extern'} <= {r[0] for r in parsed}, 'Missing historical Slurm step')
        for row in parsed:
            require(len(row) >= 4 and (row[0] == job or row[0].startswith(job+'.'))
                    and row[3] not in ('','Unknown','N/A'), 'Invalid historical accounting')
            extra = {'366681.0','366681.1','366697.0'}
            require(row[0] in {job,job+'.batch',job+'.extern'} | extra, 'Unreviewed historical step')
            expected = ('COMPLETED','0:0') if job in (HISTORY[0],HISTORY[-1]) or row[0].endswith('.extern') or row[0] in extra else ('FAILED','1:0')
            require(tuple(row[1:3]) == expected, 'Historical step changed: '+row[0])
        required_extra = {s for s in ('366681.0','366681.1','366697.0') if s.startswith(job+'.')}
        require(required_extra <= {r[0] for r in parsed}, 'Missing historical numbered step')
        results[job] = raw
    return results


def bound_core(root):
    """Rehash existing independent proof and all bound bytes, not a new numerical replay."""
    target = own(root,TARGET)
    pinned = {MARKER:MARKER_SHA, ATTEMPT+'/final_audit.json':PROOF_SHA,
              ATTEMPT+'/final_execution.jsonl':END_SHA}
    for rel,h in pinned.items():
        require(sha(own(target,rel)) == h, 'Core commit evidence changed: '+rel)
    for rel,h in ((ADMISSION,ADMISSION_SHA),(READBACK,READBACK_SHA)):
        require(sha(own(root,rel)) == h, 'Core root evidence changed: '+rel)
    marker, proof, admission = read(target/MARKER), read(target/ATTEMPT/'final_audit.json'), read(root/ADMISSION)
    require(type(marker['run_id']) is int and marker['run_id'] == 3
            and marker['slurm_job_id'] == HISTORY[-1] and marker['status'] == 'complete'
            and marker['progress_sha256'] == PROGRESS_SHA
            and marker['independent_final_audit_sha256'] == PROOF_SHA
            and marker['admission_sha256'] == ADMISSION_SHA, 'Wrong committed core')
    require(sha(target/ATTEMPT/'final_publication_gate.json') == marker['final_gate_sha256'], 'Final gate changed')
    end = rows((target/ATTEMPT/'final_execution.jsonl').read_bytes())
    require(len(end) == 1 and end[0]['status'] == 'complete'
            and end[0]['completion_marker_sha256'] == MARKER_SHA
            and end[0]['progress_sha256'] == PROGRESS_SHA
            and end[0]['report_sha256'] == marker['report_sha256'], 'Final event changed')
    require(proof['status'] == 'independent_raw_execution_cost_numeric_report_verified'
            and proof['quality_cells'] == 328 and proof['quality_responses'] == 380475
            and proof['timing_cells'] == 112 and proof['timing_responses'] == 2240
            and proof['admission_sha256'] == ADMISSION_SHA, 'Independent proof identity')
    policy = read(own(root,admission['policy']))
    require(sha(root/admission['policy']) == admission['policy_sha256'], 'Timing policy changed')
    pins = {**policy['files_sha256'], **admission['sources']}
    for rel,h in pins.items():
        require(sha(own(root,rel)) == h, 'Core source changed: '+rel)
    for rel,h in proof['protected_sha256_or_absent'].items():
        p = protected_path(root,target,rel) if rel != PROGRESS else target/ATTEMPT/'progress_before_publication.json'
        require((sha(p) if p.exists() else None) == h, 'Protected core bytes changed: '+rel)
    for rel,h in {**proof['raw_sha256'], **marker['report_sha256']}.items():
        require(sha(own(target,rel)) == h, 'Core raw/report bytes changed: '+rel)
    workspace = target/ATTEMPT/'analysis_staging'
    require({str(p.relative_to(workspace)):sha(p) for p in workspace.rglob('*') if p.is_file()}
            == proof['staged_files_sha256'], 'Private analysis staging changed')
    return marker, proof, pins


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive(path, raw):
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    fsync_dir(path.parent)
    require(path.read_bytes() == raw, 'Anchor durable readback mismatch')


def capture(root):
    """One exclusive checkpoint under both writer locks, before any extension."""
    import complete_core3_timing_20261016 as complete
    root = Path(root).resolve()
    target, dest = root/TARGET, root/ANCHOR
    with ExitStack() as stack:
        for p,mode in ((root/'artifacts/operations/repetition_writer.lock',fcntl.LOCK_SH),
                       (target/'artifacts/parallel_writer.lock',fcntl.LOCK_EX)):
            f = stack.enter_context(p.open('a'))
            fcntl.flock(f,mode|fcntl.LOCK_NB)
        require(not dest.exists(), 'Anchor already exists; never recapture or overwrite')
        complete.require_committed(target)  # The unmodified final CLI, on the actual current state.
        marker, proof, pins = bound_core(root)
        stage, progress = (target/STAGE).read_bytes(), (target/PROGRESS).read_bytes()
        require(digest(stage) == STAGE_SHA and digest(progress) == PROGRESS_SHA, 'Not the exact pre-extension commit')
        require(rows(stage)[-1] == rows((target/ATTEMPT/'final_execution.jsonl').read_bytes())[0], 'Missing final stage event')
        require(tuple(dict.fromkeys(r['slurm_job_id'] for r in rows(stage))) == HISTORY, 'Whole history identity')
        require({str(p.relative_to(target)) for p in (target/'runs').rglob('*.jsonl')} == set(proof['raw_sha256']), 'Unexpected pre-extension raw journal')
        steps = history_steps()
        dest.mkdir(); fsync_dir(dest.parent)
        exclusive(dest/'progress.json',progress)
        exclusive(dest/'stage_execution.jsonl',stage)
        value = dict(adapter=ADAPTER,run_id=3,core_job_id=HISTORY[-1],
            captured_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(root/SELF),
            progress_sha256=PROGRESS_SHA,stage_sha256=STAGE_SHA,marker_sha256=MARKER_SHA,
            independent_proof_sha256=PROOF_SHA,commit_readback_sha256=READBACK_SHA,
            final_execution_sha256=END_SHA,core_admission_sha256=ADMISSION_SHA,
            historical_steps_utc=steps,history_jobs=list(HISTORY),source_pins=len(pins),
            raw_files=len(proof['raw_sha256']),reports=len(marker['report_sha256']),
            immutable_core_only=True,admission_granted=False,numeric_proof_replayed=False,
            initial_policy_bytes_available=False)
        require((target/STAGE).read_bytes() == stage and (target/PROGRESS).read_bytes() == progress, 'Concurrent core mutation')
        exclusive(dest/'manifest.json',(json.dumps(value,sort_keys=True,indent=2)+'\n').encode())
        return value


def supplement_paths():
    return {f'runs/quality/{m}/B0/billiger-de-50un/test{s}' for m in MODELS
            for s in ('.jsonl','.failures.jsonl','.executions.jsonl','.operations.jsonl')}


def verify_prefix(target, anchor, *, started=None):
    """Only the exact core prefix plus zero or ONE explicitly bound supplement start.

The future wrapper must supply the start event from its own durable attempt,
validate its actual admission and hold both locks. No failed/completed/foreign
extension history is accepted by this first-stage adapter.
"""
    saved_stage = (anchor/'stage_execution.jsonl').read_bytes()
    saved_progress = (anchor/'progress.json').read_bytes()
    require(digest(saved_stage) == STAGE_SHA and digest(saved_progress) == PROGRESS_SHA, 'Anchor bytes changed')
    current = (target/STAGE).read_bytes()
    require(current.startswith(saved_stage), 'Core stage prefix changed')
    suffix = current[len(saved_stage):]
    if started is None:
        require(not suffix, 'An extension already exists; separate reviewed history adapter required')
    else:
        require(type(started.get('run_id')) is int and started['run_id'] == 3
                and started.get('stage') == 'supplement' and started.get('status') == 'started'
                and str(started.get('slurm_job_id','')).isdigit()
                and str(started['slurm_job_id']) not in HISTORY
                and started.get('prior_stage_journal_sha256') == STAGE_SHA
                and all(isinstance(started.get(k),str) and len(started[k]) == 64 for k in
                    ('admission_sha256','parallel_policy_sha256','launcher_sha256','prior_cache_sha256')),
                'Invalid first supplement start identity')
        require(rows(suffix) == [started], 'Unreviewed extension history')
    require((target/PROGRESS).read_bytes() == saved_progress, 'Core progress changed before supplement publication')


def verify(root, manifest_sha, *, started=None):
    """Read-only prerequisite proof. Never an inference/admission entry point."""
    root = Path(root).resolve()
    target, anchor = own(root,TARGET), own(root,ANCHOR)
    require(sha(anchor/'manifest.json') == manifest_sha, 'Anchor admission pin changed')
    manifest = read(anchor/'manifest.json')
    require(manifest['adapter'] == ADAPTER and type(manifest['run_id']) is int and manifest['run_id'] == 3
            and manifest['history_jobs'] == list(HISTORY) and manifest['source_sha256'] == sha(root/SELF)
            and manifest['admission_granted'] is False, 'Wrong anchor provenance')
    verify_prefix(target,anchor,started=started)
    marker, proof, pins = bound_core(root)
    observed = {str(p.relative_to(target)) for p in (target/'runs').rglob('*.jsonl')}
    allowed = set(proof['raw_sha256']) | (supplement_paths() if started else set())
    require(set(proof['raw_sha256']) <= observed <= allowed, 'Foreign extension raw journal')
    accounts = history_steps()
    verify_prefix(target,anchor,started=started)
    return dict(adapter=ADAPTER,anchor_sha256=manifest_sha,core_job_id=HISTORY[-1],
        source_pins=len(pins),core_raw_files_rehashed=len(proof['raw_sha256']),
        core_reports_rehashed=len(marker['report_sha256']),historical_steps_utc=accounts,
        admission_granted=False,whole_core_history_verified=True,numeric_proof_replayed=False)
