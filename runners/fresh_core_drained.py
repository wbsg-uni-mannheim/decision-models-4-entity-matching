"""Core execution component for a future separately admitted recovery launcher.

No CLI or admission is provided. The caller must hold the replica writer locks
and verify scheduler, diagnosis, budget, policy and historical stage provenance.
This component preserves the frozen core phase order and selection function.
It never resets or marks the outer core stage journal complete.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

from drained_inference import build, drain_model_tasks
from repeated_runs import shared_account

ORCHESTRATE_SHA = '36040a537fccd8d4f5e22aceb9f9b52ad5110b96b0a442998855ac9629d9c1a7'


def run_phases(root, target, run_id, inference, orchestration, protocol, journal_dir):
    """One invocation of each core phase, beginning at a diagnosed failed B0.

    All source/module loading and real admission belong to the caller. Only
    account() is cloned to the shared root; neither imported module is patched.
    Ordinary model failures drain before stopping, with no phase retry.
    """
    if run_id not in (2, 3) or type(run_id) is not int:
        raise ValueError('Fresh study run required')
    if inference.PROJECT != target or orchestration.PROJECT != target:
        raise ValueError('Wrong core workspace')
    if hashlib.sha256(Path(orchestration.__file__).read_bytes()).hexdigest() != ORCHESTRATE_SHA:
        raise ValueError('Frozen core orchestration changed')
    status = json.loads((target/'artifacts/progress.json').read_text())
    if (set(status) != {'smoke', 'AP1_baseline'} or status['smoke']['status'] != 'complete'
            or status['AP1_baseline']['status'] != 'needs_attention'):
        raise ValueError('Only an initial core failure in baseline is supported')
    orchestration.verify()
    isolated = types.SimpleNamespace(**vars(inference))
    isolated.account = shared_account(inference.account, root)
    _, cell, adapter = build(isolated, run_id)
    # Exclusive creation prevents reusing this component's invocation directory.
    journal_dir.mkdir(exist_ok=False)
    stage = 'AP1_baseline'

    def parallel(phase, aliases, variants, splits):
        def task(model):
            for variant in variants(model):
                for split in splits:
                    for benchmark in protocol.BENCHMARKS:
                        cell(model, variant, benchmark, split, workers=16)
        outcomes = drain_model_tasks(task, list(aliases), journal_dir/(phase+'.jsonl'), protocol.canonical)
        if any(row['status'] != 'complete' for row in outcomes):
            raise RuntimeError('Core phase failed after draining every model: '+phase)

    try:
        orchestration.stamp(stage, 'running')
        parallel(stage, protocol.MODELS, lambda _: ['B0'], ['prompt', 'route', 'test'])
        orchestration.stamp(stage, 'complete')
        stage = 'AP2_development'
        orchestration.stamp(stage, 'running')
        parallel(stage, protocol.ABLATION_MODELS, lambda _: ['P1', 'P2', 'S1', 'O1'], ['prompt'])
        selection = orchestration.select()
        orchestration.stamp(stage, 'complete', selection)
        stage = 'AP2_confirmation'
        orchestration.stamp(stage, 'running')
        parallel(stage, selection, lambda m: [selection[m]['variant']], ['test'])
        orchestration.stamp(stage, 'complete')
        stage = 'AP1_timing'
        orchestration.stamp(stage, 'running')
        # Preserve the original sequential model/benchmark/timing order.
        for model in protocol.MODELS:
            for benchmark in protocol.BENCHMARKS:
                for workers, name in [(1, 'serial'), (8, 'concurrent8')]:
                    cell(model, 'B0', benchmark, 'prompt', workers=workers, limit=20,
                         namespace='timing_'+name)
        orchestration.stamp(stage, 'complete')
        stage = 'AP3_AP4_analysis'
        orchestration.stamp(stage, 'running')
        subprocess.run([sys.executable, str(target/'src/analyze.py')], check=True, cwd=target)
        orchestration.stamp(stage, 'complete_automated', 'Independent human audit and author review remain pending')
    except Exception as exc:
        orchestration.stamp(stage, 'needs_attention', str(exc))
        raise
    return {'run_id': run_id, 'adapter': adapter, 'outer_core_completion_written': False,
            'historical_unjournaled_outcomes_resolved': False}
