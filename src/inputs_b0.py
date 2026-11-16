"""Input variants with the default prompt B0 on the full core test sets.

The inputs study (src/inputs_study.py) used the prompt selected on test per model and
benchmark (F0). The robustness chapter now uses B0 everywhere: nine variants (the
serializations F1-F4 and the perturbations R1-R4, R6; span deletion R5 is dropped)
x seven models x eight benchmarks, with B0 Run 1 as the reference. Record text and
perturbations are exactly inputs_study.state; only the instruction is B0.

Run 1 reuses the inputs-study cells whose F0 already was B0 (identical request,
checked at freeze). Runs 2 and 3 repeat the entry swap (R1) for all 56 cells.
"""
import json
from pathlib import Path

import inputs_study
import prompt12
import protocol

PROJECT = Path(__file__).resolve().parents[1]
RUN_MODELS = list(prompt12.RUN_MODELS)
VARIANT_IDS = [v for v in inputs_study.VARIANT_IDS if v != 'R5_span_del']
REPEAT_VARIANTS = ['R1_swap']
INPUTS = 'inputs_run01_20261019'


def register_models():
    prompt12.register_models()


def run_variants(alias, run_id=1):
    return list(VARIANT_IDS) if run_id == 1 else list(REPEAT_VARIANTS)


def benchmarks(variant):
    return inputs_study.benchmarks(variant)


def payload(case, alias, variant):
    body = prompt12.payload(case, alias, 'B0')
    value = inputs_study.state(case, variant)
    if 'questions' in body:
        body['state'] = value
    else:
        body['messages'][1]['content'] = value
    return body


def reused(root=PROJECT):
    """{(model, variant, benchmark): inputs-study journal} for run-1 cells whose F0 was B0."""
    root = Path(root)
    selected = inputs_study.load_selection(root)['selected']
    return {(m, v, b): root/'runs'/INPUTS/m/v/b/'test.jsonl'
            for m in RUN_MODELS for v in VARIANT_IDS for b in benchmarks(v) if selected[m][b] == 'B0'}
