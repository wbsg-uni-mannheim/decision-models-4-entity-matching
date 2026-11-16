"""Prompt selection on the official validation sets (one run, no test data).

The 12-prompt study selected the prompt per model and benchmark on test (leakage,
Chris, Oct 2026). This study repeats it on the validation split: ten prompts
(B0 plus the nine zero-shot variants of prompt12; the two rule prompts move to the
labelled-data chapter) x seven models x eight benchmarks. Requests are exactly
prompt12.payload; only the cases change (data/<b>/valid.json from src/valsel.py,
30,149 pairs). The Astra training pairs (rules_train) are excluded at scoring,
as in the threshold tuning.

All 560 cells come from promptval_run01_20261027. The 29 cells whose request is
identical to a cell of valsel_run01_20261019 (selected prompt F0 on validation)
serve as a repeat-run cross-check only (reused()).
"""
import json
from pathlib import Path

import prompt12
import protocol

PROJECT = Path(__file__).resolve().parents[1]
SPLIT = 'valid'
RUN_MODELS = list(prompt12.RUN_MODELS)
PROMPTS = ['B0'] + [v for v in prompt12.NEW_VARIANTS if not v.startswith('R_')]
VALSEL = 'valsel_run01_20261019'
VALSEL_SELECTION = 'configs/inputs_prompt_selection_20261019.json'


def register_models():
    prompt12.register_models()


def run_variants(alias):
    return list(PROMPTS)


def benchmarks(variant):
    return list(protocol.BENCHMARKS)


def payload(case, alias, variant):
    assert variant in PROMPTS, variant
    return prompt12.payload(case, alias, variant)


def reused(root=PROJECT):
    """{(model, prompt, benchmark): valsel journal} for cells run earlier with the identical request."""
    root = Path(root)
    selected = json.loads((root/VALSEL_SELECTION).read_text())['selected']
    out = {}
    for m, per in selected.items():
        for b, v in per.items():
            path = root/'runs'/VALSEL/m/'F0'/b/(SPLIT+'.jsonl')
            if m in RUN_MODELS and v in PROMPTS and path.exists():
                out[(m, v, b)] = path
    return out
