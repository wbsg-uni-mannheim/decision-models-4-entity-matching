"""PPLX v1.2 with the default prompt B0 on the full test sets, runs 2 and 3 (Aaron, 21 Oct 2026).

Run 1 is the B0 collection of the 12-prompt study (runs/prompt12_run01_20261019/pplx_v12/B0).
Runs 2 and 3 repeat exactly that request, so the baseline table can report PPLX v1.2 with
three collections like the other models. PPLX v1.1 lost its OpenRouter endpoint on 19 Oct 2026.
"""
import prompt12
import protocol

SPLIT = 'test'
RUN_MODELS = ['pplx_v12']
SELECTION = None


def register_models():
    prompt12.register_models()


def run_variants(alias):
    return ['B0']


def benchmarks(variant):
    return list(protocol.BENCHMARKS)


def load_selection(root=None):
    return None


def payload(case, alias, variant='B0'):
    assert alias == 'pplx_v12' and variant == 'B0', (alias, variant)
    return prompt12.payload(case, 'pplx_v12', 'B0')
