"""Qwen3.8-27B (base model of PPLX v1.2) as a generative baseline, prompt B0, full test sets, runs 1-3.

Request = the Luna generative B0 request (src/protocol.py: same messages, JSON schema with a boolean
`match`, reasoning disabled, 64 output tokens), with model qwen/qwen3.8-27b pinned to the full-precision
DeepInfra endpoint (bf16) without fallback. Parsing is the generative parser of protocol.py.
"""
import copy

import prompt12
import protocol

SPLIT = 'test'
ALIAS = 'qwen38'
MODEL = {'id': 'qwen/qwen3.8-27b', 'provider': 'deepinfra/bf16', 'kind': 'generative'}
RUN_MODELS = [ALIAS]
SELECTION = None


def register_models():
    prompt12.register_models()
    assert protocol.MODELS.get(ALIAS, MODEL) == MODEL
    protocol.MODELS[ALIAS] = MODEL


def run_variants(alias):
    return ['B0']


def benchmarks(variant):
    return list(protocol.BENCHMARKS)


def load_selection(root=None):
    return None


def payload(case, alias=ALIAS, variant='B0'):
    assert alias == ALIAS and variant == 'B0', (alias, variant)
    body = copy.deepcopy(protocol.payload(case, 'luna', 'B0'))
    body['model'] = MODEL['id']
    body['provider'] = {'only': [MODEL['provider']], 'allow_fallbacks': False}
    return body
