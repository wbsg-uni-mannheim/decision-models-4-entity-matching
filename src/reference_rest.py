"""Reference run of the three ensemble members on the full test sets (Aaron, 21 Oct 2026).

The members already answered the 6,682 escalated test pairs (runs/escalation_run01_20261020).
This module freezes the remaining test pairs (test minus escalation) as split
`reference_rest` and builds the transport-specific request bodies. Every body carries the
same frozen Luna B0 messages, JSON schema, reasoning effort and output limit as the
escalation requests (src/escalation.py); only routing and tier differ:
- GPT-6.1 Sol: direct OpenAI Chat Completions, flex tier.
- Claude Opus 5.5: OpenRouter Batch API, Anthropic endpoint, model and provider set per batch.
- Gemini 4 Argon: OpenRouter live, Google AI Studio flex, exactly the escalation body.
"""
import copy
import json
from pathlib import Path

import escalation
import protocol

PROJECT = Path(__file__).resolve().parents[1]
SPLIT = 'reference_rest'
SOL_MODEL = 'gpt-6.1-sol'
OPUS_MODEL = 'anthropic/claude-opus-5.5'


def rest_cases(root, b):
    done = {c['pair_id'] for c in json.loads((Path(root)/'data'/b/'escalation.json').read_text())}
    return [c for c in json.loads((Path(root)/'data'/b/'test.json').read_text()) if c['pair_id'] not in done]


def freeze_rest(root=PROJECT):
    """Write data/<b>/reference_rest.json once (test order) and return a report."""
    import prepare
    root = Path(root)
    report, total = {}, 0
    for b in protocol.BENCHMARKS:
        rows = rest_cases(root, b)
        prepare.write_frozen(root/'data'/b/(SPLIT+'.json'), rows)
        n_test = len(json.loads((root/'data'/b/'test.json').read_text()))
        n_esc = len(json.loads((root/'data'/b/'escalation.json').read_text()))
        assert len(rows)+n_esc == n_test, b
        report[b] = {'n': len(rows), 'positive': sum(c['label'] for c in rows), 'sha256': protocol.digest(rows)}
        total += len(rows)
    return {'benchmarks': report, 'pairs': total}


def sol_direct_body(case):
    """OpenAI Chat Completions equivalent of the escalation request for Sol, flex tier."""
    src = escalation.payload(case, 'sol61')
    return {'model': SOL_MODEL, 'messages': copy.deepcopy(src['messages']),
            'response_format': copy.deepcopy(src['response_format']),
            'reasoning_effort': src['reasoning']['effort'], 'max_completion_tokens': src['max_tokens'],
            'service_tier': 'flex', 'store': False}


def opus_batch_body(case):
    """Per-request body for the OpenRouter batch; model and provider are set at batch level."""
    src = escalation.payload(case, 'opus55')
    return {k: copy.deepcopy(v) for k, v in src.items() if k not in ('model', 'provider')}


def vote(text):
    value = json.loads(text)
    if set(value) != {'match'} or type(value['match']) is not bool:
        raise ValueError('Invalid match schema')
    return int(value['match'])
