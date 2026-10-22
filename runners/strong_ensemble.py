"""Offline strong-ensemble protocol primitives; deliberately has no network runner."""
import copy
import json
import math


MEMBERS = {
    'sol61': {'id': 'openai/gpt-6.1-sol', 'provider': 'openai',
              'input_usd_per_million': 2., 'output_usd_per_million': 10.},
    'opus55': {'id': 'anthropic/claude-opus-5.5', 'provider': 'anthropic',
               'input_usd_per_million': 4., 'output_usd_per_million': 20.},
    'argon4': {'id': 'google/gemini-4-argon', 'provider': 'google-vertex/global',
               'input_usd_per_million': 2., 'output_usd_per_million': 10.},
}
MAX_OUTPUT = 2048


def payload_from_frozen_b0(luna_b0_payload, member):
    """Retain exact B0 record text/instruction/schema; change only model settings."""
    if member not in MEMBERS:
        raise ValueError('Unknown fixed ensemble member')
    body = copy.deepcopy(luna_b0_payload)
    if body.get('model') != 'openai/gpt-6-luna' or len(body.get('messages', [])) != 2:
        raise ValueError('Expected frozen Luna B0 payload')
    spec = MEMBERS[member]
    body.update(model=spec['id'], provider={'only': [spec['provider']],
                'order': [spec['provider']], 'allow_fallbacks': False, 'require_parameters': True},
                reasoning={'effort': 'medium'}, max_tokens=MAX_OUTPUT)
    return body


def majority(votes):
    """Require all three members, even if the first two agree; never hide outages."""
    if set(votes) != set(MEMBERS):
        raise ValueError('Exactly three named members required')
    if any(type(v) is not bool for v in votes.values()):
        raise ValueError('Missing or invalid vote: cell remains incomplete')
    positives = sum(votes.values())
    return {'match': positives >= 2, 'positive_votes': positives,
            'unanimous': positives in (0, 3)}


def parse_vote(response):
    """Model/provider identity and cost must also be checked by the future runner."""
    if 'error' in response:
        raise ValueError('Provider error')
    choice, = response['choices']
    if choice.get('finish_reason') != 'stop' or choice['message'].get('refusal'):
        raise ValueError('Incomplete or refused vote')
    value = json.loads(choice['message']['content'])
    if set(value) != {'match'} or type(value['match']) is not bool:
        raise ValueError('Invalid match schema')
    return value['match']


def conditional_cost(input_tokens_by_member, output_tokens=MAX_OUTPUT):
    """Nominal price scenario, NOT a reservation bound or token-count guarantee."""
    if set(input_tokens_by_member) != set(MEMBERS):
        raise ValueError('All members required for cost scenario')
    if type(output_tokens) is not int or not 0 <= output_tokens <= MAX_OUTPUT:
        raise ValueError('Invalid output token scenario')
    costs = {}
    for member, n in input_tokens_by_member.items():
        if type(n) not in (int, float) or not math.isfinite(n) or n < 0:
            raise ValueError('Invalid input token scenario')
        spec = MEMBERS[member]
        costs[member] = (n*spec['input_usd_per_million'] + output_tokens*spec['output_usd_per_million'])/1e6
    return {'members_usd': costs, 'ensemble_usd': sum(costs.values()),
            'excludes': ['cache-write premiums', 'retries', 'unknown costs', 'live repetitions'],
            'hard_cost_bound': False}
