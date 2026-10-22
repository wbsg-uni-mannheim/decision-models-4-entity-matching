"""Offline payload freeze for two structured-output Peeters adaptations.

No inference, credentials, cache writes, selection or scheduling in this module.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

QUESTION = 'Do the two entity descriptions refer to the same real-world entity?'
ARMS = ('peeters_fields', 'peeters_values')
SOURCE_COMMIT = 'ba1691935c744bb0eacc9abf772f39271e318e4b'


def value_text(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def values_state(case):
    # Keep current frozen field order and every value; no historical truncation.
    return '\n'.join(f"Entity {i}: '" + ' '.join(value_text(v) for v in
                     case['records'][side].values()) + "'"
                     for i, side in enumerate(('left', 'right'), 1))


def payload(case, alias, arm, protocol):
    if arm not in ARMS:
        raise ValueError('Unknown Peeters control arm')
    if case['benchmark'] not in {*protocol.BENCHMARKS, 'billiger-de-50un'}:
        raise ValueError('Benchmark outside approved scope')
    body = copy.deepcopy(protocol.payload(case, alias, 'B0'))
    text = protocol.state(case, 'B0') if arm == 'peeters_fields' else values_state(case)
    if protocol.MODELS[alias]['kind'] == 'decision':
        body['state'] = text
        body['questions']['match']['instructions'] = QUESTION
    else:
        body['messages'][0]['content'] = QUESTION
        body['messages'][1]['content'] = text
    return body


def verify_source(root):
    directory = root/'artifacts/literature/peeters_source_20261014'
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['commit'] != SOURCE_COMMIT:
        raise ValueError('Unexpected source commit')
    for name, spec in manifest['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('Literature source drift: ' + name)
    task = json.loads((directory/'abt-buy-sampled-gs_general-complex-free.json').read_text())
    if task['task_prefix'] != QUESTION + '\n':
        raise ValueError('Literature question drift')
    return hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()


def freeze(root, output):
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(root/'src'))
    import protocol
    import orchestrate
    import billiger50
    source_hash = verify_source(root)
    orchestrate.verify()
    billiger50.verify()
    cells, examples = [], []
    types = {}
    for benchmark in [*protocol.BENCHMARKS, 'billiger-de-50un']:
        for split in (['test'] if benchmark == 'billiger-de-50un' else ['prompt', 'route', 'test']):
            cases = json.loads((root/'data'/benchmark/(split+'.json')).read_text())
            if len({c['pair_id'] for c in cases}) != len(cases):
                raise ValueError('Duplicate pair ID')
            for case in cases:
                if case['benchmark'] != benchmark or protocol.digest(case['records']) != case['input_hash']:
                    raise ValueError('Frozen case identity drift')
                for side in ('left', 'right'):
                    for value in case['records'][side].values():
                        key = type(value).__name__
                        types[key] = types.get(key, 0) + 1
            for alias in protocol.MODELS:
                for arm in ARMS:
                    h = hashlib.sha256()
                    for case in cases:
                        h.update(protocol.canonical(payload(case, alias, arm, protocol)).encode() + b'\n')
                    cells.append({'benchmark': benchmark, 'split': split, 'model': alias, 'arm': arm,
                                  'pairs': len(cases), 'input_data_sha256': protocol.digest(cases),
                                  'ordered_payloads_sha256': h.hexdigest()})
            # Label-independent first frozen development example; never embed targets.
            if split == 'prompt':
                examples.append({'benchmark': benchmark, 'input_hash': cases[0]['input_hash'],
                                 'payloads': {arm: {alias: payload(cases[0], alias, arm, protocol)
                                              for alias in ('jev', 'luna')} for arm in ARMS}})
    result = {'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'offline_payloads_frozen',
              'question': QUESTION, 'arms': list(ARMS), 'source_commit': SOURCE_COMMIT,
              'source_manifest_sha256': source_hash, 'cells': cells, 'examples': examples,
              'value_type_counts': types, 'calls_per_run_before_retries': sum(c['pairs'] for c in cells),
              'paid_calls': 0, 'execution_admitted': False, 'three_run_execution': False,
              'scope': 'Fixed controls, no quality selection. Inventory includes development and test; no admission or cost guarantee.',
              'adaptation': 'Current frozen attributes/order/values, no historical truncation or field subset. Strings raw, null empty, other JSON values compact. Entity markers and scalar quoting differ from field format; not an isolated field-name effect or historical replication.',
              'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256': hashlib.sha256((root/'src/protocol.py').read_bytes()).hexdigest()}
    with output.open('x') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cells', 'examples')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    freeze(args.root.resolve(), args.output)
