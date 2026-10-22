"""Freeze a small development-only connectivity/usage pilot; no network calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from strong_ensemble import MEMBERS, payload_from_frozen_b0


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def select(cases):
    if len(cases) < 2 or len({c['pair_id'] for c in cases}) != len(cases):
        raise ValueError('Need at least two unique development cases')
    ordered = sorted(cases, key=lambda c: digest(['ensemble-pilot-v1', c['pair_id']]))
    first = ordered[0]
    second = max((c for c in ordered if c['pair_id'] != first['pair_id']),
                 key=lambda c: len(canonical(c['records']).encode()))
    return [first, second]


def prepare(root, output):
    sys.path.insert(0, str(root/'src'))
    import protocol
    import orchestrate
    orchestrate.verify()
    requests, provenance = [], []
    for b in protocol.BENCHMARKS:
        source = root/'data'/b/'prompt.json'
        cases = json.loads(source.read_text())
        for index, c in enumerate(select(cases)):
            provenance.append({'benchmark': b, 'pair_id': c['pair_id'],
                'sample_role': 'hash_first' if index == 0 else 'longest_remaining',
                'source_data_sha256': digest(cases), 'records_sha256': digest(c['records'])})
            for m in MEMBERS:
                body = payload_from_frozen_b0(protocol.payload(c, 'luna', 'B0'), m)
                # Generous planning allowance, not a provider-confirmed tokenizer bound.
                byte_allowance = len(canonical(body).encode())+2048
                spec = MEMBERS[m]
                allowance = (byte_allowance*spec['input_usd_per_million']*2
                             +2048*spec['output_usd_per_million']*1.25)/1e6
                requests.append({'request_id': digest(['ensemble-pilot-v1',b,c['pair_id'],m]),
                    'member': m, 'benchmark': b, 'pair_id': c['pair_id'], 'body': body,
                    'request_sha256': digest(body), 'planning_allowance_usd': allowance})
    wire = ''.join(canonical(row)+'\n' for row in requests)
    total = sum(r['planning_allowance_usd'] for r in requests)
    if total > 10:
        raise ValueError('Conservative pilot plan exceeds USD10; do not admit automatically')
    manifest = {'purpose': 'Development-only technical pilot: identity, schema, actual usage and truncation; no matching-performance selection',
        'pairs': len(provenance), 'requests': len(requests), 'max_attempts_per_request': 1,
        'pilot_ceiling_usd': 10, 'project_ceiling_usd': 100,
        'summed_planning_allowance_usd': total, 'allowance_is_provider_hard_bound': False,
        'selection': 'Per core family: hash-first D_prompt pair plus longest remaining by serialized UTF8 records; no labels or predictions',
        'requests_sha256': hashlib.sha256(wire.encode()).hexdigest(), 'members': MEMBERS,
        'provenance': provenance, 'execution_admitted': False,
        'repetition_status': 'Technical pilot outside run01/02/03 result caches; never count as a complete study run',
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ensemble_implementation_sha256': hashlib.sha256((Path(__file__).parent/'strong_ensemble.py').read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    (output/'requests.jsonl').write_text(wire)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['provenance','members']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    prepare(a.root.resolve(), a.output)
