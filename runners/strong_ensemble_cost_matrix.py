"""Count actual frozen inputs offline; provider tokenization/output remain scenarios."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from strong_ensemble import MEMBERS, payload_from_frozen_b0


def build(root, output):
    sys.path.insert(0, str(root/'src'))
    import protocol
    import orchestrate
    import billiger50
    import tiktoken
    orchestrate.verify()
    billiger50.verify()
    enc = tiktoken.get_encoding('o200k_base')
    cells = []
    for benchmark in [*protocol.BENCHMARKS, 'billiger-de-50un']:
        splits = ['test'] if benchmark == 'billiger-de-50un' else ['prompt', 'route', 'test']
        for split in splits:
            cases = json.loads((root/'data'/benchmark/(split+'.json')).read_text())
            rows = {m: {'model': m, 'benchmark': benchmark, 'split': split, 'pairs': len(cases),
                        'proxy_input_tokens': 0, 'utf8_bytes_plus_2048_per_request': 0} for m in MEMBERS}
            hashes = {m: hashlib.sha256() for m in MEMBERS}
            for case in cases:
                base = protocol.payload(case, 'luna', 'B0')
                for m in MEMBERS:
                    body = payload_from_frozen_b0(base, m)
                    wire = protocol.canonical(body)
                    hashes[m].update(wire.encode()+b'\n')
                    # Includes entire payload, deliberately not a provider-exact tokenizer.
                    rows[m]['proxy_input_tokens'] += len(enc.encode(wire, disallowed_special=()))
                    rows[m]['utf8_bytes_plus_2048_per_request'] += len(wire.encode())+2048
            for m, row in rows.items():
                row['ordered_payloads_sha256'] = hashes[m].hexdigest()
                row['source_data_sha256'] = protocol.digest(cases)
                spec = MEMBERS[m]
                row['three_run_cost_scenarios_usd'] = {
                    str(out): 3*(row['proxy_input_tokens']*spec['input_usd_per_million']
                                +len(cases)*out*spec['output_usd_per_million'])/1e6
                    for out in [64, 256, 1024, 2048]}
                cells.append(row)
    result = {'created_at': datetime.now(timezone.utc).isoformat(), 'cells': cells,
        'member_calls_per_run': sum(c['pairs'] for c in cells),
        'proxy_tokenizer': 'o200k_base over complete wire JSON; not member tokenizer or billable server count',
        'scenario_total_three_runs_usd': {str(o): sum(c['three_run_cost_scenarios_usd'][str(o)] for c in cells) for o in [64,256,1024,2048]},
        'output_scenario_includes_reasoning': True,
        'excludes': ['cache-write premiums', 'retries', 'unknown costs', 'fresh live latency calls'],
        'hard_bound': False, 'paid_calls': 0, 'execution_admitted': False,
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with output.open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'cells'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build(a.root.resolve(), a.output)
