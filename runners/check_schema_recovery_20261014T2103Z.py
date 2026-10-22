"""Read-only schema cache provenance check and immutable terminal diagnosis."""
import sys
import json
import hashlib
import subprocess
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'artifacts/operations')]
import expanded_schema as es
import orchestrate
import billiger50
from resume_schema_once import terminal_steps, failure_hashes
from resume_confirmation import retryable
from parallel_admission import review


def main():
    root = ROOT
    acct = subprocess.check_output(['sacct', '-n', '-j', '365887', '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    terminal_steps(acct, 365887)
    es.verify()
    orchestrate.verify()
    billiger50.verify()
    config = json.loads((root/'configs/expanded_execution.json').read_text())
    hashes = dict(config['source_sha256'])
    for rel in ['src/protocol.py', 'src/run.py', 'artifacts/operations/resume_schema_once.py', 'artifacts/operations/parallel_admission.py']:
        hashes[rel] = hashlib.sha256((root/rel).read_bytes()).hexdigest()
    for rel, expected in hashes.items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest() == expected, rel
    policy = review(root, root/'configs/parallel_execution_20261014.json')
    failhash = failure_hashes(root)
    assert set(failhash) == {'runs/schema/jev/C_match_reverse/dblp-acm/prompt.failures.jsonl', 'runs/schema/jev/C_review/walmart-amazon/prompt.failures.jsonl'}
    failures = [json.loads(x) for x in (root/'runs/schema/jev/C_match_reverse/dblp-acm/prompt.failures.jsonl').read_text().splitlines()]
    assert len(failures) == 1 and retryable(failures)
    assert failures[0]['pair_id'] == 'dblp_1982#acm_1389'
    invalid = [json.loads(x) for x in (root/'runs/schema/jev/C_review/walmart-amazon/prompt.failures.jsonl').read_text().splitlines()]
    assert len(invalid) == 1
    r = invalid[0]
    assert r['pair_id'] == 'walmart_1702#amazon_16518'
    assert r['error'] == 'ValueError: Choice probabilities do not sum to one'
    assert len(r['attempts']) == 1 and r['attempts'][0]['status'] == 200
    assert r['attempts'][0]['response']['answers']['match']['probabilities'] == {'Match': .07, 'No Match': .81, 'Review': .11}
    assert r['attempts'][0]['cost'] == .000022344
    try:
        es.parse('jev', r['attempts'][0]['response'], 'C_review')
    except ValueError as exc:
        assert str(exc) == 'Choice probabilities do not sum to one'
    else:
        raise AssertionError('Invalid response unexpectedly accepted')
    counts = {}
    cells = 0
    for ns in ['schema', 'schema_pilot']:
        count = 0
        for p in (root/'runs'/ns).rglob('*.jsonl'):
            if p.name.endswith('.executions.jsonl'):
                continue
            _, _, m, v, b, name = p.relative_to(root).parts
            split = name.split('.')[0]
            cases = {c['pair_id']: c for c in json.loads((root/'data'/b/(split+'.json')).read_text())}
            rows = [json.loads(x) for x in p.read_text().splitlines()]
            seen = set()
            for r in rows:
                pid = r['pair_id']
                assert pid not in seen
                seen.add(pid)
                c = cases[pid]
                assert r['request_hash'] == es.wire_hash(es.payload(c, m, v))
                assert r['input_hash'] == c['input_hash']
                if '.failures.' not in name:
                    assert es.parse(m, r['attempts'][-1]['response'], v) == (r['prediction'], r['probability_match'])
                count += 1
            cells += 1
        counts[ns] = count
    spec = {'utc': datetime.now(timezone.utc).isoformat(), 'predecessor': 365887, 'job_id': None,
            'predecessor_accounting': acct, 'source_sha256': hashes,
            'diagnosed_failure_sha256': failhash, 'full_failure_records': failures+invalid, 'diagnosis': 'One preserved earlier HTTP499 and one new invalid probability sum0.99. Keep parser tolerance .001; no normalization or imputation. One new unchanged invocation only after complete provenance check; any further failure requires a fresh diagnosis.', 'checked_records': counts, 'checked_files': cells,
            'policy_review': policy, 'invocations_limit': 1,
            'command': '.venv/bin/python -u artifacts/operations/resume_schema_once.py',
            'budget_before': json.loads((root/'artifacts/budget.json').read_text()),
            'cache_policy': 'One unchanged invocation; preserved journals; same requests, models, selection and workers. Independent active writers allowed; shared USD100 cap.'}
    with (root/'artifacts/operations/schema_terminal_diagnosis_20261014T2103Z.json').open('x') as out:
        out.write(json.dumps(spec, indent=2)+'\n')
    with (root/'configs/schema_resume_prepared_20261014T2103Z.json').open('x') as out:
        out.write(json.dumps(spec, indent=2)+'\n')
    print(json.dumps({'counts': counts, 'files': cells, 'accounting': acct, 'budget': spec['budget_before']}))


if __name__ == '__main__':
    main()
