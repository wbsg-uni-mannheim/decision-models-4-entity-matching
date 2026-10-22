"""Offline bridge from preserved Batch responses to codebook preparation.

Recompute imports from raw files; never trust availability flags alone. This
checks local provenance consistency, not provider signatures or human semantics.
No network calls, budget changes, or approval to submit follow-up requests.
"""
import argparse
import hashlib
import json
from pathlib import Path

from astra_discovery_collect import usage_inventory, verify_batch
from astra_discovery_submit import sha
from astra_followup_package import freeze_packet, synthesis_request
from astra_observation_import import import_rows


def read(path):
    return json.loads(path.read_text())


def verified_discovery(package, retrieval, record_evidence=False):
    manifest = read(package / 'manifest.json')
    for name in ['discovery.jsonl', 'cases_private.json']:
        if sha(package / name) != manifest['files_sha256'][name]:
            raise ValueError('Frozen discovery package changed: ' + name)
    receipt = read(package / 'batch_receipt.json')
    batch = read(retrieval / 'batch_snapshot.json')
    report = read(retrieval / 'retrieval_report.json')
    package_hash = sha(package / 'discovery.jsonl')
    verify_batch(batch, receipt, package_hash)
    if batch['status'] != 'completed':
        raise ValueError('Discovery batch must be completed before synthesis')
    cases = read(package / 'cases_private.json')['discovery']
    expected = {'discover-' + c['case_id'] for c in cases}
    if not cases or len(expected) != len(cases) or any(c['source_split'] != 'prompt' for c in cases):
        raise ValueError('Nonempty unique prompt-only discovery cases required')
    requests = [json.loads(line) for line in (package / 'discovery.jsonl').read_text().splitlines() if line.strip()]
    if len(requests) != len(cases) or {r['custom_id'] for r in requests} != expected:
        raise ValueError('Submitted discovery request IDs differ from cases')
    if any(r.get('method') != 'POST' or r.get('url') != '/v1/responses'
           or r.get('body', {}).get('model') != 'gpt-6-astra' for r in requests):
        raise ValueError('Unexpected discovery request model or endpoint')
    counts = batch.get('request_counts', {})
    if any(type(counts.get(k)) is not int or counts[k] != v
           for k, v in {'total': len(cases), 'completed': len(cases), 'failed': 0}.items()):
        raise ValueError('Provider must confirm every discovery request completed')
    for key, value in {'batch_id': batch['id'], 'package_sha256': package_hash,
                       'status': 'completed', 'provider_request_counts': counts}.items():
        if report.get(key) != value:
            raise ValueError('Retrieval report disagrees with preserved batch: ' + key)
    rows, raw_hashes = [], {}
    for name in ['output_file_id', 'error_file_id']:
        path = retrieval / (name + '.jsonl')
        if batch.get(name):
            raw_hashes[name] = sha(path)
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        elif path.exists():
            raise ValueError('Unexpected raw file without provider file ID')
    if not batch.get('output_file_id') or raw_hashes != report.get('files_sha256'):
        raise ValueError('Raw output provenance differs from retrieval report')
    normalized = import_rows(rows, cases)
    strict_available = sum(v.get('available') is True for v in normalized.values())
    # Check the original collector import before applying any separately declared
    # evidence policy. Never overwrite the original retrieval or model output.
    if normalized != read(retrieval / 'observations_imported.json'):
        raise ValueError('Saved observations differ from independently recomputed raw import')
    if record_evidence:
        normalized = import_rows(rows, cases, evidence_scope='record_with_field_provenance')
    if any(result.get('available') is not True for result in normalized.values()):
        raise ValueError('Every discovery observation must pass raw-response validation')
    for row in rows:
        model = row['response']['body'].get('model')
        if not (model == 'gpt-6-astra' or isinstance(model, str) and model.startswith('gpt-6-astra-')):
            raise ValueError('Unexpected discovery response model')
    usage = usage_inventory(rows, expected)
    if not usage['all_request_usage_present'] or usage != report.get('usage'):
        raise ValueError('Incomplete or inconsistent discovery usage inventory')
    if report.get('available') != strict_available or report.get('unavailable') != len(cases) - strict_available:
        raise ValueError('Retrieval availability summary differs from raw responses')
    files = {f'package/{name}': sha(package / name) for name in
             ['manifest.json', 'discovery.jsonl', 'cases_private.json', 'batch_receipt.json']}
    files.update({f'retrieval/{name}': sha(retrieval / name) for name in
                  ['batch_snapshot.json', 'retrieval_report.json', 'observations_imported.json']})
    files.update({f'retrieval/{name}.jsonl': digest for name, digest in raw_hashes.items()})
    provenance = {'batch_id': batch['id'], 'input_file_id': batch['input_file_id'],
                  'verified_files_sha256': files, 'discovery_cases': len(cases),
                  'raw_import_recomputed': True, 'human_semantic_validation': 'pending',
                  'strict_attribute_key_available': strict_available,
                  'evidence_policy': 'record_with_field_provenance' if record_evidence else 'attribute',
                  'check_scope': 'local preserved-file consistency; no provider signature verification'}
    return cases, normalized, provenance


def prepare_synthesis(package, retrieval, output, record_evidence=False):
    cases, observations, provenance = verified_discovery(package, retrieval, record_evidence)
    wire = json.dumps(observations, indent=2, ensure_ascii=False)+'\n'
    provenance['verified_observations_sha256'] = hashlib.sha256(wire.encode()).hexdigest()
    provenance['validator_files_sha256'] = {name: sha(Path(__file__).parent / name) for name in
                                          ['astra_followup_provenance.py', 'astra_observation_import.py', 'astra_followup_package.py']}
    result = freeze_packet(output, [synthesis_request(cases, observations)], provenance)
    (output / 'discovery_observations_verified.json').write_text(wire)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--record-evidence', action='store_true', help='Explicit record evidence policy with field provenance')
    args = parser.parse_args()
    print(json.dumps(prepare_synthesis(args.package, args.retrieval, args.output, args.record_evidence), indent=2))
