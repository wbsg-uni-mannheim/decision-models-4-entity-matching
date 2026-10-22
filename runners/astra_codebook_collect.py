"""GET-only codebook retrieval and technical import; no automatic semantic acceptance."""
import argparse
import json
from pathlib import Path

from astra_codebook_submit import STAGE
from astra_discovery_collect import TERMINAL, get, identifier
from astra_discovery_submit import credential, sha
from astra_followup_package import validate_codebook


def import_codebook(row, discovery_ids):
    if row.get('custom_id') != 'synthesize-codebook-v1' or row.get('error'):
        raise ValueError('Unexpected or failed codebook response')
    http = row.get('response') or {}
    body = http.get('body') or {}
    if http.get('status_code') != 200 or body.get('status') != 'completed':
        raise ValueError('Codebook response incomplete')
    model = body.get('model')
    if not (model == 'gpt-6-astra' or isinstance(model, str) and model.startswith('gpt-6-astra-')):
        raise ValueError('Unexpected codebook model')
    texts = [part['text'] for item in body.get('output', []) if item.get('type') == 'message'
             for part in item.get('content', []) if part.get('type') == 'output_text']
    if len(texts) != 1:
        raise ValueError('Missing or ambiguous codebook output')
    book = json.loads(texts[0])
    codes = validate_codebook(book, discovery_ids)
    usage = body.get('usage') or {}
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ['input_tokens', 'output_tokens']):
        raise ValueError('Codebook usage missing; retain full commitment')
    return {'codebook': book, 'category_codes': codes, 'usage': usage,
            'schema_and_discovery_references_valid': True, 'human_semantic_validation': 'pending',
            'accepted_for_test_categorization': False, 'internal_causes_validated': False}


def collect(root, package, output, fetch=get):
    manifest = json.loads((package / 'manifest.json').read_text())
    digest = sha(package / 'requests.jsonl')
    if digest != manifest['requests_sha256']:
        raise ValueError('Frozen codebook request changed')
    requests = [json.loads(s) for s in (package / 'requests.jsonl').read_text().splitlines() if s.strip()]
    if len(requests) != 1 or requests[0]['custom_id'] != 'synthesize-codebook-v1':
        raise ValueError('Expected exactly one codebook request')
    discovery = json.loads(requests[0]['body']['input'])['discovery_cases']
    receipt = json.loads((package / 'batch_receipt.json').read_text())
    key = credential(root)
    raw = fetch(key, 'batches/'+identifier(receipt['id'], 'batch_'))
    batch = json.loads(raw)
    if batch['id'] != receipt['id'] or batch['input_file_id'] != receipt['input_file_id']:
        raise ValueError('Wrong codebook batch identity')
    expected_meta = {'study': 'decision-em-20261013', 'stage': STAGE, 'package_sha256': digest}
    if batch.get('endpoint') != '/v1/responses' or batch.get('metadata') != expected_meta:
        raise ValueError('Wrong codebook batch package or stage')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'batch_snapshot.json').write_bytes(raw)
    report = {'batch_id': batch['id'], 'status': batch['status'], 'package_sha256': digest,
              'commitment_action': 'retain; separate usage reconciliation required',
              'human_semantic_validation': 'pending', 'accepted_for_test_categorization': False}
    if batch['status'] in TERMINAL:
        rows, hashes = [], {}
        for name in ['output_file_id', 'error_file_id']:
            if batch.get(name):
                raw = fetch(key, 'files/'+identifier(batch[name], 'file-')+'/content')
                path = output / (name+'.jsonl')
                path.write_bytes(raw)
                hashes[name] = sha(path)
                rows.extend(json.loads(s) for s in raw.decode().splitlines() if s.strip())
        report['files_sha256'] = hashes
        (output / 'retrieval_report.json').write_text(json.dumps(report, indent=2)+'\n')
        counts = batch.get('request_counts', {})
        if batch['status'] != 'completed' or any(counts.get(k) != v for k, v in {'total': 1, 'completed': 1, 'failed': 0}.items()) or len(rows) != 1:
            raise ValueError('Codebook Batch incomplete or unexpected request count; raw files retained')
        imported = import_codebook(rows[0], [c['case_id'] for c in discovery])
        (output / 'codebook_candidate.json').write_text(json.dumps(imported, indent=2, ensure_ascii=False)+'\n')
        report.update(technical_import_valid=True, candidate_sha256=sha(output / 'codebook_candidate.json'))
    (output / 'retrieval_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root', 'package', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.package, args.output), indent=2))
