"""Validate downloaded discovery outputs without accepting them as human judgments."""
import json


def validate_row(response, case, evidence_scope='attribute'):
    if evidence_scope not in {'attribute', 'record_with_field_provenance'}:
        raise ValueError('Unknown evidence validation policy')
    if response.get('error'):
        return {'available': False, 'reason': 'batch_error', 'details': response['error']}
    http = response.get('response') or {}
    body = http.get('body') or {}
    if http.get('status_code') != 200 or body.get('status') != 'completed':
        return {'available': False, 'reason': 'http_or_incomplete_response'}
    texts = [part['text'] for item in body.get('output', []) if item.get('type') == 'message'
             for part in item.get('content', []) if part.get('type') == 'output_text']
    if len(texts) != 1:
        return {'available': False, 'reason': 'refusal_or_missing_output'}
    try:
        result = json.loads(texts[0])
    except (ValueError, TypeError):
        return {'available': False, 'reason': 'invalid_json'}
    if not isinstance(result, dict) or set(result) != {'observations', 'uncertainty'}:
        return {'available': False, 'reason': 'invalid_schema'}
    if not isinstance(result['observations'], list) or not isinstance(result['uncertainty'], str):
        return {'available': False, 'reason': 'invalid_schema'}
    field_matches = []
    for obs in result['observations']:
        keys = {'attribute', 'left_evidence', 'right_evidence', 'description'}
        if not isinstance(obs, dict) or set(obs) != keys or not all(isinstance(v, str) for v in obs.values()):
            return {'available': False, 'reason': 'invalid_observation_schema'}
        matches = {}
        for side in ['left', 'right']:
            field = case['records'][side].get(obs['attribute'])
            quote = obs[side+'_evidence']
            # Only check literal support, not the correctness of its interpretation.
            text = field if isinstance(field, str) else json.dumps(field, ensure_ascii=False)
            if evidence_scope == 'attribute':
                if quote and (field is None or quote not in text):
                    return {'available': False, 'reason': 'unsupported_literal_evidence'}
            else:
                # The frozen request asks for evidence from records, but does not
                # require attribute to be a literal JSON key. Preserve abstract
                # attribute labels and record every matching source field. If the
                # label IS a non-null field key, require that exact field. A null
                # brand field can coexist with a brand named in the title; the
                # source-field map records where the quote actually appears.
                fields = case['records'][side]
                candidates = [obs['attribute']] if fields.get(obs['attribute']) is not None else list(fields)
                matches[side] = [key for key in candidates if quote and fields[key] is not None
                                and quote in (fields[key] if isinstance(fields[key], str)
                                              else json.dumps(fields[key], ensure_ascii=False))]
                if quote and not matches[side]:
                    return {'available': False, 'reason': 'unsupported_literal_evidence'}
        field_matches.append(matches)
    usage = body.get('usage', {})
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ['input_tokens', 'output_tokens']):
        return {'available': False, 'reason': 'missing_usage', 'cost_unknown': True}
    validated = {'available': True, 'observations': result, 'usage': usage,
                 'human_semantic_validation': 'pending', 'internal_cause_validated': False}
    if evidence_scope != 'attribute':
        validated['evidence_validation'] = {'policy': evidence_scope, 'source_field_matches': field_matches,
                                          'empty_quotes_prove_missingness': False}
    return validated


def import_rows(rows, cases, evidence_scope='attribute'):
    expected = {'discover-'+c['case_id']: c for c in cases}
    if len(expected) != len(cases):
        raise ValueError('Duplicate source case')
    seen, result = set(), {}
    for row in rows:
        key = row.get('custom_id')
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate custom_id')
        seen.add(key)
        result[key] = validate_row(row, expected[key], evidence_scope=evidence_scope)
    for key in expected.keys()-seen:
        result[key] = {'available': False, 'reason': 'missing_batch_result'}
    return result
