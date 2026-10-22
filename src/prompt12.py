"""Twelve-prompt sensitivity set on the full core test sets (one run).

Mirrors the 10 zero-shot plus 2 rule prompts of Peeters et al. (EDBT 2025).
Only the instruction text changes. Serialization, answer schema, decision
criteria, models and provider pins stay identical to B0. B0 itself is the
existing baseline table and is not re-run.

Peeters and Narayan wordings come from the MatchGPT repository
(wbsg-uni-mannheim/MatchGPT, LLMForEM/tasks/*). The output-format suffix
("Answer with 'Yes' if they do and 'No' if they do not.") is dropped because
the answer schema is fixed by the API. Narayan's "Product A"/"Publication A"
labels become "Record A"/"Record B", the labels of our serialization.
"""
import json
from pathlib import Path

import expanded_protocol
import expanded_rules
import protocol

PROJECT = Path(__file__).resolve().parents[1]
PUBLICATION = ('dblp-acm', 'dblp-scholar', 'semi-heter')

HANDWRITTEN = {
    'product': ("Your task is to decide if two product descriptions match.\n\n"
                "The following rules regarding product features need to be observed:\n\n"
                "1. The brand of matching products must be the same if available\n"
                "2. Model names of matching products must be the same if available\n"
                "3. Model numbers of matching products must be the same if available\n"
                "4. Additional features of matching products must be the same if available\n"
                "5. Matching attributes may not have the exact same surface form due to different case, typos, value formats, etc.\n"
                "6. If an attribute is missing for one description, it is likely still a match if the existing attributes match.\n\n"
                "Do the two product descriptions match?"),
    'publication': ("Your task is to decide if two publications match.\n\n"
                    "The following rules regarding publication features need to be observed:\n\n"
                    "1. The authors must be the same.\n"
                    "2. The title must be the same.\n"
                    "3. The venue should be the same but could differ.\n"
                    "4. The year should be the same but could differ.\n"
                    "5. Matching attributes may not have the exact same surface form due to different case, typos, value formats, etc.\n"
                    "6. If an attribute is missing for one publication, it is likely still a match if the existing attributes match.\n\n"
                    "Do the two publications match?"),
}

# (variant id, display name, group). B0 is the reference and is not re-run.
PROMPTS = [
    ('B0', 'B0 (reference)', 'Own'),
    ('PE_general_simple', 'general-simple', 'Peeters et al.'),
    ('PE_general_complex', 'general-complex', 'Peeters et al.'),
    ('PE_domain_simple', 'domain-simple', 'Peeters et al.'),
    ('PE_domain_complex', 'domain-complex', 'Peeters et al.'),
    ('NA_simple', 'Narayan simple', 'Narayan et al.'),
    ('NA_complex', 'Narayan complex', 'Narayan et al.'),
    ('X_P_domain_0', 'Domain 1', 'Own'),
    ('X_P_conservative_0', 'Conservative 1', 'Own'),
    ('X_P_evidence_0', 'Evidence 1', 'Own'),
    ('R_handwritten', 'Handwritten rules', 'Rules'),
    ('R_rules', 'Astra rules', 'Rules'),
]
VARIANTS = [v for v, _, _ in PROMPTS]
NEW_VARIANTS = [v for v in VARIANTS if v != 'B0']

# perplexity/pplx-decider-v1.1-27b (baseline table) lost its OpenRouter endpoint on
# 19 Oct 2026. Its successor v1.2 joins this study with its own B0 reference
# (Aaron, 19 Oct). The baseline table keeps v1.1.
EXTRA_MODELS = {'pplx_v12': {'id': 'perplexity/pplx-decider-v1.2', 'provider': 'perplexity', 'kind': 'decision'}}
RUN_MODELS = ['jev', 'clef', 'clef_flash', 'luna_decisions', 'pplx_v12', 'kev', 'luna']


def register_models():
    """Adds the v1.2 alias to the in-process model table; existing entries stay unchanged."""
    for alias, spec in EXTRA_MODELS.items():
        assert protocol.MODELS.get(alias, spec) == spec
        protocol.MODELS[alias] = spec


def run_variants(alias):
    """Variants a model runs in this study; models without a baseline-table B0 also run B0."""
    return (['B0'] if alias in EXTRA_MODELS else []) + NEW_VARIANTS


def domain(benchmark):
    return 'publication' if benchmark in PUBLICATION else 'product'


def load_rules():
    obj = json.loads((PROJECT/'configs/learned_rules.json').read_text())
    expanded_rules.RULES.update(expanded_rules.validate_rules(obj))


def instruction(benchmark, variant):
    noun, entity = (('publications', 'publication') if domain(benchmark) == 'publication'
                    else ('product descriptions', 'product'))
    texts = {
        'PE_general_simple': 'Do the two entity descriptions match?',
        'PE_general_complex': 'Do the two entity descriptions refer to the same real-world entity?',
        'PE_domain_simple': f'Do the two {noun} match?',
        'PE_domain_complex': f'Do the two {noun} refer to the same real-world {entity}?',
        'NA_simple': 'Are Record A and Record B the same?',
        'NA_complex': 'Are Record A and Record B equivalent?',
        'R_handwritten': HANDWRITTEN[domain(benchmark)],
    }
    return texts[variant]


def payload(case, alias, variant):
    if variant == 'B0' or variant.startswith('X_P_'):
        return expanded_protocol.payload(case, alias, variant)
    if variant == 'R_rules':
        if not expanded_rules.RULES:
            load_rules()
        return expanded_rules.payload(case, alias, 'R_rules')
    assert variant in VARIANTS, variant
    body = protocol.payload(case, alias, 'B0')
    text = instruction(case['benchmark'], variant)
    if 'questions' in body:
        body['questions']['match']['instructions'] = text
    else:
        body['messages'][0]['content'] = text
    return body


def prompt_text(benchmark, variant):
    body = payload({'benchmark': benchmark, 'records': {'left': {}, 'right': {}}}, 'jev', variant)
    return body['questions']['match']['instructions']


def table():
    """Markdown table of all twelve exact instruction texts."""
    rows = ['| # | Group | Prompt | Product benchmarks | Publication benchmarks |', '|---|---|---|---|---|']
    for i, (variant, name, group) in enumerate(PROMPTS, 1):
        cells = []
        for b in ('abt-buy', 'dblp-scholar'):
            text = prompt_text(b, variant)
            if variant == 'R_rules':
                text = protocol.BASE + ' + "General rules derived from a separate labeled training sample:" + 6 to 8 Astra rules per benchmark (configs/learned_rules.json)'
            cells.append(text.replace('\n', '<br>').replace('|', '\\|'))
        if cells[0] == cells[1]:
            cells[1] = 'same'
        rows.append(f'| {i} | {group} | {name} (`{variant}`) | {cells[0]} | {cells[1]} |')
    return '\n'.join(rows)


if __name__ == '__main__':
    print(table())
