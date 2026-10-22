"""Group the 26-category Astra codebook candidate into eight categories (Aaron, 20 Oct 2026).

Human edit before freezing, made without looking at the 498 held-out cases or any results.
Every original category becomes an aspect of exactly one group and keeps its definition,
inclusion and exclusion text, so the categorizer still sees the fine-grained guidance.
Assignments are made at group level only.

    python3 artifacts/operations/astra_codebook_group8.py <v1 candidate json> <output json>
"""
import hashlib
import json
import sys
from pathlib import Path

# Revision 2 (Aaron, 20 Oct): variant and specification merged (color and size are attribute values),
# product type, role and offer scope as its own group with a substitution test, price and shipping as
# a separate context group, measurement comparability moved to information quality.
GROUPS = [
    ('G1', 'Identifier discrepancy', ['C01', 'C22'],
     'Both records supply identifying strings (model, part or product code, ISBN) that differ in letters or digits.',
     'Use G1 for the identifier strings themselves. A differing attribute value stated outside an identifier belongs to G2.'),
    ('G2', 'Attribute value discrepancy within the same product type', ['C02', 'C03', 'C04', 'C05', 'C21', 'C23'],
     'Both records describe the same type of item, but state different values for one of its attributes: product line or '
     'edition designation, color or depicted motif, size, capacity, material, other technical values, publication year, '
     'version, or page count.',
     'Substitution test: the two items would serve the same purpose and fit the same use, they differ only in a stated '
     'attribute value. Example: 16 GB versus 32 GB of the same card type, red versus blue, foam versus paper cups.'),
    ('G3', 'Product type, role, or offer scope discrepancy', ['C06', 'C07'],
     'The records describe different kinds of item or a different offer scope: device versus accessory or spare part, '
     'a component with a different role, a different form factor or device class, or a single item versus a pack, kit, or bundle.',
     'Substitution test fails: one item cannot replace the other as the same kind of thing (cartridge versus dock, laptop '
     'versus desktop memory, microSD versus SD card), or the offer includes a different quantity or set of items. Same item '
     'type with a different material belongs to G2.'),
    ('G4', 'Brand or catalog category discrepancy', ['C08', 'C09'],
     'Brand strings or populated catalog category fields differ between the records, including a second brand mentioned '
     'as a compatibility reference.',
     'Use G4 only when both records state a brand or category. A brand stated on one side only belongs to G6.'),
    ('G5', 'Publication title, author, or venue discrepancy', ['C15', 'C16', 'C17', 'C18', 'C19', 'C20'],
     'Publication titles, author lists, author-name forms, author order, or venue and publisher names differ between the records.',
     'Publication year, version, and page count belong to G2, ISBN to G1.'),
    ('G6', 'Missing, misplaced, incomparable, or noisy information', ['C10', 'C11', 'C12', 'C14', 'C24', 'C25'],
     'Information is present on one side only, written differently, placed in another field, stated in a form that '
     'cannot be compared (missing units, different measurement types), truncated or mixed with boilerplate, or inconsistent '
     'within one record.',
     'Use G6 when the evidence is absent, unequal, or not comparable as written. Two comparable but different values belong to G2.'),
    ('G7', 'Reference-label ambiguity', ['C26'],
     'The supplied reference label has an unresolved relationship to the visible identity evidence or apparent entity granularity.',
     'Retain the label. Flag only, never declare a labeling error.'),
    ('G8', 'Commercial context: price, currency, or shipping', ['C13'],
     'Prices, currencies, or shipping metadata differ between the records.',
     'Context only. A price or shipping difference does not establish different entities, and the same entity can have '
     'different prices. Record it as a signal that was available, never as the explanation of the error.'),
]

FRAMING = ('Groups describe what visibly distinguishes or fails to align the two records of an error pair. For a false '
           'positive (reference non-match predicted as match), they describe differences that were present but not '
           'decisive for the prediction. For a false negative (reference match predicted as non-match), they describe '
           'differences between records of the same entity that were present when the prediction went wrong. They do '
           'not establish why the model decided as it did.')

OVERLAP = ('Apply all independently supported groups. These are descriptive annotations, not mutually exclusive causes or '
           'matching rules. Retain prediction, reference label, and FP/FN direction separately. A group may occur in either '
           'direction. Attach each assignment to exact left/right fields and values, distinguishing explicit contradiction, '
           'surface variation, and one-sided information. Several aspects of the same group count as one assignment of that '
           'group. ISBN differences belong to G1, page counts and publication years to G2, pack quantities to G3, author order '
           'to G5. Different groups remain appropriate for distinct observations, such as a color difference (G2) plus a size difference (G2, '
           'one assignment) plus a form-factor difference (G3). G6, G7, and G8 are contextual qualifiers and must not '
           'be presented as established explanations for the error. Supporting IDs are illustrative discovery evidence, not '
           'exhaustive assignments.')


def group(v1):
    cats = {c['code']: c for c in v1['codebook']['categories']}
    used = [code for _, _, codes, _, _ in GROUPS for code in codes]
    assert sorted(used) == sorted(cats) and len(used) == len(set(used)), 'every v1 category exactly once'
    uncertainty = v1['codebook']['uncertainty_policy'].replace('apply C26 where warranted', 'apply G7 where warranted') \
        .replace('additional C-codes', 'additional group codes')
    assert 'C26' not in uncertainty
    groups = []
    for code, name, members, definition, rule in GROUPS:
        aspects = [{'from': m, 'name': cats[m]['name'], 'definition': cats[m]['definition'],
                    'include_when': cats[m]['include_when'], 'exclude_when': cats[m]['exclude_when']} for m in members]
        support = sorted({i for m in members for i in cats[m]['supporting_discovery_case_ids']})
        groups.append({'code': code, 'name': name, 'definition': definition, 'decision_rule': rule, 'merged_codes': members,
                       'aspects': aspects, 'supporting_discovery_case_ids': support})
    return {'codebook': {'categories': groups, 'framing': FRAMING, 'overlap_policy': OVERLAP, 'uncertainty_policy': uncertainty,
                         'precedence': 'Decision rules of the groups take precedence over the inclusion text of a merged aspect.'},
            'category_codes': [g['code'] for g in groups],
            'derived_from': {'candidate_sha256': hashlib.sha256(json.dumps(v1, sort_keys=True).encode()).hexdigest(),
                             'edit': 'grouped 26 categories into 8, revision 2 (Aaron, 20 Oct 2026): decision rules added, definitions of '
                                     'the original categories kept as aspects; no held-out case or result inspected'},
            'human_semantic_validation': 'pending', 'accepted_for_test_categorization': False,
            'internal_causes_validated': False}


if __name__ == '__main__':
    v1 = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2]); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(group(v1), f, indent=2, ensure_ascii=False); f.write('\n')
    print('wrote', out)
