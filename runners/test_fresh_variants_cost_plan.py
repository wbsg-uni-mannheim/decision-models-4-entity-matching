import json
from pathlib import Path
import tempfile
import unittest

import protocol
import expanded_variants
from fresh_variants_cost_plan import matrix, untouched
from fixed_extension_cost_plan import source
from test_fixed_extension_cost_plan import FixedExtensionCosts


class FreshVariantCosts(unittest.TestCase):
    def test_frozen_matrix_separates_languages_and_wdc(self):
        cells = matrix(protocol.MODELS, expanded_variants.VARIANTS)
        self.assertEqual(len(cells), 35)
        self.assertEqual({c['benchmark'] for c in cells}, set(expanded_variants.VARIANTS))
        self.assertTrue(all(c['variant'] == 'B0' and c['split'] == 'test' for c in cells))
        with self.assertRaises(ValueError):
            matrix(['jev', 'jev'], ['wdc-000un'])
        with self.assertRaises(ValueError):
            matrix([], ['wdc-000un'])

    def test_active_core_allowed_but_any_variant_attempt_or_output_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            cells = matrix(['jev'], ['wdc-000un'])
            history = target/'artifacts/stage_execution.jsonl'
            history.parent.mkdir()
            history.write_text(json.dumps({'stage': 'core', 'status': 'started'})+'\n')
            self.assertEqual(len(untouched(target, cells)), 4)
            before = history.read_bytes()
            journal = target/'runs/quality/jev/B0/wdc-000un/other.operations.jsonl'
            journal.parent.mkdir(parents=True)
            journal.touch()
            with self.assertRaises(ValueError):
                untouched(target, cells)
            journal.unlink()
            self.assertEqual(history.read_bytes(), before)
            history.write_text(json.dumps({'stage': 'variants', 'status': 'failed'})+'\n')
            with self.assertRaises(ValueError):
                untouched(target, cells)
            history.write_bytes(before)
            marker = target/'artifacts/expanded_variants_complete.json'
            marker.write_text('{}')
            with self.assertRaises(ValueError):
                untouched(target, cells)

    def test_b0_raw_source_costs_and_foreign_run_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c, row, path = FixedExtensionCosts().fixture(root, variant='B0')
            c['stage'] = 'variants'
            checked = source(root, c, protocol, protocol)
            self.assertEqual(checked['known_proxy_usd'], .02)
            self.assertEqual(checked['historical_unknown_scenario_usd'], 0)
            row['study_run_id'] = 3
            path.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):
                source(root, c, protocol, protocol)


if __name__ == '__main__':
    unittest.main()
