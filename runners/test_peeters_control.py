import copy
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import protocol
from peeters_control import ARMS, QUESTION, payload, value_text, values_state, verify_source


class PeetersTests(unittest.TestCase):
    def setUp(self):
        self.case = {'benchmark': 'abt-buy', 'pair_id': 'private-id', 'label': 1,
                     'records': {'left': {'name': "O'Brien\nTV", 'price': 12, 'empty': None},
                                 'right': {'name': 'TV', 'price': 12.5, 'empty': ''}}}

    def test_fields_change_instruction_only_all_seven_models(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('Network forbidden')):
            for alias in protocol.MODELS:
                expected = protocol.payload(self.case, alias)
                if alias == 'luna':
                    expected['messages'][0]['content'] = QUESTION
                else:
                    expected['questions']['match']['instructions'] = QUESTION
                self.assertEqual(expected, payload(self.case, alias, 'peeters_fields', protocol))

    def test_values_exact_order_null_and_quotes(self):
        self.assertEqual(values_state(self.case), "Entity 1: 'O'Brien\nTV 12 '\nEntity 2: 'TV 12.5 '")
        self.assertEqual(value_text(['x', None]), '["x",null]')
        self.assertEqual(value_text({'z': True}), '{"z":true}')
        self.assertEqual(value_text(False), 'false')
        with self.assertRaises(ValueError):
            value_text(float('nan'))

    def test_no_label_or_id_leak_and_no_mutation(self):
        before = copy.deepcopy(self.case)
        changed = copy.deepcopy(before)
        changed.update(label=0, pair_id='other-private-id', entity_ids={'left': 'secret'})
        for arm in ARMS:
            for alias in protocol.MODELS:
                self.assertEqual(payload(before, alias, arm, protocol), payload(changed, alias, arm, protocol))
        self.assertEqual(self.case, before)

    def test_values_preserve_output_schema_provider_criteria(self):
        for alias in protocol.MODELS:
            a = payload(self.case, alias, 'peeters_fields', protocol)
            b = payload(self.case, alias, 'peeters_values', protocol)
            if alias == 'luna':
                b['messages'][1]['content'] = a['messages'][1]['content']
            else:
                b['state'] = a['state']
            self.assertEqual(a, b)

    def test_no_truncation_or_field_drop(self):
        self.case['records']['left']['name'] = 'word '*80
        self.case['records']['left']['description'] = 'distinctive-long-description'
        text = values_state(self.case)
        self.assertEqual(text.count('word'), 80)
        self.assertIn('distinctive-long-description', text)

    def test_unknown_scope_rejected(self):
        with self.assertRaises(ValueError):
            payload(self.case, 'jev', 'unknown', protocol)
        self.case['benchmark'] = 'amazon-google'
        with self.assertRaises(ValueError):
            payload(self.case, 'jev', 'peeters_fields', protocol)

    def test_frozen_primary_source(self):
        self.assertEqual(len(verify_source(Path(__file__).resolve().parents[2])), 64)


if __name__ == '__main__':
    unittest.main()
