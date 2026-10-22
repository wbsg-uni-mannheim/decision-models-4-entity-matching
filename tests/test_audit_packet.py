import json
import re
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from build_audit import render


class AuditPacketTest(unittest.TestCase):
    def test_blind_and_safe_embedded_records(self):
        row = {'audit_id': 'id-1', 'benchmark': 'wdc',
               'record_left': json.dumps({'title': '</script><script>alert(1)</script>'}),
               'record_right': json.dumps({'title': 'Ä'}),
               'gold_label': 'HIDDEN_GOLD', 'model_predictions': 'HIDDEN_MODEL',
               'stratum': 'HIDDEN_STRATUM'}
        html = render([row], 'Aaron')
        self.assertNotIn('HIDDEN_', html)
        self.assertNotIn('</script><script>alert', html)
        data = re.search(r'id="cases">(.*?)</script>', html, re.S).group(1)
        parsed = json.loads(data)
        self.assertEqual(parsed[0]['record_left'], row['record_left'])
        self.assertEqual(set(parsed[0]), {'audit_id', 'benchmark', 'record_left', 'record_right'})
        self.assertNotIn('__CASES__', html)


if __name__ == '__main__':
    unittest.main()
