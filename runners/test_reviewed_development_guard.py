import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import authorized_recovery_200_reviewed as reviewed


class DevelopmentGuard(unittest.TestCase):
    def test_only_exact_reviewed_development_files_allowed_without_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            expected={name:hashlib.sha256(name.encode()).hexdigest() for name in reviewed.DEVELOPMENT}
            with patch.object(reviewed,'DEVELOPMENT',expected):
                self.assertFalse(reviewed.unexpected_report_files(directory))
                for name in expected:(directory/name).write_text(name)
                before={p.name:p.read_bytes() for p in directory.iterdir()}
                self.assertFalse(reviewed.unexpected_report_files(directory))
                self.assertEqual(before,{p.name:p.read_bytes() for p in directory.iterdir()})
                (directory/'metrics.json').write_text('[]')
                self.assertTrue(reviewed.unexpected_report_files(directory))
                (directory/'metrics.json').unlink()
                (directory/'development_metrics.csv').write_text('changed')
                self.assertTrue(reviewed.unexpected_report_files(directory))

    def test_only_two_private_guards_replaced(self):
        root=Path(__file__).resolve().parents[2]
        w=reviewed.make_wrapper(root,'confirmation')
        self.assertIn('_unexpected_report_files',w._execute_bound.__code__.co_names)
        self.assertIn('_unexpected_report_files',w.finish_report.__code__.co_names)
        self.assertIn(reviewed.SELF,w.REQUIRED)
        self.assertIs(w.run_selected_tests.__globals__['build'],w.authorized_build)


if __name__=='__main__':unittest.main()
