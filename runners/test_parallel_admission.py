import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from parallel_admission import review, sha


class ParallelAdmission(unittest.TestCase):
    def test_known_independent_writer_allowed_but_duplicate_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            approval=root/'approval.json'
            approval.write_text(json.dumps({'approved_by':'Aaron','parallel_execution_authorized':True,
                'project_cap_usd':100,'quoted_user_authorization':'synthetic'}))
            script='python independent.py'
            policy=root/'policy.json'
            policy.write_text(json.dumps({'approval_file':'approval.json','approval_sha256':sha(approval),
                'files_sha256':{},'reviewed_project_jobs':{'1':{'name':'decision-em-core',
                'scope':'runs/quality','batch_script_sha256':hashlib.sha256(script.encode()).hexdigest()}}}))
            with patch('parallel_admission.subprocess.check_output',side_effect=[
                    '1|decision-em-core\n',f'WorkDir={root} ',script]):
                self.assertTrue(review(root,policy)['all_project_inference_jobs_reviewed'])
            for outputs in [['2|decision-em-core\n',f'WorkDir={root} '],
                            ['1|decision-em-core\n',f'WorkDir={root} ','changed command']]:
                with patch('parallel_admission.subprocess.check_output',side_effect=outputs):
                    with self.assertRaises(ValueError): review(root,policy)
            approval.write_text('{}')
            with self.assertRaises(ValueError): review(root,policy)
