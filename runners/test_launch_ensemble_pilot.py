import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import launch_ensemble_pilot as launcher


class Admission(unittest.TestCase):
    def test_active_project_blocks_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'configs').mkdir()
            package=root/'artifacts/strong_ensemble/pilot_v1'; package.mkdir(parents=True)
            (root/'configs/selective_ensemble_pilot_execution.json').write_text('{"files_sha256":{}}')
            (package/'approval_aaron_selective_20261014.json').write_text(json.dumps({'approved_by':'Aaron','max_cost_usd':10,'project_cap_usd':100}))
            outputs=['123|decision-em-ensemble-pilot\n365636|decision-em-confirm-recovery\n',f'WorkDir={root}']
            with patch.object(launcher,'ROOT',root), patch.dict('os.environ',{'SLURM_JOB_ID':'123','USER':'test'}), patch.object(launcher.subprocess,'check_output',side_effect=outputs), patch.object(launcher,'run') as run:
                with self.assertRaises(ValueError): launcher.main()
                run.assert_not_called()
            self.assertFalse((package/'admission.json').exists())

    def test_terminal_chain_admitted_then_pilot_then_quote(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for folder in ['configs','docs','artifacts/operations','artifacts/strong_ensemble/pilot_v1']:
                (root/folder).mkdir(parents=True,exist_ok=True)
            package=root/'artifacts/strong_ensemble/pilot_v1'
            (root/'configs/selective_ensemble_pilot_execution.json').write_text('{"files_sha256":{}}')
            (package/'approval_aaron_selective_20261014.json').write_text(json.dumps({'approved_by':'Aaron','max_cost_usd':10,'project_cap_usd':100,'package_sha256':'synthetic'}))
            (root/'docs/APPROVED_TODO_20261014.md').write_text('synthetic')
            (root/'artifacts/operations/ensemble_pilot_run.py').write_text('synthetic')
            outputs=['123|decision-em-ensemble-pilot\n365355|statistics-full\n', ''.join(j+'|COMPLETED\n'+j+'.batch|COMPLETED\n' for j in launcher.CHAIN)]
            calls=[]
            with patch.object(launcher,'ROOT',root), patch.dict('os.environ',{'SLURM_JOB_ID':'123','USER':'test'}), patch.object(launcher.subprocess,'check_output',side_effect=outputs), patch.object(launcher,'run',side_effect=lambda *a:calls.append('pilot')), patch.object(launcher,'quote',side_effect=lambda *a:calls.append('quote')), patch.dict('sys.modules',{'run':types.SimpleNamespace(key=lambda:'fake')}):
                launcher.main()
            self.assertEqual(calls,['pilot','quote'])
            self.assertTrue(json.loads((package/'admission.json').read_text())['all_project_inference_jobs_reviewed'])


if __name__=='__main__': unittest.main()
