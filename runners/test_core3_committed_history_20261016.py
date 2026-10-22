"""No HTTP: real saved commit bytes for prefix tests; isolated capture fixture."""
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

import core3_committed_history_20261016 as history
import complete_core3_timing_20261016 as complete

ROOT = Path(__file__).resolve().parents[2]


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.target = self.root/history.TARGET
        self.anchor = self.root/history.ANCHOR
        (self.target/'artifacts').mkdir(parents=True)
        self.anchor.mkdir(parents=True)
        for rel,name in ((history.STAGE,'stage_execution.jsonl'),(history.PROGRESS,'progress.json')):
            raw = (ROOT/history.TARGET/rel).read_bytes()
            (self.target/rel).write_bytes(raw); (self.anchor/name).write_bytes(raw)
        self.started = dict(stage='supplement',status='started',run_id=3,slurm_job_id='999999',
            prior_stage_journal_sha256=history.STAGE_SHA,admission_sha256='a'*64,
            parallel_policy_sha256='b'*64,launcher_sha256='c'*64,prior_cache_sha256='d'*64)
        self.addCleanup(patch.stopall)
        patch.object(socket,'socket',side_effect=AssertionError('Network forbidden')).start()

    def append(self, *events):
        with (self.target/history.STAGE).open('ab') as f:
            for event in events:
                f.write((json.dumps(event)+'\n').encode())

    def test_exact_saved_commit_and_single_owned_start(self):
        history.verify_prefix(self.target,self.anchor)
        self.append(self.started)
        history.verify_prefix(self.target,self.anchor,started=self.started)
        with self.assertRaises(ValueError): history.verify_prefix(self.target,self.anchor)
        # The old final CLI is intentionally still strict after an extension.
        self.assertNotEqual((self.target/history.STAGE).read_bytes(),(self.anchor/'stage_execution.jsonl').read_bytes())

    def test_foreign_reordered_failed_repeated_and_partial_suffix_rejected(self):
        original = (self.target/history.STAGE).read_bytes()
        for events in ([dict(self.started,status='failed')], [self.started,self.started],
                       [dict(self.started,stage='calibration')], [dict(self.started,slurm_job_id='367108')]):
            (self.target/history.STAGE).write_bytes(original); self.append(*events)
            with self.assertRaises(ValueError): history.verify_prefix(self.target,self.anchor,started=self.started)
        (self.target/history.STAGE).write_bytes(original+b'{')
        with self.assertRaises(ValueError): history.verify_prefix(self.target,self.anchor,started=self.started)

    def test_strict_identity_and_anchor_prefix_tampering(self):
        for field,value in [('run_id',3.0),('run_id',True),('stage','core'),('prior_stage_journal_sha256','0'*64),
                            ('admission_sha256',None),('slurm_job_id','366980')]:
            bad = dict(self.started,**{field:value})
            with self.assertRaises(ValueError): history.verify_prefix(self.target,self.anchor,started=bad)
        original = (self.target/history.STAGE).read_bytes()
        (self.target/history.STAGE).write_bytes(original.replace(b'365894',b'365893',1))
        with self.assertRaises(ValueError): history.verify_prefix(self.target,self.anchor)
        (self.target/history.STAGE).write_bytes(original)
        (self.anchor/'stage_execution.jsonl').write_bytes(original+b'{}\n')
        with self.assertRaises(ValueError): history.verify_prefix(self.target,self.anchor)

    def test_core_progress_and_extension_progress_cannot_change_early(self):
        original = history.read(self.target/history.PROGRESS)
        for changed in (dict(original,AP1_timing={'status':'failed'}),
                        dict(original,supplement_billiger50={'status':'complete_automated'})):
            (self.target/history.PROGRESS).write_text(json.dumps(changed))
            with self.assertRaises(ValueError): history.verify_prefix(self.target,self.anchor)
        (self.anchor/'progress.json').write_text(json.dumps(original))
        with self.assertRaises(ValueError): history.verify_prefix(self.target,self.anchor)

    @staticmethod
    def scheduler(args, **kwargs):
        job = args[args.index('-j')+1]
        success = job in (history.HISTORY[0],history.HISTORY[-1])
        result = ''.join(f'{job}{s}|{("COMPLETED|0:0" if success or s==".extern" else "FAILED|1:0")}|now\n'
                         for s in ('','.batch','.extern'))
        for extra in ('366681.0','366681.1','366697.0'):
            if extra.startswith(job+'.'): result += extra+'|COMPLETED|0:0|now\n'
        return result

    def test_every_historical_step_including_smoke_numbered_and_final(self):
        with patch.object(history.subprocess,'check_output',side_effect=self.scheduler) as cmd:
            self.assertEqual(list(history.history_steps()),list(history.HISTORY))
            self.assertEqual(cmd.call_count,11)
        for job in history.HISTORY:
            for change in ('missing','running','bad_exit','extra'):
                def altered(args,**kw):
                    raw = self.scheduler(args,**kw)
                    if args[args.index('-j')+1] != job: return raw
                    if change == 'missing': return '\n'.join(raw.splitlines()[:-1])+'\n'
                    if change == 'running': return raw.replace('COMPLETED','RUNNING',1)
                    if change == 'bad_exit': return raw.replace('0:0','2:0',1)
                    return raw+job+'.99|COMPLETED|0:0|now\n'
                with self.subTest(job=job,change=change),patch.object(history.subprocess,'check_output',side_effect=altered):
                    with self.assertRaises(ValueError): history.history_steps()

    def capture_fixture(self, stack):
        self.anchor.rmdir() if not list(self.anchor.iterdir()) else shutil.rmtree(self.anchor)
        (self.root/'artifacts/operations').mkdir(exist_ok=True)
        (self.root/history.SELF).write_bytes((ROOT/history.SELF).read_bytes())
        p = self.target/history.ATTEMPT; p.mkdir(parents=True)
        shutil.copyfile(ROOT/history.TARGET/history.ATTEMPT/'final_execution.jsonl',p/'final_execution.jsonl')
        stack.enter_context(patch.object(complete,'require_committed',return_value={}))
        stack.enter_context(patch.object(history,'bound_core',return_value=({'report_sha256':{}},{'raw_sha256':{}},{})))
        stack.enter_context(patch.object(history.subprocess,'check_output',side_effect=self.scheduler))

    def test_capture_durable_immutable_and_second_capture_rejected(self):
        with ExitStack() as stack:
            self.capture_fixture(stack)
            original = history.exclusive
            def save(path,raw):
                for p in (self.root/'artifacts/operations/repetition_writer.lock',self.target/'artifacts/parallel_writer.lock'):
                    with p.open('a') as f:
                        with self.assertRaises(BlockingIOError): fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                original(path,raw)
            with patch.object(history,'exclusive',side_effect=save): value = history.capture(self.root)
            self.assertFalse(value['admission_granted']); self.assertFalse(value['numeric_proof_replayed'])
            self.assertEqual(history.sha(self.anchor/'progress.json'),history.PROGRESS_SHA)
            self.assertEqual(history.sha(self.anchor/'stage_execution.jsonl'),history.STAGE_SHA)
            with self.assertRaises(ValueError): history.capture(self.root)

    def test_capture_missing_actual_commit_or_storage_failure_never_commits(self):
        with ExitStack() as stack:
            self.capture_fixture(stack)
            with patch.object(complete,'require_committed',side_effect=ValueError('incomplete')):
                with self.assertRaises(ValueError): history.capture(self.root)
            self.assertFalse(self.anchor.exists())
            with patch.object(history,'exclusive',side_effect=OSError('fsync failed')):
                with self.assertRaises(OSError): history.capture(self.root)
            self.assertFalse((self.anchor/'manifest.json').exists())
            with self.assertRaises(ValueError): history.capture(self.root)

    def test_verify_manifest_pin_sources_raw_scope_and_first_start(self):
        with ExitStack() as stack:
            self.capture_fixture(stack); history.capture(self.root)
            manifest_sha = history.sha(self.anchor/'manifest.json')
            with self.assertRaises(ValueError): history.verify(self.root,'0'*64)
            history.verify(self.root,manifest_sha)
            self.append(self.started)
            rel = sorted(history.supplement_paths())[0]
            p=self.target/rel;p.parent.mkdir(parents=True);p.write_text('{}\n')
            history.verify(self.root,manifest_sha,started=self.started)
            other = self.target/'runs/foreign.jsonl'; other.write_text('{}\n')
            with self.assertRaises(ValueError): history.verify(self.root,manifest_sha,started=self.started)
            other.unlink()
            manifest=history.read(self.anchor/'manifest.json');manifest['source_sha256']='0'*64
            (self.anchor/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): history.verify(self.root,history.sha(self.anchor/'manifest.json'),started=self.started)

    def test_only_exact_shared_frozen_data_link_allowed(self):
        (self.root/'data').mkdir();(self.root/'data/case.json').write_text('{}')
        (self.target/'data').symlink_to(self.root/'data',target_is_directory=True)
        self.assertEqual(history.protected_path(self.root,self.target,'data/case.json'),self.root/'data/case.json')
        (self.target/'data').unlink();(self.target/'data').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):history.protected_path(self.root,self.target,'data/case.json')

    def test_path_traversal_and_symlink_rejected(self):
        for rel in ('../foreign','/tmp/foreign'):
            with self.assertRaises(ValueError): history.own(self.root,rel)
        (self.root/'link').symlink_to(self.target,target_is_directory=True)
        with self.assertRaises(ValueError): history.own(self.root,'link/artifacts/progress.json')


if __name__ == '__main__': unittest.main()
