import json
import os
from pathlib import Path
from unittest.mock import patch
import unittest

import grouped_inference as grouped
import test_drained_inference as old
import test_fast_resume_20261015 as fast_tests
import grouped_resume_20261015 as fast


class GroupedTests(old.DrainTests):
    def setUp(self):
        super().setUp()
        self.binding = patch.object(old, 'build', grouped.build)
        self.binding.start()
        self.addCleanup(self.binding.stop)

    def test_16_answer_batches_commit_before_next_batch_and_resume(self):
        self.budget()
        cases = [{**self.cases[0], 'pair_id':str(i)} for i in range(35)]
        (self.root/'data/abt-buy/prompt.json').write_text(json.dumps(cases))
        self.module.account = lambda *a, **k: None  # Count only journal fsync.
        _, cell, _ = grouped.build(self.module, 2)
        requests = []
        writes = []
        original = grouped.write_group
        def commit(path, lines):
            writes.append((path.name, len(lines)))
            original(path, lines)
        def http(*a, **k):
            requests.append(1)
            if len(requests) > 16:
                self.assertGreaterEqual(len(self.rows('jsonl')),16)
            if len(requests) > 32:
                self.assertGreaterEqual(len(self.rows('jsonl')),32)
            return old.Response(json.dumps(self.response).encode())
        with patch.object(grouped,'write_group',side_effect=commit), patch('urllib.request.urlopen',side_effect=http):
            cell('jev','B0','abt-buy','prompt',workers=16)
            prefix=(self.cache/'prompt.jsonl').read_bytes()
            cell('jev','B0','abt-buy','prompt',workers=16)
        self.assertEqual(len(requests),35)
        self.assertEqual([n for p,n in writes if p=='prompt.jsonl'],[16,16,3])
        self.assertEqual((self.cache/'prompt.jsonl').read_bytes(),prefix)
        self.assertEqual(len({r['pair_id'] for r in self.rows('jsonl')}),35)

    def test_one_fsync_per_existing_file_and_preserved_prefix(self):
        path=self.root/'journal.jsonl';path.write_text('old\n')
        real=os.fsync
        with patch.object(grouped.os,'fsync',side_effect=real) as fs:
            grouped.write_group(path,['a\n']*16)
        self.assertEqual(fs.call_count,1)
        self.assertEqual(path.read_text(),'old\n'+'a\n'*16)

    def test_storage_error_tries_sibling_files_and_stops(self):
        failed=[]
        g=grouped.Groups(lambda:failed.append(True));calls=[]
        def write(path,lines):
            calls.append(path.name)
            if path.name=='failed': raise OSError('simulated storage failure')
        with patch.object(grouped,'write_group',side_effect=write):
            with self.assertRaisesRegex(OSError,'audit before resume'):
                with g.batch():
                    g.append(Path('failed'),{},json.dumps)
                    g.append(Path('success'),{},json.dumps)
        self.assertEqual(calls,['failed','success'])
        self.assertEqual(failed,[True])

    def test_readable_but_uncommitted_bytes_cannot_release_credit(self):
        from leased_budget import Credit
        self.budget()
        credit=Credit(self.root,'fast-core2-durability',{})
        self.module._on_journal_failure=lambda:setattr(credit,'journal_io_failed',True)
        self.module.account=credit.account
        _,cell,_=grouped.build(self.module,2)
        original=grouped.write_group
        def fail_after_write(path,lines):
            original(path,lines)
            if path.name=='prompt.jsonl':raise OSError('fsync uncertain')
        with patch.object(grouped,'write_group',side_effect=fail_after_write),patch('urllib.request.urlopen',side_effect=lambda *a,**k:old.Response(json.dumps(self.response).encode())):
            with self.assertRaises(OSError):cell('jev','B0','abt-buy','prompt',workers=2)
        self.assertEqual(len(self.rows('jsonl')),2)
        with self.assertRaisesRegex(ValueError,'durability failed'):
            fast.verify_accounting(credit,self.root,{})
        self.assertFalse(credit.closed)


class GroupedMatrixTests(fast_tests.FastTests):
    def setUp(self):
        self.binding=patch.object(fast_tests,'credit_build',fast.credit_build)
        self.binding.start();self.addCleanup(self.binding.stop)
        self.component=patch.object(fast_tests,'core_component',fast.core_component)
        self.component.start();self.addCleanup(self.component.stop)


if __name__=='__main__':unittest.main()
