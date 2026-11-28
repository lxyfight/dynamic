import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from relay_paired_chain import audit_recovery, publish_rename_if_absent


class RelayTests(unittest.TestCase):
    def test_windows_atomic_rename_keeps_scratch_independent(self):
        if os.name != 'nt':
            self.skipTest('Windows-specific atomic rename semantics')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, final = root/'scratch.json', root/'final.json'
            source.write_bytes(b'{"net":7}')
            self.assertTrue(publish_rename_if_absent(source, final))
            self.assertEqual(source.read_bytes(), final.read_bytes())
            self.assertNotEqual(source.stat().st_ino, final.stat().st_ino)
            final.write_bytes(b'{"net":7,"wall_seconds":1}')
            self.assertEqual(source.read_bytes(), b'{"net":7}')
            self.assertFalse(publish_rename_if_absent(source, final))
            self.assertEqual(final.read_bytes(), b'{"net":7,"wall_seconds":1}')
            self.assertEqual(list(root.glob('.relay-publish-*')), [])

    def test_windows_target_created_at_rename_is_never_overwritten(self):
        if os.name != 'nt':
            self.skipTest('Windows-specific atomic rename semantics')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, final = root/'scratch.json', root/'final.json'
            source.write_bytes(b'complete scratch')
            real_rename = os.rename
            def race(a,b):
                Path(b).write_bytes(b'original worker')
                return real_rename(a,b)
            with patch('relay_paired_chain.os.rename', side_effect=race):
                self.assertFalse(publish_rename_if_absent(source,final))
            self.assertEqual(final.read_bytes(),b'original worker')
            self.assertEqual(source.read_bytes(),b'complete scratch')
            self.assertEqual(list(root.glob('.relay-publish-*')), [])

    def test_existing_partial_file_is_not_read(self):
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory)/'final.json'
            final.write_bytes(b'{partial')
            self.assertFalse(publish_rename_if_absent(Path(directory)/'absent',final))
            self.assertEqual(final.read_bytes(),b'{partial')

    def test_recovery_requires_explicit_original_exit_confirmation(self):
        with self.assertRaisesRegex(ValueError,'original executor has exited'):
            audit_recovery('unused','unused',False,'unused')


if __name__ == '__main__':
    unittest.main()
