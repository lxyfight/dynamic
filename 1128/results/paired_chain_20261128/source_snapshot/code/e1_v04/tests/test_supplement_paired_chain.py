import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from supplement_paired_chain import (audit_supplement, copy_byte_identical,
    declared_tasks, missing_reverse_tasks, publish_copy_if_absent, scientific_content)


class SupplementTests(unittest.TestCase):
    def test_create_only_and_original_rewrite_cannot_change_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, final = root / 'scratch.json', root / 'final.json'
            source.write_bytes(b'{"net":7}')
            self.assertTrue(publish_copy_if_absent(source, final))
            self.assertEqual(final.read_bytes(), source.read_bytes())
            self.assertNotEqual(source.stat().st_ino, final.stat().st_ino)
            # Simulate the already running original worker's late write_text.
            final.write_bytes(b'{"net":7,"wall_seconds":1}')
            self.assertEqual(source.read_bytes(), b'{"net":7}')
            self.assertFalse(publish_copy_if_absent(source, final))
            self.assertEqual(final.read_bytes(), b'{"net":7,"wall_seconds":1}')
            self.assertEqual(list(root.glob('.supplement-publish-*')), [])

    def test_destination_created_during_copy_wins_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, final = root / 'scratch.json', root / 'final.json'
            source.write_bytes(b'complete supplemental result')
            real_link = os.link
            def race(source_path, target_path):
                Path(target_path).write_bytes(b'original worker data')
                return real_link(source_path, target_path)
            with patch('supplement_paired_chain.os.link', side_effect=race):
                self.assertFalse(publish_copy_if_absent(source, final))
            self.assertEqual(final.read_bytes(), b'original worker data')
            self.assertEqual(source.read_bytes(), b'complete supplemental result')
            self.assertEqual(list(root.glob('.supplement-publish-*')), [])

    def test_existing_incomplete_destination_is_never_read(self):
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / 'final.json'
            final.write_bytes(b'{partial')
            # Source deliberately absent: existing target short-circuits all IO.
            self.assertFalse(publish_copy_if_absent(Path(directory) / 'absent', final))
            self.assertEqual(final.read_bytes(), b'{partial')

    def test_copied_input_has_identical_bytes_and_independent_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, copy = root / 'input.gz', root / 'scratch' / 'input.gz'
            source.write_bytes(bytes(range(256)))
            checksum = copy_byte_identical(source, copy)
            self.assertEqual(len(checksum), 64)
            self.assertEqual(source.read_bytes(), copy.read_bytes())
            source.write_bytes(b'changed')
            self.assertEqual(copy.read_bytes(), bytes(range(256)))
            with self.assertRaises(FileExistsError):
                copy_byte_identical(source, copy)

    def test_reverse_queue_is_exact_subset_without_extra_replicas(self):
        sites = [{'id': 'A'}, {'id': 'B'}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'paired').mkdir()
            (root / 'paired' / 'B_r01.json').write_bytes(b'{partial')
            queue = missing_reverse_tasks(sites, 2, root)
            self.assertEqual([(s['id'], r) for s, r in queue],
                             [('B', 0), ('B', -1), ('A', 1), ('A', 0), ('A', -1)])
            self.assertEqual(len(declared_tasks(sites, 2)), 6)
            self.assertTrue(all(r in (-1, 0, 1) for _, r in queue))

    def test_comparison_excludes_only_recorded_cpu_timings(self):
        row = dict(net=10, wall_seconds=2, branches={'0': dict(trace='same',
            continuation_search=dict(decision_seconds=1, epochs=7))})
        other = json.loads(json.dumps(row))
        other['wall_seconds'] = 9
        other['branches']['0']['continuation_search']['decision_seconds'] = 8
        self.assertEqual(scientific_content(row), scientific_content(other))
        other['branches']['0']['continuation_search']['epochs'] = 8
        self.assertNotEqual(scientific_content(row), scientific_content(other))
        self.assertEqual(row['wall_seconds'], 2)

    def test_final_audit_rejects_incomplete_main_without_reading_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_supplement(directory)
            self.assertEqual(result['status'], 'incomplete')
            self.assertFalse(result['successful'])
            self.assertEqual(result['checked'], 0)


if __name__ == '__main__':
    unittest.main()
