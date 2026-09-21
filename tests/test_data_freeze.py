import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from thesis.data_freeze import (
    DataFreezeError,
    _read_archive_member,
    audit_dataset,
    build_processed_files,
    parse_split,
    require_identical_crosschecks,
    validate_plan,
    write_bytes_new_or_identical,
)


class DataFreezeTest(unittest.TestCase):
    def test_strict_split_parser(self):
        self.assertEqual(
            parse_split(b"a\tr\tb\n", dataset="toy", split="train"),
            [("a", "r", "b")],
        )
        with self.assertRaises(DataFreezeError):
            parse_split(b"a r b\n", dataset="toy", split="train")
        with self.assertRaises(DataFreezeError):
            parse_split(b"a\tr\tb\n\ninvalid\n", dataset="toy", split="train")

    def test_dataset_audit_uses_complete_union(self):
        raw = {
            "train": b"a\tr\tb\n",
            "valid": b"c\tr\ta\n",
            "test": b"b\ts\td\n",
        }
        expected = {
            "train_rows": 1,
            "valid_rows": 1,
            "test_rows": 1,
            "entities": 4,
            "relations": 2,
        }
        audit, triples = audit_dataset("toy", raw, expected)
        self.assertEqual(audit["vocabulary"]["entities"], 4)
        self.assertEqual(audit["vocabulary"]["entities_not_in_train"], 2)
        outputs, vocabulary = build_processed_files(triples)
        self.assertEqual(vocabulary, {"entities": 4, "relations": 2})
        self.assertEqual(outputs["entities.tsv"], b"0\ta\n1\tb\n2\tc\n3\td\n")

    def test_duplicates_and_split_overlap_fail(self):
        expected = {
            "train_rows": 2,
            "valid_rows": 1,
            "test_rows": 1,
            "entities": 2,
            "relations": 1,
        }
        duplicate = {
            "train": b"a\tr\tb\na\tr\tb\n",
            "valid": b"b\tr\ta\n",
            "test": b"b\tr\tb\n",
        }
        with self.assertRaises(DataFreezeError):
            audit_dataset("toy", duplicate, expected)

        overlap = {
            "train": b"a\tr\tb\nb\tr\ta\n",
            "valid": b"a\tr\tb\n",
            "test": b"b\tr\tb\n",
        }
        with self.assertRaises(DataFreezeError):
            audit_dataset("toy", overlap, expected)

    def test_crosscheck_requires_byte_identity(self):
        canonical = {split: (split + "\n").encode() for split in ("train", "valid", "test")}
        records = require_identical_crosschecks(canonical, [("mirror", dict(canonical))])
        self.assertTrue(records[0]["splits"]["train"]["identical"])
        changed = dict(canonical)
        changed["test"] = b"different\n"
        with self.assertRaises(DataFreezeError):
            require_identical_crosschecks(canonical, [("mirror", changed)])

    def test_exact_archive_members_are_read_without_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zip_path = root / "data.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("nested/train.txt", b"zip\n")
            self.assertEqual(
                _read_archive_member(zip_path, "zip", "nested/train.txt"), b"zip\n"
            )

            tar_path = root / "data.tar.gz"
            with tarfile.open(tar_path, "w:gz") as archive:
                payload = b"tar\n"
                info = tarfile.TarInfo("train.txt")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            self.assertEqual(_read_archive_member(tar_path, "tar.gz", "train.txt"), b"tar\n")

    def test_plan_rejects_unknown_source(self):
        source_lock = {
            "schema_version": 1,
            "status": "source_lock_pending_materialization",
            "sources": [{"id": "known"}],
        }
        provider = {
            "source_id": "missing",
            "splits": {split: {} for split in ("train", "valid", "test")},
        }
        plan = {
            "schema_version": 1,
            "status": "pinned_dataset_materialization_plan",
            "datasets": [
                {"name": "Toy", "canonical": provider, "crosschecks": []}
            ],
        }
        with self.assertRaises(DataFreezeError):
            validate_plan(plan, source_lock)

    def test_output_is_create_once_or_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.txt"
            write_bytes_new_or_identical(path, b"same\n", "1")
            write_bytes_new_or_identical(path, b"same\n", "2")
            with self.assertRaises(DataFreezeError):
                write_bytes_new_or_identical(path, b"different\n", "3")


if __name__ == "__main__":
    unittest.main()
