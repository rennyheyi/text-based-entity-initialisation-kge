import tempfile
import unittest
from pathlib import Path

from thesis.evidence_audit import (
    EvidenceAuditError,
    _record,
    evidence_statistics,
    parse_two_field_mapping,
    table_bytes,
    validate_expected,
)


class EvidenceAuditTest(unittest.TestCase):
    def test_english_wrapped_mapping_is_strict_and_normalised(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptions.tsv"
            path.write_text('/m/a\t"A   description."@en\n', encoding="utf-8")
            self.assertEqual(
                {"/m/a": "A description."},
                parse_two_field_mapping(path, language_wrapped=True),
            )
            path.write_text('/m/a\t"A description."@de\n', encoding="utf-8")
            with self.assertRaises(EvidenceAuditError):
                parse_two_field_mapping(path, language_wrapped=True)

    def test_duplicate_identifiers_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.tsv"
            path.write_text("a\tone\na\ttwo\n", encoding="utf-8")
            with self.assertRaises(EvidenceAuditError):
                parse_two_field_mapping(path)

    def test_statistics_and_expected_counts_are_exact(self):
        records = [
            _record(
                dataset="Toy",
                entity_index=index,
                entity_id="e{}".format(index),
                source_artifact="source.tsv",
                mapping_status="verified",
                evidence_status=status,
                evidence_text=text,
            )
            for index, (status, text) in enumerate(
                (("full", "same"), ("full", "same"), ("name_only", "other"))
            )
        ]
        statistics = evidence_statistics(records)
        self.assertEqual(2, statistics["unique_evidence_texts"])
        self.assertEqual(1, statistics["duplicate_text_groups"])
        self.assertEqual(2, statistics["maximum_text_multiplicity"])
        self.assertEqual(
            ["e0", "e1"], statistics["duplicate_groups"][0]["entity_ids"]
        )
        plan = {
            "expected": {
                "vocabulary": 3,
                "full": 2,
                "name_only": 1,
                "missing": 0,
                "unique_evidence_texts": 2,
                "duplicate_text_groups": 1,
                "maximum_text_multiplicity": 2,
            }
        }
        validate_expected(plan, records, {}, statistics)
        plan["expected"]["mapping_gloss_alignment"] = {
            "exact_wordnet_gloss": 3
        }
        details = {
            "compatibility": {
                "mapping_gloss_alignment": {"exact_wordnet_gloss": 3}
            }
        }
        validate_expected(plan, records, details, statistics)
        details["compatibility"]["mapping_gloss_alignment"] = {
            "exact_wordnet_gloss": 2,
            "target_gloss_with_extra_text": 1,
        }
        with self.assertRaises(EvidenceAuditError):
            validate_expected(plan, records, details, statistics)
        del plan["expected"]["mapping_gloss_alignment"]
        plan["expected"]["full"] = 3
        with self.assertRaises(EvidenceAuditError):
            validate_expected(plan, records, {}, statistics)

    def test_table_is_deterministic_and_tsv_safe(self):
        record = _record(
            dataset="Toy",
            entity_index=0,
            entity_id="e0",
            source_artifact="source.tsv",
            mapping_status="verified",
            evidence_status="full",
            evidence_text="Evidence text",
        )
        first = table_bytes([record])
        self.assertEqual(first, table_bytes([record]))
        self.assertTrue(first.startswith(b"dataset\tentity_id\tentity_index\t"))
        unsafe = dict(record)
        unsafe["source_artifact"] = "bad\tfield"
        with self.assertRaises(EvidenceAuditError):
            table_bytes([unsafe])


if __name__ == "__main__":
    unittest.main()
