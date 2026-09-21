import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from thesis.text_encoder import (
    TextEncoderError,
    encoded_table_bytes,
    load_pre_encoder_table,
    normalise_embedding_matrix,
    validate_encoder_config,
)


def _config():
    return {
        "schema_version": 1,
        "status": "frozen_text_encoder_configuration",
        "model": {
            "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": "a" * 40,
            "trust_remote_code": False,
            "required_files": ["config.json"],
        },
        "runtime": {
            "device": "cuda",
            "batch_size": 256,
            "embedding_dimension": 384,
            "raw_dtype": "float32",
            "maximum_sequence_length": 256,
            "truncation": True,
            "pooling": "mean_tokens",
            "normalise_during_encoding": False,
            "post_encoding_l2_normalisation": True,
            "minimum_raw_norm": 1e-12,
            "unit_norm_absolute_tolerance": 1e-6,
        },
        "environment": {"python": "3.10.20"},
    }


class TextEncoderTest(unittest.TestCase):
    def test_encoder_revision_and_runtime_are_frozen(self):
        config = _config()
        validate_encoder_config(config)
        config["model"]["revision"] = "main"
        with self.assertRaises(TextEncoderError):
            validate_encoder_config(config)
        config = _config()
        config["runtime"]["batch_size"] = 512
        with self.assertRaises(TextEncoderError):
            validate_encoder_config(config)

    def test_pre_encoder_table_requires_hash_and_contiguous_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = "Toy"
            table_path = (
                root / "artifacts" / "descriptions" / dataset / "evidence_pre_encoder.tsv"
            )
            table_path.parent.mkdir(parents=True)
            text = 'Evidence with "literal quotes"'
            digest = hashlib.sha256(text.encode()).hexdigest()
            header = (
                "dataset\tentity_id\tentity_index\tsource_artifact\t"
                "source_identifier\tmapping_status\tevidence_status\tevidence_text\t"
                "character_length\tpretoken_word_count\tevidence_sha256\n"
            )
            row = "Toy\te0\t0\tsource\te0\tverified\tfull\t{}\t30\t4\t{}\n".format(
                text, digest
            )
            table_path.write_text(header + row, encoding="utf-8")
            manifest_path = root / "data" / "manifests" / "Toy_evidence_pre_encoder.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "evidence_verified_pending_encoder_tokenization",
                        "dataset": dataset,
                        "protocol_sha256": "protocol",
                        "audit_table": {
                            "rows": 1,
                            "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            records, _, _ = load_pre_encoder_table(
                root, dataset, protocol_sha256="protocol"
            )
            self.assertEqual("e0", records[0]["entity_id"])
            self.assertEqual(text, records[0]["evidence_text"])
            table_path.write_text(header + row.replace("\t0\t", "\t1\t"), encoding="utf-8")
            with self.assertRaises(TextEncoderError):
                load_pre_encoder_table(root, dataset, protocol_sha256="protocol")

    def test_encoded_table_records_tokens_truncation_and_vector_hash(self):
        record = {
            "dataset": "Toy",
            "entity_id": "e0",
            "entity_index": "0",
            "source_artifact": "source",
            "source_identifier": "e0",
            "mapping_status": "verified",
            "evidence_status": "full",
            "evidence_text": 'Evidence with "literal quotes"',
            "character_length": "30",
            "pretoken_word_count": "4",
            "evidence_sha256": hashlib.sha256(
                b'Evidence with "literal quotes"'
            ).hexdigest(),
        }
        row_hash = "b" * 64
        value = encoded_table_bytes(
            [record], [257], [row_hash], maximum_sequence_length=256
        ).decode()
        self.assertIn("token_length\ttruncated\tnormalised_vector_row_sha256", value)
        self.assertIn("257\ttrue\t" + row_hash, value)
        self.assertIn('Evidence with "literal quotes"', value)

    def test_l2_normalisation_rejects_zero_and_nonfinite_rows(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is tested in the frozen cluster environment")
        raw = np.zeros((2, 384), dtype=np.float32)
        raw[0, 0] = 3.0
        raw[0, 1] = 4.0
        raw[1, 2] = 2.0
        normalised, statistics = normalise_embedding_matrix(
            raw, minimum_raw_norm=1e-12, unit_norm_tolerance=1e-6
        )
        self.assertTrue(np.allclose(np.linalg.norm(normalised, axis=1), 1.0))
        self.assertLessEqual(
            statistics["normalised_norm"]["maximum_absolute_deviation_from_one"],
            1e-6,
        )
        raw[1] = 0.0
        with self.assertRaises(TextEncoderError):
            normalise_embedding_matrix(
                raw, minimum_raw_norm=1e-12, unit_norm_tolerance=1e-6
            )


if __name__ == "__main__":
    unittest.main()
