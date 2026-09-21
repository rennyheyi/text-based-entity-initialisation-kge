import copy
import unittest

from thesis.source_lock import SourceLockError, build_source_lock


SHA256 = "a" * 64
COMMIT = "b" * 40


def artifact(label="artifact.zip"):
    return {
        "label": label,
        "requested_url": "https://example.test/" + label,
        "resolved_url": "https://cdn.example.test/" + label,
        "content_type": "application/zip",
        "size_bytes": 10,
        "sha256": SHA256,
        "archive": {
            "archive_format": "zip",
            "member_count": 1,
            "total_unpacked_bytes": 20,
            "members": [{"name": "train.txt", "size_bytes": 20}],
        },
    }


def inputs():
    config = {
        "schema_version": 1,
        "status": "candidate_sources_only_not_pinned",
        "sources": [
            {
                "id": "source",
                "dataset": "Dataset",
                "role": "canonical",
                "kind": "http_file",
                "url": "https://example.test/artifact.zip",
                "filename": "artifact.zip",
                "source_page": "https://example.test/page",
                "declared_version": "1",
            }
        ],
    }
    report = {
        "status": "candidate_sources_audited_requires_pin",
        "official_result": False,
        "failure_count": 0,
        "config_sha256": SHA256,
        "execution": {
            "scheduler_job_id": "123",
            "completed_at_utc": "2026-08-04T12:00:00+00:00",
        },
        "sources": [
            {
                "id": "source",
                "dataset": "Dataset",
                "role": "canonical",
                "kind": "http_file",
                "status": "audited_candidate_requires_pin",
                "artifacts": [artifact()],
            }
        ],
    }
    return config, report


class SourceLockTest(unittest.TestCase):
    def test_successful_audit_is_promoted_without_archive_member_bulk(self):
        config, report = inputs()
        lock = build_source_lock(
            config,
            report,
            candidate_config_sha256=SHA256,
            audit_report_sha256="c" * 64,
            protocol_sha256="d" * 64,
        )
        self.assertEqual(lock["status"], "source_lock_pending_materialization")
        self.assertEqual(lock["candidate_audit_job_id"], "123")
        locked_artifact = lock["sources"][0]["artifacts"][0]
        self.assertEqual(locked_artifact["sha256"], SHA256)
        self.assertNotIn("resolved_url", locked_artifact)
        self.assertNotIn("members", locked_artifact["archive"])

    def test_failed_audit_cannot_be_promoted(self):
        config, report = inputs()
        report["status"] = "candidate_source_audit_failed"
        report["failure_count"] = 1
        with self.assertRaises(SourceLockError):
            build_source_lock(
                config,
                report,
                candidate_config_sha256=SHA256,
                audit_report_sha256="c" * 64,
                protocol_sha256="d" * 64,
            )

    def test_config_hash_mismatch_cannot_be_promoted(self):
        config, report = inputs()
        with self.assertRaises(SourceLockError):
            build_source_lock(
                config,
                report,
                candidate_config_sha256="e" * 64,
                audit_report_sha256="c" * 64,
                protocol_sha256="d" * 64,
            )

    def test_source_identity_mismatch_cannot_be_promoted(self):
        config, report = inputs()
        report["sources"][0]["id"] = "different"
        with self.assertRaises(SourceLockError):
            build_source_lock(
                config,
                report,
                candidate_config_sha256=SHA256,
                audit_report_sha256="c" * 64,
                protocol_sha256="d" * 64,
            )

    def test_github_branch_is_replaced_by_full_resolved_revision(self):
        config, report = inputs()
        config["sources"][0] = {
            "id": "source",
            "dataset": "Dataset",
            "role": "mirror",
            "kind": "github_files",
            "repository": "https://github.com/owner/repo.git",
            "ref": "master",
            "files": [{"path": "data/train.txt"}],
        }
        audited_artifact = artifact("00_data_train.txt")
        audited_artifact["repository_path"] = "data/train.txt"
        report["sources"][0].update(
            {
                "role": "mirror",
                "kind": "github_files",
                "repository": "https://github.com/owner/repo.git",
                "requested_ref": "master",
                "resolved_revision": COMMIT,
                "artifacts": [audited_artifact],
            }
        )
        lock = build_source_lock(
            config,
            report,
            candidate_config_sha256=SHA256,
            audit_report_sha256="c" * 64,
            protocol_sha256="d" * 64,
        )
        self.assertEqual(lock["sources"][0]["resolved_revision"], COMMIT)


if __name__ == "__main__":
    unittest.main()
