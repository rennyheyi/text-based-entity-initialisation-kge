import hashlib
import tempfile
import unittest
from pathlib import Path

from thesis.source_audit import (
    figshare_version_api_url,
    github_raw_url,
    parse_ls_remote,
    select_figshare_file,
    sha256_file,
    validate_archive_member_name,
    validate_config,
)


class SourceAuditTest(unittest.TestCase):
    def test_figshare_api_url_pins_article_version(self):
        self.assertEqual(
            figshare_version_api_url(11911272, 1),
            "https://api.figshare.com/v2/articles/11911272/versions/1",
        )

    def test_figshare_file_requires_one_exact_nonempty_match(self):
        metadata = {
            "id": 11911272,
            "title": "WN18RR",
            "files": [
                {
                    "id": 22044231,
                    "name": "WN18RR.zip",
                    "size": 895123,
                    "download_url": "https://ndownloader.figshare.com/files/22044231",
                }
            ],
        }
        selected = select_figshare_file(
            metadata,
            article_id=11911272,
            expected_title="WN18RR",
            filename="WN18RR.zip",
            max_download_bytes=2_000_000,
        )
        self.assertEqual(selected["id"], 22044231)

        metadata["files"][0]["download_url"] = (
            "https://figshare.com/ndownloader/files/22044231"
        )
        selected = select_figshare_file(
            metadata,
            article_id=11911272,
            expected_title="WN18RR",
            filename="WN18RR.zip",
            max_download_bytes=2_000_000,
        )
        self.assertEqual(selected["id"], 22044231)

        metadata["files"][0]["size"] = 0
        with self.assertRaises(ValueError):
            select_figshare_file(
                metadata,
                article_id=11911272,
                expected_title="WN18RR",
                filename="WN18RR.zip",
                max_download_bytes=2_000_000,
            )

    def test_figshare_file_rejects_duplicate_exact_names(self):
        file_record = {
            "id": 1,
            "name": "WN18RR.zip",
            "size": 1,
            "download_url": "https://ndownloader.figshare.com/files/1",
        }
        metadata = {
            "id": 11911272,
            "title": "WN18RR",
            "files": [file_record, dict(file_record)],
        }
        with self.assertRaises(ValueError):
            select_figshare_file(
                metadata,
                article_id=11911272,
                expected_title="WN18RR",
                filename="WN18RR.zip",
                max_download_bytes=2_000_000,
            )

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"auditable source\n")
            self.assertEqual(hashlib.sha256(b"auditable source\n").hexdigest(), sha256_file(path))

    def test_archive_member_path_validation(self):
        validate_archive_member_name("safe/subdirectory/file.txt")
        with self.assertRaises(ValueError):
            validate_archive_member_name("../../escape.txt")
        with self.assertRaises(ValueError):
            validate_archive_member_name("/absolute/path.txt")

    def test_parse_ls_remote_requires_exact_ref(self):
        sha = "1" * 40
        self.assertEqual(sha, parse_ls_remote(sha + "\trefs/heads/master\n", "refs/heads/master"))
        with self.assertRaises(ValueError):
            parse_ls_remote(sha + "\trefs/heads/main\n", "refs/heads/master")

    def test_github_raw_url_requires_pinned_commit(self):
        sha = "a" * 40
        self.assertEqual(
            "https://raw.githubusercontent.com/owner/repo/{}/data/train.txt".format(sha),
            github_raw_url("https://github.com/owner/repo.git", sha, "data/train.txt"),
        )
        with self.assertRaises(ValueError):
            github_raw_url("https://github.com/owner/repo.git", "master", "data/train.txt")

    def test_candidate_config_rejects_duplicate_ids(self):
        config = {
            "schema_version": 1,
            "status": "candidate_sources_only_not_pinned",
            "sources": [
                {"id": "same", "kind": "http_file"},
                {"id": "same", "kind": "http_file"},
            ],
        }
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
