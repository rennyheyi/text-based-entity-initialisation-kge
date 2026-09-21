"""Freeze and audit the complete paired 90-run official experiment matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PREFIX = "Integrating External Evidence to Reduce Uncertainty in Link Prediction"
PROTOCOL_SHA256 = "98d2eeb474d22beb703a8edd073996dda23fb5f772d71f9d7fb4244ecaa0e1dc"
AMENDMENT_SHA256 = "def04a542dcc51195825f1855cfa14f51986c9c9bfbd8ac0f9d264885593a18a"
CLAIMS_LOCK_SHA256 = "b9c95461918502934ae43d5b17261361c98e400909658821ee1293e4d694174e"
FINAL_HYPERPARAMETERS_SHA256 = (
    "4c56649068a80a986a5fec6198c70d97baa7ac6b28bc3a02a5f35e81f9b9b869"
)
HPO_DESIGN_SHA256 = "3487938ccafb1ad4f7f4a0be037a717ce8b8769606bb652d1ccdc66250f1bb93"
EXPECTED_OFFICIAL_DESIGN_SHA256 = (
    "48a82e2bf835942f1d76ec40ce0b9b52499aa3a0c16f40ab31c04652215a6ee5"
)
EXPECTED_RUNS_SHA256 = "92f65f088e983836ec08ad2ef316ec2b3f2946d42d80264fb0801eeac32d5107"
EXPECTED_PAIRED_BLOCKS_SHA256 = (
    "74c23ebf952eba4f33fa496b7765bb5afaa36b8eb9f2b35677e9a84530635b83"
)
EXPECTED_SHUFFLES_SHA256 = (
    "2a14622c189ee462d850289fecbfec2e723a4c76e14567a9f92814e396eb7a71"
)

DATASETS = ("FB15k-237", "WN18RR", "CoDEx-M")
MODELS = ("TransE", "DistMult")
CONDITIONS = ("random", "correct_text", "shuffled_text")
OFFICIAL_SEEDS = (3239737201, 3693500706, 4270284857, 3089462143, 3051266068)
DATASET_RESOURCES = {
    "FB15k-237": {
        "dataset_manifest": "data/manifests/FB15k-237_dataset.json",
        "dataset_manifest_sha256": "9217125b5e41758f6cb27ac72bfd82d886d131fe060e559d868c48350b3ff386",
        "text_embedding_manifest": "data/manifests/FB15k-237_text_embeddings.json",
        "text_embedding_manifest_sha256": "e8ad0abc3015aa79ab1e1a49c14d3d0db9a129b9d9b33b9773a0c8c6354ea47e",
        "normalised_embeddings_sha256": "81b1ff628ed7a0bf767be49eab74ee7fede04537cdbef4d52d19187e39b84388",
    },
    "WN18RR": {
        "dataset_manifest": "data/manifests/WN18RR_dataset.json",
        "dataset_manifest_sha256": "a108bd60cfef76db2125c86782dc11c7619bd9c9310717d9630e71fc0a53d6a2",
        "text_embedding_manifest": "data/manifests/WN18RR_text_embeddings.json",
        "text_embedding_manifest_sha256": "1c27e835af14a8ee625bcb85dcaf891dfbe9c3fa475ddae0140f97750ba88651",
        "normalised_embeddings_sha256": "79ca4ffd4f36f9059ce2ee43a6f5a1828dcc4d21eae965f2fc4665668140e555",
    },
    "CoDEx-M": {
        "dataset_manifest": "data/manifests/CoDEx-M_dataset.json",
        "dataset_manifest_sha256": "652b61286da4b96032e70f55bed7e11dcd47a76f1547bd84f3fd84577d086277",
        "text_embedding_manifest": "data/manifests/CoDEx-M_text_embeddings.json",
        "text_embedding_manifest_sha256": "9cbfc7fb1ec38968edb055058fab40d4dce963baf9221c71965d59e982d359e2",
        "normalised_embeddings_sha256": "43c158ec89399c6e0532e38ce7baa87f10eb9ef62092e899f77a912bc19d9438",
    },
}
COMMON_NAMESPACES = (
    "relation_initialisation",
    "batch_order",
    "negative_sampling",
    "data_loader_workers",
)
RANDOM_NAMESPACE = "random_entity_initialisation"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class OfficialDesignError(RuntimeError):
    """Raised when the Gate E design cannot be frozen exactly."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OfficialDesignError("JSON root is not an object: {}".format(path))
    return value


def require_file_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise OfficialDesignError(label + " is missing")
    observed = sha256_file(path)
    if observed != expected:
        raise OfficialDesignError(
            "{} hash mismatch: expected {}, observed {}".format(
                label, expected, observed
            )
        )


def derivation(*parts: str) -> Dict[str, Any]:
    input_value = PREFIX + "|" + "|".join(parts)
    digest = hashlib.sha256(input_value.encode("utf-8")).hexdigest()
    return {
        "input": input_value,
        "sha256": digest,
        "seed": int.from_bytes(bytes.fromhex(digest[:8]), "big", signed=False),
    }


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def validate_claims_lock(claims: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": 1,
        "status": "official_claims_frozen_pending_official_design",
        "official_result": False,
        "approved_by_protocol_owner": True,
        "approval_date": "2026-08-05",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_amendment_001_sha256": AMENDMENT_SHA256,
        "final_hyperparameters_sha256": FINAL_HYPERPARAMETERS_SHA256,
        "hpo_design_sha256": HPO_DESIGN_SHA256,
    }
    for key, expected in exact.items():
        if claims.get(key) != expected:
            raise OfficialDesignError("Claims lock differs in " + key)

    boundary = claims.get("information_boundary")
    if not isinstance(boundary, dict):
        raise OfficialDesignError("Claims information boundary is missing")
    for key in (
        "test_data_accessed",
        "correct_text_results_accessed",
        "shuffled_text_results_accessed",
        "random_hpo_results_used_to_set_sesoi_or_guardrail",
    ):
        if boundary.get(key) is not False:
            raise OfficialDesignError("Claims information boundary fails for " + key)

    hypotheses = claims.get("directional_hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) != 6:
        raise OfficialDesignError("Exactly six directional hypotheses are required")
    hypothesis_keys = {
        (row.get("contrast_id"), row.get("metric"), row.get("direction"))
        for row in hypotheses
        if isinstance(row, dict)
    }
    expected_keys = {
        (contrast, metric, direction)
        for contrast in (
            "correct_text_vs_random",
            "correct_text_vs_shuffled_text",
        )
        for metric, direction in (
            ("combined_filtered_test_mrr", "higher"),
            ("calibrated_test_multiclass_nll", "lower"),
            ("calibrated_test_eaurc", "lower"),
        )
    }
    if hypothesis_keys != expected_keys:
        raise OfficialDesignError("Directional hypothesis set is not frozen exactly")

    sesoi = claims.get("smallest_effects_of_interest")
    if not isinstance(sesoi, dict):
        raise OfficialDesignError("SESOI decision is missing")
    for metric in ("calibrated_test_multiclass_nll", "calibrated_test_eaurc"):
        value = sesoi.get(metric)
        if not isinstance(value, dict) or value.get("status") != "not_numerically_defined":
            raise OfficialDesignError("Unexpected SESOI state for " + metric)

    guardrail = claims.get("mrr_accuracy_guardrail")
    if (
        not isinstance(guardrail, dict)
        or guardrail.get("minimum_acceptable_mean_difference") != -0.01
        or guardrail.get("scale") != "absolute_mrr"
    ):
        raise OfficialDesignError("MRR guardrail is not frozen at -0.01")
    claim_language = claims.get("claim_language")
    if (
        not isinstance(claim_language, dict)
        or claim_language.get("supported_uncertainty_improvement_permitted") is not False
    ):
        raise OfficialDesignError("Strong uncertainty claim prohibition is missing")


def validate_final_hyperparameters(final: Mapping[str, Any]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    exact = {
        "schema_version": 1,
        "status": "final_hyperparameters_frozen_pending_official_experiment_design",
        "official_result": False,
        "test_data_accessed": False,
        "text_conditions_accessed": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "selection_condition": "random_initialisation_only",
    }
    for key, expected in exact.items():
        if final.get(key) != expected:
            raise OfficialDesignError("Final hyperparameters differ in " + key)
    units = final.get("units")
    if not isinstance(units, list) or len(units) != 6:
        raise OfficialDesignError("Final hyperparameters require six units")
    by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for unit in units:
        if not isinstance(unit, dict):
            raise OfficialDesignError("Final hyperparameter unit is invalid")
        key = (unit.get("dataset"), unit.get("model"))
        if key not in {(dataset, model) for dataset in DATASETS for model in MODELS}:
            raise OfficialDesignError("Unknown final hyperparameter unit")
        if key in by_key:
            raise OfficialDesignError("Duplicate final hyperparameter unit")
        epochs = unit.get("training_epochs")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 200:
            raise OfficialDesignError("Official epoch budget is invalid")
        parameters = unit.get("parameters")
        if not isinstance(parameters, dict):
            raise OfficialDesignError("Official parameters are missing")
        configuration_sha = unit.get("candidate_configuration_sha256")
        if not isinstance(configuration_sha, str) or not SHA256_RE.fullmatch(configuration_sha):
            raise OfficialDesignError("Candidate configuration hash is invalid")
        by_key[key] = unit
    if len(by_key) != len(DATASETS) * len(MODELS):
        raise OfficialDesignError("Final unit coverage is incomplete")
    return by_key


def _configuration_record(unit: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": unit["candidate_id"],
        "candidate_configuration_sha256": unit["candidate_configuration_sha256"],
        "training_epochs": unit["training_epochs"],
        "parameters": unit["parameters"],
    }


def build_official_design(
    final: Mapping[str, Any], claims: Mapping[str, Any]
) -> Dict[str, Any]:
    validate_claims_lock(claims)
    units = validate_final_hyperparameters(final)

    replicate_derivations = []
    for replicate, expected_seed in enumerate(OFFICIAL_SEEDS, 1):
        value = derivation("official", str(replicate))
        if value["seed"] != expected_seed:
            raise OfficialDesignError("Official replicate seed derivation changed")
        replicate_derivations.append(
            {"replicate": replicate, "seed": expected_seed, **value}
        )

    shuffle_artifacts: List[Dict[str, Any]] = []
    shuffle_by_key: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for dataset in DATASETS:
        for replicate in range(1, 6):
            seed = derivation("stream", "official", str(replicate), dataset, "shuffle")
            record = {
                "shuffle_id": "{}-official-r{}-shuffle".format(slug(dataset), replicate),
                "dataset": dataset,
                "official_replicate": replicate,
                "derivation": seed,
                "shared_across_models": list(MODELS),
                "permutation_artifact": (
                    "artifacts/embeddings/{}/official_shuffles/replicate_{:02d}.npy".format(
                        dataset, replicate
                    )
                ),
                "permutation_manifest": (
                    "data/manifests/{}_official_shuffle_replicate_{:02d}.json".format(
                        dataset, replicate
                    )
                ),
                "required_invariants": [
                    "bijection",
                    "derangement",
                    "zero_equal_evidence_hash_assignments",
                    "zero_equal_vector_hash_assignments",
                    "correct_and_shuffled_vector_multisets_identical",
                ],
            }
            shuffle_artifacts.append(record)
            shuffle_by_key[(dataset, replicate)] = record

    runs: List[Dict[str, Any]] = []
    paired_blocks: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for model in MODELS:
            unit = units[(dataset, model)]
            selected = _configuration_record(unit)
            for replicate, official_seed in enumerate(OFFICIAL_SEEDS, 1):
                block_id = "{}-{}-official-r{:02d}".format(
                    slug(dataset), slug(model), replicate
                )
                common_streams = {
                    namespace: derivation(
                        "stream",
                        "official",
                        str(replicate),
                        dataset,
                        model,
                        namespace,
                    )
                    for namespace in COMMON_NAMESPACES
                }
                random_stream = derivation(
                    "stream",
                    "official",
                    str(replicate),
                    dataset,
                    model,
                    RANDOM_NAMESPACE,
                )
                run_ids = []
                array_indices = []
                for condition in CONDITIONS:
                    array_index = len(runs)
                    run_id = block_id + "-" + condition.replace("_", "-")
                    streams: Dict[str, Any] = dict(common_streams)
                    if condition == "random":
                        streams[RANDOM_NAMESPACE] = random_stream
                    if condition == "shuffled_text":
                        streams["shuffle"] = shuffle_by_key[(dataset, replicate)][
                            "derivation"
                        ]
                    if condition == "random":
                        entity_initialisation = {
                            "kind": "random_unit_norm_float32",
                            "seed_derivation": random_stream,
                        }
                    elif condition == "correct_text":
                        entity_initialisation = {
                            "kind": "frozen_l2_normalised_correct_text",
                            "text_embedding_manifest": DATASET_RESOURCES[dataset][
                                "text_embedding_manifest"
                            ],
                            "text_embedding_manifest_sha256": DATASET_RESOURCES[dataset][
                                "text_embedding_manifest_sha256"
                            ],
                            "normalised_embeddings_sha256": DATASET_RESOURCES[dataset][
                                "normalised_embeddings_sha256"
                            ],
                        }
                    else:
                        entity_initialisation = {
                            "kind": "frozen_l2_normalised_shuffled_text",
                            "source_text_embedding_manifest": DATASET_RESOURCES[dataset][
                                "text_embedding_manifest"
                            ],
                            "source_text_embedding_manifest_sha256": DATASET_RESOURCES[
                                dataset
                            ]["text_embedding_manifest_sha256"],
                            "source_normalised_embeddings_sha256": DATASET_RESOURCES[
                                dataset
                            ]["normalised_embeddings_sha256"],
                            "shuffle_id": shuffle_by_key[(dataset, replicate)][
                                "shuffle_id"
                            ],
                        }
                    row = {
                        "array_index": array_index,
                        "run_id": run_id,
                        "paired_block_id": block_id,
                        "dataset": dataset,
                        "model": model,
                        "condition": condition,
                        "official_replicate": replicate,
                        "official_seed": official_seed,
                        "selected_configuration": selected,
                        "dataset_manifest": DATASET_RESOURCES[dataset][
                            "dataset_manifest"
                        ],
                        "dataset_manifest_sha256": DATASET_RESOURCES[dataset][
                            "dataset_manifest_sha256"
                        ],
                        "entity_initialisation": entity_initialisation,
                        "rng_streams": streams,
                        "shuffle_id": (
                            shuffle_by_key[(dataset, replicate)]["shuffle_id"]
                            if condition == "shuffled_text"
                            else None
                        ),
                        "expected_checkpoint": (
                            "runs/official/{}/{}/replicate_{:02d}/{}/checkpoint.pt".format(
                                dataset, model, replicate, condition
                            )
                        ),
                        "expected_run_report": (
                            "reports/official/{}/{}/replicate_{:02d}/{}.json".format(
                                dataset, model, replicate, condition
                            )
                        ),
                    }
                    row["run_configuration_sha256"] = sha256_bytes(
                        canonical_json_bytes(row)
                    )
                    runs.append(row)
                    run_ids.append(run_id)
                    array_indices.append(array_index)
                paired_blocks.append(
                    {
                        "paired_block_id": block_id,
                        "dataset": dataset,
                        "model": model,
                        "official_replicate": replicate,
                        "official_seed": official_seed,
                        "conditions": list(CONDITIONS),
                        "run_ids": run_ids,
                        "array_indices": array_indices,
                        "shared_rng_streams": common_streams,
                        "random_entity_initialisation": random_stream,
                        "shuffle_id": shuffle_by_key[(dataset, replicate)]["shuffle_id"],
                    }
                )

    if len(runs) != 90 or len(paired_blocks) != 30 or len(shuffle_artifacts) != 15:
        raise OfficialDesignError("Official design cardinality is invalid")

    selected_units = [
        {
            "dataset": dataset,
            "model": model,
            **_configuration_record(units[(dataset, model)]),
        }
        for dataset in DATASETS
        for model in MODELS
    ]
    design = {
        "schema_version": 1,
        "status": "official_experiment_design_frozen_pending_shuffle_materialisation_and_training",
        "official_result": False,
        "test_data_accessed": False,
        "text_condition_results_accessed": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_amendment_001_sha256": AMENDMENT_SHA256,
        "official_claims_lock_sha256": CLAIMS_LOCK_SHA256,
        "final_hyperparameters_sha256": FINAL_HYPERPARAMETERS_SHA256,
        "hpo_design_sha256": HPO_DESIGN_SHA256,
        "matrix_shape": {
            "datasets": 3,
            "models": 2,
            "conditions": 3,
            "official_replicates": 5,
            "paired_blocks": 30,
            "official_runs": 90,
        },
        "dataset_order": list(DATASETS),
        "model_order": list(MODELS),
        "condition_order": list(CONDITIONS),
        "official_replicate_derivations": replicate_derivations,
        "fixed_training": final["fixed_training"],
        "dataset_resources": DATASET_RESOURCES,
        "selected_units": selected_units,
        "shuffle_artifacts": shuffle_artifacts,
        "shuffle_artifacts_sha256": sha256_bytes(
            canonical_json_bytes(shuffle_artifacts)
        ),
        "paired_blocks": paired_blocks,
        "paired_blocks_sha256": sha256_bytes(canonical_json_bytes(paired_blocks)),
        "runs": runs,
        "runs_sha256": sha256_bytes(canonical_json_bytes(runs)),
        "execution_constraints": {
            "same_selected_configuration_across_conditions": True,
            "same_relation_initialisation_within_paired_block": True,
            "same_batch_order_within_paired_block": True,
            "same_negative_sampling_within_paired_block": True,
            "same_data_loader_worker_seed_within_paired_block": True,
            "shuffle_shared_across_models_by_dataset_replicate": True,
            "condition_specific_early_stopping": False,
            "official_random_runs_retrained_from_t0": True,
            "test_evaluation_before_checkpoint_freeze": False,
        },
    }
    return design


def validate_design_invariants(design: Mapping[str, Any]) -> None:
    runs = design.get("runs")
    blocks = design.get("paired_blocks")
    shuffles = design.get("shuffle_artifacts")
    if not isinstance(runs, list) or len(runs) != 90:
        raise OfficialDesignError("Official design does not contain 90 runs")
    if not isinstance(blocks, list) or len(blocks) != 30:
        raise OfficialDesignError("Official design does not contain 30 paired blocks")
    if not isinstance(shuffles, list) or len(shuffles) != 15:
        raise OfficialDesignError("Official design does not contain 15 shuffles")
    if [row.get("array_index") for row in runs] != list(range(90)):
        raise OfficialDesignError("Official array indices are not contiguous")

    by_block: Dict[str, List[Mapping[str, Any]]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise OfficialDesignError("Official run row is invalid")
        row_without_hash = dict(run)
        stored_row_hash = row_without_hash.pop("run_configuration_sha256", None)
        if stored_row_hash != sha256_bytes(canonical_json_bytes(row_without_hash)):
            raise OfficialDesignError("Official run configuration hash changed")
        by_block.setdefault(str(run.get("paired_block_id")), []).append(run)
    for block_runs in by_block.values():
        if {row.get("condition") for row in block_runs} != set(CONDITIONS):
            raise OfficialDesignError("A paired block does not contain all conditions")
        for namespace in COMMON_NAMESPACES:
            seeds = {
                row["rng_streams"][namespace]["seed"] for row in block_runs
            }
            if len(seeds) != 1:
                raise OfficialDesignError("Paired common RNG stream differs")
        random_rows = [row for row in block_runs if row["condition"] == "random"]
        if RANDOM_NAMESPACE not in random_rows[0]["rng_streams"]:
            raise OfficialDesignError("Random condition lacks its entity stream")
        for row in block_runs:
            if row["condition"] != "random" and RANDOM_NAMESPACE in row["rng_streams"]:
                raise OfficialDesignError("Text condition contains random entity stream")
            if ("shuffle" in row["rng_streams"]) != (
                row["condition"] == "shuffled_text"
            ):
                raise OfficialDesignError("Shuffle stream condition boundary changed")

    shuffle_seeds: Dict[Tuple[str, int], int] = {}
    for row in runs:
        if row["condition"] != "shuffled_text":
            continue
        key = (row["dataset"], row["official_replicate"])
        seed = row["rng_streams"]["shuffle"]["seed"]
        previous = shuffle_seeds.setdefault(key, seed)
        if previous != seed:
            raise OfficialDesignError("Shuffle seed differs across models")

    if design.get("runs_sha256") != sha256_bytes(canonical_json_bytes(runs)):
        raise OfficialDesignError("Run matrix hash is not reproducible")
    if design.get("runs_sha256") != EXPECTED_RUNS_SHA256:
        raise OfficialDesignError("Run matrix differs from the approved Gate E lock")
    if design.get("paired_blocks_sha256") != sha256_bytes(canonical_json_bytes(blocks)):
        raise OfficialDesignError("Paired block hash is not reproducible")
    if design.get("paired_blocks_sha256") != EXPECTED_PAIRED_BLOCKS_SHA256:
        raise OfficialDesignError("Paired blocks differ from the approved Gate E lock")
    if design.get("shuffle_artifacts_sha256") != sha256_bytes(
        canonical_json_bytes(shuffles)
    ):
        raise OfficialDesignError("Shuffle specification hash is not reproducible")
    if design.get("shuffle_artifacts_sha256") != EXPECTED_SHUFFLES_SHA256:
        raise OfficialDesignError("Shuffle specifications differ from the Gate E lock")


def require_scheduled_execution(scratch: Path) -> Dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise OfficialDesignError("Gate E freeze must run inside a Slurm allocation")
    if not scratch.is_dir():
        raise OfficialDesignError("Job-local scratch directory is missing")
    return {
        "execution_context": "slurm",
        "scheduler_job_id": job_id,
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "memory_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "hostname": socket.getfqdn(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scratch_path": str(scratch),
    }


def git_state(project_root: Path) -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not COMMIT_RE.fullmatch(commit):
        raise OfficialDesignError("Git commit is not a full digest")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise OfficialDesignError("Tracked worktree is not clean")
    return {"commit": commit, "tracked_worktree_clean": True}


def write_new_or_identical(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise OfficialDesignError("Refusing to overwrite different artifact: {}".format(path))
        return
    path.write_bytes(content)


def freeze_design(
    *,
    protocol_path: Path,
    amendment_path: Path,
    claims_path: Path,
    final_path: Path,
    hpo_design_path: Path,
    project_root: Path,
    scratch: Path,
    output_design: Path,
    output_report: Path,
) -> Dict[str, Any]:
    execution = require_scheduled_execution(scratch)
    require_file_sha256(protocol_path, PROTOCOL_SHA256, "Frozen protocol")
    require_file_sha256(amendment_path, AMENDMENT_SHA256, "Protocol amendment")
    require_file_sha256(claims_path, CLAIMS_LOCK_SHA256, "Official claims lock")
    require_file_sha256(
        final_path, FINAL_HYPERPARAMETERS_SHA256, "Final hyperparameters"
    )
    require_file_sha256(hpo_design_path, HPO_DESIGN_SHA256, "Frozen HPO design")
    claims = read_json_object(claims_path)
    final = read_json_object(final_path)
    design = build_official_design(final, claims)
    validate_design_invariants(design)
    design_bytes = canonical_json_bytes(design)
    if sha256_bytes(design_bytes) != EXPECTED_OFFICIAL_DESIGN_SHA256:
        raise OfficialDesignError("Official design differs from the approved Gate E lock")
    git = git_state(project_root)
    write_new_or_identical(output_design, design_bytes)
    report = {
        "schema_version": 1,
        "status": "gate_e_official_design_frozen_pending_shuffle_materialisation_and_training",
        "official_result": False,
        "test_data_accessed": False,
        "text_condition_results_accessed": False,
        "execution": execution,
        "git": git,
        "inputs": {
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_amendment_001_sha256": AMENDMENT_SHA256,
            "official_claims_lock_sha256": CLAIMS_LOCK_SHA256,
            "final_hyperparameters_sha256": FINAL_HYPERPARAMETERS_SHA256,
            "hpo_design_sha256": HPO_DESIGN_SHA256,
        },
        "official_design": {
            "path": str(output_design.relative_to(project_root)),
            "sha256": sha256_bytes(design_bytes),
            "runs_sha256": design["runs_sha256"],
            "paired_blocks_sha256": design["paired_blocks_sha256"],
            "shuffle_artifacts_sha256": design["shuffle_artifacts_sha256"],
            "official_runs": 90,
            "paired_blocks": 30,
            "shuffle_artifacts": 15,
        },
    }
    write_new_or_identical(output_report, canonical_json_bytes(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--claims-lock", type=Path, required=True)
    parser.add_argument("--final-hyperparameters", type=Path, required=True)
    parser.add_argument("--hpo-design", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output-design", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = freeze_design(
            protocol_path=args.protocol.resolve(),
            amendment_path=args.amendment.resolve(),
            claims_path=args.claims_lock.resolve(),
            final_path=args.final_hyperparameters.resolve(),
            hpo_design_path=args.hpo_design.resolve(),
            project_root=args.project_root.resolve(),
            scratch=args.scratch.resolve(),
            output_design=args.output_design.resolve(),
            output_report=args.output_report.resolve(),
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "gate_e_official_design_freeze_failed",
            "official_result": False,
            "test_data_accessed": False,
            "text_condition_results_accessed": False,
            "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        try:
            write_new_or_identical(
                args.output_report.resolve(), canonical_json_bytes(failure)
            )
        except Exception:
            pass
        raise
    print("Official 90-run design written to {}".format(args.output_design.resolve()))
    print("Gate E freeze recorded in {}".format(args.output_report.resolve()))
    print(json.dumps(report["official_design"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
