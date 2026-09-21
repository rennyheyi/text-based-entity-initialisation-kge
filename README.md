# Text-Based Entity Initialisation for Knowledge Graph Link Prediction

This repository contains the code and aggregate result artifacts used for the
bachelor thesis **Text-Based Entity Initialisation for Knowledge Graph Link
Prediction: Effects on Ranking and Predictive Confidence**.

The study tests a deliberately limited intervention: external entity text is
used only to initialise trainable entity embeddings. The text encoder is frozen.
No projection layer, text-graph fusion module, alignment loss, or
condition-specific architecture is introduced.

## Experimental design

- Datasets: FB15k-237, WN18RR, and CoDEx-M
- KGE models: TransE and DistMult
- Initialisation conditions: random, correct text, and shuffled text
- Text encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Official replicates: five paired seeds
- Official matrix: 3 datasets x 2 models x 3 conditions x 5 seeds = 90 runs
- Primary outcomes: filtered MRR, calibrated multiclass NLL, and calibrated
  E-AURC

Hyperparameters were selected only from random-initialisation validation runs.
The selected configuration was then held fixed across the three conditions in
each dataset-model block. Temperature was fitted separately for each trained
run using validation queries only and was frozen before test evaluation.

## Repository structure

```text
configs/                  Frozen source, encoder, HPO, and official-run designs
data/README.md            Data sources and reproduction constraints
docs/                     Frozen protocol and protocol amendment
figures/official/         Figures generated from aggregate official results
results/official_test/    Aggregate metrics and paired statistical analysis
scripts/analysis/         Figure-generation script
scripts/slurm/            Slurm entry points used on the compute cluster
src/thesis/               Data, training, calibration, and evaluation code
tests/                    Unit and integrity tests
```

## Installation

The frozen encoder environment used Python 3.10.20. Create an isolated Python
3.10 environment and install the package:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[figures]'
```

GPU training also requires a CUDA build of PyTorch compatible with the target
cluster. The official environment recorded PyTorch 2.12.1 with CUDA 13.0.

## Tests

The test suite uses Python's standard `unittest` runner:

```bash
python -m unittest discover -s tests -v
```

The tests cover data-source locking, evidence coverage, WordNet mapping, text
encoding, negative sampling, KGE scoring, the official run design, calibration,
filtered test evaluation, and paired statistical analysis.

## Reproduction workflow

The repository implements the following sequence:

1. audit and freeze dataset and metadata sources;
2. materialise the complete benchmark vocabularies;
3. construct and audit one evidence text per entity;
4. encode and L2-normalise the text with the frozen sentence encoder;
5. select hyperparameters from random-initialisation validation runs;
6. materialise fixed shuffled-text permutations;
7. train the 90 official TransE and DistMult runs;
8. fit one scalar temperature per run on validation queries;
9. evaluate the frozen model and temperature on filtered test queries;
10. compute paired effects, confidence intervals, and Holm-adjusted tests.

The exact command-line arguments used on the cluster are preserved in
`scripts/slurm/`. Submit those scripts from the repository root. Research jobs
must run on a scheduled compute node rather than a login node.

Typical entry points are:

```bash
python -m thesis.source_audit --help
python -m thesis.data_freeze --help
python -m thesis.evidence_audit --help
python -m thesis.text_encoder --help
python -m thesis.official_design --help
python -m thesis.official_shuffle --help
python -m thesis.official_training --help
python -m thesis.official_calibration --help
python -m thesis.official_test_evaluation --help
python -m thesis.official_test_evaluation_audit --help
```

The pipeline is intentionally fail-closed. Official commands expect frozen
manifests, matching SHA-256 hashes, scheduled-execution markers, and the
intermediate artifacts recorded by earlier gates. Running only the final
training command without completing those gates will fail.

## Reproducing the thesis figures

The aggregate result files required for the three main figures are included.
Generate the PDF and PNG versions with:

```bash
python scripts/analysis/generate_thesis_figures.py
```

The outputs are written to `figures/official/`.

## Data and large artifacts

This repository does not redistribute the raw datasets, external entity
metadata, encoded embedding matrices, model checkpoints, or per-query test
artifacts. See `data/README.md` for sources and integrity requirements. The
aggregate official metrics needed to verify the reported statistical findings
are included under `results/official_test/`.

## Protocol

The pre-specified design and its documented amendment are available in
`docs/PROTOCOL.md` and `docs/PROTOCOL_AMENDMENT_001.md`. The machine-readable
official design is stored in `configs/official_experiment_design.json`.

## Code availability statement

The public repository is intended to be available at:

`https://github.com/rennyheyi/text-based-entity-initialisation-kge`

For a thesis citation, use the immutable release tag and commit hash created at
submission time rather than referring only to the moving `main` branch.
