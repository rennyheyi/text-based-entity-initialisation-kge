# Data

Raw benchmark data, entity metadata, encoded text matrices, checkpoints, and
per-query prediction artifacts are not redistributed in this repository.

The data pipeline uses the sources declared in
`configs/source_candidates.json` and the deterministic materialisation rules in
`configs/materialization_plan.json`. The three benchmark datasets are:

- FB15k-237, with entity names and descriptions from the pinned KG-BERT
  metadata source;
- WN18RR, with entity mappings from the pinned KG-BERT artifact and lexical
  evidence from Princeton WordNet 3.0;
- CoDEx-M, with benchmark triples and English entity metadata from the official
  CoDEx repository.

The evidence encoder is fixed in `configs/encoder_lock.json`. It uses
`sentence-transformers/all-MiniLM-L6-v2` at revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. Raw 384-dimensional vectors are
stored separately and L2-normalised before entity initialisation.

The preparation sequence is implemented by the following modules:

1. `thesis.source_audit`
2. `thesis.data_freeze`
3. `thesis.evidence_audit`
4. `thesis.text_encoder`

The official design in `configs/official_experiment_design.json` records the
expected manifest paths and SHA-256 hashes. A regenerated artifact must match
those locked hashes before it can be treated as the artifact used for the
reported experiment.
