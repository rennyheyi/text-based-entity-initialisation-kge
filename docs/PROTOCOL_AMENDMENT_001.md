# Protocol Amendment 001: HPO execution deviation and Gate E claim freeze

Working title: **Integrating External Evidence to Reduce Uncertainty in Link Prediction**

Status: Approved before any official-condition training or test evaluation.

Approval date: 2026-08-05.

## Preserved protocol baseline

This amendment does not edit or replace `docs/PROTOCOL.md`.

The preserved protocol baseline is the 1,903-line file with SHA-256:

```text
98d2eeb474d22beb703a8edd073996dda23fb5f772d71f9d7fb4244ecaa0e1dc
```

The original protocol remains the governing document except for the explicit HPO execution differences recorded below. This amendment must be cited together with that baseline whenever the completed experiment is described as protocol-driven.

## Timing and information boundary

The deviation was identified and documented after Random-only HPO had completed but before generation of the official 90-run matrix, official-condition training, or test evaluation.

The frozen HPO artifacts record:

- `test_data_accessed: false`;
- `text_conditions_accessed: false`;
- selection using only combined filtered validation MRR from Random-initialisation runs.

No Correct-text result, Shuffled-text result, test ranking, test probability, or test uncertainty metric was available when this amendment and the Gate E claim rules were approved.

## Recorded HPO deviation

Decision 10 of the preserved protocol specifies an 8-to-2-to-1 HPO funnel with validation every 20 epochs during Stage 3. The frozen executable HPO design instead used the following design:

| Stage | Preserved protocol | Executed frozen design |
|---|---:|---:|
| Stage 1 candidates per dataset-model unit | 8 | 12 |
| Stage 1 advancing candidates | 2 | 4 |
| Stage 2 candidates | 2 | 4 |
| Stage 2 replicates per candidate | 2 | 2 |
| Stage 2 advancing candidates | 1 | 2 |
| Stage 3 candidates | 1 | 2 |
| Stage 3 replicates per candidate | 3 | 3 |
| Stage 3 validation interval | 20 epochs | 10 epochs |
| Stage 3 selection target | epoch for one frozen candidate | candidate-epoch pair |

The executed trajectory budget was therefore:

```text
12 Stage 1 + (4 x 2) Stage 2 + (2 x 3) Stage 3
= 26 tuning trajectories per dataset-model unit
= 156 tuning trajectories across six units
```

Together with the unchanged 90-run official matrix, the planned total is 246 result-producing training trajectories, excluding preflights and excluded duplicate or cancelled attempts.

The executed HPO design is identified by:

```text
configs/hpo_design.json
SHA-256 3487938ccafb1ad4f7f4a0be037a717ce8b8769606bb652d1ccdc66250f1bb93
```

The final selected configurations are identified by:

```text
configs/final_hyperparameters.json
SHA-256 4c56649068a80a986a5fec6198c70d97baa7ac6b28bc3a02a5f35e81f9b9b869
```

## Disposition

The completed Random-only HPO is retained rather than restarted.

This is acceptable for the remaining condition comparison because:

1. the executed design and candidate identities were frozen before their corresponding HPO runs;
2. selection remained confined to Random-initialisation validation MRR;
3. the same selected configuration and epoch budget will be applied unchanged to Random, Correct-text, and Shuffled-text official runs;
4. all canonical HPO runs and excluded attempts were mechanically audited;
5. neither text-condition results nor test results influenced the expanded funnel or the final selections.

This disposition does not make the deviation disappear. The thesis must describe the executed 12-to-4-to-2 design and must not state that the HPO trajectory counts or Stage 3 selection rule were executed exactly as originally specified in Decision 10.

## Gate E directional hypotheses

For each of the six dataset-model units, the following directional hypotheses are frozen. The paired tests remain two-sided as required by Decision 11.

1. Correct-text has higher combined filtered test MRR than Random.
2. Correct-text has higher combined filtered test MRR than Shuffled-text.
3. Correct-text has lower calibrated test multiclass NLL than Random.
4. Correct-text has lower calibrated test multiclass NLL than Shuffled-text.
5. Correct-text has lower calibrated test E-AURC than Random.
6. Correct-text has lower calibrated test E-AURC than Shuffled-text.

The Shuffled-text versus Random contrast remains a pre-specified secondary mechanistic contrast without a directional confirmatory hypothesis.

## Smallest effects of interest

No numerical smallest effect of interest is defined for calibrated NLL or calibrated E-AURC.

The literature supports these outcomes as proper-scoring and selective-prediction diagnostics, but it does not establish a transferable minimum important difference for temperature-calibrated, filtered knowledge-graph link prediction. A threshold chosen from the completed Random HPO results would also violate the intended information boundary.

Consequently, the phrase `supported uncertainty improvement` is prohibited for this experiment, regardless of p-values. Results must use the more limited interpretation categories already defined in Decision 11:

- `directionally consistent evidence`;
- `partial evidence`;
- `mixed evidence`;
- `no primary evidence`.

All natural-scale paired differences, five seed-level values, confidence intervals, raw p-values, Holm-adjusted p-values, and original condition means must still be reported.

## MRR accuracy guardrail

The pre-frozen MRR guardrail is an absolute mean paired difference of `-0.01`.

For a contrast in which the first-named condition is Correct-text:

```text
mean(Correct-text MRR - control MRR) >= -0.01
```

passes the guardrail. A value below `-0.01` requires the result to be described as an accuracy-uncertainty trade-off if uncertainty outcomes improve.

This one-point absolute MRR tolerance is an investigator-defined interpretation boundary, not a literature-derived universal equivalence or non-inferiority margin. Passing it does not prove practical equivalence or non-inferiority.

## Literature basis and limitation

The directional ranking hypotheses are motivated by prior work showing that entity descriptions and pretrained language representations can improve knowledge-graph completion. Temperature scaling motivates the validation-only calibration procedure, while risk-coverage work motivates E-AURC as a selective-prediction outcome. None of these sources supplies a KGE-specific smallest important NLL or E-AURC difference.

- Xie et al. (2016), *Representation Learning of Knowledge Graphs with Entity Descriptions*, DOI `10.1609/aaai.v30i1.10329`.
- Yao et al. (2019), *KG-BERT: BERT for Knowledge Graph Completion*, arXiv `1909.03193`.
- Guo et al. (2017), *On Calibration of Modern Neural Networks*, PMLR 70:1321-1330.
- Ding et al. (2020), *Revisiting the Evaluation of Uncertainty Estimation and Its Application to Explore Model Complexity-Uncertainty Trade-Off*, DOI `10.1109/CVPRW50498.2020.00010`.

## Approval record

- Amendment 001, the six directional hypotheses, the absence of numerical NLL and E-AURC SESOIs, and the absolute `-0.01` MRR guardrail were approved by the protocol owner on 2026-08-05.
