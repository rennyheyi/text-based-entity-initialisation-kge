# Official-result figures: captions and interpretation

## Figure 1 — Primary effect forest plot

**Suggested caption.** Paired effects of correct-text entity initialization relative to random and shuffled-text controls across three datasets and two knowledge-graph embedding models. Points show the mean paired difference over five official seeds and error bars show unadjusted 95% paired *t* confidence intervals. All differences are oriented so that positive values favor correct-text initialization. Filled markers indicate rejection after Holm correction within the corresponding confirmatory family; open markers indicate non-significant comparisons.

**Use in the thesis.** This should be the main Results figure because it shows effect direction, magnitude, uncertainty, both controls, and multiplicity-corrected significance in one place.

## Figure 2 — Condition means

**Suggested caption.** Official test-set performance under random, correct-text, and shuffled-text entity initialization. Bars show means over five seeds and error bars show 95% *t* confidence intervals across seeds. Higher MRR is better, whereas lower calibrated NLL and lower calibrated E-AURC are better.

**Use in the thesis.** Use this as a descriptive companion figure. It helps readers understand the absolute scale of each metric; inferential claims should still be based on the paired tests in Figure 1.

## Figure 3 — Significance map

**Suggested caption.** Direction and Holm-adjusted significance of the correct-text effects. Green cells indicate a significant benefit, red cells a significant harm, and grey cells no statistically significant difference. Values are paired mean effects oriented so that positive values favor correct-text initialization; an asterisk denotes Holm-adjusted significance.

**Use in the thesis.** This is an effective overview for the Discussion or presentation. It immediately shows that the effects are heterogeneous rather than uniformly beneficial.

## Interpretation summary

- **Link-prediction accuracy:** Correct text is not uniformly beneficial. It improves DistMult on CoDEx-M, has a very large negative effect on DistMult on WN18RR, and produces smaller mixed effects elsewhere.
- **Calibrated NLL:** TransE improves significantly in all six confirmatory comparisons, but DistMult is mixed and is significantly worse on FB15k-237 and CoDEx-M.
- **Calibrated E-AURC:** No confirmatory comparison shows a significant improvement. Eight comparisons show a significant deterioration and four are non-significant.
- **Central conclusion:** Text semantics affect the learned representation, but semantic correctness alone does not reliably improve uncertainty quality. The effect depends on the dataset, KGE architecture, uncertainty metric, and control condition.

## Reporting caution

Do not describe the study as showing a general uncertainty improvement. The strongest defensible claim is that correct-text initialization produces systematic but heterogeneous effects, including a disconnect between probability calibration (NLL) and selective-ranking quality (E-AURC).
