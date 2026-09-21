# Official Results

`official_test/` contains the aggregate outputs used in the thesis:

- `per_run_metrics.tsv`: metrics for all 90 official runs;
- `paired_differences.tsv`: seed-level paired condition differences;
- `summary_statistics_and_confidence_intervals.tsv`: paired means, standard
  deviations, confidence intervals, and hypothesis-test results;
- `holm_adjusted_p_values.tsv`: family-wise Holm corrections;
- `statistical_analysis.json`: machine-readable statistical analysis.

Large checkpoints, validation score matrices, per-query probability tables,
and raw risk-coverage tables are intentionally excluded from the public code
package.
