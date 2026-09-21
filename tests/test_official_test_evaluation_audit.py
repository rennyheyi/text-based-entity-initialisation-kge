import unittest

from thesis.official_test_evaluation_audit import holm_adjust, paired_t_summary


class OfficialTestEvaluationAuditTest(unittest.TestCase):
    def test_paired_summary_preserves_all_five_seed_differences(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("SciPy is tested in the cluster environment")
        result = paired_t_summary([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(result["paired_differences"], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(result["mean_difference"], 3.0)
        self.assertEqual(result["median_difference"], 3.0)
        self.assertEqual(result["degrees_of_freedom"], 4)
        self.assertGreater(result["raw_two_sided_p_value"], 0.0)

    def test_zero_variance_rules_are_exact(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("SciPy is tested in the cluster environment")
        zero = paired_t_summary([0.0] * 5)
        nonzero = paired_t_summary([0.2] * 5)
        self.assertEqual(zero["raw_two_sided_p_value"], 1.0)
        self.assertIsNone(nonzero["raw_two_sided_p_value"])

    def test_holm_is_monotone_and_family_local(self):
        rows = [
            {"dataset": str(index), "model": "M", "raw_two_sided_p_value": value}
            for index, value in enumerate([0.001, 0.01, 0.03, 0.04, 0.2, 0.9])
        ]
        holm_adjust(rows)
        ordered = sorted(rows, key=lambda row: row["holm_order"])
        adjusted = [row["holm_adjusted_p_value"] for row in ordered]
        self.assertEqual(adjusted, sorted(adjusted))
        self.assertAlmostEqual(ordered[0]["holm_adjusted_p_value"], 0.006)


if __name__ == "__main__":
    unittest.main()
