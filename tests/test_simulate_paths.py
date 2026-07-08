import os
import sys
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
CODE_DIR = os.path.join(REPO_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import simulate_paths as sp  # noqa: E402


class SimulatePathsTests(unittest.TestCase):
    def test_rejects_wrong_number_of_shock_columns(self):
        shocks = np.zeros((4, sp.NUM_HAZARD_STEPS + 1))

        with self.assertRaisesRegex(ValueError, "normalVariates must have shape"):
            sp.simulatePaths(shocks, useRangeAccrual=True)

    def test_zero_vol_vanilla_matches_deterministic_benchmark(self):
        original_sigma_ig = sp.sigma_h_ig
        original_sigma_hy = sp.sigma_h_hy
        try:
            sp.sigma_h_ig = 0.0
            sp.sigma_h_hy = 0.0
            shocks = np.zeros((3, sp.NUM_HAZARD_STEPS))

            pv_ig, pv_hy, vanilla_ig, vanilla_hy = sp.simulatePaths(
                shocks,
                useRangeAccrual=False,
            )

            np.testing.assert_allclose(pv_ig, vanilla_ig, rtol=0.0, atol=1e-12)
            np.testing.assert_allclose(pv_hy, vanilla_hy, rtol=0.0, atol=1e-12)
        finally:
            sp.sigma_h_ig = original_sigma_ig
            sp.sigma_h_hy = original_sigma_hy

    def test_range_accrual_outputs_are_finite_and_ordered_for_fixed_seed(self):
        rng = np.random.default_rng(0)
        shocks = rng.normal(size=(16, sp.NUM_HAZARD_STEPS))

        pv_ig, pv_hy, vanilla_ig, vanilla_hy = sp.simulatePaths(
            shocks,
            useRangeAccrual=True,
        )

        self.assertEqual(pv_ig.shape, (16,))
        self.assertEqual(pv_hy.shape, (16,))
        self.assertTrue(np.all(np.isfinite(pv_ig)))
        self.assertTrue(np.all(np.isfinite(pv_hy)))
        self.assertGreater(vanilla_ig, vanilla_hy)
        self.assertGreater(np.mean(pv_ig), np.mean(pv_hy))


if __name__ == "__main__":
    unittest.main()
