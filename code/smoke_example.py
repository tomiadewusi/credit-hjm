"""Lightweight smoke run for the CLN path simulator.

This script is intended as a quick import-and-run check, not as a production
pricing experiment. It uses a tiny fixed random sample so it finishes quickly.
"""

import numpy as np

from simulate_paths import NUM_HAZARD_STEPS, simulatePaths


def main():
    rng = np.random.default_rng(0)
    shocks = rng.normal(size=(8, NUM_HAZARD_STEPS))

    pv_ig, pv_hy, vanilla_ig, vanilla_hy = simulatePaths(
        shocks,
        useRangeAccrual=True,
    )

    print(f"IG range-accrual mean PV: {np.mean(pv_ig):.6f}")
    print(f"HY range-accrual mean PV: {np.mean(pv_hy):.6f}")
    print(f"IG deterministic vanilla PV: {vanilla_ig:.6f}")
    print(f"HY deterministic vanilla PV: {vanilla_hy:.6f}")


if __name__ == "__main__":
    main()
