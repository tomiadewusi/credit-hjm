READ ME – Credit Simulation Project (Plain Text Version)

This project contains the full codebase and notebooks for a credit derivative simulation framework. The focus is on pricing Credit-Linked Notes (CLNs) using deterministic hazard curves and a simplified HJM-style stochastic hazard-rate model. The repository includes pathwise Monte Carlo, vectorised simulation, variance reduction experiments, and figure generation.

-------------------------------------------------------------------------------
1. Python Modules
-------------------------------------------------------------------------------

simulate_paths.py
- Main Monte Carlo function used throughout the project.
- Simulates hazard-rate paths, survival probabilities, discounting, coupon legs, recovery legs, and range-accrual logic.
- Designed to be pathwise and memory-safe.
- Supports use inside CRN, control variate, and antithetic variance-reduction wrappers.
- Intended to be refactored into a pure JAX version in future work.

fig_style.py
- Centralized styling for all figures used in the report.
- Provides IEEE-style layout, text rendering setup, and helper functions.
- Used by all notebooks to generate consistent, publication-quality plots.

-------------------------------------------------------------------------------
2. Notebooks
-------------------------------------------------------------------------------

hjm_example_ram_safe.ipynb
- Memory-safe pathwise implementation of the HJM hazard-rate model.
- Evolves forward intensities on a coarse grid.
- Generates survival probabilities, discount factors, and CLN cashflows.
- Uses a pathwise implementation to keep memory usage bounded.

hjm_example_vectorised.ipynb
- Vectorised implementation of the same hazard-rate evolution.
- Much faster but memory intensive.
- Produces short-rate, hazard-rate, discount-factor, and survival-probability plots.
- Basis for several final report figures.

hjm_variance_reduction.ipynb
- Implements Monte Carlo variance reduction: antithetic sampling, common random numbers, and control variates.
- Uses a shared Z matrix to keep comparisons fair.
- Estimates control variate correlations using a subsample of paths.
- Produces grid plots of PV estimates with confidence intervals for IG and HY names.

section_two_plots.ipynb
- Notebook dedicated to producing the final plots used in the report’s Section Two.
- Generates PDFs of credit spreads, survival curves, and 2x2 term structure figures.
- Builds all figures using fig_style.py to ensure consistent formatting.

-------------------------------------------------------------------------------
3. Supporting Figures (PDF Files)
-------------------------------------------------------------------------------

credit-spreads-bps.pdf
- IG and HY credit spread term structures plotted in basis points.

orig-vs-hjm-survival.pdf
- Comparison of analytical survival curves vs HJM-simulated survival curves for IG and HY.

term-structures-2x2.pdf
- Four-panel plot showing short rate, hazard rate, discount factor, and survival probability.

-------------------------------------------------------------------------------
4. Running the Project
-------------------------------------------------------------------------------

1. Install numpy, matplotlib, and scipy.
2. Open any notebook and run it top-to-bottom.
3. simulate_paths.py is the core function for CLN pricing.
4. section_two_plots.ipynb is the best entry point for generating final figures.
5. hjm_variance_reduction.ipynb is used for all variance reduction experiments.

-------------------------------------------------------------------------------
5. Notes
-------------------------------------------------------------------------------

- Figures in the PDF files were generated through the notebooks using fig_style.py.
- Future extensions include re-writing the full simulation engine in JAX for GPU execution and adding importance sampling to the variance-reduction set.
-------------------------------------------------------------------------------
