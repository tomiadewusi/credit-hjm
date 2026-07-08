# Technical Report

This directory contains the LaTeX source and compiled technical report for the HJM-style hazard-rate Monte Carlo project.

## Files

- `main.tex`: top-level LaTeX source for the report.
- `sections/`: report sections included by `main.tex`.
- `references.bib`: bibliography entries used by the report.
- `figures/`: figures referenced by the report.
- `hjm_hazard_rate_cln_monte_carlo.pdf`: final compiled report for public review.

## Build

From this directory, rebuild the report with:

```bash
latexmk -pdf main.tex
```

Build outputs are written to `build/` when using the local LaTeX build configuration.
