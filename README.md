# HJM Hazard-Rate Monte Carlo for Credit-Linked Notes

This repository is a compact research prototype for pricing vanilla and
range-accrual credit-linked notes (CLNs) under stochastic hazard-rate dynamics.
It combines a reduced-form credit setup, an HJM-style forward hazard-rate
simulation, pathwise survival weighting, and Monte Carlo variance-reduction
experiments.

The project is intended to be readable as both code and a technical note. The
compiled report is available at:

- `docs/hjm_hazard_rate_cln_monte_carlo.pdf`

## What Is Implemented

- Deterministic short-rate, investment-grade, and high-yield hazard curves.
- Discrete forward hazard curves initialized from survival probabilities.
- One-factor HJM-style stochastic hazard-rate evolution.
- Pathwise CLN valuation using coupon, redemption, and recovery legs.
- Range-accrual redemption feature driven by a model CDS-spread proxy.
- Base Monte Carlo, antithetic variates, and vanilla-CLN control variates.
- Lightweight smoke check and unit tests for core simulator behavior.

## Repository Layout

- `code/simulate_paths.py`: core CLN path simulator and deterministic vanilla benchmark.
- `code/smoke_example.py`: quick import-and-run check using a small fixed random sample.
- `code/hjm_variance_reduction.ipynb`: main variance-reduction experiment notebook.
- `code/section_two_plots.ipynb`: notebook used to generate term-structure figures.
- `docs/main.tex`: top-level LaTeX source for the technical report.
- `docs/sections/`: report body sections.
- `docs/figures/`: figures used by the report.
- `tests/`: unit tests for the core simulator.

## Setup

Create and activate a virtual environment, then install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick Checks

Run the smoke example:

```bash
python code/smoke_example.py
```

Expected output is four present values: IG/HY range-accrual Monte Carlo means
from a tiny fixed sample, followed by deterministic vanilla CLN prices. The
Monte Carlo means are not intended to be production estimates; they are a fast
sanity check that imports and path valuation work.

Run the unit tests:

```bash
python -m unittest discover -s tests
```

## Rebuilding the Report

From the `docs/` directory:

```bash
latexmk -pdf main.tex
```

The local `.gitignore` excludes LaTeX build intermediates under `docs/build/`
and common cache files.

## Main Modeling Assumptions

This is a transparent research implementation rather than a market-calibrated
pricing library. The main assumptions are:

- Discount rates are deterministic; only credit hazard rates are stochastic.
- Initial curves are smooth parametric curves, not calibrated to traded bonds or CDS.
- CDS levels inside the range-accrual payoff use the proxy `s(t) = (1 - R)h(t)`.
- Recovery is represented on the simulation grid using interval default probabilities.
- Accrual on default, exact CDS conventions, day counts, and settlement details are omitted.
- The Gaussian HJM-style hazard model can produce negative forward hazards. The implementation clips hazards only when converting them to model spreads for the range test; survival updates use the raw simulated hazard path.

## Suggested Extensions

- Add market calibration to CDS and discount curves.
- Replace the spread proxy with full par-spread calculations from premium and protection legs.
- Add quasi-Monte Carlo and importance sampling experiments.
- Vectorize or port the simulation to JAX for larger scenario grids and automatic differentiation.
