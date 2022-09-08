# cloud-regime-feedbacks

[![DOI](https://zenodo.org/badge/534324116.svg)](https://zenodo.org/badge/latestdoi/534324116)


Code to perform calculations for Zelinka, M. D., I. Tan, L. Oreopoulos, G. Tselioudis, 2022: [Detailing Cloud Property Feedbacks with a Regime-Based Decomposition](https://doi.org/10.1007/s00382-022-06488-7), Climate Dyn., doi:10.1007/s00382-022-06488-7.

Code is broken up into 3 pieces:
- Part 1 reads in GCM fields and maps them to the observed regimes.
- Part 2 performs the detailed decomposition into within-regime, across-regime, and covariance terms, which are further partitioned by gross cloud property changes (amount, altitude, optical depth, and residual).
- Part 3 makes the figures shown in the manuscript.

Requires CDAT, which can be installed via conda following [these instructions](https://github.com/CDAT/cdat/wiki/install#installing-latest-cdat---821)
