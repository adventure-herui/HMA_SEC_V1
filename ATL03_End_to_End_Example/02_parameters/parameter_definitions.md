# Processing parameter definitions

- Pre-screening retains ATL03 photons with land signal confidence >= 1 and preserves each native ATL03 row index.
- Block SNR is calculated after pre-screening. The implementation follows the public code's signal/noise class definition and NaN handling.
- Stage 1, Stage 2 and Stage 3 use the window dimensions listed in `processing_parameters.csv`.
- Stage 3 and Stage 4 use cubic `scipy.interpolate.UnivariateSpline` fits (k=3), with the listed spar settings and the smoothing-parameter rule implemented in the script.
- Stage 5 uses the listed along-slope window and modal interval fraction.
- Stage 6 uses the public `LinearGAM` configuration only; no alternative spline fallback is used.
- Each 0.01-degree core block owns its output photons. The +/-0.001-degree overlap supplies processing context only.
- Pair candidates are generated across the pooled SE photons from the two fixed acquisitions. Candidates are sorted by horizontal distance and greedily accepted only if neither photon has already been used in this file-pair comparison.
- The one-to-one uniqueness constraint applies to the entire fixed ATL03 file-pair comparison, not independently to each block and not across unrelated acquisition-pair comparisons.
