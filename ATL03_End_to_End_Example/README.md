# ATL03 End-to-End Example

## 1. Overview

This repository provides a small-area end-to-end example of ICESat-2 ATL03 photon processing, including photon pre-screening, denoising, surface-elevation (SE) photon reconstruction, and crossover photon pairing.

The example demonstrates the complete workflow using two ATL03 acquisitions.

## 2. Processing Workflow

The processing consists of five main steps:

1. **Data pre-screening:** Retain photons with land signal confidence ≥ 1.
2. **Block-based denoising:** Process photons within consecutive 0.01° latitude blocks.
3. **Surface-elevation reconstruction:** Apply the specified filtering, spline interpolation, and GAM procedures.
4. **Crossover matching:** Identify photon pairs within the maximum horizontal distance threshold of 2 m.
5. **One-to-one validation:** Evaluate the accepted photon pairs and verify matching uniqueness.

## 3. One-to-One Pairing

One-to-one pairing is enforced across the entire fixed ATL03 file-pair comparison, rather than independently within individual processing blocks.

All candidate pairs are sorted by horizontal distance. A greedy matching procedure accepts a pair only when neither photon has been assigned to a previously accepted pair.

The constraint is applied separately to each ATL03 file-pair comparison.

## 4. Repository Structure

```text
ATL03_End_to_End_Example/
│
├── README.md
├── code/
│   └── end_to_end_example.py
├── 01_input/
├── 02_parameters/
├── 03_denoising_results/
├── 04_pairing_results/
├── 05_figures/
└── 06_logs/
```

- `01_input/`: Input data information and study-area metadata.
- `02_parameters/`: Pre-screening, denoising, spline, GAM, and pairing parameters.
- `03_denoising_results/`: Block-level processing logs and reconstructed SE photons.
- `04_pairing_results/`: Candidate pairs, accepted pairs, and validation results.
- `05_figures/`: Spatial distribution, photon reconstruction, pairing, and statistical figures.
- `06_logs/`: Execution logs and software environment information.

## 5. Running the Example

Run the Python script using the required dependencies and locally available ATL03 and auxiliary data:

```bash
python -u code/end_to_end_example.py
```

Input file paths and processing settings are configured in the script.

## 6. Data Availability

Original ATL03 HDF5 files and auxiliary geospatial datasets are not included in this repository. They should be obtained from their respective public data sources.

All processing parameters, intermediate results, and diagnostic outputs are provided to facilitate reproducibility.
