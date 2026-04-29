# HMA_SEC: Surface Elevation and Surface Elevation Change Extraction from ICESat-2 ATL03

This repository provides a Python-based processing framework for extracting surface elevation (SE) and surface elevation change (SEC) in High Mountain Asia (HMA) using ICESat-2 ATL03 raw photon data.

The workflow consists of two main modules:

1. Surface elevation extraction from ATL03 photon data
2. Surface elevation change calculation through distance-constrained photon pairing

The framework was developed to generate high-density SE and SEC observations over complex high-mountain terrain, including glaciers, permafrost regions, lakes, and other high-elevation surfaces.


## Overview

ICESat-2 ATL03 provides raw photon-level observations that contain dense elevation information, but the data also include substantial noise, especially in complex terrain. This framework first extracts reliable surface elevation photons from ATL03 data using a multi-process slope-constrained kernel density estimation strategy. Surface elevation change is then calculated by pairing spatially close photons acquired at different times.

The general workflow is:

ICESat-2 ATL03 raw photon data
        |
        |  Step 1: MSC-KDE denoising and surface elevation extraction
        v
High-density surface elevation points
        |
        |  Step 2: Distance-constrained photon pairing
        v
Surface elevation change photon pairs
