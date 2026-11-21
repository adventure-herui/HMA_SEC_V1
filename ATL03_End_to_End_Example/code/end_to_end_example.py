#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small-area end-to-end HMA_SEC demonstration tailored for reviewer response.

Purpose
-------
This script documents, in one reproducible small-area example:
1) the ATL03 pre-screening rule,
2) all fixed denoising thresholds and spline settings,
3) the level at which one-to-one pairing is enforced,
4) how the originally selected released photon pair is traced through the
   independent reconstruction and distinguished from other local crossing pairs.

Key design updates relative to the earlier demonstration
-------------------------------------------------------
A. Native ATL03 photon identity is carried through every denoising stage.
B. Overlap-expanded blocks provide processing context, but each native photon
   is emitted only by its non-overlapping core block; duplicate output IDs are
   treated as an error rather than silently removed.
C. All <=2 m candidates from the two selected beam observations are considered
   together. Distance-sorted greedy one-to-one matching is enforced globally
   within that single pairwise beam-observation comparison, not across other
   acquisition-pair comparisons.
D. SNR is calculated after the public land_conf >= 1 pre-screening, and the
   public NaN handling is retained when the SNR denominator is zero.
E. Stage 6 uses the public LinearGAM configuration only; no spline fallback
   is substituted if pygam is unavailable.

The script does NOT save data tables. It only shows figures interactively and
prints diagnostic logs. You may optionally save figures manually from the
interactive window if desired.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "8"

import gc
import re
import sys
import json
import platform
import contextlib
from importlib import metadata as importlib_metadata
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from matplotlib import font_manager
from numba import jit, prange
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
try:
    from pygam import LinearGAM, s
except Exception:
    LinearGAM = None
    s = None

# =============================================================================
# 0. PATHS AND SETTINGS
# =============================================================================

RAW_ATL03_ROOT = Path("/home/yyr/herui/ICESat-2_ATL03")

PAIR_ROOT_CANDIDATES = [
    Path("/home/yyr/herui/ICESat-2_ATL03_Dh_uncertainty_Revised"),
    Path("/home/yyr/herui/ICESat-2_ATL03_Dh_Revised"),
    Path("/home/yyr/herui/ICESat-2_ATL03_Dh"),
]

GRID_VECTOR_ROOT = Path("/home/yyr/herui/HMA_Subregion")
TP_VECTOR = Path("/home/yyr/herui/RIT/Data/TibetanPlateau/TibetanPlateau.shp")
GEOID_TIF = Path("/home/yyr/herui/us_nga_egm2008_1.tif")
CROSS_FILE_NAME = "optimized_cross_pairs.parquet"

TARGET_GRID = None           # e.g. "Grid_017"
RANDOM_SEED = 20261121
MAX_RANDOM_ATTEMPTS = 1000
MAX_CANDIDATES_TO_SCORE = 240
MAX_CANDIDATES_TO_DENOISE = 20
NEIGHBOR_BLOCKS_EACH_SIDE = 1
MIN_LOCAL_SE_RECOVERY_FRACTION = 0.40
MIN_SE_PHOTONS_PER_BEAM = 5000
TARGET_COORDINATE_TOLERANCE_M = 0.001
TARGET_ID_MIN_SECOND_NEIGHBOR_GAP_M = 0.001

# Identity tolerances
TARGET_ID_TOLERANCE_M = 0.10
STRICT_TARGET_MATCH = True

# Local crossing organization
MAX_PAIR_DISTANCE_M = 2.0
MAX_GLOBAL_CANDIDATES = 3_000_000
R_EARTH = 6371000.0

# Current public SE settings
MIN_BEAM_PHOTONS = 50
LAND_CONF_MIN = 1
MIN_BLOCK_PHOTONS = 10
SNR_MIN = 0.1

STAGE1_L = 100.0
STAGE1_W = 20.0
STAGE2_L = 20.0
STAGE2_W = 10.0
STAGE3_L = 10.0
STAGE3_W = 2.0
SPLINE_SPAR_STAGE3 = 0.5
SPLINE_SPAR_STAGE4 = 0.5
STAGE5_L = 20.0
STAGE5_MODE_FRAC = 0.10
GAM_MIN_POINTS = 30
GAM_N_SPLINES = 40
GAM_SPLINE_ORDER = 3

# Plot sampling only
MAX_MAP_DISPLAY_POINTS = 80000
MAX_PROFILE_DISPLAY_POINTS = 150000
SAVE_ANYTHING = True
OUTPUT_FOLDER_NAME = "ATL03_End_to_End_Example"


def initialize_output_package():
    """Create the reviewer-facing reproducibility package beside this script."""
    root = Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME
    folders = {
        "root": root,
        "input": root / "01_input",
        "parameters": root / "02_parameters",
        "denoising": root / "03_denoising_results",
        "pairing": root / "04_pairing_results",
        "figures": root / "05_figures",
        "logs": root / "06_logs",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


OUTPUT_PACKAGE = None


def save_figure(fig, stem):
    """Save the current figure as publication-quality PNG and vector PDF."""
    if OUTPUT_PACKAGE is None:
        return
    out = OUTPUT_PACKAGE["figures"]
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight",
                facecolor="white")
    print(f"[OUTPUT] Saved {stem}.png and {stem}.pdf", flush=True)


class Tee:
    """Write console output to both the terminal and the run log."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def package_version(name):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "not-installed-or-unavailable"


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str),
                    encoding="utf-8")


def save_case_metadata(folders, center, block_edges):
    """Write source manifest, study-area metadata, parameters and environment."""
    file1_date = extract_date_from_filename(FIXED_FILE1.name)
    file2_date = extract_date_from_filename(FIXED_FILE2.name)
    manifest = pd.DataFrame([
        {"grid": FIXED_GRID, "file": FIXED_FILE1.name, "beam": FIXED_BEAM1,
         "acquisition_date": file1_date.strftime("%Y-%m-%d") if file1_date else "",
         "local_path_on_runner": str(FIXED_FILE1), "source_product": "ICESat-2 ATL03"},
        {"grid": FIXED_GRID, "file": FIXED_FILE2.name, "beam": FIXED_BEAM2,
         "acquisition_date": file2_date.strftime("%Y-%m-%d") if file2_date else "",
         "local_path_on_runner": str(FIXED_FILE2), "source_product": "ICESat-2 ATL03"},
    ])
    manifest.to_csv(folders["input"] / "input_data_manifest.csv", index=False)

    study = {
        "grid": FIXED_GRID,
        "beam_1": FIXED_BEAM1,
        "beam_2": FIXED_BEAM2,
        "closest_approach_distance_m": float(center["distance_m"]),
        "beam_1_closest_photon": {"longitude": float(center["lon1"]), "latitude": float(center["lat1"])},
        "beam_2_closest_photon": {"longitude": float(center["lon2"]), "latitude": float(center["lat2"])},
        "crossing_center": {"longitude": float(center["center_lon"]), "latitude": float(center["center_lat"])},
        "core_block_edges_latitude_deg": [float(v) for v in block_edges],
        "core_block_width_deg": 0.01,
        "neighbor_blocks_each_side": int(NEIGHBOR_BLOCKS_EACH_SIDE),
        "processing_context_overlap_deg": 0.001,
        "coordinate_reference_note": "Coordinates are geographic longitude/latitude in degrees; elevation outputs follow the processing code's EGM2008 conversion.",
    }
    save_json(folders["input"] / "study_area_metadata.json", study)

    parameters = [
        ("minimum_raw_photons_per_beam", MIN_BEAM_PHOTONS, "photons", "Minimum local raw photons required per beam"),
        ("land_confidence_minimum", LAND_CONF_MIN, "confidence class", "Retain land signal confidence >= this value"),
        ("minimum_photons_after_prescreen", MIN_BEAM_PHOTONS, "photons", "Minimum photons after land-confidence screening"),
        ("minimum_photons_per_block_context", MIN_BLOCK_PHOTONS, "photons", "Minimum context photons for block processing"),
        ("block_snr_threshold", SNR_MIN, "dimensionless", "Block SNR threshold"),
        ("stage1_window_length", STAGE1_L, "m", "Stage 1 along-track window length"),
        ("stage1_window_width", STAGE1_W, "m", "Stage 1 perpendicular window width"),
        ("stage2_window_length", STAGE2_L, "m", "Stage 2 along-track window length"),
        ("stage2_window_width", STAGE2_W, "m", "Stage 2 perpendicular window width"),
        ("stage3_window_length", STAGE3_L, "m", "Stage 3 slope-constrained window length"),
        ("stage3_window_width", STAGE3_W, "m", "Stage 3 slope-constrained window width"),
        ("stage3_spline_spar", SPLINE_SPAR_STAGE3, "dimensionless", "Stage 3 spline spar"),
        ("stage4_spline_spar", SPLINE_SPAR_STAGE4, "dimensionless", "Stage 4 spline spar"),
        ("stage5_along_slope_window", STAGE5_L, "m", "Stage 5 along-slope window"),
        ("stage5_mode_interval_fraction", STAGE5_MODE_FRAC, "fraction", "Half-width of local perpendicular-range modal interval"),
        ("stage6_gam_minimum_points", GAM_MIN_POINTS, "points", "Minimum observations required to trigger GAM"),
        ("stage6_gam_n_splines", GAM_N_SPLINES, "splines", "Number of GAM spline basis functions"),
        ("stage6_gam_spline_order", GAM_SPLINE_ORDER, "order", "GAM spline order"),
        ("maximum_pair_distance", MAX_PAIR_DISTANCE_M, "m", "Maximum geodesic horizontal distance for candidate pairs"),
        ("core_block_width", 0.01, "degree latitude", "Non-overlapping output block width"),
        ("context_overlap", 0.001, "degree latitude", "Context overlap on either side of a core block"),
    ]
    pd.DataFrame(parameters, columns=["parameter", "value", "unit", "description"]).to_csv(
        folders["parameters"] / "processing_parameters.csv", index=False)
    save_json(folders["parameters"] / "processing_parameters.json",
              {name: {"value": value, "unit": unit, "description": desc}
               for name, value, unit, desc in parameters})
    definitions = """# Processing parameter definitions

- Pre-screening retains ATL03 photons with land signal confidence >= 1 and preserves each native ATL03 row index.
- Block SNR is calculated after pre-screening. The implementation follows the public code's signal/noise class definition and NaN handling.
- Stage 1, Stage 2 and Stage 3 use the window dimensions listed in `processing_parameters.csv`.
- Stage 3 and Stage 4 use cubic `scipy.interpolate.UnivariateSpline` fits (k=3), with the listed spar settings and the smoothing-parameter rule implemented in the script.
- Stage 5 uses the listed along-slope window and modal interval fraction.
- Stage 6 uses the public `LinearGAM` configuration only; no alternative spline fallback is used.
- Each 0.01-degree core block owns its output photons. The +/-0.001-degree overlap supplies processing context only.
- Pair candidates are generated across the pooled SE photons from the two fixed acquisitions. Candidates are sorted by horizontal distance and greedily accepted only if neither photon has already been used in this file-pair comparison.
- The one-to-one uniqueness constraint applies to the entire fixed ATL03 file-pair comparison, not independently to each block and not across unrelated acquisition-pair comparisons.
"""
    (folders["parameters"] / "parameter_definitions.md").write_text(definitions, encoding="utf-8")

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scipy": package_version("scipy"),
        "h5py": package_version("h5py"),
        "geopandas": package_version("geopandas"),
        "shapely": package_version("shapely"),
        "matplotlib": package_version("matplotlib"),
        "rasterio": package_version("rasterio"),
        "numba": package_version("numba"),
        "pygam": package_version("pygam"),
        "selected_plot_font": SELECTED_FONT,
        "matplotlib_backend": plt.get_backend(),
    }
    save_json(folders["logs"] / "environment_versions.json", versions)
    (folders["logs"] / "environment_versions.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in versions.items()) + "\n", encoding="utf-8")

# Use Arial when available. Missing fonts must never stop the scientific workflow.
# Liberation Sans has Arial-compatible metrics; DejaVu Sans is the final fallback.
_ARIAL_CANDIDATE_PATHS = [
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/arial.ttf"),
    Path("/usr/local/share/fonts/Arial.ttf"),
    Path.home() / ".fonts" / "Arial.ttf",
    Path.home() / ".local" / "share" / "fonts" / "Arial.ttf",
    Path("/mnt/c/Windows/Fonts/arial.ttf"),
]
_FONT_NAMES = {f.name for f in font_manager.fontManager.ttflist}
if "Arial" not in _FONT_NAMES:
    for _font_path in _ARIAL_CANDIDATE_PATHS:
        if _font_path.exists():
            try:
                font_manager.fontManager.addfont(str(_font_path))
            except Exception:
                pass
    _FONT_NAMES = {f.name for f in font_manager.fontManager.ttflist}

if "Arial" in _FONT_NAMES:
    SELECTED_FONT = "Arial"
elif "Liberation Sans" in _FONT_NAMES:
    SELECTED_FONT = "Liberation Sans"
    print("[FONT] Arial unavailable; using Liberation Sans (Arial-compatible metrics).")
else:
    SELECTED_FONT = "DejaVu Sans"
    print("[FONT WARNING] Arial and Liberation Sans unavailable; using DejaVu Sans.")

plt.rcParams.update(
    {
        "font.family": SELECTED_FONT,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)

BEAM1_COLOR = "steelblue"
BEAM2_COLOR = "darkorange"
TARGET_COLOR = "crimson"
OTHER_PAIR_COLOR = "0.55"
TARGET_PAIR_COLOR = "limegreen"


# =============================================================================
# 1. PRINT HELPERS
# =============================================================================

def section(title: str) -> None:
    line = "=" * 110
    print(f"\n{line}")
    print(title)
    print(line)


def sub_section(title: str) -> None:
    print("\n" + "-" * 110)
    print(title)
    print("-" * 110)


def format_float(value, digits=4) -> str:
    try:
        value = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def choose_pair_root() -> Path:
    for root in PAIR_ROOT_CANDIDATES:
        if root.is_dir():
            return root
    raise FileNotFoundError(
        "No SEC pair root found. Checked:\n" + "\n".join(f"  {p}" for p in PAIR_ROOT_CANDIDATES)
    )


def union_geometry(gdf: gpd.GeoDataFrame):
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def sample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_n, dtype=np.int64)


def read_vector_wgs84(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Vector has no CRS: {path}")
    return gdf.to_crs("EPSG:4326")


def lonlat_to_xy_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    x = (lon - lon0) * 111000.0 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 111000.0
    return x, y


# =============================================================================
# 3. RANDOM EXISTING CROSS-PAIR SELECTION
# =============================================================================

PAIR_READ_COLUMNS = [
    "photon1_file",
    "photon2_file",
    "photon1_raw_index",
    "photon2_raw_index",
    "photon1_atl03_row_index",
    "photon2_atl03_row_index",
    "photon1_beam",
    "photon2_beam",
    "photon1_lon",
    "photon1_lat",
    "photon2_lon",
    "photon2_lat",
    "photon1_elevation",
    "photon2_elevation",
    "photon1_date",
    "photon2_date",
    "distance",
    "elevation_diff",
    "months_diff",
    "years_diff",
    "monthly_change_rate",
    "yearly_change_rate",
]


def available_cross_pair_files(pair_root: Path) -> List[Tuple[str, Path]]:
    output = []
    grids = [pair_root / TARGET_GRID] if TARGET_GRID is not None else sorted(pair_root.glob("Grid_*"))
    for grid_dir in grids:
        if not grid_dir.is_dir():
            continue
        path = grid_dir / CROSS_FILE_NAME
        if not path.exists():
            continue
        try:
            pf = pq.ParquetFile(path)
            if pf.metadata.num_rows > 0:
                output.append((grid_dir.name, path))
        except Exception:
            continue
    return output


def read_random_row_from_parquet(path: Path, rng: np.random.Generator) -> pd.Series:
    pf = pq.ParquetFile(path)
    n_total = pf.metadata.num_rows
    if n_total <= 0:
        raise ValueError(f"Empty Parquet: {path}")
    global_index = int(rng.integers(0, n_total))
    offset = 0
    for rg in range(pf.metadata.num_row_groups):
        n_rg = pf.metadata.row_group(rg).num_rows
        if global_index < offset + n_rg:
            local_index = global_index - offset
            available = set(pf.schema_arrow.names)
            columns = [c for c in PAIR_READ_COLUMNS if c in available]
            table = pf.read_row_group(rg, columns=columns)
            return table.to_pandas().iloc[int(local_index)].copy()
        offset += n_rg
    raise RuntimeError("Could not locate random row in Parquet.")


def parse_atl03_keys(name: str) -> Tuple[Optional[str], Optional[str]]:
    text = str(name)
    m = re.search(r"ATL03_(\d{14})_(\d{8})_", text)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"ATL03_(\d{14})_", text)
    if m:
        return m.group(1), None
    return None, None


def locate_raw_atl03(grid_name: str, pair_file_name: str) -> Optional[Path]:
    grid_dir = RAW_ATL03_ROOT / grid_name
    if not grid_dir.is_dir():
        return None
    name = Path(str(pair_file_name)).name
    exact_names = [
        name,
        name.replace("_Elevation.parquet", ".h5"),
        name.replace("_elevation.parquet", ".h5"),
        name.replace(".parquet", ".h5"),
    ]
    for exact_name in exact_names:
        path = grid_dir / exact_name
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5", ".he5"}:
            return path
    timestamp, orbit = parse_atl03_keys(name)
    patterns = []
    if timestamp and orbit:
        patterns.extend([f"ATL03_{timestamp}_{orbit}_*.h5", f"*{timestamp}*{orbit}*.h5"])
    if timestamp:
        patterns.extend([f"ATL03_{timestamp}_*.h5", f"*{timestamp}*.h5"])
    candidates = []
    for pattern in patterns:
        candidates.extend(grid_dir.glob(pattern))
        candidates.extend(grid_dir.rglob(pattern))
    candidates = sorted({p.resolve() for p in candidates if p.is_file() and p.suffix.lower() in {".h5", ".hdf5", ".he5"}})
    if not candidates:
        return None
    if orbit is not None:
        orbit_matches = [p for p in candidates if orbit in p.name]
        if orbit_matches:
            return orbit_matches[0]
    return candidates[0]


def raw_beam_is_valid(path: Path, beam: str) -> bool:
    try:
        with h5py.File(path, "r") as f:
            if beam not in f:
                return False
            if "heights" not in f[beam]:
                return False
            grp = f[beam]["heights"]
            required = ["lat_ph", "lon_ph", "h_ph", "signal_conf_ph"]
            return all(k in grp for k in required)
    except Exception:
        return False



def _row_file_beam(row, prefix):
    return str(row[f"{prefix}_file"]), str(row[f"{prefix}_beam"])


def _optional_native_row_index(row, prefix):
    for name in (f"{prefix}_raw_index", f"{prefix}_atl03_row_index"):
        if name in row.index and pd.notna(row[name]):
            value = int(row[name])
            if value >= 0:
                return value
    return None


def _candidate_rows(pair_files, rng, max_rows):
    """Draw reproducible candidate rows spread across available Grid parquet files."""
    rows = []
    per_grid = 1  # guarantee at least one candidate draw from every available Grid
    for grid_name, pair_path in pair_files:
        n_this = min(per_grid, max_rows - len(rows))
        if n_this <= 0:
            break
        for _ in range(n_this):
            try:
                row = read_random_row_from_parquet(pair_path, rng)
                rows.append((grid_name, pair_path, row))
            except Exception:
                continue
    # Fill remaining slots by random draws across all grids.
    while len(rows) < max_rows:
        grid_name, pair_path = pair_files[int(rng.integers(0, len(pair_files)))]
        try:
            rows.append((grid_name, pair_path, read_random_row_from_parquet(pair_path, rng)))
        except Exception:
            break
    return rows


def _block_edges_around_target(target_lat, neighbor_blocks=NEIGHBOR_BLOCKS_EACH_SIDE):
    target_start = np.floor(float(target_lat) * 100.0) / 100.0
    first = target_start - neighbor_blocks * 0.01
    last = target_start + (neighbor_blocks + 1) * 0.01
    n_blocks = 2 * neighbor_blocks + 1
    return first + np.arange(n_blocks + 1, dtype=np.float64) * 0.01


def read_raw_beam_neighborhood(path: Path, beam: str, target_lon: float, target_lat: float,
                               neighbor_blocks=NEIGHBOR_BLOCKS_EACH_SIDE, expected_raw_index=None):
    """
    Read full-track latitude/longitude only for locating the target and drawing
    the spatial context. Read h_ph and signal_conf_ph only for the target block
    plus neighboring blocks and their +/-0.001 degree processing context.
    """
    with h5py.File(path, "r") as f:
        grp = f[beam]["heights"]
        lat_full = grp["lat_ph"][:].astype(np.float64)
        lon_full = grp["lon_ph"][:].astype(np.float64)

        valid_xy = np.isfinite(lat_full) & np.isfinite(lon_full)
        valid_xy &= (lat_full >= -90.0) & (lat_full <= 90.0)
        valid_xy &= (lon_full >= -180.0) & (lon_full <= 180.0)
        valid_indices = np.flatnonzero(valid_xy)
        if valid_indices.size < 2:
            raise RuntimeError(f"{path.name} {beam}: insufficient valid ATL03 coordinates")

        d_all = distance_m(lat_full[valid_indices], lon_full[valid_indices],
                           float(target_lat), float(target_lon))
        order = np.argsort(d_all)[:2]
        nearest_index = int(valid_indices[int(order[0])])
        nearest = float(d_all[int(order[0])])
        second = float(d_all[int(order[1])])
        gap = second - nearest

        if expected_raw_index is not None and int(expected_raw_index) >= 0:
            raw_target_index = int(expected_raw_index)
            if raw_target_index >= len(lat_full) or not valid_xy[raw_target_index]:
                raise RuntimeError(f"{path.name} {beam}: supplied native ATL03 row index is invalid.")
            indexed_error = float(distance_m(
                np.asarray([lat_full[raw_target_index]]), np.asarray([lon_full[raw_target_index]]),
                float(target_lat), float(target_lon)
            )[0])
            if indexed_error > TARGET_COORDINATE_TOLERANCE_M:
                raise RuntimeError(
                    f"{path.name} {beam}: supplied native row index does not agree with released coordinates "
                    f"({indexed_error:.6f} m)."
                )
            nearest = indexed_error
            second = float(d_all[int(order[1])])
            identity_method = "explicit native ATL03 row index"
        else:
            raw_target_index = nearest_index
            identity_method = "strict unique coordinate match"
            if nearest > TARGET_COORDINATE_TOLERANCE_M:
                raise RuntimeError(
                    f"{path.name} {beam}: released coordinate is {nearest:.6f} m "
                    f"from nearest ATL03 photon; strict identity tolerance is "
                    f"{TARGET_COORDINATE_TOLERANCE_M:.3f} m."
                )
            if gap < TARGET_ID_MIN_SECOND_NEIGHBOR_GAP_M:
                raise RuntimeError(
                    f"{path.name} {beam}: coordinate-to-native-index mapping is "
                    f"ambiguous; nearest/second-nearest={nearest:.6f}/{second:.6f} m."
                )

        target_raw_lat = float(lat_full[raw_target_index])
        edges = _block_edges_around_target(target_raw_lat, neighbor_blocks)
        local_min = float(edges[0] - 0.001)
        local_max = float(edges[-1] + 0.001)
        local_indices = np.flatnonzero(
            valid_xy & (lat_full >= local_min) & (lat_full <= local_max)
        )
        if local_indices.size < MIN_BEAM_PHOTONS:
            raise RuntimeError(f"{path.name} {beam}: too few photons in selected local blocks")

        heights_ds = grp["h_ph"]
        conf_ds = grp["signal_conf_ph"]
        h_local = heights_ds[local_indices].astype(np.float64)
        conf_local = conf_ds[local_indices]
        if conf_local.ndim == 1:
            land_conf = conf_local.astype(np.int16)
        else:
            # ATL03 signal_conf_ph columns: surface-type confidence classes.
            # The first column is land confidence for this workflow.
            land_conf = conf_local[:, 0].astype(np.int16)

    raw_local = {
        "lat": lat_full[local_indices].astype(np.float64),
        "lon": lon_full[local_indices].astype(np.float64),
        "h": h_local,
        "land_conf": land_conf,
        "raw_index": local_indices.astype(np.int64),
        "target_raw_index": raw_target_index,
        "target_nearest_m": nearest,
        "target_second_nearest_m": second,
        "target_identity_method": identity_method,
        "block_edges": edges,
        "target_lat_native": target_raw_lat,
    }
    track = {"lat": lat_full, "lon": lon_full}
    return raw_local, track


def _quick_beam_quality(raw, target_raw_index):
    """Fast raw-data proxy for local SE completeness; no denoising is run here."""
    lat = raw["lat"]
    conf = raw["land_conf"]
    edges = raw["block_edges"]
    core = (lat >= edges[0]) & (lat < edges[-1])
    if not np.any(core):
        raise RuntimeError("No raw photons in selected core-block neighborhood.")
    core_conf = conf[core]
    pre_ratio = float(np.mean(core_conf >= LAND_CONF_MIN))
    target_pos = np.flatnonzero(raw["raw_index"] == target_raw_index)
    if len(target_pos) != 1:
        raise RuntimeError("Target native ATL03 row is not unique in local raw subset.")
    if int(conf[int(target_pos[0])]) < LAND_CONF_MIN:
        raise RuntimeError("Target photon fails the land-confidence pre-screen.")

    block_snrs = []
    block_counts = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lat >= a - 0.001) & (lat < b + 0.001)
        c = conf[m]
        block_counts.append(int(m.sum()))
        if c.size == 0:
            block_snrs.append(0.0)
            continue
        # Public denoising code computes SNR after land_conf >= 1 pre-screening.
        c_screened = c[c >= LAND_CONF_MIN]
        n_signal = int(np.sum((c_screened >= 2) & (c_screened <= 4)))
        n_noise = int(np.sum((c_screened <= 1) | (c_screened == -1)))
        # Match the public implementation: no denominator -> NaN, not Inf/zero.
        snr = (n_signal / n_noise) if n_noise > 0 else np.nan
        block_snrs.append(float(snr))
    target_block = int(np.floor(raw["target_lat_native"] * 100.0) / 100.0)
    target_idx = int(np.argmin(np.abs((edges[:-1] + edges[1:]) * 0.5 - (target_block + 0.005))))
    target_snr = block_snrs[target_idx]
    if np.isfinite(target_snr) and target_snr < SNR_MIN:
        raise RuntimeError(f"Target core block SNR={target_snr:.4f} is below {SNR_MIN}.")
    finite_snrs = np.asarray([x for x in block_snrs if np.isfinite(x)], dtype=float)
    mean_snr = float(np.mean(finite_snrs)) if finite_snrs.size else 0.0
    # Do not cap SNR at 1: use a log transform, while prioritizing the
    # weaker beam's available photon volume across the selected core blocks.
    n_raw_core = int(core.sum())
    n_prescreen_core = int(np.sum(core & (conf >= LAND_CONF_MIN)))
    return {
        "pre_screen_fraction": pre_ratio,
        "target_block_snr": float(target_snr),
        "mean_neighbor_snr": mean_snr,
        "block_counts": block_counts,
        "score": float(np.log1p(n_prescreen_core) + 0.20 * np.log1p(mean_snr)),
        "n_raw_local": n_raw_core,
        "n_prescreen_local": n_prescreen_core,
    }


def select_best_released_pair(pair_root: Path):
    """
    Score candidate released SEC pairs using local raw ATL03 quality for both
    beams. Only a short list is fully denoised. Ranking is based on the weaker
    beam, so a pair with one poor beam is not selected merely because the other
    beam is excellent.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    pair_files = available_cross_pair_files(pair_root)
    if not pair_files:
        raise FileNotFoundError(f"No non-empty {CROSS_FILE_NAME} found under {pair_root}")
    print(f"Pair root                         : {pair_root}")
    print(f"Available Grids with cross pairs : {len(pair_files)}")
    print(f"Random seed                       : {RANDOM_SEED}")
    print(f"Candidate rows to assess          : {MAX_CANDIDATES_TO_SCORE}")
    print(f"Candidates to denoise              : {MAX_CANDIDATES_TO_DENOISE}")
    print(f"Minimum SE photons required/beam   : {MIN_SE_PHOTONS_PER_BEAM:,}")
    print("Candidate ranking objective        : maximize the weaker beam's local photon count")
    print(f"Local blocks assessed per beam    : {2 * NEIGHBOR_BLOCKS_EACH_SIDE + 1}")

    candidate_rows = _candidate_rows(pair_files, rng, MAX_CANDIDATES_TO_SCORE)
    scored = []
    seen = set()
    for attempt, (grid_name, pair_path, row) in enumerate(candidate_rows, start=1):
        try:
            file1, beam1 = _row_file_beam(row, "photon1")
            file2, beam2 = _row_file_beam(row, "photon2")
            key = (grid_name, file1, beam1, file2, beam2,
                   round(float(row["photon1_lon"]), 7), round(float(row["photon1_lat"]), 7),
                   round(float(row["photon2_lon"]), 7), round(float(row["photon2_lat"]), 7))
            if key in seen:
                continue
            seen.add(key)
            raw1_path = locate_raw_atl03(grid_name, file1)
            raw2_path = locate_raw_atl03(grid_name, file2)
            if raw1_path is None or raw2_path is None:
                continue
            if not raw_beam_is_valid(raw1_path, beam1) or not raw_beam_is_valid(raw2_path, beam2):
                continue
            raw1, _ = read_raw_beam_neighborhood(
                raw1_path, beam1, float(row["photon1_lon"]), float(row["photon1_lat"]),
                expected_raw_index=_optional_native_row_index(row, "photon1")
            )
            raw2, _ = read_raw_beam_neighborhood(
                raw2_path, beam2, float(row["photon2_lon"]), float(row["photon2_lat"]),
                expected_raw_index=_optional_native_row_index(row, "photon2")
            )
            q1 = _quick_beam_quality(raw1, int(raw1["target_raw_index"]))
            q2 = _quick_beam_quality(raw2, int(raw2["target_raw_index"]))
            # Primary ranking: maximize the weaker beam's pre-screened
            # photon count. Secondary terms reward balanced signal quality.
            min_prescreen = min(q1["n_prescreen_local"], q2["n_prescreen_local"])
            mean_prescreen = (q1["n_prescreen_local"] + q2["n_prescreen_local"]) / 2.0
            pair_score = (
                np.log1p(min_prescreen)
                + 0.10 * np.log1p(mean_prescreen)
                + 0.05 * min(q1["score"], q2["score"])
            )
            scored.append({
                "score": float(pair_score), "grid_name": grid_name, "pair_path": pair_path,
                "row": row, "file1": file1, "file2": file2, "beam1": beam1, "beam2": beam2,
                "raw1": raw1_path, "raw2": raw2_path, "q1": q1, "q2": q2,
            })
        except Exception:
            continue

    if not scored:
        raise RuntimeError(
            "No candidate pair passed strict coordinate identity, target land-confidence, "
            "and target-block SNR screening. Increase MAX_CANDIDATES_TO_SCORE or inspect "
            "the released coordinate precision."
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    print(f"Candidates passing local raw-quality screening : {len(scored)}")
    print("Top candidate quality ranking (both beams; higher is better):")
    for rank, c in enumerate(scored[:10], start=1):
        print(
            f"  {rank:02d} {c['grid_name']} | score={c['score']:.3f} | "
            f"pre-screen N={c['q1']['n_prescreen_local']:,}/{c['q2']['n_prescreen_local']:,} | "
            f"fraction={c['q1']['pre_screen_fraction']:.3f}/{c['q2']['pre_screen_fraction']:.3f} | "
            f"SNR={c['q1']['target_block_snr']:.2f}/{c['q2']['target_block_snr']:.2f} | "
            f"{Path(c['file1']).name} [{c['beam1']}] / {Path(c['file2']).name} [{c['beam2']}]"
        )
    return scored[:MAX_CANDIDATES_TO_DENOISE]


# =============================================================================
# 4. RAW ATL03 READING AND IDENTITY
# =============================================================================


def read_raw_beam(path: Path, beam: str, target_lon: float, target_lat: float):
    """Compatibility wrapper: read only target-neighborhood height/confidence arrays."""
    return read_raw_beam_neighborhood(path, beam, target_lon, target_lat)


def identify_target_raw_index(raw: Dict[str, np.ndarray], target_lon: float, target_lat: float, label: str) -> int:
    idx = int(raw["target_raw_index"])
    nearest = float(raw["target_nearest_m"])
    second = float(raw["target_second_nearest_m"])
    print(
        f"Target native-ID lookup ({label}) : nearest={nearest:.6f} m; "
        f"second-nearest={second:.6f} m; gap={second-nearest:.6f} m"
    )
    if nearest > TARGET_COORDINATE_TOLERANCE_M or second - nearest < TARGET_ID_MIN_SECOND_NEIGHBOR_GAP_M:
        raise RuntimeError(f"{label}: target coordinate does not uniquely identify a native ATL03 photon.")
    return idx


def haversine_distance_scalar(lat1, lon1, lat2, lon2) -> float:
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R_EARTH * c


def distance_m(lat_arr, lon_arr, center_lat, center_lon) -> np.ndarray:
    lat_arr = np.asarray(lat_arr, dtype=np.float64)
    lon_arr = np.asarray(lon_arr, dtype=np.float64)
    x = (lon_arr - center_lon) * 111000.0 * np.cos(np.radians(center_lat))
    y = (lat_arr - center_lat) * 111000.0
    return np.sqrt(x * x + y * y)


def identify_target_raw_index(raw: Dict[str, np.ndarray], target_lon: float, target_lat: float, label: str) -> int:
    """Map the released coordinate to one unique native ATL03 row index."""
    d = distance_m(raw["lat"], raw["lon"], target_lat, target_lon)
    order = np.argsort(d)[:2]
    idx = int(order[0])
    best = float(d[idx])
    second = float(d[order[1]]) if len(order) > 1 else np.inf
    print(f"Target native-ID lookup ({label}) : nearest={best:.6f} m; second-nearest={second:.6f} m")
    if best > TARGET_ID_TOLERANCE_M:
        raise RuntimeError(
            f"{label}: no raw ATL03 photon found within {TARGET_ID_TOLERANCE_M} m of the released target coordinate."
        )
    if second - best < 1e-5:
        raise RuntimeError(
            f"{label}: released coordinate does not uniquely identify one raw ATL03 photon (nearest-neighbour tie)."
        )
    return int(raw["raw_index"][idx])


# =============================================================================
# 5. PRE-SCREENING AND CORE DENOISING FUNCTIONS
# =============================================================================

def pre_screen_raw_beam(raw: Dict[str, np.ndarray], path: Path, beam: str) -> Dict[str, np.ndarray]:
    lat = raw["lat"]
    lon = raw["lon"]
    h = raw["h"]
    land_conf = raw["land_conf"]
    raw_index = raw["raw_index"]

    n_total = len(lat)
    print(f"Source ATL03 file                 : {path.name}")
    print(f"Target beam                       : {beam}")
    print(f"Raw photons in selected local read window : {n_total:,}")
    if n_total < MIN_BEAM_PHOTONS:
        raise RuntimeError(f"{path.name} {beam}: raw photon count < {MIN_BEAM_PHOTONS}")

    valid = land_conf >= LAND_CONF_MIN
    n_valid = int(np.sum(valid))
    print(f"Local photons with land_conf >= {LAND_CONF_MIN} : {n_valid:,}")
    print(f"Local photons removed by pre-screening       : {n_total - n_valid:,}")
    if n_valid < MIN_BEAM_PHOTONS:
        raise RuntimeError(f"{path.name} {beam}: valid photon count < {MIN_BEAM_PHOTONS}")

    result = {
        "lat": lat[valid],
        "lon": lon[valid],
        "h": h[valid],
        "signal_conf": land_conf[valid],
        "raw_index": raw_index[valid],
    }
    sort_idx = np.argsort(result["lat"])
    for key in result:
        result[key] = result[key][sort_idx]
    return result


@jit(nopython=True, parallel=True, fastmath=True)
def count_neighbors_rect_vectorized(local_coords: np.ndarray, L: float, W: float) -> np.ndarray:
    n = local_coords.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    lat_threshold = L / 2.0 / 111000.0
    elev_threshold = W / 2.0
    for i in prange(n):
        lat_diff = np.abs(local_coords[:, 1] - local_coords[i, 1])
        elev_diff = np.abs(local_coords[:, 2] - local_coords[i, 2])
        counts[i] = np.sum((lat_diff <= lat_threshold) & (elev_diff <= elev_threshold)) - 1
    return counts


@jit(nopython=True, parallel=True, fastmath=True)
def count_neighbors_slope_vectorized(local_coords: np.ndarray, slope_angles: np.ndarray, L: float, W: float) -> np.ndarray:
    n = local_coords.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    for i in prange(n):
        center = local_coords[i]
        angle = slope_angles[i]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        delta_lat = (local_coords[:, 1] - center[1]) * 111000.0
        delta_elev = local_coords[:, 2] - center[2]
        delta_along = delta_lat * cos_a + delta_elev * sin_a
        delta_perp = -delta_lat * sin_a + delta_elev * cos_a
        counts[i] = np.sum((np.abs(delta_along) <= L / 2.0) & (np.abs(delta_perp) <= W / 2.0)) - 1
    return counts


def find_local_minimum_fast(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[~np.isnan(counts)]
    n = counts.size
    if n < 10:
        return float(np.median(counts))
    sd = np.std(counts, ddof=1)
    if sd == 0:
        return float(np.median(counts))
    iqr = np.percentile(counts, 75) - np.percentile(counts, 25)
    scale = min(sd, iqr / 1.34) if iqr > 0 else sd
    h = 0.9 * scale * n ** (-1.0 / 5.0)
    h = np.clip(h, 0.5, 2.5)
    hist, bins = np.histogram(counts, bins=200, density=True)
    x = (bins[:-1] + bins[1:]) / 2.0
    dx = bins[1] - bins[0]
    y = gaussian_filter1d(hist, sigma=h / dx)
    dy = np.diff(y)
    sgn = np.sign(dy)
    dsgn = np.diff(sgn)
    minima = np.where(dsgn == 2)[0] + 1
    if minima.size == 0:
        return float(x[np.argmin(y)])
    return float(x[minima[0]])


def fit_spline_and_slopes(lat_arr: np.ndarray, elev_arr: np.ndarray, spar: float = 0.5):
    n = len(lat_arr)
    if n < 80 or np.ptp(lat_arr) < 1e-4 or np.std(elev_arr) < 0.05:
        return np.zeros(n, dtype=np.float64), None
    sort_idx = np.argsort(lat_arr)
    x = lat_arr[sort_idx]
    y = elev_arr[sort_idx]
    dx = np.diff(x)
    keep = np.concatenate(([True], dx > 1e-8))
    x = x[keep]
    y = y[keep]
    if x.size < 80:
        return np.zeros(n, dtype=np.float64), None
    s_min = max(100.0, 0.2 * x.size)
    smoothing = max(x.size * spar, s_min)
    try:
        spline = UnivariateSpline(x, y, s=smoothing, k=3)
    except Exception:
        return np.zeros(n, dtype=np.float64), None
    try:
        dy_dlat = spline.derivative(1)(x) / 111000.0
    except Exception:
        return np.zeros(n, dtype=np.float64), None
    slopes_sorted = np.arctan(dy_dlat)
    slope_angles = np.zeros(n, dtype=np.float64)
    slope_angles[sort_idx[keep]] = slopes_sorted
    return slope_angles, spline


def assign_regions_by_extrema(lat_arr: np.ndarray, spline: Optional[UnivariateSpline]):
    n = len(lat_arr)
    if spline is None or n == 0:
        return np.zeros(n, dtype=np.int32)
    sort_idx = np.argsort(lat_arr)
    x = lat_arr[sort_idx]
    d1 = spline.derivative(1)(x)
    zc = np.where(np.diff(np.sign(d1)) != 0)[0]
    extrema_lat = x[zc + 1] if zc.size > 0 else np.array([], dtype=np.float64)
    breaks = np.concatenate(([-np.inf], extrema_lat, [np.inf]))
    regions_sorted = np.zeros(n, dtype=np.int32)
    for i in range(len(breaks) - 1):
        mask = (x >= breaks[i]) & (x < breaks[i + 1])
        regions_sorted[mask] = i
    regions = np.zeros(n, dtype=np.int32)
    regions[sort_idx] = regions_sorted
    return regions


@jit(nopython=True, fastmath=True)
def final_filter_slope_union(local_coords, slope_angles, regions, L=20.0):
    n = local_coords.shape[0]
    out_mask = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        center = local_coords[i]
        angle = slope_angles[i]
        reg_i = regions[i]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        region_idx = np.where(regions == reg_i)[0]
        if region_idx.size == 0:
            continue
        pts = local_coords[region_idx]
        delta_lat = (pts[:, 1] - center[1]) * 111000.0
        delta_elev = pts[:, 2] - center[2]
        delta_along = delta_lat * cos_a + delta_elev * sin_a
        delta_perp = -delta_lat * sin_a + delta_elev * cos_a
        in_win = (delta_along >= -L / 2.0) & (delta_along <= L / 2.0)
        if np.sum(in_win) < 1:
            continue
        perp_values = delta_perp[in_win]
        m = perp_values.size
        nbins = max(10, m // 5)
        hist_counts, bin_edges = np.histogram(perp_values, bins=nbins)
        mode_bin = np.argmax(hist_counts)
        perp_mode = 0.5 * (bin_edges[mode_bin] + bin_edges[mode_bin + 1])
        perp_min = perp_values.min()
        perp_max = perp_values.max()
        perp_range = perp_max - perp_min
        if perp_range <= 0:
            continue
        perp_low = perp_mode - STAGE5_MODE_FRAC * perp_range
        perp_high = perp_mode + STAGE5_MODE_FRAC * perp_range
        keep_local = (perp_values >= perp_low) & (perp_values <= perp_high)
        in_win_idx_global = region_idx[in_win]
        selected_global = in_win_idx_global[keep_local]
        out_mask[selected_global] = True
    return out_mask


# =============================================================================
# 6. EGM2008 GEOID
# =============================================================================

_GEOID_INTERPOLATOR = None


def get_geoid_interpolator():
    global _GEOID_INTERPOLATOR
    if _GEOID_INTERPOLATOR is not None:
        return _GEOID_INTERPOLATOR
    if not GEOID_TIF.exists():
        raise FileNotFoundError(GEOID_TIF)
    print(f"Loading EGM2008 grid               : {GEOID_TIF}")
    with rasterio.open(GEOID_TIF) as src:
        geoid_data = src.read(1)
        transform = src.transform
        width = src.width
        height = src.height
    lons = np.linspace(transform[2], transform[2] + transform[0] * width, width)
    lats = np.linspace(transform[5], transform[5] + transform[4] * height, height)
    _GEOID_INTERPOLATOR = RegularGridInterpolator((lons, lats), geoid_data.T, method="linear", bounds_error=False, fill_value=0)
    return _GEOID_INTERPOLATOR


def transform_to_egm2008(lons, lats, spline_elevs, interpolator):
    if interpolator is None:
        return spline_elevs.copy(), np.zeros_like(spline_elevs)
    geoid_heights = interpolator((lons, lats))
    egm2008 = spline_elevs - geoid_heights
    return egm2008, geoid_heights


# =============================================================================
# 7. BLOCK-BASED SE PROCESSING WITH NATIVE PHOTON ID TRACKING
# =============================================================================

def make_lat_blocks(lat: np.ndarray) -> np.ndarray:
    lat_span = lat.max() - lat.min()
    if lat_span < 0.05:
        return np.array([lat.min(), lat.max()], dtype=np.float64)
    return np.arange(np.floor(lat.min() * 100.0) / 100.0, np.ceil(lat.max() * 100.0) / 100.0 + 0.01, 0.01)


def compute_block_snr_from_screened(
    screened: Dict[str, np.ndarray],
    block_start: float,
    block_end: float,
) -> float:
    """Calculate SNR exactly as the public code: after land_conf >= 1 screening."""
    in_block = (
        (screened["lat"] >= block_start - 0.001)
        & (screened["lat"] < block_end + 0.001)
    )
    bc = screened["signal_conf"][in_block]
    if bc.size == 0:
        return np.nan

    ns = int(np.sum((bc >= 2) & (bc <= 4)))
    nn = int(np.sum((bc <= 1) | (bc == -1)))

    # Because pre-screening retains confidence >= 1, the denominator is
    # effectively confidence class 1. Preserve the public code's NaN behavior
    # when there are no denominator photons.
    return float(ns / nn) if nn > 0 else np.nan


def process_one_block_screened(
    block_start: float,
    block_end: float,
    screened: Dict[str, np.ndarray],
    beam: str,
    source_file: str,
    geoid_interpolator,
    target_raw_index: int,
    snr_value: float,
    capture_plot_data: bool = False,
    is_last_block: bool = False,
):
    """Process an overlap-expanded context, but emit only core-block photons.

    The +/-0.001 degree overlap is used to stabilize neighborhood and spline
    calculations near block boundaries. Output ownership is non-overlapping:
    a photon is emitted only by the block whose [start, end) core contains it.
    """
    lat_all = screened["lat"]
    lon_all = screened["lon"]
    h_all = screened["h"]
    raw_idx_all = screened["raw_index"]

    log = {
        "source_file": source_file, "beam": beam,
        "block_start": float(block_start), "block_end": float(block_end),
        "n_block_context": 0, "n_block_core": 0, "SNR": np.nan,
        "P1": np.nan, "n_stage1": 0, "P2": np.nan, "n_stage2": 0,
        "spline1_spar": SPLINE_SPAR_STAGE3, "P3": np.nan,
        "n_stage3": 0, "spline2_spar": SPLINE_SPAR_STAGE4,
        "n_stage5_context": 0, "n_output_core": 0,
        "status": "", "target_block": False, "target_block_input": False,
        "target_stage1": False, "target_stage2": False,
        "target_stage3": False, "target_stage5": False,
        "target_emitted_by_core_block": False,
    }

    context_mask = (lat_all >= block_start - 0.001) & (lat_all < block_end + 0.001)
    core_mask = (lat_all >= block_start) & (lat_all < block_end)
    if is_last_block:
        core_mask |= np.isclose(lat_all, block_end, atol=1e-10)
    n_context = int(context_mask.sum())
    n_core = int(core_mask.sum())
    log["n_block_context"] = n_context
    log["n_block_core"] = n_core
    if n_context < MIN_BLOCK_PHOTONS:
        log["status"] = "skip_context_lt_minimum"
        return [], log, None

    raw_idx = raw_idx_all[context_mask]
    lat = lat_all[context_mask]
    lon = lon_all[context_mask]
    h = h_all[context_mask]
    log["SNR"] = snr_value
    log["target_block_input"] = bool(np.any(raw_idx == target_raw_index))
    log["target_block"] = log["target_block_input"]
    if np.isfinite(snr_value) and snr_value < SNR_MIN:
        log["status"] = "skip_SNR_below_threshold"
        return [], log, None

    coords = np.column_stack((lon, lat, h))

    c1 = count_neighbors_rect_vectorized(coords, STAGE1_L, STAGE1_W)
    P1 = find_local_minimum_fast(c1)
    keep1 = c1 >= P1
    s1, raw1 = coords[keep1], raw_idx[keep1]
    log["P1"], log["n_stage1"] = P1, len(s1)
    log["target_stage1"] = bool(np.any(raw1 == target_raw_index))
    if len(s1) < 10:
        log["status"] = "skip_stage1_lt_10"
        return [], log, None

    c2 = count_neighbors_rect_vectorized(s1, STAGE2_L, STAGE2_W)
    P2 = min(find_local_minimum_fast(c2), np.floor(P1) / 2.0)
    keep2 = c2 >= P2
    s2, raw2 = s1[keep2], raw1[keep2]
    log["P2"], log["n_stage2"] = P2, len(s2)
    log["target_stage2"] = bool(np.any(raw2 == target_raw_index))
    if len(s2) < 8:
        log["status"] = "skip_stage2_lt_8"
        return [], log, None

    slope2, spline2 = fit_spline_and_slopes(s2[:, 1], s2[:, 2], spar=SPLINE_SPAR_STAGE3)
    regions2 = assign_regions_by_extrema(s2[:, 1], spline2)
    c3 = count_neighbors_slope_vectorized(s2, slope2, STAGE3_L, STAGE3_W)
    P3 = min(find_local_minimum_fast(c3), np.floor(P2) / 2.0)
    keep3 = c3 >= P3
    s3, raw3 = s2[keep3], raw2[keep3]
    log["P3"], log["n_stage3"] = P3, len(s3)
    log["target_stage3"] = bool(np.any(raw3 == target_raw_index))
    if len(s3) < 5:
        log["status"] = "skip_stage3_lt_5"
        return [], log, None

    regions3 = regions2[keep3]
    slope_new, _ = fit_spline_and_slopes(s3[:, 1], s3[:, 2], spar=SPLINE_SPAR_STAGE4)
    mask5 = final_filter_slope_union(s3, slope_new, regions3, L=STAGE5_L)
    final_context = s3[mask5]
    raw5_context = raw3[mask5]
    log["n_stage5_context"] = int(mask5.sum())
    log["target_stage5"] = bool(np.any(raw5_context == target_raw_index))
    if len(final_context) < 3:
        log["status"] = "skip_stage5_lt_3"
        return [], log, None

    # Stage 6 fit uses the overlap-expanded context, preventing edge artifacts.
    lat_context = final_context[:, 1]
    lon_context = final_context[:, 0]
    elev_context = final_context[:, 2]
    if len(lat_context) < GAM_MIN_POINTS:
        elev_spline_context = elev_context.copy()
    else:
        x_norm = (lat_context - lat_context.mean()) / (np.ptp(lat_context) + 1e-9)
        if LinearGAM is None or s is None:
            raise ImportError(
                "The public Stage 6 algorithm requires pygam. "
                "Install it with: pip install pygam. "
                "No alternative spline fallback is used in this demonstration."
            )

        gam = LinearGAM(
            s(0, n_splines=GAM_N_SPLINES, spline_order=GAM_SPLINE_ORDER),
            fit_intercept=True,
        ).fit(x_norm.reshape(-1, 1), elev_context)
        elev_spline_context = gam.predict(x_norm.reshape(-1, 1))

    # Emit only records owned by this block's non-overlapping core interval.
    emit_mask = (lat_context >= block_start) & (lat_context < block_end)
    if is_last_block:
        emit_mask |= np.isclose(lat_context, block_end, atol=1e-10)
    if not np.any(emit_mask):
        log["status"] = "ok_no_core_photons"
        plot_data = None
        return [], log, plot_data

    final_core = final_context[emit_mask]
    raw5_core = raw5_context[emit_mask]
    lon_core = lon_context[emit_mask]
    lat_core = lat_context[emit_mask]
    elev_core = elev_context[emit_mask]
    elev_spline_core = elev_spline_context[emit_mask]
    geoid_core, geoid_height_core = transform_to_egm2008(
        lon_core, lat_core, elev_spline_core, geoid_interpolator
    )
    log["n_output_core"] = len(final_core)
    log["target_emitted_by_core_block"] = bool(np.any(raw5_core == target_raw_index))
    log["status"] = "ok"

    rows = []
    for i in range(len(final_core)):
        rows.append((
            source_file, beam, int(raw5_core[i]), float(lon_core[i]), float(lat_core[i]),
            float(elev_core[i]), float(elev_spline_core[i]), float(geoid_core[i]),
            float(geoid_height_core[i]), float(snr_value), float(block_start),
            float(block_end),
        ))

    plot_data = None
    if capture_plot_data:
        plot_data = {
            "raw_block_coords": coords, "raw_block_index": raw_idx,
            "stage1_coords": s1, "stage1_index": raw1,
            "stage2_coords": s2, "stage2_index": raw2,
            "stage3_coords": s3, "stage3_index": raw3,
            "stage5_coords": final_context, "stage5_index": raw5_context,
            "stage6_lon": lon_context, "stage6_lat": lat_context,
            "stage6_elev_raw": elev_context,
            "stage6_elev_spline": elev_spline_context,
            "core_mask_stage6": emit_mask,
            "block_start": block_start, "block_end": block_end,
        }
    return rows, log, plot_data


def deduplicate_se_by_native_index(se: pd.DataFrame, beam_label: str) -> pd.DataFrame:
    if se.empty:
        return se
    duplicated = se[se.duplicated(subset=["raw_index"], keep=False)]
    n_dup_ids = int(duplicated["raw_index"].nunique())
    if n_dup_ids:
        preview = duplicated["raw_index"].drop_duplicates().head(10).tolist()
        raise RuntimeError(
            f"Core-block ownership failed for {beam_label}: {n_dup_ids} native ATL03 photon IDs "
            f"were emitted more than once. Example raw_index values: {preview}"
        )
    print(f"Duplicate native ATL03 photon IDs after core ownership ({beam_label}) : 0")
    return se.reset_index(drop=True)



def process_target_beam(raw: Dict[str, np.ndarray], raw_path: Path, beam: str,
                        target_lon: float, target_lat: float):
    target_raw_index = identify_target_raw_index(raw, target_lon, target_lat, f"{raw_path.name} | {beam}")
    screened = pre_screen_raw_beam(raw, raw_path, beam)
    lat = screened["lat"]
    lat_blocks = raw["block_edges"].copy()
    print(f"Local raw latitude span            : {np.ptp(lat):.6f} deg")
    print(f"Full-track latitude span            : not loaded for denoising")
    print(f"Processing core blocks              : {len(lat_blocks) - 1}")
    print(f"Block range                         : {lat_blocks[0]:.4f} to {lat_blocks[-1]:.4f} deg")
    print(f"Neighbor blocks each side           : {NEIGHBOR_BLOCKS_EACH_SIDE}")
    geoid_interpolator = get_geoid_interpolator()
    all_rows, logs = [], []
    target_plot_data = None
    for block_i, (block_start, block_end) in enumerate(zip(lat_blocks[:-1], lat_blocks[1:])):
        snr_value = compute_block_snr_from_screened(screened, block_start, block_end)
        capture = (block_start <= raw["target_lat_native"] < block_end)
        if block_i == len(lat_blocks) - 2:
            capture = capture or np.isclose(raw["target_lat_native"], block_end, atol=1e-10)
        rows, block_log, plot_data = process_one_block_screened(
            block_start, block_end, screened, beam, raw_path.name,
            geoid_interpolator, target_raw_index, snr_value,
            capture_plot_data=(capture and target_plot_data is None),
            is_last_block=(block_i == len(lat_blocks) - 2),
        )
        all_rows.extend(rows)
        logs.append(block_log)
        if plot_data is not None and target_plot_data is None:
            target_plot_data = plot_data

    se = pd.DataFrame(
        all_rows,
        columns=[
            "file", "beam", "raw_index", "lon", "lat", "elevation_wgs84",
            "elevation_spline", "elevation_egm2008", "geoid_height", "SNR",
            "block_start", "block_end",
        ],
    )
    log_df = pd.DataFrame(logs)
    n_ok = int(np.sum(log_df["status"].astype(str).str.startswith("ok")))
    print(f"Accepted local blocks               : {n_ok}/{len(lat_blocks)-1}")
    print(f"SE photons emitted by core blocks    : {len(se):,}")
    se = deduplicate_se_by_native_index(se, beam)
    print(f"Unique native ATL03 photons          : {len(se):,}")

    core_all = (screened["lat"] >= lat_blocks[0]) & (screened["lat"] < lat_blocks[-1])
    if len(lat_blocks) > 1:
        core_all |= np.isclose(screened["lat"], lat_blocks[-1], atol=1e-10)
    n_prescreen_core = int(core_all.sum())
    recovery_fraction = len(se) / n_prescreen_core if n_prescreen_core else 0.0
    print(f"Pre-screened photons in selected cores: {n_prescreen_core:,}")
    print(f"Local SE recovery fraction            : {recovery_fraction:.3f}")

    target_logs = log_df[log_df["target_block_input"]].copy()
    sub_section(f"DETAILED TARGET-NEIGHBORHOOD LOG | {raw_path.name} | {beam}")
    columns = [
        "block_start", "block_end", "n_block_context", "n_block_core", "SNR",
        "P1", "n_stage1", "P2", "n_stage2", "spline1_spar", "P3", "n_stage3",
        "spline2_spar", "n_stage5_context", "n_output_core", "status",
        "target_block_input", "target_stage1", "target_stage2", "target_stage3",
        "target_stage5", "target_emitted_by_core_block",
    ]
    print(log_df[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    raw_target_pos = np.flatnonzero(raw["raw_index"] == target_raw_index)
    screened_target_pos = np.flatnonzero(screened["raw_index"] == target_raw_index)
    se_target_pos = np.flatnonzero(se["raw_index"].to_numpy() == target_raw_index)
    print(f"Target in local raw subset           : {len(raw_target_pos) == 1}")
    print(f"Target after pre-screen              : {len(screened_target_pos) == 1}")
    print(f"Target in local SE output            : {len(se_target_pos) == 1}")
    return se, screened, log_df, target_raw_index, target_plot_data, recovery_fraction


# =============================================================================
# 8. SEC PAIRING HELPERS
# =============================================================================

@jit(nopython=True, fastmath=True, nogil=True)
def haversine_distance(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R_EARTH * c


@jit(nopython=True, nogil=True)
def filter_points_in_buffer_numba(lats, lons, center_lat, center_lon, radius):
    n = len(lats)
    mask = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        mask[i] = haversine_distance(lats[i], lons[i], center_lat, center_lon) <= radius
    return mask


def extract_date_from_filename(filename: str):
    try:
        return datetime.strptime(filename[6:14], "%Y%m%d")
    except (ValueError, TypeError, IndexError):
        return None


def calculate_time_diff(date1, date2):
    if date1 is None or date2 is None:
        return 0, 0.0
    months = (date2.year - date1.year) * 12 + (date2.month - date1.month)
    years = (date2 - date1).days / 365.25
    return months, years


def lonlat_to_unitvec(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.radians(lon_deg.astype(np.float64))
    lat = np.radians(lat_deg.astype(np.float64))
    clat = np.cos(lat)
    x = clat * np.cos(lon)
    y = clat * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack((x, y, z))


def find_close_pairs_with_diagnostics(df1: pd.DataFrame, df2: pd.DataFrame, max_distance_m: float):
    if len(df1) == 0 or len(df2) == 0:
        return [], []
    P1 = lonlat_to_unitvec(df1["lon"].values, df1["lat"].values)
    P2 = lonlat_to_unitvec(df2["lon"].values, df2["lat"].values)
    theta = max_distance_m / R_EARTH
    r_chord = 2.0 * np.sin(theta * 0.5)
    tree = cKDTree(P2)
    candidate_lists = tree.query_ball_point(P1, r=r_chord, workers=1)
    lat1 = df1["lat"].values.astype(np.float64)
    lon1 = df1["lon"].values.astype(np.float64)
    lat2 = df2["lat"].values.astype(np.float64)
    lon2 = df2["lon"].values.astype(np.float64)
    candidates = []
    for i, js in enumerate(candidate_lists):
        if not js:
            continue
        for j in js:
            d = haversine_distance(lat1[i], lon1[i], lat2[j], lon2[j])
            if d <= max_distance_m:
                candidates.append((i, j, float(d)))
    candidates.sort(key=lambda x: x[2])
    matched_i = set()
    matched_j = set()
    accepted = []
    for i, j, d in candidates:
        if i not in matched_i and j not in matched_j:
            accepted.append((i, j, d))
            matched_i.add(i)
            matched_j.add(j)
    return candidates, accepted


def build_beam_object(se: pd.DataFrame, source_file: str, beam_name: str):
    required = ["raw_index", "lon", "lat", "elevation_egm2008"]
    missing = [c for c in required if c not in se.columns]
    if missing:
        raise RuntimeError(f"SE table lacks required native identity fields: {missing}")
    if se["raw_index"].duplicated().any():
        raise RuntimeError(f"Duplicate native ATL03 photon IDs remain in {source_file} {beam_name}.")
    df = se[required].copy().reset_index(drop=True)
    return {
        "df": df,
        "date_obj": extract_date_from_filename(source_file),
        "file": source_file,
        "beam_name": beam_name,
    }


def pairing_all_crossings(beam1: dict, beam2: dict, target_raw_idx1: int, target_raw_idx2: int):
    """Global one-to-one greedy matching for ONE two-beam observation comparison.

    All <=2 m photon candidates from the two reconstructed target-neighborhood datasets
    are ranked by distance. Each SE photon can be assigned at most once within
    this beam-observation pair. This constraint is not shared across other
    acquisition-pair comparisons.
    """
    section("STEP 5 | ALL CROSSING CANDIDATES AND ONE-TO-ONE MATCHING")
    df1, df2 = beam1["df"], beam2["df"]
    candidates, accepted = find_close_pairs_with_diagnostics(
        df1[["lon", "lat", "elevation_egm2008"]].rename(columns={"elevation_egm2008": "elevation"}),
        df2[["lon", "lat", "elevation_egm2008"]].rename(columns={"elevation_egm2008": "elevation"}),
        MAX_PAIR_DISTANCE_M,
    )
    print(f"Candidate photon pairs <= {MAX_PAIR_DISTANCE_M:.1f} m in selected target-neighbor blocks : {len(candidates):,}")
    print(f"Accepted one-to-one pairs for this beam-observation comparison                         : {len(accepted):,}")
    used1 = [i for i, _, _ in accepted]
    used2 = [j for _, j, _ in accepted]
    print(f"Repeated beam-1 SE indices within this comparison : {len(used1) - len(set(used1))}")
    print(f"Repeated beam-2 SE indices within this comparison : {len(used2) - len(set(used2))}")
    if len(used1) != len(set(used1)) or len(used2) != len(set(used2)):
        raise RuntimeError("One-to-one matching invariant failed.")

    d1, d2 = beam1["date_obj"], beam2["date_obj"]
    if d1 is None or d2 is None:
        raise RuntimeError("Could not parse acquisition dates from source filenames.")
    months, years = calculate_time_diff(d1, d2)
    rows = []
    for i, j, distance in accepted:
        r1, r2 = df1.iloc[i], df2.iloc[j]
        if d1 <= d2:
            p_early, p_late = r1, r2
            f_early, f_late = beam1, beam2
        else:
            p_early, p_late = r2, r1
            f_early, f_late = beam2, beam1
        dh = float(p_late.elevation_egm2008 - p_early.elevation_egm2008)
        rows.append({
            "group_id": 1,  # all accepted pairs belong to this single acquisition-pair comparison
            "photon1_file": f_early["file"], "photon1_beam": f_early["beam_name"],
            "photon1_raw_index": int(p_early.raw_index),
            "photon1_lon": float(p_early.lon), "photon1_lat": float(p_early.lat),
            "photon1_elevation": float(p_early.elevation_egm2008),
            "photon2_file": f_late["file"], "photon2_beam": f_late["beam_name"],
            "photon2_raw_index": int(p_late.raw_index),
            "photon2_lon": float(p_late.lon), "photon2_lat": float(p_late.lat),
            "photon2_elevation": float(p_late.elevation_egm2008),
            "distance": float(distance), "elevation_diff": dh,
            "months_diff": abs(months), "years_diff": abs(years),
            "monthly_change_rate": dh / abs(months) if abs(months) else np.nan,
            "yearly_change_rate": dh / abs(years) if abs(years) else np.nan,
        })
    reproduced = pd.DataFrame(rows)

    # Target candidate diagnostics are based on native ATL03 photon IDs.
    if d1 <= d2:
        target_early, target_late = target_raw_idx1, target_raw_idx2
    else:
        target_early, target_late = target_raw_idx2, target_raw_idx1
    target_candidate = any(
        int(df1.iloc[i].raw_index) == target_raw_idx1 and int(df2.iloc[j].raw_index) == target_raw_idx2
        for i, j, _ in candidates
    )
    target_accepted = (
        not reproduced.empty
        and bool(((reproduced.photon1_raw_index == target_early) &
                  (reproduced.photon2_raw_index == target_late)).any())
    )
    print(f"Exact target raw-index pair in <=2 m candidate set : {target_candidate}")
    print(f"Exact target raw-index pair accepted               : {target_accepted}")
    if target_candidate and not target_accepted:
        print("[TARGET] Candidate was rejected by one-to-one competition with a closer candidate.")
    elif not target_candidate:
        print("[TARGET] Target photons survived SE, but their reconstructed separation exceeds the candidate threshold or pair orientation needs review.")
    return reproduced, candidates, accepted


def select_exact_target_row(selected: pd.Series, reproduced: pd.DataFrame, target_raw_idx1: int, target_raw_idx2: int):
    section("STEP 6 | VERIFY EXACTLY THE ORIGINALLY SELECTED PAIR")
    if reproduced.empty:
        print("TARGET RECOVERED = False (no accepted pairs)")
        return None
    # Selected Parquet photon1/photon2 are tied to file1/file2 from the selected row.
    # Convert to chronological output order using the filenames in the output table.
    file1_date = extract_date_from_filename(str(selected["photon1_file"])) if "photon1_file" in selected.index else None
    file2_date = extract_date_from_filename(str(selected["photon2_file"])) if "photon2_file" in selected.index else None
    if file1_date is not None and file2_date is not None and file1_date > file2_date:
        early_id, late_id = target_raw_idx2, target_raw_idx1
    else:
        early_id, late_id = target_raw_idx1, target_raw_idx2
    exact = reproduced[(reproduced.photon1_raw_index == early_id) & (reproduced.photon2_raw_index == late_id)]
    print(f"Accepted pairs matching BOTH original native ATL03 IDs : {len(exact)}")
    if len(exact) != 1:
        print("TARGET RECOVERED = False; no nearby substitute is counted as the target.")
        return None
    row = exact.iloc[0]
    # Compare released coordinates in chronological order.
    if file1_date is not None and file2_date is not None and file1_date > file2_date:
        released_early = (float(selected["photon2_lon"]), float(selected["photon2_lat"]))
        released_late = (float(selected["photon1_lon"]), float(selected["photon1_lat"]))
    else:
        released_early = (float(selected["photon1_lon"]), float(selected["photon1_lat"]))
        released_late = (float(selected["photon2_lon"]), float(selected["photon2_lat"]))
    e1 = float(distance_m(row.photon1_lat, row.photon1_lon, released_early[1], released_early[0]))
    e2 = float(distance_m(row.photon2_lat, row.photon2_lon, released_late[1], released_late[0]))
    print("TARGET RECOVERED = True (both native ATL03 photon IDs retained)")
    print(f"  photon 1 coordinate error : {e1:.6f} m")
    print(f"  photon 2 coordinate error : {e2:.6f} m")
    print(f"  released dh               : {float(selected['elevation_diff']):+.6f} m")
    print(f"  reconstructed dh          : {float(row.elevation_diff):+.6f} m")
    print(f"  elevation difference delta: {float(row.elevation_diff)-float(selected['elevation_diff']):+.6f} m")
    return {"index": int(exact.index[0]), "row": row, "d1": e1, "d2": e2, "recovered": True}


# =============================================================================
# 9. TARGET LINEAGE CHECK
# =============================================================================

def check_target_beam(raw, screened, se, target_raw_index, beam_tag, log_df):
    sub_section(f"TARGET NATIVE PHOTON IDENTITY | {beam_tag}")
    row_raw = np.flatnonzero(raw["raw_index"] == target_raw_index)
    row_scr = np.flatnonzero(screened["raw_index"] == target_raw_index)
    row_se = np.flatnonzero(se["raw_index"].to_numpy() == target_raw_index)
    print(f"  Original ATL03                 {'PRESENT' if len(row_raw)==1 else 'ABSENT'} | row={int(row_raw[0]) if len(row_raw)==1 else 'NA'}")
    print(f"  After pre-screen               {'PRESENT' if len(row_scr)==1 else 'ABSENT'} | row={int(row_scr[0]) if len(row_scr)==1 else 'NA'}")
    print(f"  After SE core-block output      {'PRESENT' if len(row_se)==1 else 'ABSENT'} | row={int(row_se[0]) if len(row_se)==1 else 'NA'}")
    target_logs = log_df[log_df["target_block_input"]].copy()
    if not target_logs.empty:
        print(f"  Overlap contexts containing target : {len(target_logs)}")
        for stage in ["target_block_input", "target_stage1", "target_stage2", "target_stage3", "target_stage5"]:
            print(f"  {stage:<32} {bool(target_logs[stage].any())}")
        n_owner = int(target_logs["target_emitted_by_core_block"].sum())
        print(f"  Core block(s) emitting target       : {n_owner}")
        if n_owner != 1 and len(row_se) == 1:
            raise RuntimeError(f"{beam_tag}: target SE row exists but core-block ownership count is {n_owner}, expected exactly one.")
    raw_pos = int(row_raw[0]) if len(row_raw) == 1 else None
    if raw_pos is not None:
        print(f"  Target raw land_conf           : {int(raw['land_conf'][raw_pos])}")
    if len(row_se) > 1:
        raise RuntimeError(f"{beam_tag}: target native photon ID was emitted more than once.")
    return (len(row_raw) == 1) and (len(row_scr) == 1) and (len(row_se) == 1)




# =============================================================================
# 10. VISUALIZATION
# =============================================================================

def plot_spatial_context(tp_gdf, grid_gdf, raw1, raw2, selected, grid_name, file1, file2, beam1, beam2):
    """Left: full Tibetan Plateau context; right: selected-grid detail."""
    section("FIGURE 1 | REGIONAL AND GRID-SCALE ATL03 SPATIAL CONTEXT")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    tp_gdf.boundary.plot(ax=ax, color="0.35", linewidth=0.65, zorder=1)
    grid_gdf.boundary.plot(ax=ax, color="black", linewidth=1.35, zorder=3)
    i1 = sample_indices(len(raw1["lon"]), MAX_MAP_DISPLAY_POINTS)
    i2 = sample_indices(len(raw2["lon"]), MAX_MAP_DISPLAY_POINTS)
    ax.scatter(raw1["lon"][i1], raw1["lat"][i1], s=0.22, color=BEAM1_COLOR, alpha=0.75, rasterized=True, zorder=4)
    ax.scatter(raw2["lon"][i2], raw2["lat"][i2], s=0.22, color=BEAM2_COLOR, alpha=0.75, rasterized=True, zorder=5)
    # Full TP extent, not the selected-grid extent.
    xmin, ymin, xmax, ymax = tp_gdf.total_bounds
    dx, dy = (xmax - xmin) * 0.025, (ymax - ymin) * 0.025
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)
    ax.set_xlabel("Longitude (\u00b0E)")
    ax.set_ylabel("Latitude (\u00b0N)")
    ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean([ymin, ymax])))), adjustable="box")
    # Filename labels are color-matched and placed in a neat legend outside the map.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=BEAM1_COLOR, lw=2.2, label=Path(file1).name),
        Line2D([0], [0], color=BEAM2_COLOR, lw=2.2, label=Path(file2).name),
        Line2D([0], [0], color="black", lw=1.4, label=grid_name),
        Line2D([0], [0], color="0.35", lw=0.8, label="Tibetan Plateau boundary"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.9, fontsize=7.2, borderpad=0.45, handlelength=1.8)

    ax = axes[1]
    grid_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=2)
    ax.scatter(raw1["lon"][i1], raw1["lat"][i1], s=0.65, color=BEAM1_COLOR, alpha=0.65, rasterized=True, zorder=3)
    ax.scatter(raw2["lon"][i2], raw2["lat"][i2], s=0.65, color=BEAM2_COLOR, alpha=0.65, rasterized=True, zorder=4)
    ax.scatter([selected["photon1_lon"], selected["photon2_lon"]],
               [selected["photon1_lat"], selected["photon2_lat"]],
               marker="x", s=78, linewidths=1.8, color=TARGET_COLOR, zorder=10)
    gx0, gy0, gx1, gy1 = grid_gdf.total_bounds
    padx = max((gx1 - gx0) * 0.035, 0.004)
    pady = max((gy1 - gy0) * 0.035, 0.004)
    ax.set_xlim(gx0 - padx, gx1 + padx)
    ax.set_ylim(gy0 - pady, gy1 + pady)
    ax.set_xlabel("Longitude (\u00b0E)")
    ax.set_ylabel("Latitude (\u00b0N)")
    ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean([gy0, gy1])))), adjustable="box")
    # Dates and beam IDs are explicitly color-coded in the right panel.
    d1 = extract_date_from_filename(file1)
    d2 = extract_date_from_filename(file2)
    date1 = d1.strftime("%d %b %Y") if d1 else Path(file1).name
    date2 = d2.strftime("%d %b %Y") if d2 else Path(file2).name
    ax.text(0.025, 0.975, f"{date1}  |  {beam1}", transform=ax.transAxes,
            ha="left", va="top", color=BEAM1_COLOR, fontsize=9.2, weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2.0))
    ax.text(0.025, 0.925, f"{date2}  |  {beam2}", transform=ax.transAxes,
            ha="left", va="top", color=BEAM2_COLOR, fontsize=9.2, weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2.0))
    ax.text(0.975, 0.025, "Released target pair", transform=ax.transAxes,
            ha="right", va="bottom", color=TARGET_COLOR, fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2.0))
    fig.tight_layout()
    plt.show()



def plot_denoising_progress(plot1: dict, plot2: dict, target_raw_idx1: int, target_raw_idx2: int):
    """One final target-block profile per beam; no separate plot for each filter stage."""
    section("FIGURE 2 | TARGET-BLOCK PHOTONS AND RECONSTRUCTED SURFACE")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    for ax, pdata, color, target_idx, beam_label in [
        (axes[0], plot1, BEAM1_COLOR, target_raw_idx1, "Beam 1"),
        (axes[1], plot2, BEAM2_COLOR, target_raw_idx2, "Beam 2"),
    ]:
        raw = pdata["raw_block_coords"]
        idx = sample_indices(len(raw), min(80000, len(raw)))
        ax.scatter(raw[idx, 1], raw[idx, 2], s=0.45, color="0.72", alpha=0.32,
                   rasterized=True, label="Pre-screened ATL03 photons")
        final = pdata["stage5_coords"]
        ax.scatter(final[:, 1], final[:, 2], s=2.0, color=color, alpha=0.72,
                   rasterized=True, label="Retained surface photons")
        ax.plot(pdata["stage6_lat"], pdata["stage6_elev_spline"], color="black",
                linewidth=1.45, label="Stage 6 reconstructed surface")
        pos = np.flatnonzero(pdata["stage5_index"] == target_idx)
        if len(pos):
            q = final[int(pos[0])]
            ax.scatter([q[1]], [q[2]], marker="*", s=115, color=TARGET_COLOR,
                       edgecolor="white", linewidth=0.55, zorder=12, label="Target photon")
        ax.set_xlabel("Latitude (\u00b0N)")
        ax.set_ylabel("Elevation (m, WGS84 ellipsoid)")
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.text(0.02, 0.97, beam_label, transform=ax.transAxes, ha="left", va="top",
                color=color, weight="bold", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "Figure2_photon_reconstruction")
    plt.show()


def plot_target_and_other_pairs(se1, se2, reproduced, candidates, selected, target):
    """Clear all-pairs overview plus target-centered metric close-up."""
    section("FIGURE 3 | SPATIAL DISTRIBUTION OF CANDIDATE AND ACCEPTED PAIRS")
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Panel 1: full reconstructed footprint and all accepted pair midpoints.
    ax = axes[0]
    ix1 = sample_indices(len(se1), min(65000, len(se1)))
    ix2 = sample_indices(len(se2), min(65000, len(se2)))
    ax.scatter(se1.lon.to_numpy()[ix1], se1.lat.to_numpy()[ix1], s=0.25, color=BEAM1_COLOR, alpha=0.35, rasterized=True, label="Beam 1 SE photons")
    ax.scatter(se2.lon.to_numpy()[ix2], se2.lat.to_numpy()[ix2], s=0.25, color=BEAM2_COLOR, alpha=0.35, rasterized=True, label="Beam 2 SE photons")
    if not reproduced.empty:
        mx = 0.5 * (reproduced.photon1_lon.to_numpy() + reproduced.photon2_lon.to_numpy())
        my = 0.5 * (reproduced.photon1_lat.to_numpy() + reproduced.photon2_lat.to_numpy())
        ax.scatter(mx, my, s=18, facecolors="none", edgecolors="0.25", linewidths=0.65, label=f"Accepted pairs (n={len(reproduced)})", zorder=8)
    target_lon = 0.5 * (float(selected.photon1_lon) + float(selected.photon2_lon))
    target_lat = 0.5 * (float(selected.photon1_lat) + float(selected.photon2_lat))
    ax.scatter([target_lon], [target_lat], marker="*", s=145, color=TARGET_COLOR, edgecolor="white", linewidth=0.6, zorder=15, label="Released target location")
    ax.set_xlabel("Longitude (\u00b0E)")
    ax.set_ylabel("Latitude (\u00b0N)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=True, fontsize=8, loc="best")

    # Panel 2: metric local ENU coordinates, all <=2m candidates and accepted links.
    ax = axes[1]
    lat0, lon0 = target_lat, target_lon
    def to_local_xy(lon, lat):
        x = (np.asarray(lon) - lon0) * 111000.0 * np.cos(np.radians(lat0))
        y = (np.asarray(lat) - lat0) * 111000.0
        return x, y
    radius = 15.0
    m1 = distance_m(se1.lat.to_numpy(), se1.lon.to_numpy(), lat0, lon0) <= radius
    m2 = distance_m(se2.lat.to_numpy(), se2.lon.to_numpy(), lat0, lon0) <= radius
    x1, y1 = to_local_xy(se1.loc[m1, "lon"], se1.loc[m1, "lat"])
    x2, y2 = to_local_xy(se2.loc[m2, "lon"], se2.loc[m2, "lat"])
    ax.scatter(x1, y1, s=17, color=BEAM1_COLOR, alpha=0.8, label="Beam 1 SE photons", zorder=4)
    ax.scatter(x2, y2, s=17, color=BEAM2_COLOR, alpha=0.8, label="Beam 2 SE photons", zorder=4)

    # Plot every candidate connection in a very light neutral tone.
    if candidates:
        for i, j, d in candidates:
            p1 = se1.iloc[int(i)]
            p2 = se2.iloc[int(j)]
            if float(distance_m(0.5*(p1.lat+p2.lat), 0.5*(p1.lon+p2.lon), lat0, lon0)) <= radius:
                xa, ya = to_local_xy([p1.lon, p2.lon], [p1.lat, p2.lat])
                ax.plot(xa, ya, color="0.78", lw=0.55, alpha=0.35, zorder=1)

    # Accepted pairs in dark teal; target pair in red and thicker.
    if not reproduced.empty:
        for _, r in reproduced.iterrows():
            mid_dist = float(distance_m(0.5*(r.photon1_lat+r.photon2_lat), 0.5*(r.photon1_lon+r.photon2_lon), lat0, lon0))
            if mid_dist > radius:
                continue
            xa, ya = to_local_xy([r.photon1_lon, r.photon2_lon], [r.photon1_lat, r.photon2_lat])
            ax.plot(xa, ya, color="#34495e", lw=1.05, alpha=0.85, zorder=6)
    if target is not None:
        r = target["row"]
        xa, ya = to_local_xy([r.photon1_lon, r.photon2_lon], [r.photon1_lat, r.photon2_lat])
        ax.plot(xa, ya, color=TARGET_COLOR, lw=2.8, marker="o", markersize=5.5, zorder=12, label="Recovered target pair")
    else:
        xa, ya = to_local_xy([selected.photon1_lon, selected.photon2_lon], [selected.photon1_lat, selected.photon2_lat])
        ax.plot(xa, ya, color=TARGET_COLOR, lw=2.4, marker="x", markersize=8, zorder=12, label="Released target pair (not accepted)")

    ax.axhline(0, color="0.88", lw=0.6, zorder=0)
    ax.axvline(0, color="0.88", lw=0.6, zorder=0)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    # Slightly widen the local geometry panel for readability.
    # The horizontal axis is intentionally expanded relative to the vertical axis;
    # coordinates and pairing distances in the underlying data remain unchanged.
    ax.set_aspect(0.55, adjustable="box")
    ax.set_xlabel("East\u2013west distance from target (m)")
    ax.set_ylabel("North\u2013south distance from target (m)")
    ax.legend(frameon=True, fontsize=8, loc="best")
    fig.tight_layout()
    plt.show()


# =============================================================================
# 11. PARAMETER LOG
# =============================================================================

def print_fixed_parameter_log():
    section("FIXED PROCESSING SETTINGS AND MATCHING LEVEL")
    print("PRE-SCREENING")
    print(f"  Minimum raw photons per beam       : {MIN_BEAM_PHOTONS}")
    print(f"  Land signal confidence             : >= {LAND_CONF_MIN}")
    print(f"  Minimum photons after pre-screen   : {MIN_BEAM_PHOTONS}")
    print(f"  Minimum photons per processing context : {MIN_BLOCK_PHOTONS}")
    print(f"  Block SNR threshold                : {SNR_MIN}")
    print("  SNR definition                     : calculated after land_conf >= 1 pre-screening")
    print("                                      N(signal_conf=2..4) / N(signal_conf<=1 or signal_conf=-1)")
    print("  SNR zero-denominator handling       : NaN; as in public code, NaN does not trigger SNR rejection")
    print("  Pre-screen operation               : retain photons with land_conf >= 1; preserve original ATL03 row index")
    print("\nDENOISING")
    print(f"  Stage 1 rectangular window         : {STAGE1_L:.0f} x {STAGE1_W:.0f} m")
    print(f"  Stage 2 rectangular window         : {STAGE2_L:.0f} x {STAGE2_W:.0f} m")
    print(f"  Stage 3 slope-constrained window   : {STAGE3_L:.0f} x {STAGE3_W:.0f} m")
    print(f"  Stage 3 spline spar                : {SPLINE_SPAR_STAGE3}")
    print(f"  Stage 4 spline spar                : {SPLINE_SPAR_STAGE4}")
    print("  Stage 3/4 spline implementation    : scipy.interpolate.UnivariateSpline, cubic (k=3)")
    print("  Stage 3/4 smoothing parameter      : s=max(n*spar, 100, 0.2*n), n=unique latitude positions")
    print(f"  Stage 5 along-slope window         : {STAGE5_L:.0f} m")
    print(f"  Stage 5 modal interval             : +/- {STAGE5_MODE_FRAC:.2f} of local perpendicular range")
    print(f"  Stage 6 GAM                        : {GAM_N_SPLINES} splines, order {GAM_SPLINE_ORDER}, used when n >= {GAM_MIN_POINTS}")
    print("  Block edge handling                : +/-0.001 degree overlap used as context; output emitted only by non-overlapping core block")
    print("\nCROSSOVER PAIRING")
    print(f"  Maximum photon-pair distance       : {MAX_PAIR_DISTANCE_M:.1f} m")
    print("  Candidate search domain            : all reconstructed SE photons emitted by the three selected non-overlapping core blocks in each fixed beam")
    print("  Candidate distance                 : geodesic horizontal distance (haversine), threshold <= 2.0 m")
    print("  One-to-one enforcement level       : entire fixed file-pair comparison (all three core blocks pooled per beam)")
    print("  Matching rule                      : sort all <=2 m candidates by ascending distance; greedily accept only if neither endpoint photon has been used")
    print("  Uniqueness scope                   : each SE photon used at most once within this one file-pair comparison")
    print("  Across other acquisition pairs     : no cross-comparison/global uniqueness constraint")
    print("  Pair elevation difference          : later-acquisition EGM2008 elevation minus earlier-acquisition EGM2008 elevation")
    print("  Outputs                            : candidate count, accepted count, duplicate-use checks, distance and elevation-difference summaries")


# =============================================================================
# 12. MAIN
# =============================================================================




# =============================================================================
# FIXED-FILE DEMONSTRATION (NO PRESELECTED TARGET PAIR)
# =============================================================================

FIXED_GRID = "Grid_003"
FIXED_FILE1 = RAW_ATL03_ROOT / FIXED_GRID / "ATL03_20181118201153_07820106_006_02_subsetted.h5"
FIXED_FILE2 = RAW_ATL03_ROOT / FIXED_GRID / "ATL03_20221124100820_09881702_006_02_subsetted.h5"
FIXED_BEAM1 = "gt3r"
FIXED_BEAM2 = "gt1r"


def read_full_track_xy(path: Path, beam: str) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        grp = f[beam]["heights"]
        lat = grp["lat_ph"][:].astype(np.float64)
        lon = grp["lon_ph"][:].astype(np.float64)
    valid = np.isfinite(lat) & np.isfinite(lon)
    valid &= (lat >= -90.0) & (lat <= 90.0) & (lon >= -180.0) & (lon <= 180.0)
    return {"lat": lat, "lon": lon, "valid": valid}


def estimate_crossing_center(track1: Dict[str, np.ndarray], track2: Dict[str, np.ndarray],
                             coarse_step: int = 200, refine_half_window: int = 25000) -> Dict[str, float]:
    valid1 = np.flatnonzero(track1["valid"])
    valid2 = np.flatnonzero(track2["valid"])
    if valid1.size == 0 or valid2.size == 0:
        raise RuntimeError("One of the two fixed ATL03 tracks has no valid coordinates.")

    idx1c = valid1[::coarse_step]
    idx2c = valid2[::coarse_step]
    P1c = lonlat_to_unitvec(track1["lon"][idx1c], track1["lat"][idx1c])
    P2c = lonlat_to_unitvec(track2["lon"][idx2c], track2["lat"][idx2c])
    tree_c = cKDTree(P2c)
    d_c, j_c = tree_c.query(P1c, k=1, workers=1)
    i_best = int(np.argmin(d_c))
    idx1_seed = int(idx1c[i_best])
    idx2_seed = int(idx2c[int(j_c[i_best])])

    s1 = max(0, idx1_seed - refine_half_window)
    e1 = min(len(track1["lat"]), idx1_seed + refine_half_window + 1)
    s2 = max(0, idx2_seed - refine_half_window)
    e2 = min(len(track2["lat"]), idx2_seed + refine_half_window + 1)

    idx1r = np.arange(s1, e1, dtype=np.int64)
    idx2r = np.arange(s2, e2, dtype=np.int64)
    idx1r = idx1r[track1["valid"][idx1r]]
    idx2r = idx2r[track2["valid"][idx2r]]
    if idx1r.size == 0 or idx2r.size == 0:
        raise RuntimeError("No valid coordinates available in the refined crossing window.")

    P2r = lonlat_to_unitvec(track2["lon"][idx2r], track2["lat"][idx2r])
    tree_r = cKDTree(P2r)
    P1r = lonlat_to_unitvec(track1["lon"][idx1r], track1["lat"][idx1r])
    _, j_r = tree_r.query(P1r, k=1, workers=1)

    lat1 = track1["lat"][idx1r]
    lon1 = track1["lon"][idx1r]
    lat2 = track2["lat"][idx2r[np.asarray(j_r, dtype=int)]]
    lon2 = track2["lon"][idx2r[np.asarray(j_r, dtype=int)]]
    d_m = np.array([haversine_distance(float(a), float(b), float(c), float(d))
                    for a, b, c, d in zip(lat1, lon1, lat2, lon2)], dtype=np.float64)
    k = int(np.argmin(d_m))
    idx1 = int(idx1r[k])
    idx2 = int(idx2r[int(j_r[k])])
    best_d = float(d_m[k])
    center_lat = float((track1["lat"][idx1] + track2["lat"][idx2]) / 2.0)
    center_lon = float((track1["lon"][idx1] + track2["lon"][idx2]) / 2.0)
    return {
        "idx1": idx1, "idx2": idx2,
        "lat1": float(track1["lat"][idx1]), "lon1": float(track1["lon"][idx1]),
        "lat2": float(track2["lat"][idx2]), "lon2": float(track2["lon"][idx2]),
        "center_lat": center_lat, "center_lon": center_lon,
        "distance_m": best_d,
    }


def local_block_edges_from_center(center_lat: float, neighbor_blocks: int = NEIGHBOR_BLOCKS_EACH_SIDE) -> np.ndarray:
    target_start = np.floor(float(center_lat) * 100.0) / 100.0
    first = target_start - neighbor_blocks * 0.01
    last = target_start + (neighbor_blocks + 1) * 0.01
    n_blocks = 2 * neighbor_blocks + 1
    return first + np.arange(n_blocks + 1, dtype=np.float64) * 0.01


def read_local_beam_from_center(path: Path, beam: str, center_lat: float,
                                neighbor_blocks: int = NEIGHBOR_BLOCKS_EACH_SIDE):
    track = read_full_track_xy(path, beam)
    edges = local_block_edges_from_center(center_lat, neighbor_blocks)
    local_min = float(edges[0] - 0.001)
    local_max = float(edges[-1] + 0.001)
    valid_xy = track["valid"]
    lat_full = track["lat"]
    lon_full = track["lon"]
    local_indices = np.flatnonzero(valid_xy & (lat_full >= local_min) & (lat_full <= local_max))
    if local_indices.size < MIN_BEAM_PHOTONS:
        raise RuntimeError(f"{path.name} {beam}: too few photons in the local crossing window.")
    with h5py.File(path, "r") as f:
        grp = f[beam]["heights"]
        h_local = grp["h_ph"][local_indices].astype(np.float64)
        conf_local = grp["signal_conf_ph"][local_indices]
        if conf_local.ndim == 1:
            land_conf = conf_local.astype(np.int16)
        else:
            land_conf = conf_local[:, 0].astype(np.int16)
    raw_local = {
        "lat": lat_full[local_indices].astype(np.float64),
        "lon": lon_full[local_indices].astype(np.float64),
        "h": h_local,
        "land_conf": land_conf,
        "raw_index": local_indices.astype(np.int64),
        "block_edges": edges,
        "center_lat": float(center_lat),
        "center_lon": float(np.nanmedian(lon_full[local_indices])),
    }
    return raw_local, track


def process_local_beam_no_target(raw: Dict[str, np.ndarray], raw_path: Path, beam: str):
    screened = pre_screen_raw_beam(raw, raw_path, beam)
    lat_blocks = raw["block_edges"].copy()
    center_lat = float(raw["center_lat"])
    print(f"Source ATL03 file                 : {raw_path.name}")
    print(f"Target beam                       : {beam}")
    print(f"Raw photons in selected local read window : {len(raw['lat']):,}")
    print(f"Local photons with land_conf >= 1 : {int(np.sum(raw['land_conf'] >= LAND_CONF_MIN)):,}")
    print(f"Local photons removed by pre-screening       : {int(np.sum(raw['land_conf'] < LAND_CONF_MIN)):,}")
    print(f"Local raw latitude span            : {np.ptp(raw['lat']):.6f} deg")
    print("Full-track latitude span            : not loaded for denoising")
    print(f"Processing core blocks              : {len(lat_blocks)-1}")
    print(f"Block range                         : {lat_blocks[0]:.4f} to {lat_blocks[-1]:.4f} deg")
    print(f"Neighbor blocks each side           : {NEIGHBOR_BLOCKS_EACH_SIDE}")

    geoid_interpolator = get_geoid_interpolator()
    all_rows, logs = [], []
    plot_data = None
    for block_i, (block_start, block_end) in enumerate(zip(lat_blocks[:-1], lat_blocks[1:])):
        snr_value = compute_block_snr_from_screened(screened, block_start, block_end)
        capture = (block_start <= center_lat < block_end)
        if block_i == len(lat_blocks) - 2:
            capture = capture or np.isclose(center_lat, block_end, atol=1e-10)
        rows, block_log, maybe_plot = process_one_block_screened(
            block_start, block_end, screened, beam, raw_path.name,
            geoid_interpolator, target_raw_index=-1, snr_value=snr_value,
            capture_plot_data=(capture and plot_data is None),
            is_last_block=(block_i == len(lat_blocks) - 2),
        )
        all_rows.extend(rows)
        logs.append(block_log)
        if maybe_plot is not None and plot_data is None:
            plot_data = maybe_plot

    se = pd.DataFrame(
        all_rows,
        columns=[
            "file", "beam", "raw_index", "lon", "lat", "elevation_wgs84",
            "elevation_spline", "elevation_egm2008", "geoid_height", "SNR",
            "block_start", "block_end",
        ],
    )
    log_df = pd.DataFrame(logs)
    n_ok = int(np.sum(log_df["status"].astype(str).str.startswith("ok")))
    print(f"Accepted local blocks               : {n_ok}/{len(lat_blocks)-1}")
    print(f"SE photons emitted by core blocks    : {len(se):,}")
    se = deduplicate_se_by_native_index(se, beam)
    print(f"Unique native ATL03 photons          : {len(se):,}")

    core_all = (screened["lat"] >= lat_blocks[0]) & (screened["lat"] < lat_blocks[-1])
    if len(lat_blocks) > 1:
        core_all |= np.isclose(screened["lat"], lat_blocks[-1], atol=1e-10)
    n_prescreen_core = int(core_all.sum())
    recovery_fraction = len(se) / n_prescreen_core if n_prescreen_core else 0.0
    print(f"Pre-screened photons in selected cores: {n_prescreen_core:,}")
    print(f"Local SE recovery fraction            : {recovery_fraction:.3f}")

    sub_section(f"LOCAL DENOISING LOG | {raw_path.name} | {beam}")
    columns = [
        "block_start", "block_end", "n_block_context", "n_block_core", "SNR",
        "P1", "n_stage1", "P2", "n_stage2", "spline1_spar", "P3", "n_stage3",
        "spline2_spar", "n_stage5_context", "n_output_core", "status",
    ]
    print(log_df[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return se, screened, log_df, plot_data, recovery_fraction


def pairing_local_no_target(beam1: dict, beam2: dict):
    section("STEP 4 | ONE-TO-ONE MATCHING OF THE FIXED-FILE LOCAL SE PHOTONS")
    df1, df2 = beam1["df"], beam2["df"]
    candidates, accepted = find_close_pairs_with_diagnostics(
        df1[["lon", "lat", "elevation_egm2008"]].rename(columns={"elevation_egm2008": "elevation"}),
        df2[["lon", "lat", "elevation_egm2008"]].rename(columns={"elevation_egm2008": "elevation"}),
        MAX_PAIR_DISTANCE_M,
    )
    print(f"Candidate photon pairs <= {MAX_PAIR_DISTANCE_M:.1f} m in selected local blocks : {len(candidates):,}")
    print(f"Accepted one-to-one pairs                                            : {len(accepted):,}")

    used1 = [int(i) for i, _, _ in accepted]
    used2 = [int(j) for _, j, _ in accepted]
    raw1 = [int(df1.iloc[i]["raw_index"]) for i in used1]
    raw2 = [int(df2.iloc[j]["raw_index"]) for j in used2]
    dup_pos1 = len(used1) - len(set(used1))
    dup_pos2 = len(used2) - len(set(used2))
    dup_raw1 = len(raw1) - len(set(raw1))
    dup_raw2 = len(raw2) - len(set(raw2))
    print(f"Repeated Beam-1 SE row positions                  : {dup_pos1}")
    print(f"Repeated Beam-2 SE row positions                  : {dup_pos2}")
    print(f"Repeated Beam-1 native ATL03 row indices          : {dup_raw1}")
    print(f"Repeated Beam-2 native ATL03 row indices          : {dup_raw2}")
    if any((dup_pos1, dup_pos2, dup_raw1, dup_raw2)):
        raise RuntimeError("One-to-one invariant failed: an SE/native photon was reused.")

    d1, d2 = beam1["date_obj"], beam2["date_obj"]
    months, years = calculate_time_diff(d1, d2)
    rows = []
    for pair_id, (i, j, distance) in enumerate(accepted, start=1):
        r1, r2 = df1.iloc[i], df2.iloc[j]
        if d1 <= d2:
            p_early, p_late = r1, r2
            f_early, f_late = beam1, beam2
        else:
            p_early, p_late = r2, r1
            f_early, f_late = beam2, beam1
        dh = float(p_late.elevation_egm2008 - p_early.elevation_egm2008)
        rows.append({
            "pair_id": pair_id,
            "photon1_file": f_early["file"], "photon1_beam": f_early["beam_name"],
            "photon1_raw_index": int(p_early.raw_index),
            "photon1_lon": float(p_early.lon), "photon1_lat": float(p_early.lat),
            "photon1_elevation": float(p_early.elevation_egm2008),
            "photon2_file": f_late["file"], "photon2_beam": f_late["beam_name"],
            "photon2_raw_index": int(p_late.raw_index),
            "photon2_lon": float(p_late.lon), "photon2_lat": float(p_late.lat),
            "photon2_elevation": float(p_late.elevation_egm2008),
            "distance": float(distance), "elevation_diff": dh,
            "months_diff": abs(months), "years_diff": abs(years),
            "monthly_change_rate": dh / abs(months) if abs(months) else np.nan,
            "yearly_change_rate": dh / abs(years) if abs(years) else np.nan,
        })
    matched = pd.DataFrame(rows)
    if not matched.empty:
        assert matched["photon1_raw_index"].is_unique
        assert matched["photon2_raw_index"].is_unique
        print(f"Accepted distance summary (m)      : min={matched['distance'].min():.3f}, median={matched['distance'].median():.3f}, max={matched['distance'].max():.3f}")
        print(f"Accepted elevation-difference summary (m) : min={matched['elevation_diff'].min():+.3f}, median={matched['elevation_diff'].median():+.3f}, max={matched['elevation_diff'].max():+.3f}")
    else:
        print("No accepted pairs were produced.")
    return matched, candidates, accepted


def plot_spatial_context_fixed(tp_gdf, grid_gdf, track1, track2, center, file1, file2, beam1, beam2, block_edges):
    """Fast regional and Grid-scale context plot.

    The previous version built/intersected Shapely LineStrings from millions of
    ATL03 coordinates. That operation could make the script appear frozen at
    FIGURE 1. This version uses vectorized bounding-box filtering and capped,
    rasterized scatter layers. It preserves the native row order when sampling
    and never connects unrelated track fragments with artificial lines.
    """
    import time as _time

    t0 = _time.perf_counter()
    section("FIGURE 1 | FIXED-FILE SPATIAL CONTEXT")
    print("[FIGURE 1] Preparing map layers (fast vectorized mode)...", flush=True)

    bounds = np.asarray(grid_gdf.total_bounds, dtype=float)
    gx0, gy0, gx1, gy1 = map(float, bounds)
    if not np.all(np.isfinite(bounds)) or gx1 <= gx0 or gy1 <= gy0:
        raise RuntimeError(f"Invalid Grid bounds: {bounds}")
    mean_lat = 0.5 * (gy0 + gy1)
    geo_aspect = 1.0 / max(np.cos(np.radians(mean_lat)), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), constrained_layout=True)

    # Left: regional context. Subsample the full tracks to avoid millions of markers.
    ax = axes[0]
    print("[FIGURE 1] Drawing TP and selected Grid...", flush=True)
    tp_gdf.boundary.plot(ax=ax, color="0.78", linewidth=0.7, zorder=1)
    grid_gdf.boundary.plot(ax=ax, color="black", linewidth=1.3, zorder=4)
    for tr, color, label, y in [
        (track1, BEAM1_COLOR, f"{Path(file1).name} | {beam1}", 0.98),
        (track2, BEAM2_COLOR, f"{Path(file2).name} | {beam2}", 0.91),
    ]:
        ids = np.flatnonzero(tr["valid"])
        if ids.size > MAX_MAP_DISPLAY_POINTS:
            ids = ids[np.linspace(0, ids.size - 1, MAX_MAP_DISPLAY_POINTS).astype(np.int64)]
        ax.scatter(tr["lon"][ids], tr["lat"][ids], s=0.22, color=color,
                   alpha=0.42, linewidths=0, rasterized=True, zorder=2)
        ax.text(0.015, y, label, transform=ax.transAxes, ha="left", va="top",
                color=color, fontsize=7.2, wrap=True,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5))
    ax.scatter([center["center_lon"]], [center["center_lat"]], marker="x",
               s=65, linewidths=1.7, color="black", zorder=6)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_aspect(geo_aspect, adjustable="datalim")

    # Right: points falling within the selected Grid's geographic bounding box.
    # Do not construct millions of Shapely vertices or intersect LineStrings.
    ax = axes[1]
    print("[FIGURE 1] Filtering track photons to Grid extent...", flush=True)
    pad_x = max((gx1 - gx0) * 0.025, 0.0005)
    pad_y = max((gy1 - gy0) * 0.025, 0.0005)
    xlo, xhi = gx0 - pad_x, gx1 + pad_x
    ylo, yhi = gy0 - pad_y, gy1 + pad_y

    for tr, color, label in [
        (track1, BEAM1_COLOR, f"{extract_date_from_filename(file1).strftime('%d %b %Y') if extract_date_from_filename(file1) else Path(file1).name} | {beam1}"),
        (track2, BEAM2_COLOR, f"{extract_date_from_filename(file2).strftime('%d %b %Y') if extract_date_from_filename(file2) else Path(file2).name} | {beam2}"),
    ]:
        valid = tr["valid"]
        # Boolean filtering is vectorized; plotting is capped to keep rendering responsive.
        ids = np.flatnonzero(valid & (tr["lon"] >= xlo) & (tr["lon"] <= xhi)
                             & (tr["lat"] >= ylo) & (tr["lat"] <= yhi))
        n_inside = int(ids.size)
        if ids.size > 120000:
            ids = ids[np.linspace(0, ids.size - 1, 120000).astype(np.int64)]
        print(f"[FIGURE 1] {label}: {n_inside:,} photons in Grid extent; plotting {ids.size:,}.", flush=True)
        if ids.size:
            ax.scatter(tr["lon"][ids], tr["lat"][ids], s=0.65, color=color,
                       alpha=0.72, linewidths=0, rasterized=True, label=label, zorder=3)
        else:
            print(f"[FIGURE 1 WARNING] No track photons found inside Grid extent for {label}.", flush=True)

    print("[FIGURE 1] Drawing Grid outline and local block guides...", flush=True)
    grid_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=5)
    # Latitude guides are clipped horizontally to the Grid's bounding extent.
    for edge in block_edges:
        ax.plot([gx0, gx1], [edge, edge], color="0.65",
                ls="--" if np.isclose(edge, block_edges[0]) or np.isclose(edge, block_edges[-1]) else ":",
                lw=0.65, zorder=2)
    ax.scatter([center["center_lon"]], [center["center_lat"]], marker="x",
               s=65, linewidths=1.7, color="black", zorder=8, label="Closest approach")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_aspect(geo_aspect, adjustable="box")
    ax.legend(loc="lower left", frameon=True, fontsize=7.2)

    print(f"[FIGURE 1] Layout complete in {_time.perf_counter() - t0:.1f} s. Opening figure window...", flush=True)
    save_figure(fig, "Figure1_spatial_context")
    fig.canvas.draw_idle()
    plt.show()
    plt.close(fig)
    print("[FIGURE 1] Figure window closed; continuing to denoising.", flush=True)


def plot_raw_and_final_nodes(plot1: dict, plot2: dict, raw1: dict, raw2: dict, label1: str, label2: str):
    section("FIGURE 2 | RAW PHOTONS AND FINAL RECONSTRUCTED SURFACE NODES")
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), constrained_layout=True)
    for ax, pdata, raw_source, color, label in [
        (axes[0], plot1, raw1, BEAM1_COLOR, label1),
        (axes[1], plot2, raw2, BEAM2_COLOR, label2),
    ]:
        ax.scatter(raw_source["lat"], raw_source["h"], s=2, c="0.78", alpha=0.38, linewidths=0,
                   label="Original ATL03 photons")
        lat_context = pdata["stage6_lat"]
        elev_context = pdata["stage6_elev_spline"]
        emit = pdata["core_mask_stage6"]
        block_start = float(pdata["block_start"])
        block_end = float(pdata["block_end"])

        ax.axvspan(block_start, block_end, color=color, alpha=0.08, zorder=0)
        ax.scatter(lat_context[emit], elev_context[emit], s=8, c=color, alpha=0.88, linewidths=0,
                   zorder=3)
        ax.set_xlabel("Latitude (deg)", fontsize=12)
        ax.set_ylabel("Elevation (m)", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top",
                color=color, fontsize=12, fontweight="bold")

        y0, y1 = ax.get_ylim()
        yr = y1 - y0
        y_arrow = y1 - 0.12 * yr
        y_text = y1 - 0.07 * yr
        ax.annotate("", xy=(block_start, y_arrow), xytext=(block_end, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.5))
        ax.text((block_start + block_end) / 2.0, y_text,
                "One core block (0.01\xb0 latitude)",
                ha="center", va="top", color=color, fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, alpha=0.92))
    save_figure(fig, "Figure2_photon_reconstruction")
    plt.show()
    plt.close(fig)


def plot_pair_matching_simple(se1: pd.DataFrame, se2: pd.DataFrame, matched: pd.DataFrame,
                              file1: str, beam1: str, file2: str, beam2: str):
    section("FIGURE 3 | ACCEPTED ONE-TO-ONE PHOTON PAIRS NEAR THE CROSSING")
    if matched.empty:
        print("No accepted pairs were produced. Pairing figure skipped.")
        return

    x_all = np.r_[matched["photon1_lon"].to_numpy(dtype=float), matched["photon2_lon"].to_numpy(dtype=float)]
    y_all = np.r_[matched["photon1_lat"].to_numpy(dtype=float), matched["photon2_lat"].to_numpy(dtype=float)]
    lon0 = float(np.mean(x_all))
    lat0 = float(np.mean(y_all))
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * max(np.cos(np.radians(lat0)), 1e-6)

    def to_local_xy(lon, lat):
        x = (np.asarray(lon, dtype=float) - lon0) * meters_per_deg_lon
        y = (np.asarray(lat, dtype=float) - lat0) * meters_per_deg_lat
        return x, y

    pair_df = matched.copy()
    pair_df["x1_m"], pair_df["y1_m"] = to_local_xy(pair_df["photon1_lon"], pair_df["photon1_lat"])
    pair_df["x2_m"], pair_df["y2_m"] = to_local_xy(pair_df["photon2_lon"], pair_df["photon2_lat"])

    se1_local = se1.copy()
    se2_local = se2.copy()
    se1_local["x_m"], se1_local["y_m"] = to_local_xy(se1_local["lon"], se1_local["lat"])
    se2_local["x_m"], se2_local["y_m"] = to_local_xy(se2_local["lon"], se2_local["lat"])

    x_pairs = np.r_[pair_df["x1_m"].to_numpy(dtype=float), pair_df["x2_m"].to_numpy(dtype=float)]
    y_pairs = np.r_[pair_df["y1_m"].to_numpy(dtype=float), pair_df["y2_m"].to_numpy(dtype=float)]
    pad_x = max(float(np.ptp(x_pairs)) * 0.45, 0.75)
    pad_y = max(float(np.ptp(y_pairs)) * 0.20, 0.60)
    xmin, xmax = float(np.min(x_pairs) - pad_x), float(np.max(x_pairs) + pad_x)
    ymin, ymax = float(np.min(y_pairs) - pad_y), float(np.max(y_pairs) + pad_y)
    se1_local = se1_local[(se1_local["x_m"] >= xmin) & (se1_local["x_m"] <= xmax) &
                          (se1_local["y_m"] >= ymin) & (se1_local["y_m"] <= ymax)].copy()
    se2_local = se2_local[(se2_local["x_m"] >= xmin) & (se2_local["x_m"] <= xmax) &
                          (se2_local["y_m"] >= ymin) & (se2_local["y_m"] <= ymax)].copy()

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 7.4), constrained_layout=True)
    if len(se1_local):
        ax.scatter(se1_local["x_m"], se1_local["y_m"], s=16, c=BEAM1_COLOR, alpha=0.15,
                   linewidths=0, zorder=1)
    if len(se2_local):
        ax.scatter(se2_local["x_m"], se2_local["y_m"], s=16, c=BEAM2_COLOR, alpha=0.15,
                   linewidths=0, zorder=1)

    cmap = plt.get_cmap("tab20")
    pair_colors = [cmap(i % cmap.N) for i in range(len(pair_df))]
    for idx, (_, r) in enumerate(pair_df.iterrows()):
        line_color = pair_colors[idx]
        x1, y1, x2, y2 = float(r["x1_m"]), float(r["y1_m"]), float(r["x2_m"]), float(r["y2_m"])
        ax.plot([x1, x2], [y1, y2], color=line_color, lw=1.8, alpha=0.95, zorder=3)
        ax.scatter([x1], [y1], s=60, marker="o", facecolor=BEAM1_COLOR,
                   edgecolor=line_color, linewidth=1.0, zorder=4)
        ax.scatter([x2], [y2], s=64, marker="s", facecolor=BEAM2_COLOR,
                   edgecolor=line_color, linewidth=1.0, zorder=4)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Relative easting (m)", fontsize=12)
    ax.set_ylabel("Relative northing (m)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.02, 0.02,
            f"Accepted pairs: {len(pair_df)}\nCircle = {beam1}; square = {beam2}\nEach pair is shown by a unique line color",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.7", alpha=0.94))
    save_figure(fig, "Figure3_one_to_one_pairing")
    plt.show()


def plot_pair_distance_dh_statistics(matched: pd.DataFrame):
    section("FIGURE 4 | STATISTICS OF ACCEPTED PAIR DISTANCE AND ELEVATION DIFFERENCE")
    if matched.empty:
        print("No accepted pairs: statistical figure skipped.")
        return
    dist = matched["distance"].to_numpy(dtype=float)
    dh = matched["elevation_diff"].to_numpy(dtype=float)
    cmap = plt.get_cmap("tab20")
    pair_colors = [cmap(i % cmap.N) for i in range(len(matched))]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)

    axes[0].hist(dist, bins="auto", edgecolor="black", linewidth=0.6, alpha=0.8)
    axes[0].axvline(np.median(dist), color="crimson", ls="--", lw=1.2,
                    label=f"Median = {np.median(dist):.3f} m")
    axes[0].set_xlabel("Horizontal pair distance (m)", fontsize=11)
    axes[0].set_ylabel("Number of accepted pairs", fontsize=11)
    axes[0].tick_params(labelsize=10)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].hist(dh, bins="auto", edgecolor="black", linewidth=0.6, alpha=0.8)
    axes[1].axvline(0.0, color="0.35", ls=":", lw=1.0)
    axes[1].axvline(np.median(dh), color="crimson", ls="--", lw=1.2,
                    label=f"Median = {np.median(dh):+.3f} m")
    axes[1].set_xlabel("Elevation difference (m; later - earlier)", fontsize=11)
    axes[1].set_ylabel("Number of accepted pairs", fontsize=11)
    axes[1].tick_params(labelsize=10)
    axes[1].legend(frameon=False, fontsize=9)

    for i in range(len(dist)):
        axes[2].scatter(dist[i], dh[i], s=42, color=pair_colors[i], edgecolors="black", linewidths=0.35)
    axes[2].axhline(0.0, color="0.35", ls=":", lw=1.0)
    axes[2].set_xlabel("Horizontal pair distance (m)", fontsize=11)
    axes[2].set_ylabel("Elevation difference (m; later - earlier)", fontsize=11)
    axes[2].tick_params(labelsize=10)
    axes[2].set_title("Pair-wise distance vs. elevation difference", fontsize=12)

    summary = (
        f"n = {len(matched)}\n"
        f"Distance: {np.min(dist):.3f}-{np.max(dist):.3f} m\n"
        f"Median |dh| = {np.median(np.abs(dh)):.3f} m\n"
        f"Mean dh = {np.mean(dh):+.3f} m"
    )
    axes[2].text(0.98, 0.02, summary, transform=axes[2].transAxes,
                 ha="right", va="bottom", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    save_figure(fig, "Figure4_pair_statistics")
    plt.show()


def _main_impl():
    global OUTPUT_PACKAGE
    OUTPUT_PACKAGE = initialize_output_package()
    print(f"[OUTPUT] Case package directory: {OUTPUT_PACKAGE['root']}", flush=True)

    section("STEP 1 | FIXED FILE PAIR AND AUTOMATIC CROSSING LOCALIZATION")
    print(f"Grid                             : {FIXED_GRID}")
    print(f"ATL03 1                          : {FIXED_FILE1}")
    print(f"ATL03 2                          : {FIXED_FILE2}")
    print(f"Beam 1 / Beam 2                  : {FIXED_BEAM1} / {FIXED_BEAM2}")

    if not FIXED_FILE1.exists() or not FIXED_FILE2.exists():
        raise FileNotFoundError("One or both fixed ATL03 files do not exist.")
    if not raw_beam_is_valid(FIXED_FILE1, FIXED_BEAM1):
        raise RuntimeError(f"Invalid beam group: {FIXED_FILE1.name} {FIXED_BEAM1}")
    if not raw_beam_is_valid(FIXED_FILE2, FIXED_BEAM2):
        raise RuntimeError(f"Invalid beam group: {FIXED_FILE2.name} {FIXED_BEAM2}")

    track1 = read_full_track_xy(FIXED_FILE1, FIXED_BEAM1)
    track2 = read_full_track_xy(FIXED_FILE2, FIXED_BEAM2)
    print(f"Full-track coordinate counts     : {int(track1['valid'].sum()):,}, {int(track2['valid'].sum()):,}")
    center = estimate_crossing_center(track1, track2)
    print(f"Estimated closest-approach distance : {center['distance_m']:.6f} m")
    print(f"Beam 1 closest photon (lon, lat)    : {center['lon1']:.6f}, {center['lat1']:.6f}")
    print(f"Beam 2 closest photon (lon, lat)    : {center['lon2']:.6f}, {center['lat2']:.6f}")
    print(f"Local crossing center (lon, lat)    : {center['center_lon']:.6f}, {center['center_lat']:.6f}")
    block_edges = local_block_edges_from_center(center['center_lat'], NEIGHBOR_BLOCKS_EACH_SIDE)
    print(f"Local processing block range        : {block_edges[0]:.4f} to {block_edges[-1]:.4f} deg latitude")

    # Write the input manifest, study-area metadata, parameter tables,
    # parameter definitions, and software environment before processing.
    save_case_metadata(OUTPUT_PACKAGE, center, block_edges)
    print("[OUTPUT] Input manifest and parameter documentation saved.", flush=True)

    section("STEP 2 | LOAD LOCAL ATL03 WINDOWS AROUND THE CROSSING")
    raw1, _ = read_local_beam_from_center(FIXED_FILE1, FIXED_BEAM1, center['center_lat'])
    raw2, _ = read_local_beam_from_center(FIXED_FILE2, FIXED_BEAM2, center['center_lat'])
    print(f"Local raw photons loaded, beam 1 / beam 2 : {len(raw1['lat']):,} / {len(raw2['lat']):,}")

    grid_vector = GRID_VECTOR_ROOT / f"{FIXED_GRID}.shp"
    if not grid_vector.exists():
        raise FileNotFoundError(f"Grid vector not found: {grid_vector}")
    grid_gdf = read_vector_wgs84(grid_vector)
    tp_gdf = read_vector_wgs84(TP_VECTOR)
    print(f"[PLOT] Matplotlib backend: {plt.get_backend()}", flush=True)
    print("[PLOT] If the figure window appears, close it to continue to denoising.", flush=True)
    plot_spatial_context_fixed(tp_gdf, grid_gdf, track1, track2, center,
                               str(FIXED_FILE1), str(FIXED_FILE2), FIXED_BEAM1, FIXED_BEAM2, block_edges)

    print_fixed_parameter_log()

    section("STEP 3 | DENOISING THE TWO FIXED FILES (THREE CORE BLOCKS ONLY)")
    se1, screened1, logs1, plot1, recovery1 = process_local_beam_no_target(raw1, FIXED_FILE1, FIXED_BEAM1)
    se2, screened2, logs2, plot2, recovery2 = process_local_beam_no_target(raw2, FIXED_FILE2, FIXED_BEAM2)
    print(f"Recovered SE photon counts, beam 1 / beam 2 : {len(se1):,} / {len(se2):,}")
    print(f"Local SE recovery fraction, beam 1 / beam 2 : {recovery1:.3f} / {recovery2:.3f}")

    # Preserve the actual per-block processing logs and per-beam screening counts.
    logs1_out = logs1.copy()
    logs2_out = logs2.copy()

    # process_one_block_screened() already records both "beam" and
    # "source_file" in every log row. Assigning is safe whether the columns
    # already exist or not; DataFrame.insert() would fail on duplicate names.
    logs1_out["beam"] = FIXED_BEAM1
    logs1_out["source_file"] = FIXED_FILE1.name
    logs2_out["beam"] = FIXED_BEAM2
    logs2_out["source_file"] = FIXED_FILE2.name
    pd.concat([logs1_out, logs2_out], ignore_index=True).to_csv(
        OUTPUT_PACKAGE["denoising"] / "block_processing_log.csv", index=False)

    screening_rows = []
    for beam_name, raw_data, screened_data, se_data, recovery in [
        (FIXED_BEAM1, raw1, screened1, se1, recovery1),
        (FIXED_BEAM2, raw2, screened2, se2, recovery2),
    ]:
        n_raw = int(len(raw_data["lat"]))
        n_screened = int(len(screened_data["lat"]))
        n_core = int(np.sum((screened_data["lat"] >= raw_data["block_edges"][0]) &
                            (screened_data["lat"] <= raw_data["block_edges"][-1])))
        screening_rows.append({
            "beam": beam_name, "raw_local_photons": n_raw,
            "land_conf_ge_1_photons": n_screened,
            "removed_by_prescreen": n_raw - n_screened,
            "prescreen_retention_fraction": n_screened / n_raw if n_raw else np.nan,
            "SE_photons_emitted": int(len(se_data)),
            "SE_recovery_fraction_reported_by_processor": float(recovery),
            "core_prescreened_photons_recounted": n_core,
        })
        # Native ATL03 identity is retained in the SE table; this is the auditable lineage table.
        se_data.to_csv(OUTPUT_PACKAGE["denoising"] /
                       f"SE_photons_{beam_name}.csv", index=False)
    pd.DataFrame(screening_rows).to_csv(
        OUTPUT_PACKAGE["denoising"] / "photon_screening_summary.csv", index=False)
    lineage_cols = ["file", "beam", "raw_index", "lon", "lat", "elevation_wgs84",
                    "elevation_spline", "elevation_egm2008", "geoid_height",
                    "SNR", "block_start", "block_end"]
    lineage = pd.concat([se1[lineage_cols], se2[lineage_cols]], ignore_index=True)
    lineage.to_csv(OUTPUT_PACKAGE["denoising"] / "photon_lineage.csv", index=False)

    plot_raw_and_final_nodes(plot1, plot2, raw1, raw2, f"{FIXED_FILE1.name} | {FIXED_BEAM1}", f"{FIXED_FILE2.name} | {FIXED_BEAM2}")

    beam1 = build_beam_object(se1, FIXED_FILE1.name, FIXED_BEAM1)
    beam2 = build_beam_object(se2, FIXED_FILE2.name, FIXED_BEAM2)
    matched, candidates, accepted = pairing_local_no_target(beam1, beam2)

    # Expand the complete candidate list to native ATL03 photon identities.
    candidate_rows = []
    accepted_lookup = {(int(i), int(j)): pair_id
                       for pair_id, (i, j, _d) in enumerate(accepted, start=1)}
    for candidate_id, (i, j, distance) in enumerate(candidates, start=1):
        accepted_pair_id = accepted_lookup.get((int(i), int(j)))
        p1 = beam1["df"].iloc[int(i)]
        p2 = beam2["df"].iloc[int(j)]
        candidate_rows.append({
            "candidate_id": candidate_id,
            "beam1": FIXED_BEAM1,
            "beam1_raw_index": int(p1["raw_index"]),
            "beam1_longitude": float(p1["lon"]),
            "beam1_latitude": float(p1["lat"]),
            "beam1_elevation_egm2008": float(p1["elevation_egm2008"]),
            "beam2": FIXED_BEAM2,
            "beam2_raw_index": int(p2["raw_index"]),
            "beam2_longitude": float(p2["lon"]),
            "beam2_latitude": float(p2["lat"]),
            "beam2_elevation_egm2008": float(p2["elevation_egm2008"]),
            "horizontal_distance_m": float(distance),
            "accepted": accepted_pair_id is not None,
            "accepted_pair_id": accepted_pair_id if accepted_pair_id is not None else "",
            "rejection_reason": "" if accepted_pair_id is not None else "one_to_one_conflict",
        })
    candidate_table = pd.DataFrame(candidate_rows)
    candidate_table.to_csv(OUTPUT_PACKAGE["pairing"] / "pairing_candidates.csv", index=False)
    matched.to_csv(OUTPUT_PACKAGE["pairing"] / "accepted_pairs.csv", index=False)

    # Explicit one-to-one validation report, derived from the accepted records.
    dup_se1 = int(matched["photon1_raw_index"].duplicated().sum()) if not matched.empty else 0
    dup_se2 = int(matched["photon2_raw_index"].duplicated().sum()) if not matched.empty else 0
    validation_lines = [
        "ONE-TO-ONE PAIRING VALIDATION",
        f"Grid: {FIXED_GRID}",
        f"ATL03 file pair: {FIXED_FILE1.name} ({FIXED_BEAM1}) vs {FIXED_FILE2.name} ({FIXED_BEAM2})",
        f"Candidate pairs within {MAX_PAIR_DISTANCE_M:.3f} m: {len(candidates)}",
        f"Accepted one-to-one pairs: {len(matched)}",
        f"Repeated native ATL03 row indices in earlier acquisition: {dup_se1}",
        f"Repeated native ATL03 row indices in later acquisition: {dup_se2}",
        "Enforcement level: entire fixed ATL03 file-pair comparison, after pooling all core-block SE photons per beam.",
        "Algorithm: sort all candidates by ascending geodesic horizontal distance; greedily accept a pair only if neither endpoint has been used.",
        "Cross-comparison scope: no uniqueness constraint is shared across separate acquisition-pair comparisons.",
        f"RESULT: {'PASS' if dup_se1 == 0 and dup_se2 == 0 else 'FAIL'}",
    ]
    (OUTPUT_PACKAGE["pairing"] / "one_to_one_validation.txt").write_text(
        "\\n".join(validation_lines) + "\\n", encoding="utf-8")
    pairing_summary = {
        "candidate_pairs": int(len(candidates)),
        "accepted_pairs": int(len(matched)),
        "rejected_candidates": int(len(candidates) - len(matched)),
        "maximum_pair_distance_m": float(MAX_PAIR_DISTANCE_M),
        "repeated_beam1_native_indices": dup_se1,
        "repeated_beam2_native_indices": dup_se2,
        "one_to_one_validation": bool(dup_se1 == 0 and dup_se2 == 0),
        "enforcement_level": "entire fixed ATL03 file-pair comparison",
    }
    save_json(OUTPUT_PACKAGE["pairing"] / "pairing_summary.json", pairing_summary)
    plot_pair_matching_simple(se1, se2, matched, FIXED_FILE1.name, FIXED_BEAM1, FIXED_FILE2.name, FIXED_BEAM2)
    plot_pair_distance_dh_statistics(matched)

    section("FINAL SUMMARY")
    print(f"Recovered SE photon counts, beam 1 / beam 2 : {len(se1):,} / {len(se2):,}")
    print(f"Local pre-screened photon counts, beam 1 / beam 2 : {len(screened1['lat']):,} / {len(screened2['lat']):,}")
    print(f"Local recovery fractions, beam 1 / beam 2   : {recovery1:.4f} / {recovery2:.4f}")
    print(f"Accepted one-to-one pairs                    : {len(matched):,}")
    print(f"Candidate pairs within {MAX_PAIR_DISTANCE_M:.1f} m         : {len(candidates):,}")
    if not matched.empty:
        print("Pair-level distance and elevation-difference statistics:")
        for col, label in [("distance", "Horizontal distance (m)"), ("elevation_diff", "Elevation difference (m; later-earlier)")]:
            v = matched[col].to_numpy(dtype=float)
            print(f"  {label}: mean={np.mean(v):+.6f}, SD={np.std(v, ddof=1) if len(v)>1 else 0.0:.6f}, median={np.median(v):+.6f}, min={np.min(v):+.6f}, max={np.max(v):+.6f}")
        print("Accepted pair records (pair ID, raw indices, distance, dh):")
        show = matched.copy()
        # pair_id is already assigned when accepted pairs are created in
        # pairing_local_no_target(); do not insert a duplicate column here.
        display_columns = [
            'pair_id', 'photon1_raw_index', 'photon2_raw_index',
            'distance', 'elevation_diff'
        ]
        missing_columns = [c for c in display_columns if c not in show.columns]
        if missing_columns:
            raise KeyError(
                f"Accepted-pair table is missing display columns: {missing_columns}"
            )
        print(show[display_columns].to_string(
            index=False, float_format=lambda x: f"{x:.6f}"
        ))
    execution_summary = {
        "grid": FIXED_GRID,
        "file_1": FIXED_FILE1.name,
        "beam_1": FIXED_BEAM1,
        "file_2": FIXED_FILE2.name,
        "beam_2": FIXED_BEAM2,
        "raw_local_photons": [int(len(raw1["lat"])), int(len(raw2["lat"]))],
        "prescreened_photons": [int(len(screened1["lat"])), int(len(screened2["lat"]))],
        "recovered_SE_photons": [int(len(se1)), int(len(se2))],
        "recovery_fractions": [float(recovery1), float(recovery2)],
        "candidate_pairs": int(len(candidates)),
        "accepted_pairs": int(len(matched)),
        "pairing_one_to_one_pass": bool(dup_se1 == 0 and dup_se2 == 0),
        "output_directory": str(OUTPUT_PACKAGE["root"]),
        "completed_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    save_json(OUTPUT_PACKAGE["logs"] / "execution_summary.json", execution_summary)
    print(f"[OUTPUT] Complete reproducibility package: {OUTPUT_PACKAGE['root']}", flush=True)
    gc.collect()


def main():
    """Capture the complete console transcript while preserving live terminal output."""
    output = initialize_output_package()
    log_path = output["logs"] / "complete_run_log.txt"
    with log_path.open("w", encoding="utf-8") as log_handle:
        tee = Tee(sys.stdout, log_handle)
        with contextlib.redirect_stdout(tee):
            _main_impl()


if __name__ == "__main__":
    main()
