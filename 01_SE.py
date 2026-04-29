#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import pandas as pd
import multiprocessing as mp
from joblib import Parallel, delayed
import os
import time
import gc
from numba import jit, prange
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import resource
import rasterio
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline
import ctypes

# ================== 并行 & 内存设置 ==================

# 每个 worker 内部再开很多 BLAS / OpenMP 线程会很浪费，这里限制为 2
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMBA_NUM_THREADS'] = '2'

# 文件层最大并行 worker 数
MAX_FILE_WORKERS = 64

def clear_memory():
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

# ================== 全局 geoid 插值器 ==================
GEOID_INTERPOLATOR = None
GEOID_GRID_LOADED = False

############################################
# Geoid height interpolation setup
############################################

def load_geoid_interpolator(geoid_tiff_path="/home/yyr/herui/us_nga_egm2008_1.tif"):
    """Load EGM2008 geoid grid and create interpolator"""
    global GEOID_INTERPOLATOR, GEOID_GRID_LOADED

    if GEOID_GRID_LOADED:
        return GEOID_INTERPOLATOR

    try:
        print(f"Loading geoid TIFF file: {geoid_tiff_path}")
        with rasterio.open(geoid_tiff_path) as src:
            geoid_data = src.read(1)
            transform = src.transform
            width, height = src.width, src.height

            lons = np.linspace(transform[2], transform[2] + transform[0] * width, width)
            lats = np.linspace(transform[5], transform[5] + transform[4] * height, height)

            GEOID_INTERPOLATOR = RegularGridInterpolator(
                (lons, lats),
                geoid_data.T,
                method='linear',
                bounds_error=False,
                fill_value=0
            )
            GEOID_GRID_LOADED = True
            print("Geoid interpolator created successfully")
            return GEOID_INTERPOLATOR
    except Exception as e:
        print(f"Error loading geoid grid: {e}")
        return None

def transform_to_egm2008(lons, lats, spline_elevs, interpolator):
    """Transform spline-fitted WGS84 elevations to EGM2008"""
    if interpolator is None:
        return spline_elevs, np.zeros_like(spline_elevs)

    try:
        geoid_heights = interpolator((lons, lats))
        egm2008_elevations = spline_elevs - geoid_heights
        return egm2008_elevations, geoid_heights
    except Exception as e:
        print(f"Error in elevation transformation: {e}")
        return spline_elevs, np.zeros_like(spline_elevs)

############################################
# Neighbor counting and filters
############################################

@jit(nopython=True, parallel=True, fastmath=True)
def count_neighbors_rect_vectorized(local_coords: np.ndarray, L: float, W: float) -> np.ndarray:
    n = local_coords.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    lat_threshold = L / 2 / 111000
    elev_threshold = W / 2
    for i in prange(n):
        lat_diff = np.abs(local_coords[:, 1] - local_coords[i, 1])
        elev_diff = np.abs(local_coords[:, 2] - local_coords[i, 2])
        counts[i] = np.sum((lat_diff <= lat_threshold) & (elev_diff <= elev_threshold)) - 1
    return counts


@jit(nopython=True, parallel=True, fastmath=True)
def count_neighbors_slope_vectorized(local_coords: np.ndarray, slope_angles: np.ndarray,
                                     L: float, W: float) -> np.ndarray:
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
        delta_perp  = -delta_lat * sin_a + delta_elev * cos_a

        counts[i] = np.sum((np.abs(delta_along) <= L/2) & (np.abs(delta_perp) <= W/2)) - 1
    return counts


@jit(nopython=True, fastmath=True)
def final_filter_slope_union(local_coords, slope_angles, regions, L=20.0):
    n = local_coords.shape[0]
    out_mask = np.zeros(n, dtype=np.bool_)

    for i in range(n):

        center = local_coords[i]
        angle  = slope_angles[i]
        reg_i  = regions[i]

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        region_idx = np.where(regions == reg_i)[0]
        if region_idx.size == 0:
            continue

        pts = local_coords[region_idx]

        delta_lat  = (pts[:, 1] - center[1]) * 111000.0
        delta_elev =  pts[:, 2] - center[2]

        delta_along =  delta_lat * cos_a + delta_elev * sin_a
        delta_perp  = -delta_lat * sin_a + delta_elev * cos_a

        in_win = (delta_along >= -L / 2.0) & (delta_along <= L / 2.0)
        if np.sum(in_win) < 5:
            continue

        delta_perp_in_window = delta_perp[in_win]

        m = delta_perp_in_window.size

        nbins = max(10, m // 5)

        hist_counts, bin_edges = np.histogram(
            delta_perp_in_window, bins=nbins
        )

        mode_bin = np.argmax(hist_counts)

        perp_mode = 0.5 * (bin_edges[mode_bin] +
                           bin_edges[mode_bin + 1])

        # ===== perp 范围 =====
        perp_min = delta_perp_in_window.min()
        perp_max = delta_perp_in_window.max()
        perp_range = perp_max - perp_min

        if perp_range <= 0:
            continue

        # R: perp_mode ± 10% * perp_range
        perp_low  = perp_mode - 0.10 * perp_range
        perp_high = perp_mode + 0.10 * perp_range

        keep_local = (
            (delta_perp_in_window >= perp_low) &
            (delta_perp_in_window <= perp_high)
        )

        in_win_idx_global = region_idx[in_win]
        selected_global  = in_win_idx_global[keep_local]

        out_mask[selected_global] = True

    return out_mask


############################################
# Sheather–Jones bandwidth 
############################################
def find_local_minimum_fast(counts):

    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[~np.isnan(counts)]
    n = counts.size

    if n < 10:
        return float(np.median(counts))

    # ===============================
    # 1. Silverman bandwidth
    # ===============================
    sd = np.std(counts, ddof=1)
    if sd == 0:
        return float(np.median(counts))

    iqr = np.percentile(counts, 75) - np.percentile(counts, 25)
    scale = min(sd, iqr / 1.34) if iqr > 0 else sd

    h = 0.9 * scale * n ** (-1.0 / 5.0)

    h = np.clip(h, 0.5, 2.5)

    hist, bins = np.histogram(counts, bins=200, density=True)
    x = (bins[:-1] + bins[1:]) / 2
    dx = bins[1] - bins[0]

    y = gaussian_filter1d(hist, sigma=h / dx)

    dy = np.diff(y)
    sgn = np.sign(dy)
    dsgn = np.diff(sgn)

    minima = np.where(dsgn == 2)[0] + 1

    if minima.size == 0:
        return float(x[np.argmin(y)])

    return float(x[minima[0]])

############################################
# 
############################################

def _fit_spline_and_slopes(lat_arr: np.ndarray,
                           elev_arr: np.ndarray,
                           spar: float = 0.5) -> tuple:

    if len(lat_arr) < 5:
        slope_angles = np.zeros(len(lat_arr), dtype=np.float64)
        spline = None
        return slope_angles, spline

    sort_idx = np.argsort(lat_arr)
    x = lat_arr[sort_idx]
    y = elev_arr[sort_idx]

    spline = UnivariateSpline(x, y, s=len(x) * spar, k=3)

    dy_dlat = spline.derivative(1)(x) / 111000.0
    slopes_sorted = np.arctan(dy_dlat)

    slope_angles = np.zeros(len(lat_arr), dtype=np.float64)
    slope_angles[sort_idx] = slopes_sorted
    return slope_angles, spline

def _assign_regions_by_extrema(lat_arr: np.ndarray,
                              spline: UnivariateSpline) -> np.ndarray:
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
        m = (x >= breaks[i]) & (x < breaks[i + 1])
        regions_sorted[m] = i

    regions = np.zeros(n, dtype=np.int32)
    regions[sort_idx] = regions_sorted
    return regions

############################################
# Beam-level processing
############################################

def process_single_beam(beam: str, beam_data: dict, ATL03_path: str) -> pd.DataFrame:
    print(f"Processing Beam: {beam}")
    beam_start = time.time()
    try:
        lat, lon, h = beam_data['lat'], beam_data['lon'], beam_data['h']
        signal_conf = beam_data.get('signal_conf', None)

        if len(lat) < 20:
            return pd.DataFrame()

        sort_idx = np.argsort(lat)
        lat, lon, h = lat[sort_idx], lon[sort_idx], h[sort_idx]
        if signal_conf is not None:
            signal_conf = signal_conf[sort_idx]

        lat_blocks = np.arange(np.floor(lat.min() * 100) / 100,
                               np.ceil(lat.max() * 100) / 100 + 0.01, 0.01)

        beam_results = []
        geoid_interpolator = load_geoid_interpolator()

        for block_start, block_end in zip(lat_blocks[:-1], lat_blocks[1:]):
            in_block = (lat >= block_start) & (lat < block_end)
            if np.sum(in_block) < 25:
                continue

            # ---------- block-level SNR and skip rule (MODIFIED) ----------
            snr = np.nan
            if signal_conf is not None:
                block_conf = signal_conf[in_block]
                num_signal = np.sum((block_conf >= 2) & (block_conf <= 4))
                num_noise = np.sum((block_conf == 0) | (block_conf == 1) | (block_conf == -1))
                if num_noise > 0 and num_signal / num_noise < 0.2:
                    continue
            # -------------------------------------------------------------

            block_coords = np.column_stack((lon[in_block], lat[in_block], h[in_block]))

            # ===== 1st denoise =====
            counts1 = count_neighbors_rect_vectorized(block_coords, L=100, W=20)
            P1 = find_local_minimum_fast(counts1)
            signal_coords = block_coords[counts1 >= P1]
            if len(signal_coords) < 10:
                continue

            # ===== 2nd denoise =====
            counts2 = count_neighbors_rect_vectorized(signal_coords, L=20, W=10)
            P2 = min(find_local_minimum_fast(counts2), np.floor(P1) / 5.0)
            final_coords = signal_coords[counts2 >= P2]
            if len(final_coords) < 8:
                continue

            slope_angles2, spline2 = _fit_spline_and_slopes(final_coords[:, 1], final_coords[:, 2], spar=0.5)
            regions2 = _assign_regions_by_extrema(final_coords[:, 1], spline2)

            # ===== 3rd denoise (slope rect) =====
            counts3 = count_neighbors_slope_vectorized(final_coords, slope_angles2, L=10, W=2)
            P3 = min(find_local_minimum_fast(counts3), np.floor(P2) / 5.0)

            final_signal_coords = final_coords[counts3 >= P3]
            if len(final_signal_coords) < 5:
                continue

            slope_angles3 = slope_angles2[counts3 >= P3]
            regions3 = regions2[counts3 >= P3]

            # 再拟合一次样条得到 slope_angles_new
            slope_angles_new, _spline3 = _fit_spline_and_slopes(final_signal_coords[:, 1],
                                                               final_signal_coords[:, 2],
                                                               spar=0.5)

            # final_filter_slope（region 内、min+0.25*(max-min)、并集）
            bottom25_mask = final_filter_slope_union(
                final_signal_coords,
                slope_angles_new,
                regions3,
                L=20.0
            )

            if np.sum(bottom25_mask) < 3:
                continue

            bottom25_coords = final_signal_coords[bottom25_mask]
            lons_coords = bottom25_coords[:, 0]
            lats_coords = bottom25_coords[:, 1]
            wgs84_elevs = bottom25_coords[:, 2]

            # spline on bottom25
            try:
                spline = UnivariateSpline(lats_coords, wgs84_elevs, s=len(lats_coords))
                spline_elevs = spline(lats_coords)
            except Exception as e:
                print(f"Spline fitting failed: {e}, fallback to raw elevations")
                spline_elevs = wgs84_elevs

            egm2008_elevs, geoid_heights = transform_to_egm2008(
                lons_coords, lats_coords, spline_elevs, geoid_interpolator
            )

            # ---------- add SNR column (MODIFIED) ----------
            beam_results.append(pd.DataFrame({
                "file": os.path.basename(ATL03_path),
                "beam": beam,
                "lat_block": f"{block_start:.3f}-{block_end:.3f}",
                "lon": bottom25_coords[:, 0],
                "lat": bottom25_coords[:, 1],
                "elevation_wgs84": bottom25_coords[:, 2],
                "elevation_spline": spline_elevs,
                "elevation_egm2008": egm2008_elevs,
                "geoid_height": geoid_heights
            }))
            # ----------------------------------------------

        if beam_results:
            beam_final = pd.concat(beam_results, ignore_index=True)
            print(f"Completed {beam}: {len(beam_final)} points, Time: {time.time()-beam_start:.1f}s")
            return beam_final
        return pd.DataFrame()

    except Exception as e:
        print(f"Error in {beam}: {e}")
        return pd.DataFrame()


############################################
# File-level processing
############################################

def process_single_file(ATL03_path: str, output_dir: str, index: int, total: int) -> None:
    start_time = time.time()
    file_name = os.path.basename(ATL03_path)
    output_file = os.path.join(output_dir, file_name.replace(".h5", "_Elevation.csv"))
    if os.path.exists(output_file):
        print(f"[Skip] {file_name} already processed")
        return
    print(f"\n=== [File {index}/{total}] Processing: {file_name} ===")
    beam_list = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
    h5_data = {}
    try:
        with h5py.File(ATL03_path, "r") as f:
            for beam in beam_list:
                if beam in f:
                    try:
                        lat = f[f"{beam}/heights/lat_ph"][:]
                        lon = f[f"{beam}/heights/lon_ph"][:]
                        h = f[f"{beam}/heights/h_ph"][:]
                        signal_conf = f[f"{beam}/heights/signal_conf_ph"][:]

                        land_conf = signal_conf[:, 0]
                     
                        #water_conf = signal_conf[:, 4]
                        #signal_mask = (land_conf >= 0) & (land_conf <= 4) & (water_conf <= 0)

                        #lat_filtered = lat[signal_mask]
                        #lon_filtered = lon[signal_mask]
                        #h_filtered = h[signal_mask]
                        #land_conf_filtered = land_conf[signal_mask]


                        if len(lat) > 50:
                            h5_data[beam] = {
                                "lat": lat,
                                "lon": lon,
                                "h": h,
                                "signal_conf": land_conf
                            }
                            print(f"Loaded {beam}: {len(lat)} photons")
                    except Exception as e:
                        print(f"Error loading {beam}: {e}")
    except Exception as e:
        print(f"Error opening {file_name}: {e}")
        return

    if not h5_data:
        print(f"[Warning] {file_name} has no valid data")
        return

    results = []
    for beam, data in h5_data.items():
        df = process_single_beam(beam, data, ATL03_path)
        if not df.empty:
            results.append(df)

    if results:
        final_data = pd.concat(results, ignore_index=True)
        final_data.to_csv(output_file, index=False)
        total_time = time.time() - start_time
        print(f"[Done] {file_name}: {len(final_data)} points, {total_time:.1f}s, saved -> {output_file}")
    else:
        print(f"[No Data] {file_name} has no results")

    clear_memory()

############################################
# Batch process all files
############################################

def main_batch():
    input_dir = "/home/yyr/herui/ICESat-2_ATL03/Grid_015/"
    output_dir = "/home/yyr/herui/ICESat-2_ATL03_Elevation/Grid_015/"
    os.makedirs(output_dir, exist_ok=True)
    all_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".h5")
    ])
    total_files = len(all_files)
    print(f"Found {total_files} HDF5 files")
    print("Loading geoid interpolation grid...")
    load_geoid_interpolator("/home/yyr/herui/us_nga_egm2008_1.tif")

    #  Parallel
    n_jobs_files = min(MAX_FILE_WORKERS, max(1, mp.cpu_count() - 10))
    print(f"Using {n_jobs_files} workers at file level")

    Parallel(n_jobs=n_jobs_files, prefer="processes", verbose=10)(
        delayed(process_single_file)(f, output_dir, i, total_files)
        for i, f in enumerate(all_files, start=1)
    )
    print("All files processed.")

if __name__ == "__main__":
    try:
        resource.setrlimit(resource.RLIMIT_AS, (80 * 1024**3, 80 * 1024**3))
    except Exception:
        pass

    main_batch()
    gc.collect()
