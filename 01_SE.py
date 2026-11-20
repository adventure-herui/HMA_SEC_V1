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
from scipy.spatial import cKDTree
from tqdm import tqdm
from joblib import parallel_backend

# ================== 并行 & 内存设置 =================
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 

os.environ["NUMBA_NUM_THREADS"] = "8" 

MAX_FILE_WORKERS = 32 

BLOCK_THREAD_WORKERS = 12 

# ================== PRINT CONTROL ==================
def quiet_print(*args, **kwargs):
    return



def clear_memory():
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


# ================== 全局 geoid 缓存 ==================
_GEOID_CACHE = None


#####################################

def get_geoid_interpolator(geoid_tiff_path="/home/yyr/herui/us_nga_egm2008_1.tif"):
    global _GEOID_CACHE
    if _GEOID_CACHE is not None:
        return _GEOID_CACHE

    try:
        quiet_print(f"Loading geoid TIFF file: {geoid_tiff_path}")
        with rasterio.open(geoid_tiff_path) as src:
            geoid_data = src.read(1)
            transform = src.transform
            width, height = src.width, src.height

            lons = np.linspace(transform[2], transform[2] + transform[0] * width, width)
            lats = np.linspace(transform[5], transform[5] + transform[4] * height, height)

            _GEOID_CACHE = RegularGridInterpolator(
                (lons, lats),
                geoid_data.T,
                method='linear',
                bounds_error=False,
                fill_value=0
            )
            quiet_print("Geoid interpolator created successfully")
            return _GEOID_CACHE
    except Exception as e:
        quiet_print(f"Error loading geoid grid: {e}")
        return None


def transform_to_egm2008(lons, lats, spline_elevs, interpolator):
    if interpolator is None:
        return spline_elevs, np.zeros_like(spline_elevs)

    try:
        geoid_heights = interpolator((lons, lats))
        egm2008_elevations = spline_elevs - geoid_heights
        return egm2008_elevations, geoid_heights
    except Exception as e:
        quiet_print(f"Error in elevation transformation: {e}")
        return spline_elevs, np.zeros_like(spline_elevs)


#####################################

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


#####################################

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
        delta_perp = -delta_lat * sin_a + delta_elev * cos_a

        counts[i] = np.sum((np.abs(delta_along) <= L / 2) & (np.abs(delta_perp) <= W / 2)) - 1
    return counts

############ 函数测试区 #################


#####################################

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
        if np.sum(in_win) < 1:  # 已经验证：设置太大，如果地面点不够多的话，会导致点数骤降
            continue

        delta_perp_in_window = delta_perp[in_win]
        m = delta_perp_in_window.size
        nbins = max(10, m // 5)

        hist_counts, bin_edges = np.histogram(delta_perp_in_window, bins=nbins)
        mode_bin = np.argmax(hist_counts)
        perp_mode = 0.5 * (bin_edges[mode_bin] + bin_edges[mode_bin + 1])

        perp_min = delta_perp_in_window.min()
        perp_max = delta_perp_in_window.max()
        perp_range = perp_max - perp_min

        if perp_range <= 0:
            continue

        perp_low = perp_mode - 0.10 * perp_range
        perp_high = perp_mode + 0.10 * perp_range

        keep_local = ((delta_perp_in_window >= perp_low) & (delta_perp_in_window <= perp_high))
        in_win_idx_global = region_idx[in_win]
        selected_global = in_win_idx_global[keep_local]

        out_mask[selected_global] = True

    return out_mask



#####################################

def find_local_minimum_fast(counts):
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


#####################################
'''
def _fit_spline_and_slopes(lat_arr: np.ndarray, elev_arr: np.ndarray, spar: float = 0.5) -> tuple:
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
'''

def _fit_spline_and_slopes(lat_arr: np.ndarray, elev_arr: np.ndarray, spar: float = 0.5) -> tuple:

    n = len(lat_arr)

    if n < 80:
        return np.zeros(n, dtype=np.float64), None

    if np.ptp(lat_arr) < 1e-4:
        return np.zeros(n, dtype=np.float64), None

    if np.std(elev_arr) < 0.05:
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


    S_MIN = max(100.0, 0.2 * x.size)
    s = max(x.size * spar, S_MIN)


    try:
        spline = UnivariateSpline(x, y, s=s, k=3)
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


#####################################

def _assign_regions_by_extrema(lat_arr: np.ndarray, spline: UnivariateSpline) -> np.ndarray:
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



####################################################
####################################################

def _process_single_block_internal(
    block_start, block_end,
    lat, lon, h, signal_conf,
    beam, ATL03_path, geoid_interpolator
):
    # 注意：这里保持你 Stage1–6 的完整逻辑，不做删改
    from pygam import LinearGAM, s

    out_rows = []

    in_block = (lat >= block_start - 0.001) & (lat < block_end + 0.001)
    if np.sum(in_block) < 10:
        return out_rows

    # ---------- SNR ----------
    snr = np.nan
    if signal_conf is not None:
        bc = signal_conf[in_block]
        ns = np.sum((bc >= 2) & (bc <= 4))
        nn = np.sum((bc <= 1) | (bc == -1))
        snr = (ns / nn) if nn > 0 else np.nan
        if nn > 0 and snr < 0.1:
            return out_rows

    block_coords = np.column_stack((lon[in_block], lat[in_block], h[in_block]))

    # ================= Stage 1 =================
    c1 = count_neighbors_rect_vectorized(block_coords, 100, 20)
    P1 = find_local_minimum_fast(c1)
    s1 = block_coords[c1 >= P1]
    if len(s1) < 10:
        return out_rows

    # ================= Stage 2 =================
    c2 = count_neighbors_rect_vectorized(s1, 20, 10)
    P2 = min(find_local_minimum_fast(c2), np.floor(P1) / 2.0)
    s2 = s1[c2 >= P2]
    if len(s2) < 8:
        return out_rows

    # ================= Stage 3 =================
    slope2, spline2 = _fit_spline_and_slopes(s2[:, 1], s2[:, 2], spar=0.5)
    regions2 = _assign_regions_by_extrema(s2[:, 1], spline2)

    c3 = count_neighbors_slope_vectorized(s2, slope2, 10, 2)
    P3 = min(find_local_minimum_fast(c3), np.floor(P2) / 2.0)
    s3 = s2[c3 >= P3]
    if len(s3) < 5:
        return out_rows

    # ================= Stage 4 =================
    regions3 = regions2[c3 >= P3]

    slope_new, _ = _fit_spline_and_slopes(s3[:, 1], s3[:, 2], spar=0.5)

    # ================= Stage 5 =================
    mask = final_filter_slope_union(s3, slope_new, regions3, L=20.0)
    if np.sum(mask) < 3:
        return out_rows

    bottom25 = s3[mask]

    # ================= Stage 6 : block 内部 GAM =================
    lat_b = bottom25[:, 1]
    lon_b = bottom25[:, 0]
    elev_b = bottom25[:, 2]

    if len(lat_b) < 30:
        elev_spline_b = elev_b.copy()
    else:
        x_norm = (lat_b - lat_b.mean()) / (lat_b.ptp() + 1e-9)
        Xb = x_norm.reshape(-1, 1)

        gam = LinearGAM(
            s(0, n_splines=40, spline_order=3),
            fit_intercept=True
        ).fit(Xb, elev_b)

        elev_spline_b = gam.predict(Xb)

    egm2008_b, geoid_b = transform_to_egm2008(
        lon_b, lat_b, elev_spline_b, geoid_interpolator
    )

    for i in range(len(lat_b)):
        out_rows.append((
            os.path.basename(ATL03_path),
            beam,
            lon_b[i],
            lat_b[i],
            elev_b[i],
            elev_spline_b[i],
            egm2008_b[i],
            geoid_b[i],
            snr
        ))

    return out_rows


def process_single_beam(beam: str, beam_data: dict, ATL03_path: str) -> pd.DataFrame:
    # 保留你的算法逻辑：只改 block loop -> threading 并行

    try:
        lat = beam_data["lat"]
        lon = beam_data["lon"]
        h   = beam_data["h"]
        signal_conf = beam_data.get("signal_conf", None)

        if len(lat) < 20:
            return pd.DataFrame()

        # sort by latitude
        sort_idx = np.argsort(lat)
        lat = lat[sort_idx]
        lon = lon[sort_idx]
        h   = h[sort_idx]
        if signal_conf is not None:
            signal_conf = signal_conf[sort_idx]

        # block strategy + single-block fallback
        lat_span = lat.max() - lat.min()
        if lat_span < 0.05:
            lat_blocks = np.array([lat.min(), lat.max()])
        else:
            lat_blocks = np.arange(
                np.floor(lat.min() * 100) / 100,
                np.ceil(lat.max() * 100) / 100 + 0.01,
                0.01
            )

        geoid_interpolator = get_geoid_interpolator()

        # =====================================================
        # block loop (Stage1–6) -> 结构性并行：threading
        # =====================================================
        block_pairs = list(zip(lat_blocks[:-1], lat_blocks[1:]))

        out_rows = []

        # threading 不会创建新进程，不会叠加 loky 灾难
        # 注意：这里 n_jobs=BLOCK_THREAD_WORKERS
        with parallel_backend("threading", n_jobs=BLOCK_THREAD_WORKERS):
            block_results = Parallel(batch_size=1)(
                delayed(_process_single_block_internal)(
                    bs, be, lat, lon, h, signal_conf,
                    beam, ATL03_path, geoid_interpolator
                )
                for bs, be in block_pairs
            )

        for rows in block_results:
            if rows:
                out_rows.extend(rows)

        if not out_rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            out_rows,
            columns=[
                "file", "beam",
                "lon", "lat",
                "elevation_wgs84",
                "elevation_spline",
                "elevation_egm2008",
                "geoid_height",
                "SNR"
            ]
        )
        return df

    except Exception:
        return pd.DataFrame()

    finally:
        gc.collect()

####################################################
####################################################
'''
def process_single_beam(beam: str, beam_data: dict, ATL03_path: str) -> pd.DataFrame:
    from pygam import LinearGAM, s

    quiet_print(f"Processing Beam: {beam}")
    beam_start = time.time()

    try:
        lat = beam_data["lat"]
        lon = beam_data["lon"]
        h = beam_data["h"]
        signal_conf = beam_data.get("signal_conf", None)

        if len(lat) < 20:
            return pd.DataFrame()

        # =====================================================
        # sort by latitude
        # =====================================================
        sort_idx = np.argsort(lat)
        lat = lat[sort_idx]
        lon = lon[sort_idx]
        h = h[sort_idx]
        if signal_conf is not None:
            signal_conf = signal_conf[sort_idx]

        # =====================================================
        # block strategy + single-block fallback 
        # =====================================================
        lat_span = lat.max() - lat.min()

        if lat_span < 0.05:
            lat_blocks = np.array([lat.min(), lat.max()])
            quiet_print(
                f"[{beam}]  short track (Δlat={lat_span:.4f}), "
                f"use SINGLE-BLOCK mode"
            )
        else:
            lat_blocks = np.arange(
                np.floor(lat.min() * 100) / 100,
                np.ceil(lat.max() * 100) / 100 + 0.01, 0.01)

        geoid_interpolator = get_geoid_interpolator()

        out_rows = []

        # =====================================================
        # block loop (Stage1–6)
        # =====================================================
        for block_start, block_end in zip(lat_blocks[:-1], lat_blocks[1:]):

            in_block = (lat >= block_start - 0.001) & (lat < block_end + 0.001)
            if np.sum(in_block) < 10:
                continue

            # ---------- SNR ----------
            snr = np.nan
            if signal_conf is not None:
                bc = signal_conf[in_block]
                ns = np.sum((bc >= 2) & (bc <= 4))
                nn = np.sum((bc <= 1) | (bc == -1))
                snr = (ns / nn) if nn > 0 else np.nan
                if nn > 0 and snr < 0.1:
                    continue

            block_coords = np.column_stack((lon[in_block], lat[in_block], h[in_block]))

            # ================= Stage 1 =================
            c1 = count_neighbors_rect_vectorized(block_coords, 100, 20)
            P1 = find_local_minimum_fast(c1)
            s1 = block_coords[c1 >= P1]
            if len(s1) < 10:
                continue

            # ================= Stage 2 =================
            c2 = count_neighbors_rect_vectorized(s1, 20, 10)
            P2 = min(find_local_minimum_fast(c2), np.floor(P1) / 2.0) # 限制阈值过大/小
            s2 = s1[c2 >= P2]
            if len(s2) < 8:
                continue

            # ================= Stage 3 =================
            slope2, spline2 = _fit_spline_and_slopes(
                s2[:, 1], s2[:, 2], spar=0.5
            )
            regions2 = _assign_regions_by_extrema(s2[:, 1], spline2)

            c3 = count_neighbors_slope_vectorized(s2, slope2, 10, 2)
            P3 = min(find_local_minimum_fast(c3), np.floor(P2) / 2.0)
            s3 = s2[c3 >= P3]
            if len(s3) < 5:
                continue

            # ================= Stage 4 =================
            slope3 = slope2[c3 >= P3]
            regions3 = regions2[c3 >= P3]

            slope_new, _ = _fit_spline_and_slopes(
                s3[:, 1], s3[:, 2], spar=0.5
            )

            # ================= Stage 5 =================
            mask = final_filter_slope_union(
                s3, slope_new, regions3, L=20.0
            )
            if np.sum(mask) < 3:
                continue

            bottom25 = s3[mask]

            # =================================================
            # Stage 6 : block 内部 GAM
            # =================================================
            lat_b = bottom25[:, 1]
            lon_b = bottom25[:, 0]
            elev_b = bottom25[:, 2]

            # ---------- 点数判断
            if len(lat_b) < 30:
                elev_spline_b = elev_b.copy()
            else:
                # ---------- 标准化输入
                x_norm = (lat_b - lat_b.mean()) / (lat_b.ptp() + 1e-9)
                Xb = x_norm.reshape(-1, 1)

                gam = LinearGAM(
                    s(0, n_splines=40, spline_order=3),
                    fit_intercept=True
                ).fit(Xb, elev_b)

                elev_spline_b = gam.predict(Xb)

            egm2008_b, geoid_b = transform_to_egm2008(
                lon_b, lat_b, elev_spline_b, geoid_interpolator
            )

            # ---------- collect ----------
            for i in range(len(lat_b)):
                out_rows.append((
                    os.path.basename(ATL03_path),
                    beam,
                    lon_b[i],
                    lat_b[i],
                    elev_b[i],
                    elev_spline_b[i],
                    egm2008_b[i],
                    geoid_b[i],
                    snr
                ))

        if not out_rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            out_rows,
            columns=[
                "file", "beam",
                "lon", "lat",
                "elevation_wgs84",
                "elevation_spline",
                "elevation_egm2008",
                "geoid_height",
                "SNR"
            ]
        )

        quiet_print(
            f"Completed {beam}: {len(df)} points, "
            f"Time: {time.time() - beam_start:.1f}s"
        )
        return df

    except Exception as e:
        print(f"Error in {beam}: {e}")
        return pd.DataFrame()

    finally:
        gc.collect()
'''

####################################################
####################################################
def process_single_file(ATL03_path: str, output_dir: str, index: int, total: int) -> None:

    start_time = time.time()
    file_name = os.path.basename(ATL03_path)
    output_file = os.path.join(output_dir, file_name.replace(".h5", "_Elevation.parquet"))

    if os.path.exists(output_file):
        quiet_print(f"[Skip] {file_name} already processed")
        return

    print(f"\n=== [File {index}/{total}] Processing: {file_name} ===")

    beam_list = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
    h5_data = {}

    try:
        with h5py.File(ATL03_path, "r") as f:

            for beam in beam_list:

                if beam not in f:
                    quiet_print(f"[Skip] {beam}: group not found")
                    continue

                if "heights" not in f[beam]:
                    quiet_print(f"[Skip] {beam}: 'heights' group missing")
                    continue

                grp = f[beam]["heights"]

                required = ["lat_ph", "lon_ph", "h_ph", "signal_conf_ph"]
                missing = [k for k in required if k not in grp]
                if missing:
                    quiet_print(f"[Skip] {beam}: missing datasets {missing}")
                    continue

                try:
                    lat = grp["lat_ph"][:]
                    lon = grp["lon_ph"][:]
                    h   = grp["h_ph"][:]
                    signal_conf = grp["signal_conf_ph"][:]
                except Exception as e:
                    quiet_print(f"[Skip] {beam}: dataset read error: {e}")
                    continue

                n_total = lat.size
                if n_total < 50:
                    quiet_print(f"[Skip] {beam}: too few photons ({n_total})")
                    continue

                # land signal confidence（第 1 列）
                land_conf = signal_conf[:, 0]

                # 置信度筛选
                valid_mask = land_conf >= 1
                n_valid = np.sum(valid_mask)

                if n_valid < 50:
                    quiet_print(
                        f"[Skip] {beam}: too few valid photons "
                        f"after confidence filter ({n_valid}/{n_total})"
                    )
                    continue

                lat = lat[valid_mask]
                lon = lon[valid_mask]
                h   = h[valid_mask]
                land_conf = land_conf[valid_mask]

                removed = n_total - n_valid
                quiet_print(
                    f"Loaded {beam}: {n_valid:,} photons "
                    f"(filtered out {removed:,})"
                )

                h5_data[beam] = {
                    "lat": lat,
                    "lon": lon,
                    "h": h,
                    "signal_conf": land_conf
                }

    except Exception as e:
        quiet_print(f"[Fatal] Failed to open {file_name}: {e}")
        return

    if not h5_data:
        quiet_print(f"[No Valid Beams] {file_name} after confidence filtering")
        return

    results = []

    for beam, data in h5_data.items():
        df = process_single_beam(beam, data, ATL03_path)
        if not df.empty:
            results.append(df)

    if results:
        final_data = pd.concat(results, ignore_index=True)
        final_data.to_parquet(
            output_file,
            engine="pyarrow",
            compression="zstd",
            index=False
        )

        print(
            f"[Done] {file_name}: {len(final_data)} points, "
            f"{time.time() - start_time:.1f}s, saved -> {output_file}"
        )
    else:
        print(f"[No Data After Processing] {file_name}")

    clear_memory()


####################################################
####################################################

from contextlib import contextmanager
import joblib

@contextmanager
def tqdm_joblib(tqdm_object):

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

####################################################
####################################################

def main_batch():

    # 要处理的区域["Grid_001", "Grid_002", "Grid_003", "Grid_004"]
    
    target_regions = ["Grid_005"]
    
    
    # 预加载geoid
    get_geoid_interpolator("/home/yyr/herui/us_nga_egm2008_1.tif")
    
    # 处理每个区域
    for region in target_regions:
        print(f"\n{'='*60}")
        print(f"开始处理区域: {region}")
        print(f"{'='*60}")
        
        input_dir = f"/home/yyr/herui/ICESat-2_ATL03/{region}/"
        output_dir = f"/home/yyr/herui/ICESat-2_ATL03_Elevation/{region}/"
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取该区域的所有h5文件
        all_files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".h5")
        ])
        
        if not all_files:
            print(f"警告: {input_dir} 中没有找到.h5文件，跳过")
            continue
        
        total_files = len(all_files)
        print(f"找到 {total_files} 个文件")

        # 进度条
        pbar = tqdm(total=total_files, desc=f"处理 {region}", ncols=90, leave=True)
        
        with tqdm_joblib(pbar):
            Parallel(
                n_jobs=MAX_FILE_WORKERS,
                backend="loky",
                batch_size=1,
                pre_dispatch="1 * n_jobs",
                verbose=0
            )(
                delayed(process_single_file)(f, output_dir, i, total_files)
                for i, f in enumerate(all_files, start=1)
            )
        
        print(f"✅ 区域 {region} 处理完成")

    print("\n✅ 所有指定区域处理完成！")


if __name__ == "__main__":
    main_batch()
