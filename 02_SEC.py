# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# ================================================================
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.prepared import prep  # Performance
from shapely import wkb            # Performance
from pathlib import Path
from tqdm import tqdm
import warnings
import multiprocessing as mp
from datetime import datetime
from scipy.spatial import cKDTree
import gc

from numba import jit, prange
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

# Global path configuration
BASE_ELEV_DIR = "/home/yyr/herui/ICESat-2_ATL03_Elevation/"
BASE_DH_DIR   = "/home/yyr/herui/ICESat-2_ATL03_Dh/"
BASE_SHP_DIR  = "/home/yyr/herui/HMA_Subregion/"

CROSS_BUFFER_RADIUS = 5
MAX_PAIR_DISTANCE   = 2
SAMPLE_RATE = 0.1
MIN_POINTS_FOR_SAMPLING = 1000

NUM_CORES = max(1, mp.cpu_count() // 6)

# Avoid overly fine task granularity for large numbers of micro-tasks
# Parquet reading uses moderate task granularity
IMAP_CHUNKSIZE = 1

# Cross-track and repeat-track pairing use moderate batch sizes
BATCH_SIZE_CROSS  = 64
BATCH_SIZE_REPEAT = 2

SLOPE_EPS = 1e-9
R_EARTH = 6371000.0

# Worker-local prepared geometry and beam cache
_GLOBAL_PREP_GEOM = None
_GLOBAL_BEAMS = None

# ============================================
# Basic functions
# ============================================

@jit(nopython=True, fastmath=True, nogil=True)
def haversine_distance(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R_EARTH * c


@jit(nopython=True, nogil=True)
def filter_points_in_buffer_numba(lats, lons, center_lat, center_lon, radius):
    n = len(lats)
    mask = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        mask[i] = haversine_distance(lats[i], lons[i], center_lat, center_lon) <= radius
    return mask


def sample_dataframe(df):
    if len(df) <= MIN_POINTS_FOR_SAMPLING:
        return df
    return df.sample(frac=SAMPLE_RATE, random_state=42)


@jit(nopython=True, fastmath=True, nogil=True)
def linear_fit_beam_numba(lons, lats):
    n = len(lons)
    if n < 2:
        return 0.0, 0.0
    sx = 0.0; sy = 0.0; sxy = 0.0; sxx = 0.0
    for i in range(n):
        x = lons[i]; y = lats[i]
        sx += x; sy += y
        sxy += x * y; sxx += x * x
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def linear_fit_beam(df):
    if len(df) < 2:
        return 0.0, 0.0
    dff = df
    if len(df) > MIN_POINTS_FOR_SAMPLING:
        dff = sample_dataframe(df)
    return linear_fit_beam_numba(dff["lon"].values.astype(np.float64),
                                 dff["lat"].values.astype(np.float64))


def extract_date_from_filename(filename):
    """Extract ATL03 acquisition date (YYYYMMDD) from the SE filename."""
    try:
        return datetime.strptime(filename[6:14], "%Y%m%d")
    except (ValueError, TypeError, IndexError):
        return None


def format_date_yyyymmdd(date_obj):
    """Return acquisition date as integer YYYYMMDD for SEC output."""
    if date_obj is None:
        return None
    return int(date_obj.strftime("%Y%m%d"))


def calculate_time_diff(date1, date2):
    """Return calendar-month and year differences; absolute values are written to output."""
    if date1 is None or date2 is None:
        return 0, 0
    months = (date2.year - date1.year) * 12 + (date2.month - date1.month)
    years = (date2 - date1).days / 365.25
    return months, years


def _lonlat_to_unitvec(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Convert longitude and latitude to float64 unit-sphere coordinates."""
    lon = np.radians(lon_deg.astype(np.float64))
    lat = np.radians(lat_deg.astype(np.float64))
    clat = np.cos(lat)
    x = clat * np.cos(lon)
    y = clat * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack((x, y, z))


def find_close_pairs_kdtree(df1, df2, max_distance_m: float):
    """Find photon pairs using KDTree candidates, Haversine distance, and one-to-one matching."""
    n1, n2 = len(df1), len(df2)
    if n1 == 0 or n2 == 0:
        return []

    # Unit-sphere coordinates
    P1 = _lonlat_to_unitvec(df1["lon"].values, df1["lat"].values)
    P2 = _lonlat_to_unitvec(df2["lon"].values, df2["lat"].values)

    # Angular and chord-distance search radius
    theta = max_distance_m / R_EARTH
    r_chord = 2.0 * np.sin(theta * 0.5)

    # KDTree candidate search
    # Use one KDTree worker because process-level parallelism is already enabled
    tree = cKDTree(P2)
    cand = tree.query_ball_point(P1, r=r_chord, workers=1)

    # Coordinate arrays
    lat1 = df1["lat"].values.astype(np.float64)
    lon1 = df1["lon"].values.astype(np.float64)
    lat2 = df2["lat"].values.astype(np.float64)
    lon2 = df2["lon"].values.astype(np.float64)

    # Collect candidate pairs
    candidates = []
    for i, js in enumerate(cand):
        if not js:
            continue
        la1, lo1 = lat1[i], lon1[i]
        for j in js:
            d = haversine_distance(la1, lo1, lat2[j], lon2[j])
            if d <= max_distance_m:
                candidates.append((i, j, d))

    # Sort candidates by increasing distance
    candidates.sort(key=lambda x: x[2])

    # Enforce one-to-one matching within this beam-pair comparison
    matched_i = set()
    matched_j = set()
    pairs = []
    for i, j, d in candidates:
        if i not in matched_i and j not in matched_j:
            pairs.append((i, j, d))
            matched_i.add(i)
            matched_j.add(j)

    return pairs


def find_intersection(s1, c1, s2, c2):
    if abs(s1 - s2) < 1e-12:
        return None
    try:
        x = (c2 - c1) / (s1 - s2)
        y = s1 * x + c1
        return (x, y)
    except:
        return None


def beam_polygon(beam, buffer_m=5.0):
    """Create a narrow along-track polygon from the fitted beam line."""
    slope, intercept = beam["slope"], beam["intercept"]
    lon_min, lon_max = beam["lon_min"], beam["lon_max"]

    # Beam endpoints
    p1 = np.array([lon_min, slope*lon_min + intercept])
    p2 = np.array([lon_max, slope*lon_max + intercept])

    # Along-track and normal vectors
    v = p2 - p1
    if np.linalg.norm(v) == 0:
        return None
    v = v / np.linalg.norm(v)
    n = np.array([-v[1], v[0]])

    # Approximate degree-to-meter conversion
    lat_mean = (beam["lat_min"] + beam["lat_max"]) / 2
    meters_per_deg_lat = 111000.0
    meters_per_deg_lon = 111000.0 * np.cos(np.radians(lat_mean))

    dlon = (buffer_m / meters_per_deg_lon) * n[0]
    dlat = (buffer_m / meters_per_deg_lat) * n[1]
    offset = np.array([dlon, dlat])

    # Polygon corners
    c1 = p1 + offset
    c2 = p2 + offset
    c3 = p2 - offset
    c4 = p1 - offset

    return Polygon([c1, c2, c3, c4])


def beam_polygons_overlap(b1, b2, buffer_m=10.0):
    poly1 = beam_polygon(b1, buffer_m)
    poly2 = beam_polygon(b2, buffer_m)
    if poly1 is None or poly2 is None:
        return False
    return poly1.intersects(poly2)


def process_single_parquet(file_path):
    out = []
    try:
        table = pq.read_table(
            file_path,
            columns=[
                "lon",
                "lat",
                "elevation_egm2008",
                "beam"
            ]
        )

        df = table.to_pandas(split_blocks=True, self_destruct=True)

        if df.empty:
            return out

        df["beam"] = df["beam"].astype("category")

        filename = Path(file_path).name
        date_obj = extract_date_from_filename(filename)

        for beam_name, beam_df in df.groupby("beam", observed=True):
            if len(beam_df) < 2:
                continue

            slope, intercept = linear_fit_beam(beam_df)

            out.append({
                "df": beam_df[["lon", "lat", "elevation_egm2008"]].copy(),
                "slope": float(slope),
                "intercept": float(intercept),
                "date_obj": date_obj,
                "file": filename,
                "beam_name": beam_name,
                "lon_min": float(beam_df["lon"].min()),
                "lon_max": float(beam_df["lon"].max()),
                "lat_min": float(beam_df["lat"].min()),
                "lat_max": float(beam_df["lat"].max()),
            })

        return out

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return out


# Pool initializer for prepared geometry, beam cache, and Numba warm-up
def _pool_initializer(prep_geom_wkb: bytes, beams):
    global _GLOBAL_PREP_GEOM, _GLOBAL_BEAMS

    try:
        geom = wkb.loads(prep_geom_wkb)
        _GLOBAL_PREP_GEOM = prep(geom)
    except Exception:
        _GLOBAL_PREP_GEOM = None

    _GLOBAL_BEAMS = beams

    _ = haversine_distance(0.0, 0.0, 0.0, 0.0)
    _ = linear_fit_beam_numba(
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64)
    )


# Cross-track photon pairs
def process_cross_orbit_pair_wrapper(args):

    results = []

    try:
        # Key change fetch beams from worker-local cache
        i, j = args
        a = _GLOBAL_BEAMS[i]
        d = _GLOBAL_BEAMS[j]

        # -------- geometry overlap --------
        if not beam_polygons_overlap(a, d, buffer_m=10.0):
            return results

        # -------- intersection --------
        inter = find_intersection(a["slope"], a["intercept"],
                                  d["slope"], d["intercept"])
        if inter is None:
            return results

        # -------- study area check (prepared geometry only) --------
        global _GLOBAL_PREP_GEOM
        if _GLOBAL_PREP_GEOM is not None:
            if not _GLOBAL_PREP_GEOM.contains(Point(inter[0], inter[1])):
                return results

        # -------- buffer filtering --------
        df1 = a["df"]
        df2 = d["df"]

        mask1 = filter_points_in_buffer_numba(
            df1["lat"].values, df1["lon"].values,
            inter[1], inter[0], CROSS_BUFFER_RADIUS
        )
        mask2 = filter_points_in_buffer_numba(
            df2["lat"].values, df2["lon"].values,
            inter[1], inter[0], CROSS_BUFFER_RADIUS
        )

        df1f = df1[mask1].reset_index(drop=True)
        df2f = df2[mask2].reset_index(drop=True)

        if len(df1f) == 0 or len(df2f) == 0:
            return results

        # -------- KDTree matching --------
        pairs = find_close_pairs_kdtree(df1f, df2f, MAX_PAIR_DISTANCE)

        # -------- acquisition dates and time difference --------
        d1, d2 = a["date_obj"], d["date_obj"]
        if d1 is None or d2 is None:
            return results

        m, y = calculate_time_diff(d1, d2)
        abs_m = abs(m)
        abs_y = abs(y)

        # -------- arrays --------
        lat1 = df1f["lat"].values.astype(np.float64)
        lon1 = df1f["lon"].values.astype(np.float64)
        h1   = df1f["elevation_egm2008"].values.astype(np.float64)

        lat2 = df2f["lat"].values.astype(np.float64)
        lon2 = df2f["lon"].values.astype(np.float64)
        h2   = df2f["elevation_egm2008"].values.astype(np.float64)

        # -------- assemble results --------
        for ii, jj, _ in pairs:
            d_check = haversine_distance(
                lat1[ii], lon1[ii],
                lat2[jj], lon2[jj]
            )
            if d_check > MAX_PAIR_DISTANCE:
                continue

            if d1 <= d2:
                p1_lon, p1_lat, p1_h = float(lon1[ii]), float(lat1[ii]), float(h1[ii])
                p2_lon, p2_lat, p2_h = float(lon2[jj]), float(lat2[jj]), float(h2[jj])
                ph1_file, ph1_beam = a["file"], a["beam_name"]
                ph2_file, ph2_beam = d["file"], d["beam_name"]
                ph1_date = format_date_yyyymmdd(d1)
                ph2_date = format_date_yyyymmdd(d2)
            else:
                p1_lon, p1_lat, p1_h = float(lon2[jj]), float(lat2[jj]), float(h2[jj])
                p2_lon, p2_lat, p2_h = float(lon1[ii]), float(lat1[ii]), float(h1[ii])
                ph1_file, ph1_beam = d["file"], d["beam_name"]
                ph2_file, ph2_beam = a["file"], a["beam_name"]
                ph1_date = format_date_yyyymmdd(d2)
                ph2_date = format_date_yyyymmdd(d1)

            dh = p2_h - p1_h

            results.append({
                "photon1_file": ph1_file,
                "photon1_beam": ph1_beam,
                "photon1_date": ph1_date,
                "photon1_lon":  p1_lon,
                "photon1_lat":  p1_lat,
                "photon1_elevation": p1_h,
                "photon2_file": ph2_file,
                "photon2_beam": ph2_beam,
                "photon2_date": ph2_date,
                "photon2_lon":  p2_lon,
                "photon2_lat":  p2_lat,
                "photon2_elevation": p2_h,
                "distance": d_check,
                "elevation_diff": float(dh),
                "months_diff": abs_m,
                "years_diff": float(abs_y),
                "monthly_change_rate": float(dh / abs_m) if abs_m > 0 else np.nan,
                "yearly_change_rate":  float(dh / abs_y) if abs_y > 0 else 0.0,
                "type": "cross"
            })

    except Exception as e:
        print(f"Error processing cross-track pair: {e}")

    return results



# Repeat-track photon pairs
def process_repeat_orbit_pair_wrapper(args):
    b1, b2, orbit_type = args
    results = []
    try:
        if b1["file"] == b2["file"]:  # Skip identical source files
            return results
        if not beam_polygons_overlap(b1, b2, buffer_m=10.0):  # Skip non-overlapping beams
            return results

        min_lon = max(b1["lon_min"], b2["lon_min"])
        max_lon = min(b1["lon_max"], b2["lon_max"])
        min_lat = max(b1["lat_min"], b2["lat_min"])
        max_lat = min(b1["lat_max"], b2["lat_max"])

        df1 = b1["df"]
        df2 = b2["df"]

        # df1 = df1[(df1["lon"] >= min_lon) & (df1["lon"] <= max_lon) & (df1["lat"] >= min_lat) & (df1["lat"] <= max_lat)].reset_index(drop=True)
        # df2 = df2[(df2["lon"] >= min_lon) & (df2["lon"] <= max_lon) & (df2["lat"] >= min_lat) & (df2["lat"] <= max_lat)].reset_index(drop=True)
        
        # Restrict to the overlapping latitude range
        df1 = df1[(df1["lat"] >= min_lat) & (df1["lat"] <= max_lat)].reset_index(drop=True)
        df2 = df2[(df2["lat"] >= min_lat) & (df2["lat"] <= max_lat)].reset_index(drop=True)


        if len(df1) == 0 or len(df2) == 0:
            return results

        pairs = find_close_pairs_kdtree(df1, df2, MAX_PAIR_DISTANCE)

        d1, d2 = b1["date_obj"], b2["date_obj"]
        if d1 is None or d2 is None:
            return results

        m, y = calculate_time_diff(d1, d2)
        abs_m = abs(m); abs_y = abs(y)

        lat1 = df1["lat"].values.astype(np.float64)
        lon1 = df1["lon"].values.astype(np.float64)
        h1   = df1["elevation_egm2008"].values.astype(np.float64)
        lat2 = df2["lat"].values.astype(np.float64)
        lon2 = df2["lon"].values.astype(np.float64)
        h2   = df2["elevation_egm2008"].values.astype(np.float64)

        for i, j, _ in pairs:
            d_check = haversine_distance(lat1[i], lon1[i], lat2[j], lon2[j])
            if d_check > MAX_PAIR_DISTANCE:
                continue

            if d1 <= d2:
                p1_lon, p1_lat, p1_h = float(lon1[i]), float(lat1[i]), float(h1[i])
                p2_lon, p2_lat, p2_h = float(lon2[j]), float(lat2[j]), float(h2[j])
                ph1_file, ph1_beam = b1["file"], b1["beam_name"]
                ph2_file, ph2_beam = b2["file"], b2["beam_name"]
                ph1_date = format_date_yyyymmdd(d1)
                ph2_date = format_date_yyyymmdd(d2)
            else:
                p1_lon, p1_lat, p1_h = float(lon2[j]), float(lat2[j]), float(h2[j])
                p2_lon, p2_lat, p2_h = float(lon1[i]), float(lat1[i]), float(h1[i])
                ph1_file, ph1_beam = b2["file"], b2["beam_name"]
                ph2_file, ph2_beam = b1["file"], b1["beam_name"]
                ph1_date = format_date_yyyymmdd(d2)
                ph2_date = format_date_yyyymmdd(d1)

            dh = p2_h - p1_h

            results.append({
                "photon1_file": ph1_file, "photon1_beam": ph1_beam,
                "photon1_date": ph1_date,
                "photon1_lon": p1_lon, "photon1_lat": p1_lat, "photon1_elevation": p1_h,
                "photon2_file": ph2_file, "photon2_beam": ph2_beam,
                "photon2_date": ph2_date,
                "photon2_lon": p2_lon, "photon2_lat": p2_lat, "photon2_elevation": p2_h,
                "distance": d_check,
                "elevation_diff": float(dh),
                "months_diff": abs_m,
                "years_diff": float(abs_y),
                "monthly_change_rate": float(dh / abs_m) if abs_m > 0 else np.nan,
                "yearly_change_rate":  float(dh / abs_y) if abs_y > 0 else 0.0,
                "type": f"repeat_{orbit_type}"
            })
    except Exception as e:
        print(f"Error processing repeat-track pair: {e}")
    return results


def process_repeat_orbit_pair_and_write(args):
    """Write one repeat-track Parquet part directly from a worker."""
    idx, task, out_dir = args
    results = process_repeat_orbit_pair_wrapper(task)
    if not results:
        return 0
    df = pd.DataFrame(results)
    part_file = out_dir / f"part_{idx:06d}.parquet"
    df.to_parquet(part_file, engine="pyarrow", compression="snappy", index=False)
    return len(df)


def write_repeat_pairs_to_parquet(tasks, path, desc, chunksize=1, merge=True):
    """Generate Parquet parts in parallel and optionally merge them."""
    out_dir = path.parent / f"{path.stem}_parts"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [(i, tasks[i], out_dir) for i in range(len(tasks))]
    total_count = 0
    with mp.Pool(NUM_CORES) as pool:
        for count in tqdm(pool.imap(process_repeat_orbit_pair_and_write, args, chunksize=chunksize),
                          total=len(tasks), desc=desc):
            total_count += count

    print(f"{desc} parts written to: {out_dir}")

    if merge:
        files = sorted(out_dir.glob("part_*.parquet"))
        if not files:
            return total_count

        schema = None
        writer = None
        try:
            for f in tqdm(files, desc=f"Merging {desc}"):
                table = pq.read_table(f)
                if schema is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(path, schema=schema, compression="snappy")
                writer.write_table(table)
        finally:
            if writer:
                writer.close()

        print(f"{desc} merged and saved to: {path}")
        # Optional cleanup of the part directory
        # import shutil; shutil.rmtree(out_dir)

    return total_count


# Region-level processing
def process_region(region_name: str):

    print("\n" + "=" * 60)
    print(f"Starting region with Parquet SE products: {region_name}")
    print("=" * 60)

    # Build input and output paths
    parquet_dir = Path(BASE_ELEV_DIR) / region_name
    out_dir     = Path(BASE_DH_DIR) / region_name
    shp_path    = Path(BASE_SHP_DIR) / f"{region_name}.shp"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not parquet_dir.exists():
        print(f"Warning: Parquet SE directory does not exist; skipping: {parquet_dir}")
        return

    if not shp_path.exists():
        print(f"Warning: shapefile does not exist; skipping: {shp_path}")
        return

    # Read the study-area boundary
    print(f"Reading study-area boundary: {shp_path}")
    study_area = gpd.read_file(shp_path)

    # Build a union geometry for prepared spatial containment tests
    union_geom = study_area.geometry.unary_union
    prep_wkb = union_geom.wkb

    # Read Parquet SE files
    files = list(parquet_dir.glob("*.parquet"))
    print(f"Found {len(files)} Parquet SE files; parsing with {NUM_CORES} processes...")

    if len(files) == 0:
        print(f"No Parquet SE files found for {region_name}; skipping.")
        return

    # Parse Parquet files into beam-level structures in parallel
    with mp.Pool(NUM_CORES) as pool:
        beams_nested = list(tqdm(
            pool.imap(process_single_parquet, files, chunksize=IMAP_CHUNKSIZE),
            total=len(files),
            desc=f"Parsing Parquet ({region_name})"
        ))

    beams = [b for sub in beams_nested for b in sub]
    print(f"Parsing completed: {len(beams)} beam observations")

    if len(beams) == 0:
        print(f"No valid beam observations found for {region_name}; skipping.")
        return

    # Classify ascending and descending beam observations
    asc_idx  = [i for i, b in enumerate(beams) if b["slope"] < -SLOPE_EPS]
    desc_idx = [i for i, b in enumerate(beams) if b["slope"] >  SLOPE_EPS]
    print(f"Beam classification completed: ascending={len(asc_idx)}, descending={len(desc_idx)}")

    # Build cross-track and repeat-track candidate tasks

    cross_tasks = []
    for i in asc_idx:
        for j in desc_idx:
            bi, bj = beams[i], beams[j]
            if bi["file"] == bj["file"]:
                continue
            if beam_polygons_overlap(bi, bj, buffer_m=10.0):
                cross_tasks.append((i, j))  # Pass beam indices only
            
                
    print(f"Cross-track candidate tasks: {len(cross_tasks)}")

    rep_asc_tasks = []
    for a in range(len(asc_idx)):
        for b in range(a + 1, len(asc_idx)):
            i, j = asc_idx[a], asc_idx[b]
            bi, bj = beams[i], beams[j]
            if bi["file"] == bj["file"]:
                continue
            if beam_polygons_overlap(bi, bj, buffer_m=10.0):
                rep_asc_tasks.append((bi, bj, "asc"))

    rep_desc_tasks = []
    for a in range(len(desc_idx)):
        for b in range(a + 1, len(desc_idx)):
            i, j = desc_idx[a], desc_idx[b]
            bi, bj = beams[i], beams[j]
            if bi["file"] == bj["file"]:
                continue
            if beam_polygons_overlap(bi, bj, buffer_m=10.0):
                rep_desc_tasks.append((bi, bj, "desc"))

    print(f"Repeat-track candidates: ascending={len(rep_asc_tasks)}, descending={len(rep_desc_tasks)}")

    # Cross-track photon pairs
    cross_pairs = []
    if cross_tasks:
        with mp.Pool(NUM_CORES, 
                    initializer=_pool_initializer, 
                    initargs=(prep_wkb, beams)
        ) as pool:
            cross_chunks = list(tqdm(
                pool.imap_unordered(
                    process_cross_orbit_pair_wrapper,
                    cross_tasks,
                    chunksize=BATCH_SIZE_CROSS
                ),
                total=len(cross_tasks),
                desc=f"Cross-track pairing ({region_name})"
            ))
        cross_pairs = [r for sub in cross_chunks for r in sub]
        del cross_chunks
        gc.collect()

    print(f"Cross-track photon pairs: {len(cross_pairs)}")
    if cross_pairs:
        cross_path = out_dir / "optimized_cross_pairs.parquet"
        pd.DataFrame(cross_pairs).to_parquet(
            cross_path,
            engine="pyarrow",
            compression="snappy",
            index=False
        )
        print(f"Saved cross-track photon pairs: {cross_path}")

    del cross_pairs
    gc.collect()

    # Ascending repeat-track photon pairs
    rep_asc_path = out_dir / "optimized_repeat_asc_pairs.parquet"
    if rep_asc_tasks:
        rep_asc_count = write_repeat_pairs_to_parquet(
            rep_asc_tasks,
            rep_asc_path,
            f"Ascending repeat-track pairs ({region_name})",
            chunksize=BATCH_SIZE_REPEAT,
            merge=True
        )
        print(f"Saved ascending repeat-track pairs: {rep_asc_path}, count={rep_asc_count}")

    gc.collect()

    # Descending repeat-track photon pairs
    rep_desc_path = out_dir / "optimized_repeat_desc_pairs.parquet"
    if rep_desc_tasks:
        rep_desc_count = write_repeat_pairs_to_parquet(
            rep_desc_tasks,
            rep_desc_path,
            f"Descending repeat-track pairs ({region_name})",
            chunksize=BATCH_SIZE_REPEAT,
            merge=True
        )
        print(f"Saved descending repeat-track pairs: {rep_desc_path}, count={rep_desc_count}")

    gc.collect()

    print(
        f"Region {region_name} completed | "
        f"cross_candidates={len(cross_tasks)}, "
        f"repeat_ascending_candidates={len(rep_asc_tasks)}, "
        f"repeat_descending_candidates={len(rep_desc_tasks)}"
    )


# Main program

def main():
    Path(BASE_DH_DIR).mkdir(parents=True, exist_ok=True)

    target_regions = [f"Grid_{i:03d}" for i in range(1, 47)]

    print("SEC pairing will run for the following regions:")
    for region_name in target_regions:
        print(" -", region_name)

    for i, region_name in enumerate(target_regions, start=1):
        print(f"\n[{i}/{len(target_regions)}] Starting region: {region_name}")

        try:
            process_region(region_name)
        except Exception as e:
            print(f"Region {region_name} failed: {e}")
            continue

        try:
            gc.collect()

            try:
                import pyarrow as pa
                pa.default_memory_pool().release_unused()
            except Exception:
                pass

            time.sleep(1)

        except Exception as e:
            print(f"Memory cleanup warning: {e}")

        print(f"Region {region_name} completed and memory was released")

    print("\nAll specified regions completed")


if __name__ == "__main__":
    main()
