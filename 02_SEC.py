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
from shapely.prepared import prep  # <<< PERF >>>
from shapely import wkb            # <<< PERF >>>
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

# ================= 全局路径配置 =================
BASE_ELEV_DIR = "/home/yyr/herui/ICESat-2_ATL03_Elevation/"
BASE_DH_DIR   = "/home/yyr/herui/ICESat-2_ATL03_Dh/"
BASE_SHP_DIR  = "/home/yyr/herui/HMA_Subregion/"

CROSS_BUFFER_RADIUS = 5
MAX_PAIR_DISTANCE   = 2
SAMPLE_RATE = 0.1
MIN_POINTS_FOR_SAMPLING = 1000

NUM_CORES = max(1, mp.cpu_count() // 6)

# <<< PERF >>> 关键：不要极细粒度 (chunksize=1) 处理百万级 micro-task 会被调度开销拖死
# 读取 Parquet 属于“重任务”，适合中等粒度
IMAP_CHUNKSIZE = 1

# 交叉/重复属于“微任务百万级”，适合中等粒度，才能吃满 CPU
BATCH_SIZE_CROSS  = 64
BATCH_SIZE_REPEAT = 2

SLOPE_EPS = 1e-9
R_EARTH = 6371000.0

# <<< PERF >>> worker 共享 prepared geometry（避免每任务 GeoDataFrame contains 的 Python/GeoPandas 开销）
_GLOBAL_PREP_GEOM = None
_GLOBAL_BEAMS = None

# ============================================
# ================= 基础函数 =================
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
    try:
        return datetime.strptime(filename[6:14], "%Y%m%d")
    except:
        return None


def calculate_time_diff(date1, date2):
    """返回 (months, years)，可能为负；最终写表时取绝对值。"""
    if date1 is None or date2 is None:
        return 0, 0
    months = (date2.year - date1.year) * 12 + (date2.month - date1.month)
    years = (date2 - date1).days / 365.25
    return months, years


def _lonlat_to_unitvec(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """(lon, lat) -> 单位球面 (x,y,z)，float64"""
    lon = np.radians(lon_deg.astype(np.float64))
    lat = np.radians(lat_deg.astype(np.float64))
    clat = np.cos(lat)
    x = clat * np.cos(lon)
    y = clat * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack((x, y, z))


def find_close_pairs_kdtree(df1, df2, max_distance_m: float):
    """KDTree + Haversine 查找点对，双向唯一匹配"""
    n1, n2 = len(df1), len(df2)
    if n1 == 0 or n2 == 0:
        return []

    # 单位球坐标
    P1 = _lonlat_to_unitvec(df1["lon"].values, df1["lat"].values)
    P2 = _lonlat_to_unitvec(df2["lon"].values, df2["lat"].values)

    # 球心角 & 弦长半径
    theta = max_distance_m / R_EARTH
    r_chord = 2.0 * np.sin(theta * 0.5)

    # KDTree 搜索
    # <<< PERF >>> 进程并行已经足够，禁止 KDTree 内部 workers=-1 造成进程×线程爆炸
    tree = cKDTree(P2)
    cand = tree.query_ball_point(P1, r=r_chord, workers=1)

    # 经纬度数组
    lat1 = df1["lat"].values.astype(np.float64)
    lon1 = df1["lon"].values.astype(np.float64)
    lat2 = df2["lat"].values.astype(np.float64)
    lon2 = df2["lon"].values.astype(np.float64)

    # 收集所有候选点对
    candidates = []
    for i, js in enumerate(cand):
        if not js:
            continue
        la1, lo1 = lat1[i], lon1[i]
        for j in js:
            d = haversine_distance(la1, lo1, lat2[j], lon2[j])
            if d <= max_distance_m:
                candidates.append((i, j, d))

    # 距离升序排序
    candidates.sort(key=lambda x: x[2])

    # 双向唯一匹配
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
    """根据线性拟合结果，生成沿轨道方向的窄矩形 polygon"""
    slope, intercept = beam["slope"], beam["intercept"]
    lon_min, lon_max = beam["lon_min"], beam["lon_max"]

    # 起点和终点
    p1 = np.array([lon_min, slope*lon_min + intercept])
    p2 = np.array([lon_max, slope*lon_max + intercept])

    # 方向和法向量
    v = p2 - p1
    if np.linalg.norm(v) == 0:
        return None
    v = v / np.linalg.norm(v)
    n = np.array([-v[1], v[0]])

    # 经纬度到米的换算
    lat_mean = (beam["lat_min"] + beam["lat_max"]) / 2
    meters_per_deg_lat = 111000.0
    meters_per_deg_lon = 111000.0 * np.cos(np.radians(lat_mean))

    dlon = (buffer_m / meters_per_deg_lon) * n[0]
    dlat = (buffer_m / meters_per_deg_lat) * n[1]
    offset = np.array([dlon, dlat])

    # 矩形四个角点
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


def process_single_csv(file_path):
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
        print(f"处理文件 {file_path} 出错: {e}")
        return out


# <<< PERF >>> Pool initializer：prepared geometry + numba 预热（避免 96 次冷启动）
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


# ================= 交叉轨道对 =================
def process_cross_orbit_pair_wrapper(args):

    results = []

    try:
        # <<< KEY CHANGE >>> fetch beams from worker-local cache
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

        # -------- time difference --------
        d1, d2 = a["date_obj"], d["date_obj"]
        m, y = calculate_time_diff(d1, d2) if (d1 and d2) else (0, 0)
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

            if (d1 is not None and d2 is not None and d1 <= d2):
                p1_lon, p1_lat, p1_h = float(lon1[ii]), float(lat1[ii]), float(h1[ii])
                p2_lon, p2_lat, p2_h = float(lon2[jj]), float(lat2[jj]), float(h2[jj])
                ph1_file, ph1_beam = a["file"], a["beam_name"]
                ph2_file, ph2_beam = d["file"], d["beam_name"]
            else:
                p1_lon, p1_lat, p1_h = float(lon2[jj]), float(lat2[jj]), float(h2[jj])
                p2_lon, p2_lat, p2_h = float(lon1[ii]), float(lat1[ii]), float(h1[ii])
                ph1_file, ph1_beam = d["file"], d["beam_name"]
                ph2_file, ph2_beam = a["file"], a["beam_name"]

            dh = p2_h - p1_h

            results.append({
                "photon1_file": ph1_file,
                "photon1_beam": ph1_beam,
                "photon1_lon":  p1_lon,
                "photon1_lat":  p1_lat,
                "photon1_elevation": p1_h,
                "photon2_file": ph2_file,
                "photon2_beam": ph2_beam,
                "photon2_lon":  p2_lon,
                "photon2_lat":  p2_lat,
                "photon2_elevation": p2_h,
                "distance": d_check,
                "elevation_diff": float(dh),
                "months_diff": abs_m,
                "years_diff": float(abs_y),
                "monthly_change_rate": float(dh / abs_m) if abs_m > 0 else 0.0,
                "yearly_change_rate":  float(dh / abs_y) if abs_y > 0 else 0.0,
                "type": "cross"
            })

    except Exception as e:
        print(f"处理交叉轨道对时出错: {e}")

    return results



# ================= 重复轨道对 =================
def process_repeat_orbit_pair_wrapper(args):
    b1, b2, orbit_type = args
    results = []
    try:
        if b1["file"] == b2["file"]: # 同轨道 + 同文件 ==》 直接跳过
            return results
        if not beam_polygons_overlap(b1, b2, buffer_m=10.0): # 两波束外包矩形没有交集 ==》 直接跳过
            return results

        min_lon = max(b1["lon_min"], b2["lon_min"])
        max_lon = min(b1["lon_max"], b2["lon_max"])
        min_lat = max(b1["lat_min"], b2["lat_min"])
        max_lat = min(b1["lat_max"], b2["lat_max"])

        df1 = b1["df"]
        df2 = b2["df"]

        # df1 = df1[(df1["lon"] >= min_lon) & (df1["lon"] <= max_lon) & (df1["lat"] >= min_lat) & (df1["lat"] <= max_lat)].reset_index(drop=True)
        # df2 = df2[(df2["lon"] >= min_lon) & (df2["lon"] <= max_lon) & (df2["lat"] >= min_lat) & (df2["lat"] <= max_lat)].reset_index(drop=True)
        
        # 只限制纬度
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
            else:
                p1_lon, p1_lat, p1_h = float(lon2[j]), float(lat2[j]), float(h2[j])
                p2_lon, p2_lat, p2_h = float(lon1[i]), float(lat1[i]), float(h1[i])
                ph1_file, ph1_beam = b2["file"], b2["beam_name"]
                ph2_file, ph2_beam = b1["file"], b1["beam_name"]

            dh = p2_h - p1_h

            results.append({
                "photon1_file": ph1_file, "photon1_beam": ph1_beam,
                "photon1_lon": p1_lon, "photon1_lat": p1_lat, "photon1_elevation": p1_h,
                "photon2_file": ph2_file, "photon2_beam": ph2_beam,
                "photon2_lon": p2_lon, "photon2_lat": p2_lat, "photon2_elevation": p2_h,
                "distance": d_check,
                "elevation_diff": float(dh),
                "months_diff": abs_m,
                "years_diff": float(abs_y),
                "monthly_change_rate": float(dh / abs_m) if abs_m > 0 else 0.0,
                "yearly_change_rate":  float(dh / abs_y) if abs_y > 0 else 0.0,
                "type": f"repeat_{orbit_type}"
            })
    except Exception as e:
        print(f"处理重复轨道对时出错: {e}")
    return results


def process_repeat_orbit_pair_and_write(args):
    """worker 直接写 parquet 分片"""
    idx, task, out_dir = args
    results = process_repeat_orbit_pair_wrapper(task)
    if not results:
        return 0
    df = pd.DataFrame(results)
    part_file = out_dir / f"part_{idx:06d}.parquet"
    df.to_parquet(part_file, engine="pyarrow", compression="snappy", index=False)
    return len(df)


def write_repeat_pairs_to_parquet(tasks, path, desc, chunksize=1, merge=True):
    """并行生成分片 → 可选合并为一个大文件"""
    out_dir = path.parent / f"{path.stem}_parts"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [(i, tasks[i], out_dir) for i in range(len(tasks))]
    total_count = 0
    with mp.Pool(NUM_CORES) as pool:
        for count in tqdm(pool.imap(process_repeat_orbit_pair_and_write, args, chunksize=chunksize),
                          total=len(tasks), desc=desc):
            total_count += count

    print(f"{desc} 分片已写入目录: {out_dir}")

    if merge:
        files = sorted(out_dir.glob("part_*.parquet"))
        if not files:
            return total_count

        schema = None
        writer = None
        try:
            for f in tqdm(files, desc=f"合并 {desc}"):
                table = pq.read_table(f)
                if schema is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(path, schema=schema, compression="snappy")
                writer.write_table(table)
        finally:
            if writer:
                writer.close()

        print(f"{desc} 已合并保存为单文件: {path}")
        # 可选：清理分片目录
        # import shutil; shutil.rmtree(out_dir)

    return total_count


# ================= 区域级处理函数 =================
def process_region(region_name: str):

    print("\n" + "=" * 60)
    print(f"开始处理区域（Parquet 高程产品）：{region_name}")
    print("=" * 60)

    # ================= 路径构造 =================
    parquet_dir = Path(BASE_ELEV_DIR) / region_name
    out_dir     = Path(BASE_DH_DIR) / region_name
    shp_path    = Path(BASE_SHP_DIR) / f"{region_name}.shp"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not parquet_dir.exists():
        print(f"[警告] Parquet 高程目录不存在，跳过：{parquet_dir}")
        return

    if not shp_path.exists():
        print(f"[警告] Shapefile 不存在，跳过：{shp_path}")
        return

    # ================= 读取研究区 =================
    print(f"读取研究区矢量边界：{shp_path}")
    study_area = gpd.read_file(shp_path)

    # <<< PERF >>> 为 prepared geometry 准备 union（判断仍等价于“在任一面内”）
    union_geom = study_area.geometry.unary_union
    prep_wkb = union_geom.wkb

    # ================= 读取 Parquet 文件 ================
    files = list(parquet_dir.glob("*.parquet"))
    print(f"发现 {len(files)} 个 Parquet 高程文件，使用 {NUM_CORES} 核心并行解析...")

    if len(files) == 0:
        print(f"[提示] 区域 {region_name} 无 Parquet 高程文件，跳过。")
        return

    # ================= 并行解析 Parquet → beam 级结构 =================
    with mp.Pool(NUM_CORES) as pool:
        beams_nested = list(tqdm(
            pool.imap(process_single_csv, files, chunksize=IMAP_CHUNKSIZE),
            total=len(files),
            desc=f"解析 Parquet（{region_name}）"
        ))

    beams = [b for sub in beams_nested for b in sub]
    print(f"解析完成：共获得 {len(beams)} 条 beam 轨道")

    if len(beams) == 0:
        print(f"[提示] 区域 {region_name} 无有效 beam，跳过。")
        return

    # ================= 上升 / 下降束分类 =================
    asc_idx  = [i for i, b in enumerate(beams) if b["slope"] < -SLOPE_EPS]
    desc_idx = [i for i, b in enumerate(beams) if b["slope"] >  SLOPE_EPS]
    print(f"轨道分类完成：上升束 {len(asc_idx)}，下降束 {len(desc_idx)}")

    # ================= 构建交叉 / 重复候选任务 =================

    cross_tasks = []
    for i in asc_idx:
        for j in desc_idx:
            bi, bj = beams[i], beams[j]
            if bi["file"] == bj["file"]:
                continue
            if beam_polygons_overlap(bi, bj, buffer_m=10.0):
                cross_tasks.append((i, j))   # <<< 只传 index
            
                
    print(f"交叉轨道候选任务数：{len(cross_tasks)}")

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

    print(f"重复轨道候选：上升 {len(rep_asc_tasks)}，下降 {len(rep_desc_tasks)}")

    # ================== 1. 交叉点对 ==================
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
                desc=f"交叉点对计算（{region_name}）"
            ))
        cross_pairs = [r for sub in cross_chunks for r in sub]
        del cross_chunks
        gc.collect()

    print(f"交叉点对数量：{len(cross_pairs)}")
    if cross_pairs:
        cross_path = out_dir / "optimized_cross_pairs.parquet"
        pd.DataFrame(cross_pairs).to_parquet(
            cross_path,
            engine="pyarrow",
            compression="snappy",
            index=False
        )
        print(f"✔ 已保存交叉点对：{cross_path}")

    del cross_pairs
    gc.collect()

    # ================== 2. 上升重复点对 ==================
    rep_asc_path = out_dir / "optimized_repeat_asc_pairs.parquet"
    if rep_asc_tasks:
        rep_asc_count = write_repeat_pairs_to_parquet(
            rep_asc_tasks,
            rep_asc_path,
            f"上升重复点对（{region_name}）",
            chunksize=BATCH_SIZE_REPEAT,
            merge=True
        )
        print(f"✔ 已保存上升重复点对：{rep_asc_path}，数量：{rep_asc_count}")

    gc.collect()

    # ================== 3. 下降重复点对 ==================
    rep_desc_path = out_dir / "optimized_repeat_desc_pairs.parquet"
    if rep_desc_tasks:
        rep_desc_count = write_repeat_pairs_to_parquet(
            rep_desc_tasks,
            rep_desc_path,
            f"下降重复点对（{region_name}）",
            chunksize=BATCH_SIZE_REPEAT,
            merge=True
        )
        print(f"✔ 已保存下降重复点对：{rep_desc_path}，数量：{rep_desc_count}")

    gc.collect()

    print(
        f"✅ 区域 {region_name} 处理完成 | "
        f"交叉候选={len(cross_tasks)}, "
        f"上升重复候选={len(rep_asc_tasks)}, "
        f"下降重复候选={len(rep_desc_tasks)}"
    )


# ================= 主程序：循环所有区域 =================
from pathlib import Path
import gc
import time

def main():
    base_elev = Path(BASE_ELEV_DIR)
    base_dh   = Path(BASE_DH_DIR)
    base_shp  = Path(BASE_SHP_DIR)

    base_dh.mkdir(parents=True, exist_ok=True)

    # 排除问题区域
    exclude_grids = {
        "Grid_001", "Grid_002", "Grid_003", "Grid_004", "Grid_006", "Grid_007", "Grid_008", "Grid_009","Grid_010", 
        "Grid_011", "Grid_012", "Grid_013", "Grid_014", "Grid_015", "Grid_016", "Grid_017", "Grid_018", "Grid_019", "Grid_020", 
        "Grid_021", "Grid_022", "Grid_023", "Grid_024", "Grid_025", "Grid_026", "Grid_027", "Grid_028", "Grid_029", "Grid_030",
        "Grid_031", "Grid_032", "Grid_033", "Grid_034", "Grid_035", "Grid_036", "Grid_037", "Grid_038", "Grid_039", "Grid_040",
        "Grid_041", "Grid_042", "Grid_043", "Grid_044", "Grid_045", "Grid_046"
    }

    region_folders = sorted(
        p for p in base_elev.iterdir()
        if p.is_dir() and p.name not in exclude_grids
    )

    if not region_folders:
        print(f"[错误] 在 {base_elev} 下没有找到任何区域子文件夹。")
        return

    print("将在以下区域上运行点对检测：")
    for p in region_folders:
        print(" -", p.name)

    # ====================================================
    # 主循环：逐区域处理 + 强制内存清理
    # ====================================================
    for i, region_dir in enumerate(region_folders, start=1):
        region_name = region_dir.name
        print(f"\n▶ [{i}/{len(region_folders)}] 开始处理区域：{region_name}")

        try:
            process_region(region_name)
        except Exception as e:
            print(f"❌ 区域 {region_name} 处理失败：{e}")
            continue

        # ====================================================
        # ⭐ 关键：区域级内存清理（防止后续区域崩溃）
        # ====================================================
        try:
            # Python 垃圾回收
            gc.collect()

            # 如果你使用了 PyArrow / Parquet（强烈推荐）
            try:
                import pyarrow as pa
                pa.default_memory_pool().release_unused()
            except Exception:
                pass

            # 给 OS 一点时间回收内存
            time.sleep(1)

        except Exception as e:
            print(f"⚠ 内存清理阶段出现异常：{e}")

        print(f"✔ 区域 {region_name} 完成并已清理内存")

    print("\n✅ 所选区域处理完成！")


if __name__ == "__main__":
    main()