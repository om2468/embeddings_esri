import argparse
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import psycopg
import rasterio
import rasterio.features
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely import wkb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stepped_workings import analyze_multi_year_health as base

YEAR_START = 2018
YEAR_END = 2025
CHART_DIR = "report_images/tfl_par_land"
REPORT_PATH = "tfl_par_land_health_report.md"
CACHE_DIR = "stepped_workings/cache/tfl_par_land_2018_2025"
WRITEBACK_COLUMNS = {
    "aef_18_25_px_count": "integer",
    "aef_18_25_mean_dist": "double precision",
    "aef_18_25_pct_stable": "double precision",
    "aef_18_25_pct_mild": "double precision",
    "aef_18_25_pct_degraded": "double precision",
    "aef_18_25_hotspot_px": "integer",
    "aef_18_25_hotspot_ha": "double precision",
    "aef_18_25_dom_cluster": "smallint",
    "aef_18_25_interp": "text",
    "aef_18_25_borough": "text",
}

base.CHART_DIR = CHART_DIR
base.REPORT_PATH = REPORT_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse AEF embedding change for TFL/GLA land clipped to TOW.")
    parser.add_argument("--start-year", type=int, default=YEAR_START)
    parser.add_argument("--end-year", type=int, default=YEAR_END)
    parser.add_argument("--skip-writeback", action="store_true", help="Skip updating tfl.tfl_par_land with parcel-level analysis columns.")
    return parser.parse_args()


def load_env_file(env_path):
    env = {}
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def discover_years(start_year, end_year):
    years_files = []
    for year in range(start_year, end_year + 1):
        tif_path = ROOT / f"london_aef_clipped_10m_{year}.tif"
        if tif_path.exists():
            years_files.append((year, str(tif_path)))
    return years_files


def load_land_gdf():
    env = load_env_file(ROOT / ".env")
    conn = psycopg.connect(
        host=env["PG_HOST"],
        port=env["PG_PORT"],
        dbname=env["PG_DATABASE"],
        user=env["PG_USERNAME"],
        password=env["PG_PASSWORD"],
    )
    sql = '''
        SELECT
            "OBJECTID",
            "PAR_ID",
            COALESCE("COMPANY_DESC", 'Unknown') AS company_desc,
            COALESCE("INTEREST_DESC", 'Unknown') AS interest_desc,
            COALESCE("CAT_LABEL", 'Unknown') AS cat_label,
            ST_AsBinary(ST_Transform("geometry", 27700)) AS geom_wkb
        FROM tfl.tfl_par_land
        WHERE "geometry" IS NOT NULL
    '''
    rows = []
    with conn, conn.cursor() as cur:
        cur.execute(sql)
        for object_id, par_id, company_desc, interest_desc, cat_label, geom_wkb in cur.fetchall():
            geom = wkb.loads(bytes(geom_wkb))
            if geom.is_empty:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            rows.append(
                {
                    "OBJECTID": object_id,
                    "PAR_ID": par_id,
                    "company_desc": company_desc,
                    "interest_desc": interest_desc,
                    "cat_label": cat_label,
                    "geometry": geom,
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:27700")


def rasterize_categories(gdf, meta, column):
    values = sorted(gdf[column].dropna().astype(str).unique())
    code_map = {value: idx + 1 for idx, value in enumerate(values)}
    shapes = [
        (geom, code_map[str(value)])
        for geom, value in zip(gdf.geometry, gdf[column])
        if geom is not None and not geom.is_empty
    ]
    grid = rasterio.features.rasterize(
        shapes,
        out_shape=meta["shape"],
        transform=meta["transform"],
        fill=0,
        dtype=np.uint16,
        all_touched=True,
    )
    reverse_map = {code: value for value, code in code_map.items()}
    return grid, reverse_map


def build_land_surfaces(meta):
    print("\nLoading TFL/GLA land parcels from Postgres...")
    land_gdf = load_land_gdf()
    print(f"  Loaded {len(land_gdf):,} land parcels.")

    mask_shapes = [(geom, 1) for geom in land_gdf.geometry if geom is not None and not geom.is_empty]
    land_mask = rasterio.features.rasterize(
        mask_shapes,
        out_shape=meta["shape"],
        transform=meta["transform"],
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    company_grid, company_map = rasterize_categories(land_gdf, meta, "company_desc")
    interest_grid, interest_map = rasterize_categories(land_gdf, meta, "interest_desc")
    category_grid, category_map = rasterize_categories(land_gdf, meta, "cat_label")
    return land_gdf, land_mask, company_grid, company_map, interest_grid, interest_map, category_grid, category_map


def attach_boroughs(land_gdf):
    borough_geojson = ROOT / "stepped_workings/london-boroughs.geojson"
    if not borough_geojson.exists():
        land_gdf["borough_name"] = None
        return land_gdf
    boroughs_gdf = gpd.read_file(borough_geojson)
    if boroughs_gdf.crs is None or boroughs_gdf.crs.to_epsg() != 27700:
        boroughs_gdf = boroughs_gdf.to_crs(epsg=27700)
    joined = gpd.sjoin(
        land_gdf,
        boroughs_gdf[["name", "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined = joined.sort_values(["OBJECTID", "name"], na_position="last")
    joined = joined.drop_duplicates(subset=["OBJECTID"], keep="first")
    return land_gdf.merge(joined[["OBJECTID", "name"]].rename(columns={"name": "borough_name"}), on="OBJECTID", how="left")


def rasterize_object_ids(land_gdf, meta):
    shapes = [
        (geom, int(object_id))
        for geom, object_id in zip(land_gdf.geometry, land_gdf["OBJECTID"])
        if geom is not None and not geom.is_empty
    ]
    return rasterio.features.rasterize(
        shapes,
        out_shape=meta["shape"],
        transform=meta["transform"],
        fill=0,
        dtype=np.int32,
        all_touched=True,
    )


def save_land_mask(meta, land_mask):
    grid = np.where(land_mask == 1, 1, 0).astype(np.uint8)
    base.save_geotiff(grid, "tfl_par_land_mask.tif", meta, "uint8", nodata=0)


def compute_valid_mask(years_files, land_mask):
    mask = None
    meta = None
    for _, tif_path in years_files:
        with rasterio.open(tif_path) as src:
            if meta is None:
                meta = {
                    "bounds": src.bounds,
                    "shape": src.shape,
                    "transform": src.transform,
                    "crs": src.crs,
                }
            year_mask = (src.read(1) != -128) & (land_mask == 1)
            mask = year_mask if mask is None else (mask & year_mask)
    vy, vx = np.where(mask)
    return vy, vx, meta


def serialize_meta(meta):
    return {
        "bounds": [meta["bounds"].left, meta["bounds"].bottom, meta["bounds"].right, meta["bounds"].top],
        "shape": list(meta["shape"]),
        "transform": list(meta["transform"]),
        "crs": meta["crs"].to_string() if hasattr(meta["crs"], "to_string") else str(meta["crs"]),
    }


def deserialize_meta(meta_json):
    return {
        "bounds": BoundingBox(*meta_json["bounds"]),
        "shape": tuple(meta_json["shape"]),
        "transform": Affine(*meta_json["transform"]),
        "crs": CRS.from_string(meta_json["crs"]),
    }


def load_cache(years):
    cache_dir = ROOT / CACHE_DIR
    required = ["dist_matrix.npy", "vy.npy", "vx.npy", "meta.json", "results.json", "borough_stats.json"]
    if not all((cache_dir / name).exists() for name in required):
        return None
    with open(cache_dir / "meta.json", "r", encoding="utf-8") as handle:
        meta = deserialize_meta(json.load(handle))
    vy = np.load(cache_dir / "vy.npy")
    vx = np.load(cache_dir / "vx.npy")
    dist_matrix = np.load(cache_dir / "dist_matrix.npy")
    with open(cache_dir / "results.json", "r", encoding="utf-8") as handle:
        results_serialized = json.load(handle)
    results = {}
    for index, year in enumerate(years):
        result = results_serialized[str(year)].copy()
        result["raw_bl"] = dist_matrix[:, index]
        results[year] = result
    with open(cache_dir / "borough_stats.json", "r", encoding="utf-8") as handle:
        borough_stats = json.load(handle)
    return vy, vx, meta, dist_matrix, results, borough_stats


def save_cache(vy, vx, meta, dist_matrix, results, borough_stats):
    cache_dir = ROOT / CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "vy.npy", vy)
    np.save(cache_dir / "vx.npy", vx)
    np.save(cache_dir / "dist_matrix.npy", dist_matrix)
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(serialize_meta(meta), handle)
    serializable_results = {}
    for year, result in results.items():
        serializable_results[str(year)] = {key: value for key, value in result.items() if key != "raw_bl"}
    with open(cache_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(serializable_results, handle)
    with open(cache_dir / "borough_stats.json", "w", encoding="utf-8") as handle:
        json.dump(borough_stats, handle)


def module_1_trajectory_clustering(years, dist_matrix, vy, vx, meta):
    print("\n── Module 1: Temporal Trajectory Clustering ──")
    n_years = dist_matrix.shape[1]
    if n_years < 4:
        print("  ⚠ Fewer than 4 years available — skipping trajectory clustering.")
        return None, {}

    cache_dir = ROOT / CACHE_DIR
    labels_path = cache_dir / "cluster_labels.npy"
    stats_path = cache_dir / "cluster_stats.json"
    centroids_path = cache_dir / "cluster_centroids.npy"
    palette = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c", "#8e44ad"]
    archetype_names = [
        "Stable / Resilient",
        "Minor Variation",
        "Gradual Decline",
        "Drought Stress & Recovery",
        "Sudden / Significant Loss",
        "Extreme Change",
    ]

    if labels_path.exists() and stats_path.exists() and centroids_path.exists():
        labels = np.load(labels_path)
        centroids = np.load(centroids_path)
        with open(stats_path, "r", encoding="utf-8") as handle:
            cached_data = json.load(handle)
        best_k = cached_data["best_k"]
        stats = cached_data["stats"]
    else:
        mn = dist_matrix.min(axis=1, keepdims=True)
        mx = dist_matrix.max(axis=1, keepdims=True)
        rng = mx - mn
        rng[rng == 0] = 1.0
        normed = (dist_matrix - mn) / rng

        print("  Running elbow analysis (k=2..8)...")
        inertias = []
        k_range = range(2, min(9, n_years + 1))
        for k in k_range:
            km = base.KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
            km.fit(normed)
            inertias.append(km.inertia_)

        if len(inertias) >= 3:
            diffs2 = np.diff(np.diff(inertias))
            best_idx = int(np.argmax(diffs2)) + 2
            best_k = list(k_range)[best_idx]
        else:
            best_k = min(5, max(k_range))
        best_k = max(3, min(best_k, 6))
        print(f"  Selected k={best_k}")

        fig, ax = base.plt.subplots(figsize=(8, 5))
        ax.plot(list(k_range), inertias, "o-", color="steelblue", linewidth=2)
        ax.axvline(best_k, color="crimson", linestyle="--", linewidth=1.5, label=f"Selected k={best_k}")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("Trajectory Clustering — Elbow Method")
        ax.legend()
        base.save_chart(fig, "trajectory_elbow.png")

        km = base.KMeans(n_clusters=best_k, n_init=10, max_iter=300, random_state=42)
        labels = km.fit_predict(normed)
        centroids = km.cluster_centers_
        order = np.argsort(centroids[:, -1])
        label_map = {old: new for new, old in enumerate(order)}
        labels = np.array([label_map[label] for label in labels])
        centroids = centroids[order]
        counts = np.bincount(labels, minlength=best_k)
        pcts = counts / counts.sum() * 100
        stats = {
            archetype_names[idx]: {"count": int(counts[idx]), "pct": float(pcts[idx])}
            for idx in range(best_k)
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(labels_path, labels.astype(np.uint8))
        np.save(centroids_path, centroids)
        with open(stats_path, "w", encoding="utf-8") as handle:
            json.dump({"best_k": best_k, "stats": stats}, handle)

    fig, ax = base.plt.subplots(figsize=(10, 6))
    for idx in range(len(centroids)):
        ax.plot(years, centroids[idx], "o-", color=palette[idx], linewidth=2.5, label=f"Cluster {idx + 1}: {archetype_names[idx]}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalised Cosine Distance from Baseline")
    ax.set_title("Trajectory Cluster Centroids")
    ax.set_xticks(years)
    ax.legend(fontsize=9, loc="upper left")
    base.save_chart(fig, "trajectory_cluster_centroids.png")

    grid = np.full(meta["shape"], np.nan, dtype=np.float32)
    grid[vy, vx] = labels.astype(np.float32)
    cmap = base.ListedColormap(palette[: len(centroids)])
    norm = base.BoundaryNorm(np.arange(-0.5, len(centroids) + 0.5, 1), cmap.N)
    fig, ax = base.plt.subplots(figsize=(12, 10))
    im = ax.imshow(grid, cmap=cmap, norm=norm, extent=meta["bounds"], origin="upper")
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(centroids)), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([f"{idx + 1}: {archetype_names[idx]}" for idx in range(len(centroids))], fontsize=9)
    ax.set_title("Spatial Distribution of Canopy Trajectory Archetypes")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    base.save_chart(fig, "trajectory_cluster_map.png")

    cluster_grid = np.full(meta["shape"], 255, dtype=np.uint8)
    cluster_grid[vy, vx] = labels.astype(np.uint8)
    base.save_geotiff(cluster_grid, "canopy_trajectory_clusters.tif", meta, "uint8", nodata=255)

    counts = np.bincount(labels, minlength=len(centroids))
    fig, ax = base.plt.subplots(figsize=(8, 8))
    ax.pie(counts, labels=archetype_names[: len(centroids)], colors=palette[: len(centroids)], autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title("Canopy Pixel Distribution by Trajectory Archetype")
    base.save_chart(fig, "trajectory_cluster_pie.png")
    return labels, stats


def summarise_category(dist_bl, category_values, code_to_name, min_pixels=10):
    stats = {}
    for code, name in code_to_name.items():
        mask = category_values == code
        if int(mask.sum()) < min_pixels:
            continue
        values = dist_bl[mask]
        stats[name] = {
            "count": int(mask.sum()),
            "mean_dist": float(np.mean(values)),
            "pct_stable": float(np.mean(values < 0.05) * 100),
            "pct_mild": float(np.mean((values >= 0.05) & (values < 0.15)) * 100),
            "pct_degraded": float(np.mean(values >= 0.15) * 100),
        }
    return stats


def plot_category_stats(stats, filename, title, top_n=10):
    if not stats:
        return
    names = sorted(stats.keys(), key=lambda name: stats[name]["pct_degraded"], reverse=True)[:top_n]
    fig, ax = base.plt.subplots(figsize=(12, max(5, 0.6 * len(names))))
    x = np.arange(len(names))
    bar_w = 0.25
    stable = [stats[name]["pct_stable"] for name in names]
    mild = [stats[name]["pct_mild"] for name in names]
    degraded = [stats[name]["pct_degraded"] for name in names]
    ax.bar(x - bar_w, stable, bar_w, color="#2ecc71", edgecolor="black", alpha=0.85, label="Stable")
    ax.bar(x, mild, bar_w, color="#f1c40f", edgecolor="black", alpha=0.85, label="Mild Stress")
    ax.bar(x + bar_w, degraded, bar_w, color="#e74c3c", edgecolor="black", alpha=0.85, label="Degraded")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of Canopy Pixels")
    ax.set_title(title)
    ax.legend()
    base.save_chart(fig, filename)


def module_4_land_vulnerability(dist_bl, vy, vx, company_grid, company_map, interest_grid, interest_map, category_grid, category_map):
    print("\n── Module 4: TFL/GLA Land Attribute Analysis ──")
    company_stats = summarise_category(dist_bl, company_grid[vy, vx], company_map)
    interest_stats = summarise_category(dist_bl, interest_grid[vy, vx], interest_map)
    category_stats = summarise_category(dist_bl, category_grid[vy, vx], category_map)
    plot_category_stats(company_stats, "vulnerability_by_company.png", "Canopy Vulnerability by Owning Organisation")
    plot_category_stats(interest_stats, "vulnerability_by_interest.png", "Canopy Vulnerability by Land Interest", top_n=6)
    plot_category_stats(category_stats, "vulnerability_by_category.png", "Canopy Vulnerability by Company and Interest Class")
    return {
        "by_company": company_stats,
        "by_interest": interest_stats,
        "by_category": category_stats,
    }


def compute_hotspot_mask(dist_bl, vy, vx, meta):
    grid = np.full(meta["shape"], np.nan, dtype=np.float32)
    grid[vy, vx] = dist_bl
    fill = np.where(np.isnan(grid), 0.0, grid)
    count = np.where(np.isnan(grid), 0.0, 1.0)
    smooth_sum = base.gaussian_filter(fill, sigma=3)
    smooth_cnt = base.gaussian_filter(count, sigma=3)
    smooth_cnt[smooth_cnt == 0] = 1.0
    smoothed = smooth_sum / smooth_cnt
    smoothed[np.isnan(grid)] = np.nan
    vals = smoothed[~np.isnan(smoothed)]
    mu, sigma = vals.mean(), vals.std()
    if sigma == 0:
        sigma = 1.0
    zscore = (smoothed - mu) / sigma
    return zscore > 2.0


def compute_parcel_level_stats(land_gdf, objectid_grid, vy, vx, dist_bl_latest, cluster_labels, hotspot_mask):
    print("\nComputing parcel-level write-back metrics...")
    object_ids = objectid_grid[vy, vx]
    valid = object_ids > 0
    object_ids = object_ids[valid].astype(np.int64)
    distances = dist_bl_latest[valid]
    hotspots = hotspot_mask[vy, vx][valid].astype(np.int64)

    max_object_id = int(object_ids.max()) if len(object_ids) else 0
    pixel_count = np.bincount(object_ids, minlength=max_object_id + 1)
    sum_distance = np.bincount(object_ids, weights=distances, minlength=max_object_id + 1)
    stable_count = np.bincount(object_ids, weights=(distances < 0.05).astype(np.int64), minlength=max_object_id + 1)
    mild_count = np.bincount(object_ids, weights=((distances >= 0.05) & (distances < 0.15)).astype(np.int64), minlength=max_object_id + 1)
    degraded_count = np.bincount(object_ids, weights=(distances >= 0.15).astype(np.int64), minlength=max_object_id + 1)
    hotspot_count = np.bincount(object_ids, weights=hotspots, minlength=max_object_id + 1)

    cluster_lookup = {}
    if cluster_labels is not None:
        clusters = cluster_labels[valid].astype(np.int64) + 1
        unique_pairs = np.stack([object_ids, clusters], axis=1)
        for object_id in np.unique(unique_pairs[:, 0]):
            cluster_values = unique_pairs[unique_pairs[:, 0] == object_id, 1]
            counts = np.bincount(cluster_values)
            dominant = int(np.argmax(counts[1:]) + 1) if len(counts) > 1 else None
            cluster_lookup[int(object_id)] = dominant

    all_object_ids = set(int(v) for v in land_gdf["OBJECTID"].tolist())
    borough_lookup = {
        int(row.OBJECTID): (None if row.borough_name is None or (isinstance(row.borough_name, float) and np.isnan(row.borough_name)) else str(row.borough_name))
        for row in land_gdf[["OBJECTID", "borough_name"]].itertuples(index=False)
    }
    stats = []
    for object_id in sorted(all_object_ids):
        count = int(pixel_count[object_id]) if object_id <= max_object_id else 0
        if count == 0:
            stats.append((object_id, 0, None, None, None, None, 0, 0.0, None, None, borough_lookup.get(object_id)))
            continue
        mean_dist = float(sum_distance[object_id] / count)
        pct_stable = float(stable_count[object_id] / count * 100)
        pct_mild = float(mild_count[object_id] / count * 100)
        pct_degraded = float(degraded_count[object_id] / count * 100)
        hot_px = int(hotspot_count[object_id])
        hot_ha = float(hot_px * 100 / 10000)
        if pct_degraded >= 50.0:
            interpretation = "degraded"
        elif pct_mild + pct_degraded >= 50.0:
            interpretation = "mild stress"
        else:
            interpretation = "stable"
        stats.append(
            (
                object_id,
                count,
                mean_dist,
                pct_stable,
                pct_mild,
                pct_degraded,
                hot_px,
                hot_ha,
                cluster_lookup.get(object_id),
                interpretation,
                borough_lookup.get(object_id),
            )
        )
    print(f"  Prepared parcel metrics for {len(stats):,} rows.")
    return stats


def connect_postgres():
    env = load_env_file(ROOT / ".env")
    return psycopg.connect(
        host=env["PG_HOST"],
        port=env["PG_PORT"],
        dbname=env["PG_DATABASE"],
        user=env["PG_USERNAME"],
        password=env["PG_PASSWORD"],
    )


def writeback_parcel_stats(parcel_stats):
    print("\nWriting parcel-level metrics back to Postgres...")
    with connect_postgres() as conn, conn.cursor() as cur:
        for column_name, data_type in WRITEBACK_COLUMNS.items():
            cur.execute(f'ALTER TABLE tfl.tfl_par_land ADD COLUMN IF NOT EXISTS "{column_name}" {data_type}')

        cur.execute(
            '''
            UPDATE tfl.tfl_par_land
            SET
                "aef_18_25_px_count" = 0,
                "aef_18_25_mean_dist" = NULL,
                "aef_18_25_pct_stable" = NULL,
                "aef_18_25_pct_mild" = NULL,
                "aef_18_25_pct_degraded" = NULL,
                "aef_18_25_hotspot_px" = 0,
                "aef_18_25_hotspot_ha" = 0,
                "aef_18_25_dom_cluster" = NULL,
                "aef_18_25_interp" = NULL,
                "aef_18_25_borough" = NULL
            '''
        )

        cur.executemany(
            '''
            UPDATE tfl.tfl_par_land
            SET
                "aef_18_25_px_count" = %s,
                "aef_18_25_mean_dist" = %s,
                "aef_18_25_pct_stable" = %s,
                "aef_18_25_pct_mild" = %s,
                "aef_18_25_pct_degraded" = %s,
                "aef_18_25_hotspot_px" = %s,
                "aef_18_25_hotspot_ha" = %s,
                "aef_18_25_dom_cluster" = %s,
                "aef_18_25_interp" = %s,
                "aef_18_25_borough" = %s
            WHERE "OBJECTID" = %s
            ''',
            [
                (count, mean_dist, pct_stable, pct_mild, pct_degraded, hot_px, hot_ha, dom_cluster, interpretation, borough_name, object_id)
                for object_id, count, mean_dist, pct_stable, pct_mild, pct_degraded, hot_px, hot_ha, dom_cluster, interpretation, borough_name in parcel_stats
            ],
        )
        conn.commit()
    print("  ✓ Postgres table updated.")


def write_report(years, total_valid, results, cluster_stats, pc_info, hotspot_stats, vuln_stats, borough_stats, parcel_count, baseline_year, latest_year):
    print(f"\n── Writing report to {REPORT_PATH} ──")
    trend_rows = []
    for year in years:
        result = results[year]
        yoy = "N/A" if year == baseline_year else f"{result['mean_yoy']:.4f}"
        trend_rows.append(
            f"| {year} | {result['mean_bl']:.4f} | {result['p90_bl']:.4f} | {yoy} | {result['pct_stable']:.1f}% | {result['pct_mild']:.1f}% | {result['pct_degraded']:.1f}% |"
        )
    trend_table = "\n".join(trend_rows)

    cluster_table = ""
    if cluster_stats:
        cluster_table = "\n".join(
            f"| {name} | {stats['count']:,} | {stats['pct']:.1f}% |"
            for name, stats in cluster_stats.items()
        )

    pca_table = ""
    if pc_info:
        pca_table = "\n".join(
            f"| PC{pc['pc']} | {pc['variance_pct']:.1f}% | {', '.join(str(dim) for dim in pc['top_dims'])} | {pc['corr_with_distance']:.3f} |"
            for pc in pc_info
        )

    def stats_table(stats):
        if not stats:
            return ""
        rows = []
        for name in sorted(stats.keys(), key=lambda key: stats[key]["pct_degraded"], reverse=True):
            item = stats[name]
            rows.append(
                f"| {name} | {item['count']:,} | {item['mean_dist']:.4f} | {item['pct_stable']:.1f}% | {item['pct_mild']:.1f}% | {item['pct_degraded']:.1f}% |"
            )
        return "\n".join(rows)

    company_table = stats_table(vuln_stats.get("by_company", {}))
    interest_table = stats_table(vuln_stats.get("by_interest", {}))
    category_table = stats_table(vuln_stats.get("by_category", {}))

    borough_table = ""
    if borough_stats:
        borough_table = "\n".join(
            f"| {name} | {stats['count']:,} | {stats['mean_dist']:.4f} | {stats['pct_stable']:.1f}% | {stats['pct_mild']:.1f}% | {stats['pct_degraded']:.1f}% |"
            for name, stats in sorted(borough_stats.items(), key=lambda item: item[1]["pct_degraded"], reverse=True)
        )

    hs = hotspot_stats or {}
    overall_stable = results[latest_year]["pct_stable"]
    overall_mild = results[latest_year]["pct_mild"]
    overall_degraded = results[latest_year]["pct_degraded"]
    area_ha = total_valid * 100 / 10000

    report = f"""# TFL/GLA Land within TOW: Advanced Canopy Health Analysis ({baseline_year}–{latest_year})

## Executive Summary

This report analyses **TFL / GLA land parcels intersecting the existing TOW mask** using Google DeepMind **Alpha Earth Foundations** 64-band embeddings at 10m resolution. The study covers **{len(years)} annual observations** from **{baseline_year}** to **{latest_year}**, over **{parcel_count:,} land parcels** and **{total_valid:,} valid canopy pixels** (~**{area_ha:,.0f} hectares**) where TFL/GLA land and TOW overlap.

By {latest_year}, **{overall_stable:.1f}%** of valid canopy pixels remained stable against the {baseline_year} baseline, **{overall_mild:.1f}%** showed mild stress or thinning, and **{overall_degraded:.1f}%** showed significant change or loss. Hotspot analysis isolates **{hs.get('n_hotspot', 0):,} pixels** (~**{hs.get('ha_hotspot', 0):.0f} hectares**) as statistically significant degradation clusters.

---

## 1. Data and Method

- Input rasters: existing London-wide **TOW-clipped** AEF GeoTIFFs for {baseline_year}–{latest_year}
- Land geometry source: Postgres table **tfl.tfl_par_land** reprojected from EPSG:4326 to EPSG:27700
- Analysis mask: intersection of valid TOW-clipped pixels with rasterised TFL/GLA parcel land
- Change metric: cosine distance between yearly dequantized embedding vectors and the {baseline_year} baseline

### Years Analysed
{', '.join(str(year) for year in years)}

---

## 2. Multi-Year Trend Analysis

| Year | Mean Dist. from Baseline | 90th Pct. | YoY Mean Dist. | Stable (< 0.05) | Mild (0.05–0.15) | Significant (≥ 0.15) |
|---|---|---|---|---|---|---|
{trend_table}

![Divergence Trend]({CHART_DIR}/tree_health_trend_line.png)

![Categories]({CHART_DIR}/tree_health_categories_bar.png)

![YoY]({CHART_DIR}/tree_health_yoy_trend.png)

![Spatial Maps]({CHART_DIR}/tree_health_multi_panel_map.png)

---

## 3. Trajectory Clustering
"""

    if cluster_table:
        report += f"""
| Archetype | Pixel Count | % of Canopy |
|---|---|---|
{cluster_table}

![Centroids]({CHART_DIR}/trajectory_cluster_centroids.png)

![Cluster Map]({CHART_DIR}/trajectory_cluster_map.png)

![Cluster Pie]({CHART_DIR}/trajectory_cluster_pie.png)

![Elbow]({CHART_DIR}/trajectory_elbow.png)
"""
    else:
        report += "\nTrajectory clustering was skipped because fewer than 4 annual observations were available.\n"

    report += """
---

## 4. Directional PCA
"""

    if pca_table:
        report += f"""
| Component | Variance Explained | Top Embedding Dimensions | Correlation with Cosine Distance |
|---|---|---|---|
{pca_table}

![Variance]({CHART_DIR}/pca_explained_variance.png)

![PC Maps]({CHART_DIR}/pca_spatial_maps.png)

![Biplot]({CHART_DIR}/pca_biplot.png)
"""

    report += f"""
---

## 5. Spatial Hotspots

- Degradation hotspots (Z > 2.0): **{hs.get('n_hotspot', 0):,} pixels** (~{hs.get('ha_hotspot', 0):.0f} ha)
- Resilience coldspots (Z < -1.0): **{hs.get('n_coldspot', 0):,} pixels** (~{hs.get('ha_coldspot', 0):.0f} ha)
- Background pixels: **{hs.get('n_background', 0):,}**

![Hotspot Map]({CHART_DIR}/hotspot_map.png)

![Z Histogram]({CHART_DIR}/hotspot_histogram.png)

---

## 6. Land Attribute Vulnerability
"""

    if company_table:
        report += f"""
### By Owning Organisation

| Organisation | Pixel Count | Mean Cosine Dist. | Stable | Mild Stress | Degraded |
|---|---|---|---|---|---|
{company_table}

![By Company]({CHART_DIR}/vulnerability_by_company.png)
"""

    if interest_table:
        report += f"""
### By Land Interest

| Interest | Pixel Count | Mean Cosine Dist. | Stable | Mild Stress | Degraded |
|---|---|---|---|---|---|
{interest_table}

![By Interest]({CHART_DIR}/vulnerability_by_interest.png)
"""

    if category_table:
        report += f"""
### By Company and Interest Class

| Category | Pixel Count | Mean Cosine Dist. | Stable | Mild Stress | Degraded |
|---|---|---|---|---|---|
{category_table}

![By Category]({CHART_DIR}/vulnerability_by_category.png)
"""

    if borough_table:
        report += f"""
---

## 7. Borough-Level Distribution

| Borough | Pixel Count | Mean Cosine Dist. | Stable | Mild Stress | Degraded |
|---|---|---|---|---|---|
{borough_table}

![Borough Performance]({CHART_DIR}/vulnerability_by_borough.png)
"""

    report += """
---

## 8. Conclusions

1. The analysis isolates the subset of London canopy pixels that are both within the precomputed TOW footprint and inside TFL/GLA land parcels.
2. The time-series quantifies where those estate lands remained stable, where gradual divergence accumulated, and where abrupt loss signatures emerged.
3. The organisation and tenure splits show whether canopy stress is concentrated in particular ownership or interest classes rather than being evenly distributed across the estate.
4. Hotspot and borough outputs provide a spatial triage layer for follow-up inspection, maintenance planning, or estate-level intervention prioritisation.
"""

    with open(ROOT / REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"  ✓ Report written to {REPORT_PATH}")


def main():
    args = parse_args()
    years_files = discover_years(args.start_year, args.end_year)
    if len(years_files) < 2:
        raise SystemExit(f"Need at least 2 yearly rasters between {args.start_year} and {args.end_year}.")

    years = [year for year, _ in years_files]
    baseline_year = years[0]
    latest_year = years[-1]
    print("=" * 70)
    print("  TFL/GLA Land Canopy Health Analysis")
    print("=" * 70)
    print(f"Years available: {years}")
    print(f"Baseline: {baseline_year}, Latest: {latest_year}")

    with rasterio.open(years_files[0][1]) as src:
        base_meta = {"bounds": src.bounds, "shape": src.shape, "transform": src.transform, "crs": src.crs}

    land_gdf, land_mask, company_grid, company_map, interest_grid, interest_map, category_grid, category_map = build_land_surfaces(base_meta)
    land_gdf = attach_boroughs(land_gdf)
    objectid_grid = rasterize_object_ids(land_gdf, base_meta)
    save_land_mask(base_meta, land_mask)

    cache_payload = load_cache(years)
    if cache_payload is None:
        print("\nComputing valid TFL/TOW pixel intersection...")
        vy, vx, meta = compute_valid_mask(years_files, land_mask)
        if len(vy) == 0:
            raise SystemExit("No overlapping valid pixels found between TOW rasters and TFL/GLA land parcels.")
        print(f"Valid canopy pixels: {len(vy):,}")

        print(f"\nLoading baseline embeddings ({baseline_year})...")
        emb_baseline = base.load_dequantized(years_files[0][1], vy, vx)
        print(f"Loading latest embeddings ({latest_year})...")
        emb_latest = base.load_dequantized(years_files[-1][1], vy, vx)

        results = {}
        dist_matrix_cols = []
        prev_emb = emb_baseline
        for year, tif_path in years_files:
            if year == baseline_year:
                emb = emb_baseline
                d_bl = np.zeros(len(vy), dtype=np.float32)
                d_yoy = np.zeros(len(vy), dtype=np.float32)
            elif year == latest_year:
                emb = emb_latest
                d_bl = base.cosine_distance(emb_baseline, emb)
                d_yoy = base.cosine_distance(prev_emb, emb)
            else:
                emb = base.load_dequantized(tif_path, vy, vx)
                d_bl = base.cosine_distance(emb_baseline, emb)
                d_yoy = base.cosine_distance(prev_emb, emb)

            results[year] = {
                "mean_bl": float(np.mean(d_bl)),
                "p50_bl": float(np.percentile(d_bl, 50)),
                "p90_bl": float(np.percentile(d_bl, 90)),
                "p95_bl": float(np.percentile(d_bl, 95)),
                "mean_yoy": float(np.mean(d_yoy)),
                "pct_stable": float(np.mean(d_bl < 0.05) * 100),
                "pct_mild": float(np.mean((d_bl >= 0.05) & (d_bl < 0.15)) * 100),
                "pct_degraded": float(np.mean(d_bl >= 0.15) * 100),
                "raw_bl": d_bl,
            }
            dist_matrix_cols.append(d_bl)
            prev_emb = emb
            print(
                f"  {year}: mean={results[year]['mean_bl']:.4f}, stable={results[year]['pct_stable']:.1f}%, mild={results[year]['pct_mild']:.1f}%, degraded={results[year]['pct_degraded']:.1f}%"
            )

        dist_matrix = np.column_stack(dist_matrix_cols)
        borough_stats = {}
        save_cache(vy, vx, meta, dist_matrix, results, borough_stats)
    else:
        vy, vx, meta, dist_matrix, results, borough_stats = cache_payload
        print(f"\nLoaded cached distance surfaces for {len(vy):,} valid pixels.")
        print(f"Loading baseline embeddings ({baseline_year}) for PCA...")
        emb_baseline = base.load_dequantized(years_files[0][1], vy, vx)
        print(f"Loading latest embeddings ({latest_year}) for PCA...")
        emb_latest = base.load_dequantized(years_files[-1][1], vy, vx)

    base.generate_trend_charts(years, results, meta, vy, vx, baseline_year)
    cluster_labels, cluster_stats = module_1_trajectory_clustering(years, dist_matrix, vy, vx, meta)

    dist_bl_latest = results[latest_year]["raw_bl"]
    change_grid = np.full(meta["shape"], np.nan, dtype=np.float32)
    change_grid[vy, vx] = dist_bl_latest
    base.save_geotiff(change_grid, f"canopy_change_{baseline_year}_{latest_year}.tif", meta, "float32", nodata=-9999.0)

    pc_info = base.module_2_pca(emb_baseline, emb_latest, dist_bl_latest, vy, vx, meta, baseline_year, latest_year)
    del emb_baseline, emb_latest

    hotspot_mask = compute_hotspot_mask(dist_bl_latest, vy, vx, meta)
    hotspot_stats = base.module_3_hotspots(dist_bl_latest, vy, vx, meta)
    vuln_stats = module_4_land_vulnerability(dist_bl_latest, vy, vx, company_grid, company_map, interest_grid, interest_map, category_grid, category_map)
    borough_stats = base.module_5_boroughs(dist_bl_latest, vy, vx, meta)
    save_cache(vy, vx, meta, dist_matrix, results, borough_stats)
    if not args.skip_writeback:
        parcel_stats = compute_parcel_level_stats(land_gdf, objectid_grid, vy, vx, dist_bl_latest, cluster_labels, hotspot_mask)
        writeback_parcel_stats(parcel_stats)
    write_report(years, len(vy), results, cluster_stats, pc_info, hotspot_stats, vuln_stats, borough_stats, len(land_gdf), baseline_year, latest_year)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()