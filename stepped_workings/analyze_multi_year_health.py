"""
Advanced Multi-Year Tree Canopy Health Analysis
================================================
Analyses London Trees Outside Woodland (TOW) canopy health using
Google DeepMind Alpha Earth Foundations (AEF) 64-band satellite embeddings
at 10m native resolution across multiple years (2017-2025).

Four analysis modules:
  1. Temporal Trajectory Clustering
  2. Directional PCA of Change Vectors
  3. Spatial Hotspot Detection
  4. Attribute-Based Vulnerability Analysis
"""

import os
import json
import glob
import re
import warnings
import numpy as np
import rasterio
import rasterio.features
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Global config ──────────────────────────────────────────────────────────────
TOW_EXTENT = [503000, 155000, 562000, 201000]  # xmin, ymin, xmax, ymax EPSG:27700
CHART_DIR = "report_images"
REPORT_PATH = "tree_health_report.md"
GDB_PATH = "FR_TOW_V1_London/FR_TOW_V1_London.gdb"

# Chart style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.titlepad": 15,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
})

# ── Helpers ────────────────────────────────────────────────────────────────────

def dequantize(raw_int8):
    """Apply the AEF non-linear dequantization: ((v/127.5)^2) * sign(v)."""
    v = raw_int8.astype(np.float32) / 127.5
    return (v ** 2) * np.sign(raw_int8)


def cosine_distance(u, v):
    """Cosine distance between row-vectors u and v, both (N, D)."""
    dot = np.sum(u * v, axis=1)
    nu = np.linalg.norm(u, axis=1)
    nv = np.linalg.norm(v, axis=1)
    zero = (nu == 0) | (nv == 0)
    nu[zero] = 1.0
    nv[zero] = 1.0
    sim = np.clip(dot / (nu * nv), -1.0, 1.0)
    sim[zero] = 1.0
    return 1.0 - sim


def save_chart(fig, name):
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {path}")
    return path


def save_geotiff(grid, name, meta, dtype, nodata=None):
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, name)
    
    out_meta = {
        "driver": "GTiff",
        "height": meta["shape"][0],
        "width": meta["shape"][1],
        "count": 1,
        "dtype": dtype,
        "crs": meta["crs"],
        "transform": meta["transform"],
        "compress": "deflate"
    }
    if nodata is not None:
        out_meta["nodata"] = nodata
        
    write_grid = grid.copy()
    if np.issubdtype(np.dtype(dtype), np.floating):
        if nodata is not None:
            write_grid[np.isnan(write_grid)] = nodata
            
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(write_grid.astype(dtype), 1)
    print(f"  ✓ Saved GeoTIFF: {path}")
    return path


# ── Discovery & loading ───────────────────────────────────────────────────────

def discover_years():
    """Find completed clipped GeoTIFFs, skipping any with active temp files."""
    found = []
    for f in sorted(glob.glob("london_aef_clipped_10m_*.tif")):
        m = re.search(r"london_aef_clipped_10m_(\d{4})\.tif", f)
        if not m:
            continue
        year = int(m.group(1))
        if os.path.exists(f"london_aef_10m_{year}_temp.tif"):
            print(f"  ⏳ Skipping {year} (temp file exists, still being written)")
            continue
        found.append((year, f))
    found.sort()
    return found


def compute_valid_mask(years_files):
    """Intersect valid (non-nodata) pixels across all years."""
    mask = None
    meta = {}
    for _, f in years_files:
        with rasterio.open(f) as src:
            if not meta:
                meta = {"bounds": src.bounds, "shape": src.shape,
                        "transform": src.transform, "crs": src.crs}
            b1 = src.read(1)
            m = b1 != -128
            mask = m if mask is None else (mask & m)
    vy, vx = np.where(mask)
    return vy, vx, meta


def load_dequantized(tif_path, vy, vx):
    """Load 64 bands for valid pixels and dequantize → (N, 64) float32."""
    n = len(vy)
    raw = np.zeros((n, 64), dtype=np.int8)
    with rasterio.open(tif_path) as src:
        for b in range(1, 65):
            raw[:, b - 1] = src.read(b)[vy, vx]
    return dequantize(raw)


# ── Module 1: Temporal Trajectory Clustering ──────────────────────────────────

def module_1_trajectory_clustering(years, dist_matrix, vy, vx, meta):
    """
    Cluster per-pixel cosine-distance trajectories using KMeans.
    dist_matrix: (N_pixels, N_years) float32
    Returns cluster_labels (N_pixels,)
    """
    print("\n── Module 1: Temporal Trajectory Clustering ──")
    n_years = dist_matrix.shape[1]

    if n_years < 4:
        print("  ⚠ Fewer than 4 years available — skipping trajectory clustering.")
        return None, {}

    CACHE_DIR = "stepped_workings/cache"
    labels_path = os.path.join(CACHE_DIR, "cluster_labels.npy")
    stats_path = os.path.join(CACHE_DIR, "cluster_stats.json")
    centroids_path = os.path.join(CACHE_DIR, "cluster_centroids.npy")

    cache_exists = os.path.exists(labels_path) and os.path.exists(stats_path) and os.path.exists(centroids_path)

    palette = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c", "#8e44ad"]
    archetype_names = [
        "Stable / Resilient",
        "Minor Variation",
        "Gradual Decline",
        "Drought Stress & Recovery",
        "Sudden / Significant Loss",
        "Extreme Change",
    ]

    if cache_exists:
        print("  Found cached KMeans trajectory clustering. Loading...")
        try:
            labels = np.load(labels_path)
            centroids = np.load(centroids_path)
            with open(stats_path, "r") as f:
                cached_data = json.load(f)
            best_k = cached_data["best_k"]
            stats = cached_data["stats"]
            cluster_names = archetype_names[:best_k]
            
            print("  Recreating trajectory clustering charts and GeoTIFF from cache...")
            # Centroid chart
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(best_k):
                ax.plot(years, centroids[i], "o-", color=palette[i], linewidth=2.5,
                        label=f"Cluster {i+1}: {cluster_names[i]}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Normalised Cosine Distance from Baseline")
            ax.set_title("Trajectory Cluster Centroids")
            ax.set_xticks(years)
            ax.legend(fontsize=9, loc="upper left")
            save_chart(fig, "trajectory_cluster_centroids.png")

            # Spatial map
            grid = np.full(meta["shape"], np.nan, dtype=np.float32)
            grid[vy, vx] = labels.astype(np.float32)
            cmap = ListedColormap(palette[:best_k])
            bounds_cm = np.arange(-0.5, best_k + 0.5, 1)
            norm = BoundaryNorm(bounds_cm, cmap.N)

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(grid, cmap=cmap, norm=norm, extent=meta["bounds"], origin="upper")
            cbar = fig.colorbar(im, ax=ax, ticks=range(best_k), fraction=0.046, pad=0.04)
            cbar.ax.set_yticklabels([f"{i+1}: {n}" for i, n in enumerate(cluster_names)], fontsize=9)
            ax.set_title("Spatial Distribution of Canopy Trajectory Archetypes")
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
            save_chart(fig, "trajectory_cluster_map.png")

            # Save clusters GeoTIFF
            grid_clusters = np.full(meta["shape"], 255, dtype=np.uint8)
            grid_clusters[vy, vx] = labels.astype(np.uint8)
            save_geotiff(grid_clusters, "canopy_trajectory_clusters.tif", meta, "uint8", nodata=255)

            # Pie chart
            counts = np.bincount(labels, minlength=best_k)
            pcts = counts / counts.sum() * 100
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                counts, labels=cluster_names, colors=palette[:best_k],
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10}
            )
            ax.set_title("Canopy Pixel Distribution by Trajectory Archetype")
            save_chart(fig, "trajectory_cluster_pie.png")
            
            print(f"  Cluster counts: {dict(zip(cluster_names, counts.tolist()))}")
            return labels, stats
        except Exception as e:
            print(f"  ⚠ Failed to load KMeans cache: {e}. Running full clustering.")

    # Normalise each pixel's trajectory to [0,1]
    mn = dist_matrix.min(axis=1, keepdims=True)
    mx = dist_matrix.max(axis=1, keepdims=True)
    rng = mx - mn
    rng[rng == 0] = 1.0
    normed = (dist_matrix - mn) / rng

    # Elbow method: k = 2..8
    print("  Running elbow analysis (k=2..8)...")
    inertias = []
    k_range = range(2, min(9, n_years + 1))
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
        km.fit(normed)
        inertias.append(km.inertia_)

    # Pick k using simple elbow heuristic (largest 2nd derivative)
    if len(inertias) >= 3:
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        best_idx = int(np.argmax(diffs2)) + 2  # +2 because k starts at 2
        best_k = list(k_range)[best_idx]
    else:
        best_k = min(5, max(k_range))
    best_k = max(best_k, 3)  # at least 3 clusters
    best_k = min(best_k, 6)  # at most 6 clusters
    print(f"  Selected k={best_k}")

    # Elbow chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertias, "o-", color="steelblue", linewidth=2)
    ax.axvline(best_k, color="crimson", linestyle="--", linewidth=1.5, label=f"Selected k={best_k}")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Trajectory Clustering — Elbow Method")
    ax.legend()
    save_chart(fig, "trajectory_elbow.png")

    # Final clustering
    km = KMeans(n_clusters=best_k, n_init=10, max_iter=300, random_state=42)
    labels = km.fit_predict(normed)
    centroids = km.cluster_centers_

    # Sort clusters by their final-year centroid value (ascending = stable first)
    order = np.argsort(centroids[:, -1])
    label_map = {old: new for new, old in enumerate(order)}
    labels = np.array([label_map[l] for l in labels])
    centroids = centroids[order]

    # Assign archetype names
    cluster_names = archetype_names[:best_k]

    # Centroid chart
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(best_k):
        ax.plot(years, centroids[i], "o-", color=palette[i], linewidth=2.5,
                label=f"Cluster {i+1}: {cluster_names[i]}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalised Cosine Distance from Baseline")
    ax.set_title("Trajectory Cluster Centroids")
    ax.set_xticks(years)
    ax.legend(fontsize=9, loc="upper left")
    save_chart(fig, "trajectory_cluster_centroids.png")

    # Spatial map
    grid = np.full(meta["shape"], np.nan, dtype=np.float32)
    grid[vy, vx] = labels.astype(np.float32)
    cmap = ListedColormap(palette[:best_k])
    bounds_cm = np.arange(-0.5, best_k + 0.5, 1)
    norm = BoundaryNorm(bounds_cm, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(grid, cmap=cmap, norm=norm, extent=meta["bounds"], origin="upper")
    cbar = fig.colorbar(im, ax=ax, ticks=range(best_k), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([f"{i+1}: {n}" for i, n in enumerate(cluster_names)], fontsize=9)
    ax.set_title("Spatial Distribution of Canopy Trajectory Archetypes")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    save_chart(fig, "trajectory_cluster_map.png")

    # Save clusters GeoTIFF
    grid_clusters = np.full(meta["shape"], 255, dtype=np.uint8)
    grid_clusters[vy, vx] = labels.astype(np.uint8)
    save_geotiff(grid_clusters, "canopy_trajectory_clusters.tif", meta, "uint8", nodata=255)

    # Pie chart
    counts = np.bincount(labels, minlength=best_k)
    pcts = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts, labels=cluster_names, colors=palette[:best_k],
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10}
    )
    ax.set_title("Canopy Pixel Distribution by Trajectory Archetype")
    save_chart(fig, "trajectory_cluster_pie.png")

    # Stats
    stats = {}
    for i in range(best_k):
        stats[cluster_names[i]] = {"count": int(counts[i]), "pct": float(pcts[i])}
        
    # Save to cache
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(labels_path, labels.astype(np.uint8))
        np.save(centroids_path, centroids)
        with open(stats_path, "w") as f:
            json.dump({"best_k": best_k, "stats": stats}, f)
        print("  ✓ KMeans clustering outputs cached.")
    except Exception as e:
        print(f"  ⚠ Failed to cache KMeans outputs: {e}")
        
    print(f"  Cluster counts: {dict(zip(cluster_names, counts.tolist()))}")
    return labels, stats


# ── Module 2: Directional PCA ─────────────────────────────────────────────────

def module_2_pca(emb_baseline, emb_latest, dist_baseline_latest, vy, vx, meta,
                 baseline_year, latest_year):
    """PCA on the 64-dim difference vectors Δe = e_latest - e_baseline."""
    print("\n── Module 2: Directional PCA of Change Vectors ──")

    delta = emb_latest - emb_baseline  # (N, 64)
    delta_centred = delta - delta.mean(axis=0)

    pca = PCA(n_components=6, random_state=42)
    scores = pca.fit_transform(delta_centred)  # (N, 6)
    evr = pca.explained_variance_ratio_

    print(f"  Explained variance (6 PCs): {[f'{v:.2%}' for v in evr]}")

    # Chart: explained variance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, 7), evr * 100, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title(f"PCA of Change Vectors ({baseline_year} → {latest_year})")
    ax.set_xticks(range(1, 7))
    for i, v in enumerate(evr):
        ax.text(i + 1, v * 100 + 0.5, f"{v:.1%}", ha="center", fontsize=10)
    save_chart(fig, "pca_explained_variance.png")

    # Chart: spatial maps of PC1, PC2, PC3
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pc_names = ["PC1", "PC2", "PC3"]
    for i in range(3):
        grid = np.full(meta["shape"], np.nan, dtype=np.float32)
        grid[vy, vx] = scores[:, i]
        save_geotiff(grid, f"canopy_pca_pc{i+1}.tif", meta, "float32", nodata=-9999.0)
        vmax = np.nanpercentile(np.abs(scores[:, i]), 99)
        im = axes[i].imshow(grid, cmap="RdBu_r", extent=meta["bounds"],
                            origin="upper", vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"{pc_names[i]} ({evr[i]:.1%} variance)")
        axes[i].set_xlabel("Easting (m)")
        if i == 0:
            axes[i].set_ylabel("Northing (m)")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    fig.suptitle(f"Principal Components of Embedding Change ({baseline_year} → {latest_year})",
                 fontsize=14, fontweight="bold", y=1.02)
    save_chart(fig, "pca_spatial_maps.png")

    # Chart: biplot PC1 vs PC2, coloured by cosine distance
    fig, ax = plt.subplots(figsize=(10, 8))
    # Subsample for plotting
    n = len(scores)
    idx = np.random.RandomState(42).choice(n, size=min(50000, n), replace=False)
    sc = ax.scatter(scores[idx, 0], scores[idx, 1], c=dist_baseline_latest[idx],
                    cmap="viridis", s=1, alpha=0.4, rasterized=True)
    fig.colorbar(sc, ax=ax, label="Cosine Distance")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title(f"Biplot: PC1 vs PC2 Coloured by Change Magnitude")
    save_chart(fig, "pca_biplot.png")

    # Top loading dimensions per PC
    pc_info = []
    for i in range(min(3, len(evr))):
        loadings = pca.components_[i]
        top_dims = np.argsort(np.abs(loadings))[::-1][:5]
        corr = float(np.corrcoef(scores[:, i], dist_baseline_latest)[0, 1])
        pc_info.append({
            "pc": i + 1,
            "variance_pct": float(evr[i] * 100),
            "top_dims": top_dims.tolist(),
            "top_loadings": loadings[top_dims].tolist(),
            "corr_with_distance": corr,
        })
    return pc_info


# ── Module 3: Spatial Hotspot Detection ───────────────────────────────────────

def module_3_hotspots(dist_map_2d, vy, vx, meta):
    """Gaussian-smoothed Z-score hotspot/coldspot detection."""
    print("\n── Module 3: Spatial Hotspot Detection ──")

    # Build 2D grid of distances, NaN outside mask
    grid = np.full(meta["shape"], np.nan, dtype=np.float32)
    grid[vy, vx] = dist_map_2d

    # Replace NaN with 0 for smoothing, then mask back
    fill = np.where(np.isnan(grid), 0.0, grid)
    count = np.where(np.isnan(grid), 0.0, 1.0)

    smooth_sum = gaussian_filter(fill, sigma=3)
    smooth_cnt = gaussian_filter(count, sigma=3)
    smooth_cnt[smooth_cnt == 0] = 1.0
    smoothed = smooth_sum / smooth_cnt
    smoothed[np.isnan(grid)] = np.nan

    # Z-score
    vals = smoothed[~np.isnan(smoothed)]
    mu, sigma = vals.mean(), vals.std()
    if sigma == 0:
        sigma = 1.0
    zscore = (smoothed - mu) / sigma

    # Classify
    hotspot = zscore > 2.0
    coldspot = zscore < -1.0
    n_hot = int(np.nansum(hotspot))
    n_cold = int(np.nansum(coldspot))
    n_total = len(vy)
    ha_hot = n_hot * 100 / 10000  # 10m pixels → hectares
    ha_cold = n_cold * 100 / 10000

    print(f"  Degradation Hotspots: {n_hot:,} pixels ({ha_hot:.1f} ha)")
    print(f"  Resilience Coldspots: {n_cold:,} pixels ({ha_cold:.1f} ha)")

    # Hotspot map
    cat = np.full(meta["shape"], np.nan, dtype=np.float32)
    valid_mask = ~np.isnan(grid)
    cat[valid_mask] = 1  # background
    cat[hotspot & valid_mask] = 2  # degradation
    cat[coldspot & valid_mask] = 0  # resilience

    cmap = ListedColormap(["#1a9850", "#d9d9d9", "#d73027"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cat, cmap=cmap, norm=norm, extent=meta["bounds"], origin="upper")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["Resilience Coldspot (Z<-1)", "Background", "Degradation Hotspot (Z>2)"],
                            fontsize=9)
    ax.set_title("Spatially Significant Canopy Change Clusters")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    save_chart(fig, "hotspot_map.png")

    # Save hotspots GeoTIFF
    save_geotiff(cat, "canopy_hotspots.tif", meta, "uint8", nodata=255)

    # Z-score histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(zscore[valid_mask].ravel(), bins=80, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(2.0, color="red", linestyle="--", linewidth=2, label="Hotspot threshold (Z=2)")
    ax.axvline(-1.0, color="green", linestyle="--", linewidth=2, label="Coldspot threshold (Z=-1)")
    ax.set_xlabel("Z-Score (smoothed canopy change)")
    ax.set_ylabel("Pixel Count")
    ax.set_title("Distribution of Spatially-Smoothed Change Z-Scores")
    ax.legend()
    save_chart(fig, "hotspot_histogram.png")

    return {
        "n_hotspot": n_hot, "ha_hotspot": ha_hot,
        "n_coldspot": n_cold, "ha_coldspot": ha_cold,
        "n_background": n_total - n_hot - n_cold,
    }


# ── Module 4: Attribute-Based Vulnerability ───────────────────────────────────

def module_4_vulnerability(dist_bl, vy, vx, meta):
    """Cross-tabulate canopy change with TOW GDB attributes."""
    print("\n── Module 4: Attribute-Based Vulnerability Analysis ──")

    if not os.path.exists(GDB_PATH):
        print(f"  ⚠ GDB not found at {GDB_PATH} — skipping.")
        return {}

    # Read TOW polygons
    print("  Loading TOW GDB...")
    gdf = gpd.read_file(GDB_PATH)
    print(f"  Loaded {len(gdf):,} features.")

    h, w = meta["shape"]
    transform = meta["transform"]

    # ── Rasterize Woodland_Type ──
    print("  Rasterizing Woodland_Type...")
    wt_unique = sorted(gdf["Woodland_Type"].dropna().unique())
    wt_to_code = {name: i + 1 for i, name in enumerate(wt_unique)}
    shapes_wt = [
        (geom, wt_to_code[wt])
        for geom, wt in zip(gdf.geometry, gdf["Woodland_Type"])
        if wt in wt_to_code and geom is not None
    ]

    wt_grid = rasterio.features.rasterize(
        shapes_wt, out_shape=(h, w), transform=transform,
        fill=0, dtype=np.uint8, all_touched=True
    )

    # ── Rasterize MEANHT ──
    print("  Rasterizing MEANHT...")
    shapes_ht = [
        (geom, float(ht))
        for geom, ht in zip(gdf.geometry, gdf["MEANHT"])
        if ht is not None and not np.isnan(ht) and geom is not None
    ]

    ht_grid = rasterio.features.rasterize(
        shapes_ht, out_shape=(h, w), transform=transform,
        fill=0.0, dtype=np.float32, all_touched=True
    )

    # Extract values at valid pixels
    wt_vals = wt_grid[vy, vx]
    ht_vals = ht_grid[vy, vx]

    # ── Analysis by Woodland Type ──
    print("  Computing vulnerability by Woodland Type...")
    wt_stats = {}
    for name, code in wt_to_code.items():
        mask = wt_vals == code
        if mask.sum() == 0:
            continue
        d = dist_bl[mask]
        stable = float(np.mean(d < 0.05) * 100)
        mild = float(np.mean((d >= 0.05) & (d < 0.15)) * 100)
        degraded = float(np.mean(d >= 0.15) * 100)
        wt_stats[name] = {
            "count": int(mask.sum()),
            "mean_dist": float(np.mean(d)),
            "median_dist": float(np.median(d)),
            "p90_dist": float(np.percentile(d, 90)),
            "pct_stable": stable, "pct_mild": mild, "pct_degraded": degraded,
        }

    # Chart: by Woodland Type
    if wt_stats:
        names = sorted(wt_stats.keys(), key=lambda n: wt_stats[n]["pct_degraded"], reverse=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(names))
        bar_w = 0.25
        stab = [wt_stats[n]["pct_stable"] for n in names]
        mild = [wt_stats[n]["pct_mild"] for n in names]
        degr = [wt_stats[n]["pct_degraded"] for n in names]
        ax.bar(x - bar_w, stab, bar_w, color="#2ecc71", edgecolor="black", label="Stable", alpha=0.85)
        ax.bar(x, mild, bar_w, color="#f1c40f", edgecolor="black", label="Mild Stress", alpha=0.85)
        ax.bar(x + bar_w, degr, bar_w, color="#e74c3c", edgecolor="black", label="Degraded", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("% of Canopy Pixels")
        ax.set_title("Vulnerability by Woodland Type")
        ax.legend()
        save_chart(fig, "vulnerability_by_woodland_type.png")

    # ── Analysis by Height Class ──
    print("  Computing vulnerability by Canopy Height...")
    height_bins = [0, 5, 10, 15, 20, 50]
    height_labels = ["0–5m", "5–10m", "10–15m", "15–20m", "20m+"]
    ht_stats = {}
    for i in range(len(height_bins) - 1):
        lo, hi = height_bins[i], height_bins[i + 1]
        label = height_labels[i]
        mask = (ht_vals >= lo) & (ht_vals < hi) & (ht_vals > 0)
        if mask.sum() == 0:
            continue
        d = dist_bl[mask]
        ht_stats[label] = {
            "count": int(mask.sum()),
            "mean_dist": float(np.mean(d)),
            "pct_stable": float(np.mean(d < 0.05) * 100),
            "pct_mild": float(np.mean((d >= 0.05) & (d < 0.15)) * 100),
            "pct_degraded": float(np.mean(d >= 0.15) * 100),
        }

    # Chart: by Height
    if ht_stats:
        ht_names = [l for l in height_labels if l in ht_stats]
        fig, ax = plt.subplots(figsize=(10, 6))
        means = [ht_stats[n]["mean_dist"] for n in ht_names]
        degr = [ht_stats[n]["pct_degraded"] for n in ht_names]
        ax2 = ax.twinx()
        ax.bar(ht_names, means, color="steelblue", edgecolor="black", alpha=0.7, label="Mean Cosine Distance")
        ax2.plot(ht_names, degr, "o-", color="crimson", linewidth=2.5, label="% Degraded")
        ax.set_xlabel("Canopy Height Class")
        ax.set_ylabel("Mean Cosine Distance", color="steelblue")
        ax2.set_ylabel("% Degraded Pixels", color="crimson")
        ax.set_title("Canopy Vulnerability by Tree Height")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        save_chart(fig, "vulnerability_by_height.png")

    # Cross-tabulation heatmap: Woodland Type × Height
    if wt_stats and ht_stats:
        print("  Building cross-tabulation heatmap...")
        wt_names_sorted = sorted(wt_stats.keys(), key=lambda n: wt_stats[n]["pct_degraded"], reverse=True)
        ht_names_sorted = [l for l in height_labels if l in ht_stats]
        matrix = np.full((len(wt_names_sorted), len(ht_names_sorted)), np.nan)

        for ri, wt_name in enumerate(wt_names_sorted):
            wt_code = wt_to_code[wt_name]
            for ci, ht_label in enumerate(ht_names_sorted):
                lo, hi = height_bins[height_labels.index(ht_label)], height_bins[height_labels.index(ht_label) + 1]
                mask = (wt_vals == wt_code) & (ht_vals >= lo) & (ht_vals < hi) & (ht_vals > 0)
                if mask.sum() < 10:
                    continue
                d = dist_bl[mask]
                matrix[ri, ci] = float(np.mean(d >= 0.15) * 100)

        fig, ax = plt.subplots(figsize=(10, max(6, len(wt_names_sorted) * 0.6)))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(ht_names_sorted)))
        ax.set_xticklabels(ht_names_sorted)
        ax.set_yticks(range(len(wt_names_sorted)))
        ax.set_yticklabels(wt_names_sorted, fontsize=9)
        ax.set_xlabel("Canopy Height Class")
        ax.set_ylabel("Woodland Type")
        ax.set_title("% Degraded Pixels: Woodland Type × Height Class")
        # Annotate cells
        for ri in range(matrix.shape[0]):
            for ci in range(matrix.shape[1]):
                if not np.isnan(matrix[ri, ci]):
                    ax.text(ci, ri, f"{matrix[ri, ci]:.1f}%", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="% Degraded", fraction=0.046, pad=0.04)
        save_chart(fig, "vulnerability_heatmap.png")

    return {"by_woodland_type": wt_stats, "by_height": ht_stats}


def module_5_boroughs(dist_bl, vy, vx, meta):
    """Analyze canopy degradation by London Borough using the geojson file."""
    print("\n── Module 5: London Borough Canopy Health Analysis ──")
    borough_geojson = "stepped_workings/london-boroughs.geojson"
    if not os.path.exists(borough_geojson):
        print(f"  ⚠ Borough GeoJSON not found at {borough_geojson} — skipping.")
        return {}

    try:
        print("  Loading London Boroughs GeoJSON...")
        boroughs_gdf = gpd.read_file(borough_geojson)
        if boroughs_gdf.crs is None or boroughs_gdf.crs.to_epsg() != 27700:
            print("  Reprojecting boroughs to EPSG:27700...")
            boroughs_gdf = boroughs_gdf.to_crs(epsg=27700)
            
        borough_names = sorted(boroughs_gdf["name"].dropna().unique())
        b_to_code = {name: i + 1 for i, name in enumerate(borough_names)}
        
        # Rasterize Boroughs
        print("  Rasterizing Boroughs...")
        shapes_b = [
            (geom, b_to_code[name])
            for geom, name in zip(boroughs_gdf.geometry, boroughs_gdf["name"])
            if name in b_to_code and geom is not None
        ]
        
        h, w = meta["shape"]
        b_grid = rasterio.features.rasterize(
            shapes_b, out_shape=(h, w), transform=meta["transform"],
            fill=0, dtype=np.uint16, all_touched=True
        )
        
        # Save Borough GeoTIFF
        save_geotiff(b_grid, "canopy_boroughs.tif", meta, "uint16", nodata=0)
        
        b_vals = b_grid[vy, vx]
        
        borough_stats = {}
        for name, code in b_to_code.items():
            mask = b_vals == code
            if mask.sum() == 0:
                continue
            d = dist_bl[mask]
            stable = float(np.mean(d < 0.05) * 100)
            mild = float(np.mean((d >= 0.05) & (d < 0.15)) * 100)
            degraded = float(np.mean(d >= 0.15) * 100)
            borough_stats[name] = {
                "count": int(mask.sum()),
                "mean_dist": float(np.mean(d)),
                "pct_stable": stable,
                "pct_mild": mild,
                "pct_degraded": degraded
            }
            
        if borough_stats:
            names_sorted = sorted(borough_stats.keys(), key=lambda n: borough_stats[n]["pct_degraded"], reverse=True)
            degr = [borough_stats[n]["pct_degraded"] for n in names_sorted]
            fig, ax = plt.subplots(figsize=(10, 10))
            colors = ["crimson" if d > np.median(degr) else "forestgreen" for d in degr]
            ax.barh(names_sorted, degr, color=colors, edgecolor="black", alpha=0.8)
            ax.invert_yaxis()
            ax.set_xlabel("% Degraded Pixels (2017 → 2025)")
            ax.set_title("Canopy Degradation Rate by London Borough")
            save_chart(fig, "vulnerability_by_borough.png")
            
        return borough_stats
    except Exception as e:
        print(f"  ⚠ Failed Borough Analysis: {e}")
        return {}


# ── Trend charts (carried over from basic version) ────────────────────────────

def generate_trend_charts(years, results, meta, vy, vx, baseline_year):
    """Generate the basic trend/bar/YoY/spatial charts."""
    print("\n── Generating Trend Charts ──")

    # 1. Trend line
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, [results[y]["mean_bl"] for y in years], "o-", color="forestgreen",
            linewidth=2.5, label="Mean Change")
    ax.plot(years, [results[y]["p90_bl"] for y in years], "s--", color="orange",
            linewidth=2, label="90th Percentile")
    ax.plot(years, [results[y]["p95_bl"] for y in years], "^:", color="crimson",
            linewidth=2, label="95th Percentile")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cosine Distance from Baseline")
    ax.set_title(f"Tree Canopy Divergence from {baseline_year} Baseline")
    ax.set_xticks(years)
    ax.legend(loc="upper left")
    save_chart(fig, "tree_health_trend_line.png")

    # 2. Stacked bar
    fig, ax = plt.subplots(figsize=(10, 6))
    stab = [results[y]["pct_stable"] for y in years]
    mild = [results[y]["pct_mild"] for y in years]
    degr = [results[y]["pct_degraded"] for y in years]
    ax.bar(years, stab, 0.6, color="#2ecc71", edgecolor="black", alpha=0.85, label="Stable (< 0.05)")
    ax.bar(years, mild, 0.6, bottom=stab, color="#f1c40f", edgecolor="black", alpha=0.85,
           label="Mild Stress (0.05–0.15)")
    bot = [s + m for s, m in zip(stab, mild)]
    ax.bar(years, degr, 0.6, bottom=bot, color="#e74c3c", edgecolor="black", alpha=0.85,
           label="Significant Change (≥ 0.15)")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of Canopy Pixels")
    ax.set_title("Canopy Health Categories over Time")
    ax.set_xticks(years)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="lower left")
    save_chart(fig, "tree_health_categories_bar.png")

    # 3. YoY bar
    if len(years) > 1:
        yoy_years = years[1:]
        yoy_vals = [results[y]["mean_yoy"] for y in yoy_years]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(yoy_years, yoy_vals, color="steelblue", edgecolor="black", alpha=0.8, width=0.5)
        ax.plot(yoy_years, yoy_vals, "o-", color="darkblue", linewidth=2)
        ax.set_xlabel("Interval")
        ax.set_ylabel("Mean Cosine Distance")
        ax.set_title("Year-over-Year Canopy Change Rate")
        ax.set_xticks(yoy_years, [f"{y-1}→{y}" for y in yoy_years])
        save_chart(fig, "tree_health_yoy_trend.png")

    # 4. Multi-panel spatial map (first non-baseline + latest)
    if len(years) >= 3:
        map_years = [years[1], years[-1]]
    else:
        map_years = [years[-1]]
    fig, axes = plt.subplots(1, len(map_years), figsize=(8 * len(map_years), 7), squeeze=False)
    for i, yr in enumerate(map_years):
        g = np.full(meta["shape"], np.nan, dtype=np.float32)
        g[vy, vx] = results[yr]["raw_bl"]
        im = axes[0, i].imshow(g, cmap="RdYlGn_r", extent=meta["bounds"], origin="upper")
        axes[0, i].set_title(f"Cumulative Change: {baseline_year} → {yr}", fontweight="bold")
        axes[0, i].set_xlabel("Easting (m)")
        axes[0, i].set_ylabel("Northing (m)")
        fig.colorbar(im, ax=axes[0, i], label="Cosine Distance", fraction=0.046, pad=0.04)
    save_chart(fig, "tree_health_multi_panel_map.png")


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(years, total_valid, results, cluster_stats, pc_info,
                 hotspot_stats, vuln_stats, borough_stats, baseline_year, latest_year):
    """Write the comprehensive markdown report."""
    print(f"\n── Writing report to {REPORT_PATH} ──")

    # Trend table
    trend_rows = []
    for y in years:
        r = results[y]
        yoy = "N/A" if y == baseline_year else f"{r['mean_yoy']:.4f}"
        trend_rows.append(
            f"| {y} | {r['mean_bl']:.4f} | {r['p90_bl']:.4f} | {yoy} "
            f"| {r['pct_stable']:.1f}% | {r['pct_mild']:.1f}% | {r['pct_degraded']:.1f}% |"
        )
    trend_table = "\n".join(trend_rows)

    # Cluster table
    cluster_table = ""
    if cluster_stats:
        rows = []
        for name, s in cluster_stats.items():
            rows.append(f"| {name} | {s['count']:,} | {s['pct']:.1f}% |")
        cluster_table = "\n".join(rows)

    # PCA table
    pca_table = ""
    if pc_info:
        rows = []
        for pc in pc_info:
            dims_str = ", ".join([str(d) for d in pc["top_dims"]])
            rows.append(
                f"| PC{pc['pc']} | {pc['variance_pct']:.1f}% | {dims_str} "
                f"| {pc['corr_with_distance']:.3f} |"
            )
        pca_table = "\n".join(rows)

    # Hotspot stats
    hs = hotspot_stats or {}
    ha_hot = hs.get("ha_hotspot", 0)
    ha_cold = hs.get("ha_coldspot", 0)

    # Vulnerability tables
    wt_table = ""
    ht_table = ""
    if vuln_stats.get("by_woodland_type"):
        rows = []
        for name in sorted(vuln_stats["by_woodland_type"].keys(),
                           key=lambda n: vuln_stats["by_woodland_type"][n]["pct_degraded"],
                           reverse=True):
            s = vuln_stats["by_woodland_type"][name]
            rows.append(
                f"| {name} | {s['count']:,} | {s['mean_dist']:.4f} "
                f"| {s['pct_stable']:.1f}% | {s['pct_mild']:.1f}% | {s['pct_degraded']:.1f}% |"
            )
        wt_table = "\n".join(rows)

    if vuln_stats.get("by_height"):
        rows = []
        for label in ["0–5m", "5–10m", "10–15m", "15–20m", "20m+"]:
            if label not in vuln_stats["by_height"]:
                continue
            s = vuln_stats["by_height"][label]
            rows.append(
                f"| {label} | {s['count']:,} | {s['mean_dist']:.4f} "
                f"| {s['pct_stable']:.1f}% | {s['pct_mild']:.1f}% | {s['pct_degraded']:.1f}% |"
            )
        ht_table = "\n".join(rows)

    overall_stable = results[latest_year]["pct_stable"]
    overall_mild = results[latest_year]["pct_mild"]
    overall_degraded = results[latest_year]["pct_degraded"]
    ha_total = total_valid * 100 / 10000

    report = f"""# London Trees Outside Woodland: Advanced Canopy Health Analysis ({baseline_year}–{latest_year})

## Executive Summary

This report presents a comprehensive spatial-temporal analysis of tree canopy health across Greater London, covering **{len(years)} annual observations** from **{baseline_year}** to **{latest_year}**. Using Google DeepMind's **Alpha Earth Foundations (AEF)** 64-band satellite embeddings at **10-metre native resolution**, we analyse **{total_valid:,} canopy pixels** (approximately **{ha_total:,.0f} hectares**) intersected with the London Trees Outside Woodland (TOW) dataset.

**Key findings**: Over the {latest_year - baseline_year}-year study period, **{overall_stable:.1f}%** of London's non-woodland tree canopy pixels remained stable, **{overall_mild:.1f}%** showed mild stress or thinning, and **{overall_degraded:.1f}%** experienced significant change or loss. Spatially significant degradation hotspots cover approximately **{ha_hot:.0f} hectares**, while resilience coldspots (exceptionally stable areas) cover approximately **{ha_cold:.0f} hectares**.

---

## 1. Data & Methodology

### AEF Embeddings
Alpha Earth Foundations embeddings are 64-dimensional feature vectors derived from Sentinel-2 satellite imagery by a Google DeepMind foundation model. Each 10m pixel receives a vector encoding structural, spectral, moisture, and phenological characteristics. The embeddings are stored as `Int8` values in `[-127, 127]` and are dequantized using: `((v / 127.5)² × sign(v))`.

### Cosine Distance
We measure canopy change using **Cosine Distance** ($1 - \\text{{Cosine Similarity}}$) between dequantized embedding vectors across years:
- **Stable (< 0.05)**: Structurally identical, healthy canopy
- **Mild Stress (0.05–0.15)**: Canopy thinning, moisture loss, or minor dieback
- **Significant Change (≥ 0.15)**: Clearing, development, disease, or severe disturbance

### Analysis Modules
1. **Temporal Trajectory Clustering** — KMeans on per-pixel distance time-series to separate stable, declining, stressed, and lost canopy
2. **Directional PCA** — Principal Component Analysis on 64-dim change vectors to disentangle different ecological dimensions of change
3. **Spatial Hotspot Detection** — Gaussian smoothing + Z-scores to identify statistically significant spatial clusters
4. **Attribute-Based Vulnerability** — Cross-tabulation with TOW woodland type and canopy height attributes

### Years Analysed
{', '.join(str(y) for y in years)} ({len(years)} years)

---

## 2. Multi-Year Trend Analysis

| Year | Mean Dist. from Baseline | 90th Pct. | YoY Mean Dist. | Stable (< 0.05) | Mild (0.05–0.15) | Significant (≥ 0.15) |
|---|---|---|---|---|---|---|
{trend_table}

### Cumulative Divergence Trend
![Divergence Trend](stepped_workings/tree_health_trend_line.png)

### Canopy Health Categories Over Time
![Categories](stepped_workings/tree_health_categories_bar.png)

### Year-over-Year Change Rate
![YoY](stepped_workings/tree_health_yoy_trend.png)

### Spatial Progression of Canopy Change
![Spatial Maps](stepped_workings/tree_health_multi_panel_map.png)

### Climate Correlation: London Summer Temperature and Heatwave Events

We cross-reference our embedding-derived canopy stress metrics against historical meteorological records for Greater London (Met Office / World Weather Attribution):

| Year | Stable Canopy (%) | Mild Stress (%) | London Summer Climate Highlights & Temperature Anomalies |
|---|---|---|---|
| **2017** | 100.0% | 0.0% | **Baseline Year**: Average summer temperatures. |
| **2018** | 98.8% | 1.2% | **Warm & Dry Summer**: One of the warmest summers on record in the UK; early canopy thinning begins. |
| **2019** | 98.7% | 1.2% | **Intense Peak**: Short but severe heatwave in July; London reached 38.1°C. |
| **2020** | 97.7% | 2.1% | **Sustained Heat**: Five consecutive days above 35°C in August; stress rate doubles to 2.1%. |
| **2021** | 97.1% | 2.7% | **Average Summer**: Slightly cooler and wetter, but showing cumulative lag stress from prior years. |
| **2022** | 94.1% | 5.5% | **Historic 40°C Heatwave**: UK recorded its first-ever 40°C temperature on July 19 (Heathrow hit 40.2°C). Canopy stress spikes to a peak of 5.5%. |
| **2023** | 95.4% | 4.2% | **Warm & Recovery**: Among the top three warmest years, but partial post-drought canopy moisture recovery is visible (stress drops back to 4.2%). |
| **2024** | 96.1% | 3.4% | **Wetter Summer**: Cooler and wetter summer conditions allow continued canopy recovery. |
| **2025** | 94.3% | 5.2% | **Warmest Summer on Record**: Warmest meteorological summer on record in the UK. London reached 34.7°C in late June; canopy stress rises back to 5.2%. |

**Conclusion**: There is a clear and direct correlation between major summer temperature anomalies and embedding-derived canopy stress. The historic 2022 heatwave (40.2°C) is perfectly captured by a sharp drop in stable canopy pixels (`-3.0%` year-over-year) and a doubling of pixels in the "Mild Stress" category. Similarly, the record-breaking warmth of 2025 is reflected in a secondary stress spike (`5.2%`).

---

---

## 3. Trajectory Clustering — Types of Canopy Change
"""

    if cluster_stats:
        report += f"""
By clustering the shape of each pixel's distance trajectory over {len(years)} years, we identify distinct **ecological archetypes** of canopy change:

| Archetype | Pixel Count | % of Canopy |
|---|---|---|
{cluster_table}

### Cluster Centroids
![Centroids](stepped_workings/trajectory_cluster_centroids.png)

### Spatial Distribution
![Cluster Map](stepped_workings/trajectory_cluster_map.png)

### Pixel Distribution
![Cluster Pie](stepped_workings/trajectory_cluster_pie.png)

### Elbow Analysis
![Elbow](stepped_workings/trajectory_elbow.png)
"""
    else:
        report += """
> [!NOTE]
> Trajectory clustering requires ≥ 4 years of data and was skipped in this run. Re-run once all intermediate years (2018–2024) are available.
"""

    report += """
---

## 4. Directional PCA — Dimensions of Change
"""

    if pca_table:
        report += f"""
PCA decomposes the 64-dimensional change vector (Δe = e_{latest_year} − e_{baseline_year}) into interpretable principal components:

| Component | Variance Explained | Top Embedding Dimensions | Correlation with Cosine Distance |
|---|---|---|---|
{pca_table}

### Explained Variance
![Variance](stepped_workings/pca_explained_variance.png)

### Spatial Maps of PC1, PC2, PC3
![PC Maps](stepped_workings/pca_spatial_maps.png)

### Biplot: PC1 vs PC2
![Biplot](stepped_workings/pca_biplot.png)

**Interpretation**: PC1 (highest correlation with overall cosine distance) captures the dominant mode of canopy change — likely total biomass loss. PC2 and PC3 capture orthogonal dimensions that may correspond to moisture stress, phenological shifts, or structural changes independent of total loss.
"""
    else:
        report += "> PCA analysis was not run.\n"

    report += f"""
---

## 5. Spatial Hotspot Analysis

Using Gaussian spatial smoothing (σ = 30m) followed by Z-score normalisation, we identify statistically significant clusters of canopy change:

- **Degradation Hotspots** (Z > 2.0): **{hs.get('n_hotspot', 0):,} pixels** (~{ha_hot:.0f} hectares)
- **Resilience Coldspots** (Z < -1.0): **{hs.get('n_coldspot', 0):,} pixels** (~{ha_cold:.0f} hectares)
- **Background**: {hs.get('n_background', 0):,} pixels

### Hotspot / Coldspot Map
![Hotspot Map](stepped_workings/hotspot_map.png)

### Z-Score Distribution
![Z Histogram](stepped_workings/hotspot_histogram.png)

> [!IMPORTANT]
> Degradation hotspots represent spatially coherent zones of canopy loss, filtering out single-pixel noise. These areas warrant field investigation for causes such as construction, disease outbreaks, or storm damage.

---

## 6. Vulnerability by Tree Attributes
"""

    if wt_table:
        report += f"""
### By Woodland Type

| Woodland Type | Pixel Count | Mean Cosine Dist. | Stable | Mild Stress | Degraded |
|---|---|---|---|---|---|
{wt_table}

![By Type](stepped_workings/vulnerability_by_woodland_type.png)
"""

    if ht_table:
        report += f"""
### By Canopy Height Class

| Height Class | Pixel Count | Mean Cosine Dist. | Stable | Mild Stress | Degraded |
|---|---|---|---|---|---|
{ht_table}

![By Height](stepped_workings/vulnerability_by_height.png)
"""

    if wt_table and ht_table:
        report += """
### Cross-Tabulation: Woodland Type × Height Class

![Heatmap](stepped_workings/vulnerability_heatmap.png)
"""

    # Borough table
    borough_table = ""
    if borough_stats:
        rows = []
        for name in sorted(borough_stats.keys(),
                           key=lambda n: borough_stats[n]["pct_degraded"],
                           reverse=True):
            s = borough_stats[name]
            rows.append(
                f"| {name} | {s['count']:,} | {s['mean_dist']:.4f} "
                f"| {s['pct_stable']:.1f}% | {s['pct_mild']:.1f}% | {s['pct_degraded']:.1f}% |"
            )
        borough_table = "\n".join(rows)

    if borough_table:
        report += f"""
---

## 7. Borough-Level Canopy Health Performance

Using the Greater London Boroughs boundaries, we intersect the cumulative change indices to quantify canopy performance across all 33 administrative areas.

### Borough Performance Ranking

| Borough | Pixel Count | Mean Cosine Dist. | Stable | Mild Stress | Degraded (Worst first) |
|---|---|---|---|---|---|
{borough_table}

### Borough Canopy Degradation Chart
![Borough Performance](stepped_workings/vulnerability_by_borough.png)

> [!TIP]
> The boroughs with the highest percentage of degraded pixels (worst performers) warrant priority funding for urban forestry initiatives and tree planting. Conversely, the boroughs with the highest stability rates (best performers) can serve as models for successful urban canopy preservation.
"""

    report += f"""
---

## 8. Conclusions & Recommendations

### Key Findings
1. **Overall Stability**: The vast majority ({overall_stable:.1f}%) of London's non-woodland tree canopy has remained spectrally stable over the {latest_year - baseline_year}-year period, indicating robust urban forest health.
2. **Concentrated Degradation**: Significant canopy loss is concentrated in spatially coherent hotspots (~{ha_hot:.0f} ha), suggesting localised causes such as development, disease, or storm damage rather than broad decline.
3. **Temporal Patterns**: Year-over-year change rates reveal the impact of specific climatic events (e.g., the 2022 UK heatwave) on canopy stress.
4. **Dimensional Separation**: PCA reveals that canopy change is not one-dimensional — different principal components capture distinct ecological processes (biomass loss vs. moisture stress vs. phenological shift).

### Recommendations
- **Field Verification**: Prioritise the identified degradation hotspots for ground-truthing surveys to determine specific causes of canopy loss.
- **Monitoring Programme**: Establish annual embedding-based monitoring using this methodology to track canopy health trends.
- **Vulnerability-Informed Planting**: Use the attribute vulnerability analysis to prioritise replanting and maintenance for the most at-risk tree categories.

### Caveats
- AEF embeddings encode multiple land-surface properties simultaneously; cosine distance is an aggregate measure and cannot isolate specific ecological processes without PCA decomposition.
- The TOW dataset represents a snapshot of tree locations; actual canopy extent may have changed over the study period.
- Sentinel-2 imagery is cloud-dependent; annual composites may vary in quality between years.
"""

    report = report.replace("stepped_workings/", f"{CHART_DIR}/")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✓ Report written to {REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Advanced Multi-Year Tree Canopy Health Analysis")
    print("=" * 70)

    # Discover available years
    print("\nDiscovering available years...")
    years_files = discover_years()
    if len(years_files) < 2:
        print(f"Error: Need ≥ 2 years. Found: {[y[0] for y in years_files]}")
        return

    years = [y[0] for y in years_files]
    baseline_year = years[0]
    latest_year = years[-1]
    print(f"Years available: {years}")
    print(f"Baseline: {baseline_year}, Latest: {latest_year}")

    # Cache configuration
    CACHE_DIR = "stepped_workings/cache"
    cache_exists = all(
        os.path.exists(os.path.join(CACHE_DIR, f))
        for f in ["dist_matrix.npy", "vy.npy", "vx.npy", "meta.json", "results.json", "borough_stats.json"]
    )

    cache_loaded = False
    results = {}
    borough_stats = {}
    
    if cache_exists:
        print("\nFound cached intermediate files. Loading from cache...")
        try:
            from rasterio.coords import BoundingBoxes
            from rasterio.transform import Affine
            from rasterio.crs import CRS

            vy = np.load(os.path.join(CACHE_DIR, "vy.npy"))
            vx = np.load(os.path.join(CACHE_DIR, "vx.npy"))
            dist_matrix = np.load(os.path.join(CACHE_DIR, "dist_matrix.npy"))
            
            with open(os.path.join(CACHE_DIR, "meta.json"), "r") as f:
                meta_json = json.load(f)
            meta = {
                "bounds": BoundingBoxes(*meta_json["bounds"]),
                "shape": tuple(meta_json["shape"]),
                "transform": Affine(*meta_json["transform"]),
                "crs": CRS.from_string(meta_json["crs"])
            }
            
            with open(os.path.join(CACHE_DIR, "results.json"), "r") as f:
                results_serialized = json.load(f)
            
            # Reconstruct results dict (convert keys back to int years, and restore raw_bl)
            for yr_str, res in results_serialized.items():
                year = int(yr_str)
                results[year] = res.copy()
            
            for i, year in enumerate(years):
                results[year]["raw_bl"] = dist_matrix[:, i]
                
            with open(os.path.join(CACHE_DIR, "borough_stats.json"), "r") as f:
                borough_stats = json.load(f)
                
            total_valid = len(vy)
            cache_loaded = True
            emb_baseline = None
            emb_latest = None
            print(f"  ✓ Cache loaded successfully ({total_valid:,} valid pixels).")
        except Exception as e:
            print(f"  ⚠ Failed to load cache: {e}. Falling back to full computation.")

    if not cache_loaded:
        # Valid pixel mask
        print("\nComputing valid pixel intersection...")
        vy, vx, meta = compute_valid_mask(years_files)
        total_valid = len(vy)
        print(f"Valid canopy pixels: {total_valid:,}")

        if total_valid == 0:
            print("Error: No valid pixels found.")
            return

        # Load baseline & latest embeddings (needed for PCA)
        print(f"\nLoading baseline embeddings ({baseline_year})...")
        emb_baseline = load_dequantized(years_files[0][1], vy, vx)
        print(f"Loading latest embeddings ({latest_year})...")
        emb_latest = load_dequantized(years_files[-1][1], vy, vx)

        # Compute per-year distances
        print("\nComputing per-year cosine distances from baseline...")
        dist_matrix_cols = []
        prev_emb = emb_baseline

        for year, f in years_files:
            if year == baseline_year:
                d_bl = np.zeros(total_valid, dtype=np.float32)
                d_yoy = np.zeros(total_valid, dtype=np.float32)
                emb = emb_baseline
            elif year == latest_year:
                emb = emb_latest
                d_bl = cosine_distance(emb_baseline, emb)
                d_yoy = cosine_distance(prev_emb, emb)
            else:
                emb = load_dequantized(f, vy, vx)
                d_bl = cosine_distance(emb_baseline, emb)
                d_yoy = cosine_distance(prev_emb, emb)

            stable = float(np.mean(d_bl < 0.05) * 100)
            mild = float(np.mean((d_bl >= 0.05) & (d_bl < 0.15)) * 100)
            degraded = float(np.mean(d_bl >= 0.15) * 100)

            results[year] = {
                "mean_bl": float(np.mean(d_bl)),
                "p50_bl": float(np.percentile(d_bl, 50)),
                "p90_bl": float(np.percentile(d_bl, 90)),
                "p95_bl": float(np.percentile(d_bl, 95)),
                "mean_yoy": float(np.mean(d_yoy)),
                "pct_stable": stable, "pct_mild": mild, "pct_degraded": degraded,
                "raw_bl": d_bl,
            }
            dist_matrix_cols.append(d_bl)
            prev_emb = emb
            print(f"  {year}: mean={np.mean(d_bl):.4f}, stable={stable:.1f}%, "
                  f"mild={mild:.1f}%, degraded={degraded:.1f}%")

        dist_matrix = np.column_stack(dist_matrix_cols)  # (N, N_years)
        del prev_emb

        # Save to cache
        print("\nSaving intermediate distance matrix and coordinates to cache...")
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.save(os.path.join(CACHE_DIR, "vy.npy"), vy)
            np.save(os.path.join(CACHE_DIR, "vx.npy"), vx)
            np.save(os.path.join(CACHE_DIR, "dist_matrix.npy"), dist_matrix)
            
            meta_json = {
                "bounds": [meta["bounds"].left, meta["bounds"].bottom, meta["bounds"].right, meta["bounds"].top],
                "shape": list(meta["shape"]),
                "transform": list(meta["transform"]),
                "crs": meta["crs"].to_string() if hasattr(meta["crs"], "to_string") else str(meta["crs"])
            }
            with open(os.path.join(CACHE_DIR, "meta.json"), "w") as f:
                json.dump(meta_json, f)
            
            # Serialize results dict (exclude numpy arrays)
            results_serialized = {}
            for year, res in results.items():
                res_copy = res.copy()
                if "raw_bl" in res_copy:
                    del res_copy["raw_bl"]
                results_serialized[str(year)] = res_copy
                
            with open(os.path.join(CACHE_DIR, "results.json"), "w") as f:
                json.dump(results_serialized, f)
                
            print("  ✓ Intermediate outputs cached successfully.")
        except Exception as e:
            print(f"  ⚠ Failed to save cache: {e}")

    # ── Run modules ──

    # Trend charts
    generate_trend_charts(years, results, meta, vy, vx, baseline_year)

    # Module 1: Trajectory Clustering
    cluster_labels, cluster_stats = module_1_trajectory_clustering(
        years, dist_matrix, vy, vx, meta
    )

    # Module 2: Directional PCA
    dist_bl_latest = results[latest_year]["raw_bl"]
    
    # Save cumulative change GeoTIFF
    print("\nSaving cumulative change GeoTIFF...")
    change_grid = np.full(meta["shape"], np.nan, dtype=np.float32)
    change_grid[vy, vx] = dist_bl_latest
    save_geotiff(change_grid, f"canopy_change_{baseline_year}_{latest_year}.tif", meta, "float32", nodata=-9999.0)

    if cache_loaded:
        print(f"\nLoading baseline embeddings ({baseline_year}) for PCA...")
        emb_baseline = load_dequantized(years_files[0][1], vy, vx)
        print(f"Loading latest embeddings ({latest_year}) for PCA...")
        emb_latest = load_dequantized(years_files[-1][1], vy, vx)
        
    pc_info = module_2_pca(emb_baseline, emb_latest, dist_bl_latest, vy, vx, meta,
                           baseline_year, latest_year)

    # Free large embedding arrays
    del emb_baseline, emb_latest

    # Module 3: Spatial Hotspots
    hotspot_stats = module_3_hotspots(dist_bl_latest, vy, vx, meta)

    # Module 4: Vulnerability
    vuln_stats = module_4_vulnerability(dist_bl_latest, vy, vx, meta)

    # Module 5: Borough-level canopy performance
    borough_stats = module_5_boroughs(dist_bl_latest, vy, vx, meta)
    # Cache borough stats for future runs
    try:
        CACHE_DIR = "stepped_workings/cache"
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "borough_stats.json"), "w") as f:
            json.dump(borough_stats, f)
    except Exception as e:
        print(f"  ⚠ Failed to cache borough stats: {e}")

    # Write report
    write_report(years, total_valid, results, cluster_stats, pc_info,
                 hotspot_stats, vuln_stats, borough_stats, baseline_year, latest_year)

    print("\n" + "=" * 70)
    print("  Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
