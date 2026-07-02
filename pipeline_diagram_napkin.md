# London TOW Canopy Health Pipeline — Stepped Diagram

A stepped pipeline diagram for napkin.ai. The diagram reads top-to-bottom in 5
numbered stages. Each stage lists its **inputs**, **processing steps**, and
**outputs**, plus a short note on the key technology or optimisation used.
Arrows flow from one stage's output into the next stage's input.

---

## Diagram Specification

**Title:** London Trees Outside Woodland (TOW) — AEF Satellite Embeddings Health Pipeline (2017–2025)

**Subtitle:** From remote cloud-optimised GeoTIFFs to per-parcel canopy health report, across 9 annual observations and 1.4 million canopy geometries.

**Style:** Stepped / waterfall layout. Each stage is a wide rounded box. Number badges on the left edge of each stage. Down-arrows connect stages. Inputs and outputs shown as small labelled chips inside each box. Key technology shown as a tag on the right edge of each box.

---

### Stage 1 — Source Data Acquisition

**Inputs**
- Google DeepMind Alpha Earth Foundations (AEF) 10m resolution 64-band satellite embeddings, hosted as remote Cloud-Optimised GeoTIFFs (~2.3 GB per year)
- `london_index.csv` — manifest mapping each year (2017–2025) to its remote tile URLs on `data.source.coop`
- London Trees Outside Woodland (TOW) vector polygons — 1.4M canopy geometries in Esri File Geodatabase `FR_TOW_V1_London.gdb`

**Processing steps**
1. Read `london_index.csv` and group tile paths by year.
2. Convert `s3://` URIs to HTTPS and prefix with GDAL `/vsicurl/` virtual filesystem paths.

**Outputs**
- Per-year list of remote COG tile URLs ready for streaming.

**Key technology:** GDAL Virtual File System (`/vsicurl/`)

---

### Stage 2 — Streaming, Reprojection & Clipping

**Inputs**
- Per-year remote COG tile URL lists (from Stage 1)
- TOW extent bounding box in EPSG:27700: `[503000, 155000, 562000, 201000]`
- London TOW vector polygons (from Stage 1)

**Processing steps**
1. **Generate static 10m binary TOW mask** once using `gdal_rasterize` with `OGR_ORGANIZE_POLYGONS=SKIP` to bypass expensive polygon ring nesting (rasterisation completes in <10 seconds for 1.4M geometries).
2. **Stream and warp** each year's remote AEF tiles with `gdalwarp`:
   - Reproject to EPSG:27700
   - Resample to 10m with nearest-neighbour
   - Merge all tiles for the year into a temporary local GeoTIFF
3. **Apply the binary TOW mask** with NumPy: zero out all non-canopy pixels and write a compressed final clipped GeoTIFF.
4. **Delete the temporary warped file** immediately to guard against low disk space.

**Outputs**
- `tow_mask_10m.tif` — static 10m binary canopy mask (generated once)
- `london_aef_clipped_10m_{year}.tif` — 64-band Int8 clipped embedding raster for each year (2017–2025)

**Key technology:** GDAL `gdalwarp` + `gdal_rasterize`, HTTP/2 multiplexing, 1 GB cache, range-merging

---

### Stage 3 — Dequantization & File Geodatabase Loading

**Inputs**
- `london_aef_clipped_10m_{year}.tif` clipped 64-band Int8 rasters (from Stage 2)

**Processing steps**
1. **Identify valid canopy pixels** by reading band 1 and masking out the `-128` NoData sentinel.
2. **Load all 64 bands** into memory (~120 MB per year for the London extent).
3. **Dequantize** each pixel's 64-band Int8 vector to Float32 using DeepMind's non-linear formula: `f(v) = (v / 127.5)² × sign(v)`, restoring values to the `[-1.0, 1.0]` physical range.
4. **Pack** each 64-float vector into a 256-byte little-endian binary BLOB using Python `struct.pack`.
5. **Insert** pixel rows (x, y, Embedding BLOB, WKT geometry) into a temporary SQLite database in 100k-row chunks.
6. **Export** to an Esri File Geodatabase per year via an OGR VRT wrapper and `ogr2ogr`.

**Outputs**
- `london_tow_embeddings_{year}.gdb` — one File Geodatabase per year containing a `pixel_embeddings` feature class with point geometry and a 256-byte `Embedding` BLOB field, consumable by ArcGIS Pro embedding tools.

**Key technology:** Non-linear dequantization, `struct.pack` little-endian Float32, SQLite → OGR VRT → File GDB

---

### Stage 4 — Multi-Year Canopy Health Analysis

**Inputs**
- `london_aef_clipped_10m_{year}.tif` clipped embedding rasters for 2017–2025 (from Stage 2)
- `stepped_workings/cache/` — intermediate results cache (`dist_matrix.npy`, `vx.npy`, `vy.npy`, `meta.json`, `results.json`) to bypass heavy I/O on re-runs
- London TOW attribute vectors: `Woodland_Type` and `MEANHT` canopy height, rasterised onto the same 10m grid

**Processing steps**
1. **Dequantize** all years and extract valid canopy pixels (5.64M pixels).
2. **Compute cosine distance** of each pixel's embedding from its 2017 baseline for every year, yielding a 9-dimensional trajectory vector per pixel. Cosine distance is chosen over Euclidean to filter illumination and atmospheric artefacts.
3. **Cache** the distance matrix and coordinate indices to `stepped_workings/cache/` (248 MB total) so subsequent re-runs load in <1 second.
4. **Run four parallel analysis modules:**
   - **Module 1 — Trajectory Clustering:** Min-Max normalise each trajectory, run KMeans elbow analysis over k=2..8, select optimal k via second-derivative test, sort centroids by 2025 value, and label archetypes (Stable, Minor Variation, Gradual Decline, Drought Stress & Recovery, Sudden Loss).
   - **Module 2 — Directional PCA:** Compute delta vectors between 2017 baseline and 2025 latest embeddings, run PCA on the centred delta matrix, and extract PC1–PC3 as orthogonal ecological change axes.
   - **Module 3 — Spatial Hotspot Detection:** Smooth the cumulative cosine distance grid with a Gaussian filter (σ=3, ~30m window), compute Z-scores, and flag degradation hotspots (Z > 2.0) and resilience coldspots (Z < -1.0).
   - **Module 4 — Attribute Vulnerability:** Rasterise `Woodland_Type` and `MEANHT` onto the 10m grid, cross-tabulate Stable / Stressed / Degraded proportions against woodland type × height class to identify the most at-risk sub-populations.
5. **Generate charts** (PNG) and GeoTIFFs into `report_images/`.
6. **Assemble** `tree_health_report.md` with summary tables, embedded charts, and methodology notes.

**Outputs**
- `tree_health_report.md` and `tree_health_report_standalone.html` — London-wide canopy health report
- `report_images/` — trajectory cluster map, PCA spatial maps, hotspot map, vulnerability heatmap, trend lines, pie charts

**Key technology:** NumPy vectorised cosine distance, scikit-learn KMeans + PCA, SciPy Gaussian filter, Z-score statistics

---

### Stage 5 — TFL / GLA Parcel-Level Health & Postgres Writeback

**Inputs**
- London-wide clipped AEF GeoTIFFs 2018–2025 (from Stage 2)
- Postgres table `tfl.tfl_par_land` — 12,670 TFL/GLA land parcels in EPSG:4326
- London boroughs GeoJSON for borough attribution
- `stepped_workings/cache/tfl_par_land_2018_2025/` — parcel-specific intermediate cache

**Processing steps**
1. **Load TFL/GLA parcels** from Postgres via `psycopg` and reproject from EPSG:4326 to EPSG:27700.
2. **Rasterise parcel geometries** onto the identical 10m AEF grid and intersect with the TOW canopy mask, yielding 132,787 valid canopy pixels (~1,328 ha).
3. **Compute cosine distance** of each pixel from the 2018 baseline across all 8 years.
4. **Reuse the four analysis modules** from Stage 4 (trajectory clustering, PCA, hotspots, vulnerability) at parcel scale.
5. **Aggregate per-parcel metrics:** pixel count, mean distance, % stable / mild / degraded, hotspot pixel count and hectares, dominant cluster, interpretation text, and borough name.
6. **Write back** 10 analysis columns to `tfl.tfl_par_land` in Postgres (`aef_18_25_px_count`, `aef_18_25_mean_dist`, `aef_18_25_pct_stable`, `aef_18_25_pct_mild`, `aef_18_25_pct_degraded`, `aef_18_25_hotspot_px`, `aef_18_25_hotspot_ha`, `aef_18_25_dom_cluster`, `aef_18_25_interp`, `aef_18_25_borough`).
7. **Generate** `tfl_par_land_health_report.md` and the standalone HTML variant with parcel-level trend tables, trajectory centroids, PCA biplots, and spatial maps.

**Outputs**
- `tfl_par_land_health_report.md` and `tfl_par_land_health_report_standalone.html` — TFL/GLA parcel canopy health report
- Updated `tfl.tfl_par_land` Postgres table with 10 embedding-derived analysis columns for downstream GIS queries

**Key technology:** psycopg Postgres connection, geopandas reproject, rasterio centroid-aligned rasterisation, parcel-level cache bypass

---

## Connection Arrows

```
Stage 1  ──┐
           ▼
        Stage 2 ──┬──────────────►  Stage 3  (per-year GDB load)
                  │
                  └──────────────►  Stage 4  (London-wide analysis)
                                      │
                                      ▼
                                   Stage 5  (TFL parcel analysis + Postgres writeback)
```

- **Stage 1 → Stage 2:** remote tile URLs + TOW polygons feed streaming and clipping.
- **Stage 2 → Stage 3:** clipped 64-band GeoTIFFs are dequantized and packed into per-year File Geodatabases.
- **Stage 2 → Stage 4:** the same clipped GeoTIFFs feed the multi-year cosine distance and clustering pipeline.
- **Stage 4 → Stage 5:** the analysis module functions are reused at parcel scale, and parcel results are written back to Postgres.

---

## Side-Panel: Parallel Text-Embedding Sub-Pipeline (FME)

A smaller parallel pipeline exists for reference / tooling comparison, documented in `fme_embeddings_blob_guide.md` and `generate_gdb_embeddings.py`. It is not part of the satellite canopy pipeline but follows the same BLOB-packing pattern:

1. **HTTPCaller** — POST text to Ollama (`nomic-embed-text`) local API or Nomic Cloud API.
2. **JSONExtractor** — parse the 768-dimensional float array from the JSON response.
3. **PythonCaller** — `struct.pack` the float list into a 3072-byte little-endian BLOB.
4. **OpenFileGDB Writer** — write point features with a `binary` (BLOB) attribute into a File Geodatabase, consumable by ArcGIS Pro's `Extract Embeddings To Fields` geoprocessing tool.

Output: `nomic_embeddings_example.gdb` — example File GDB with 768-dim text-embedding BLOBs.

---

## Global Optimisations & Safeguards (annotation layer)

These can be shown as small callout badges attached to the relevant stages:

- **Polygon skip** (`OGR_ORGANIZE_POLYGONS=SKIP`) — Stage 2 — rasterise 1.4M polygons in <10 s.
- **Streaming COG config** (`GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES`, `GDAL_HTTP_MULTIPLEX=YES`, `GDAL_CACHEMAX=1024`) — Stage 2 — avoid full 2.3 GB downloads.
- **Non-linear dequantization** `((v/127.5)² × sign(v))` — Stage 3 — preserve DeepMind's Int8 → Float32 mapping.
- **Intermediate cache** (`stepped_workings/cache/`) — Stage 4 & 5 — reduce re-run from ~45 min to <1 s.
- **Cosine distance over Euclidean** — Stage 4 & 5 — illumination-invariant change metric.
- **Gaussian smoothing (σ=3)** — Stage 4 & 5 — mitigate Sentinel-2 5–10m geolocation error before Z-score hotspots.
- **Deterministic seeds** (`random_state=42`) — Stage 4 — reproducible KMeans and PCA.
- **Low-disk guard** — Stage 2 — delete temp warped GeoTIFFs immediately after masking.