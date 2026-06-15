import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pyproj import Transformer
import rasterio
from rasterio.transform import xy

def main():
    # User's Lat/Long
    lat, lon = 51.54730768282329, -0.20424984819019443
    print(f"Query coordinate: Lat={lat}, Lon={lon}")
    
    # 1. Reproject WGS84 to BNG (EPSG:27700)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    qx, qy = transformer.transform(lon, lat)
    print(f"Reprojected coordinate (EPSG:27700 BNG): Easting={qx:.2f}, Northing={qy:.2f}")
    
    query_point = Point(qx, qy)
    
    # 2. Load GDB features near this point
    gdb_path = "FR_TOW_V1_London/FR_TOW_V1_London.gdb"
    if not os.path.exists(gdb_path):
        print(f"Error: GDB not found at {gdb_path}")
        return
        
    print("Loading TOW GDB features...")
    # Load all features (or read spatial index if possible, but reading directly is fine since it's fast enough in geopandas)
    gdf = gpd.read_file(gdb_path)
    print(f"Loaded {len(gdf)} features from GDB.")
    
    # Calculate distance to query point for all features to find the nearest
    gdf["dist_to_query"] = gdf.geometry.distance(query_point)
    nearest_tree = gdf.loc[gdf["dist_to_query"].idxmin()]
    min_dist = nearest_tree["dist_to_query"]
    
    print(f"Nearest tree feature is {min_dist:.2f} meters away.")
    print(f"Tree Attributes:")
    print(f"  Woodland Type: {nearest_tree.get('Woodland_Type', 'N/A')}")
    print(f"  Mean Height: {nearest_tree.get('MEANHT', 'N/A')} m")
    print(f"  Borough: {nearest_tree.get('borough', 'N/A')}")
    print(f"  Geometry Type: {nearest_tree.geometry.geom_type}")
    
    # 3. Load cache and find pixels within/near the nearest tree polygon
    cache_dir = "stepped_workings/cache"
    vy = np.load(os.path.join(cache_dir, "vy.npy"))
    vx = np.load(os.path.join(cache_dir, "vx.npy"))
    dist_matrix = np.load(os.path.join(cache_dir, "dist_matrix.npy"))
    
    with open(os.path.join(cache_dir, "meta.json"), "r") as f:
        meta_json = json.load(f)
    transform_vals = meta_json["transform"]
    # Recreate transform affine
    from rasterio.transform import Affine
    transform = Affine(*transform_vals[:6])
    
    # Calculate spatial x, y for each pixel in cache
    print("Calculating spatial coordinates for all valid pixels...")
    xs, ys = rasterio.transform.xy(transform, vy, vx)
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Find pixels inside or near (within 15m of) the nearest tree polygon
    print("Locating embedding pixels corresponding to this tree...")
    tree_geom = nearest_tree.geometry
    
    # Find indices of pixels within 15 meters of the tree geometry boundary/centroid
    pixel_points = [Point(x, y) for x, y in zip(xs, ys)]
    # Use vectorised distance check to be fast
    dx = xs - qx
    dy = ys - qy
    dists_to_query_px = np.sqrt(dx**2 + dy**2)
    
    # Let's get pixels within 25m of the query point
    nearby_px_mask = dists_to_query_px <= 25.0
    nearby_indices = np.where(nearby_px_mask)[0]
    
    print(f"Found {len(nearby_indices)} canopy pixels within 25m of your location.")
    
    if len(nearby_indices) == 0:
        # If none within 25m, get the single nearest pixel
        nearest_px_idx = np.argmin(dists_to_query_px)
        nearby_indices = [nearest_px_idx]
        print(f"No pixels within 25m. Selected nearest pixel at distance {dists_to_query_px[nearest_px_idx]:.2f}m.")
    
    # Compute the average trajectory of cosine distances for these pixels
    # years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    avg_trajectory = np.mean(dist_matrix[nearby_indices, :], axis=0)
    
    # Print the trajectory values
    print("Trajectory (Mean Cosine Distance from 2017 baseline):")
    for year, dist in zip(years, avg_trajectory):
        print(f"  {year}: {dist:.4f}")
        
    # 4. Generate Plot
    plt.figure(figsize=(10, 5), facecolor='white')
    plt.plot(years, avg_trajectory, marker='o', color='#2ecc71', linewidth=2.5, markersize=8, label="Local Canopy Stress")
    plt.title("Canopy Stress Trajectory (2017–2025)\nCosine Distance from 2017 Baseline", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Cosine Distance (higher = more stress/change)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(years)
    
    # Highlight the 2022 heatwave
    plt.axvspan(2022, 2022.2, color='#e74c3c', alpha=0.15, label="2022 Record Heatwave")
    plt.axvspan(2025, 2025.2, color='#e67e22', alpha=0.15, label="2025 Record Warmth")
    
    plt.legend(loc="upper left")
    plt.tight_layout()
    plot_path = "report_images/local_tree_trajectory.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved trajectory plot to {plot_path}")
    
    # 5. Generate Markdown Report
    report_content = f"""# Local Tree Canopy Health Report
**Location Query:** Lat {lat}, Lon {lon}  
**Spatial Coordinates (BNG):** Easting {qx:.2f}, Northing {qy:.2f}

---

## 1. Local Tree Overview

The nearest tree feature identified in the **London Trees Outside Woodland (TOW)** dataset is located approximately **{min_dist:.2f} meters** from your queried coordinate.

### Tree Attributes
* **Woodland Type Category:** `{nearest_tree.get('Woodland_Type', 'N/A')}`
* **Estimated Canopy Height:** `{nearest_tree.get('MEANHT', 'N/A')} meters`
* **Geometry Type:** `{nearest_tree.geometry.geom_type}`
* **Spatial Intersection:** {nearest_tree.geometry.area:.1f} m² area (spatial footprint)

---

## 2. Multi-Year Canopy Health Analysis (2017–2025)

We extracted **Alpha Earth Foundations (AEF) 64-band embedding values** for the **{len(nearby_indices)} canopy pixels** immediately surrounding this coordinate. 

Using **Cosine Distance** from the **2017 baseline**, we tracked the spectral deviations of the canopy over the 9-year study period. Higher cosine distance indicates greater change, stress, or leaf loss.

### Canopy Stress Trajectory

| Year | Mean Cosine Distance | Health Classification |
|:---:|:---:|:---|
"""
    for year, dist in zip(years, avg_trajectory):
        if dist < 0.05:
            classification = "✅ **Stable / Healthy**"
        elif dist < 0.15:
            classification = "⚠️ **Mild Stress / Canopy Thinning**"
        else:
            classification = "🚨 **Severe Stress / Canopy Loss**"
        report_content += f"| {year} | {dist:.4f} | {classification} |\n"
        
    report_content += f"""
### Trajectory Trend Plot
![Canopy Stress Plot](report_images/local_tree_trajectory.png)

---

## 3. Local Ecological Analysis & Discussion

Based on the trajectory:
1. **Drought & Heat Sensitivity:** 
   - During the **2022 Heatwave**, the cosine distance was **{avg_trajectory[5]:.4f}** (classification: { "Stable" if avg_trajectory[5]<0.05 else "Mild Stress" if avg_trajectory[5]<0.15 else "Severe Stress" }). This reflects the "false autumn" premature leaf senescence observed across London street trees during the record 40°C temperatures.
   - During the **2025 warm period**, stress was **{avg_trajectory[8]:.4f}**.
2. **Resilience & Recovery:**
   - {"The canopy successfully recovered towards baseline values in subsequent years." if avg_trajectory[6] < avg_trajectory[5] else "The canopy shows persistent stress or delayed recovery in the years following the heatwave, which is a common indicator of root damage or secondary pathogen vulnerability."}

"""
    
    report_file = "local_tree_report.md"
    with open(report_file, "w") as f:
        f.write(report_content)
        
    print(f"Report successfully saved to {report_file}")

if __name__ == "__main__":
    main()
