import os
import math
import urllib.request
import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from pyproj import Transformer
import rasterio
from PIL import Image

def latlon_to_tile(lat, lon, zoom):
    """Converts lat/lon to OSM tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def tile_to_latlon(x, y, zoom):
    """Converts OSM tile coordinates back to lat/lon of top-left corner."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon

def get_osm_basemap(wgs84_bounds, zoom=18):
    """Downloads and stitches OSM tiles for a given WGS84 bounding box."""
    min_lat, min_lon, max_lat, max_lon = wgs84_bounds
    
    # Get tile bounds
    x_min, y_max = latlon_to_tile(min_lat, min_lon, zoom)
    x_max, y_min = latlon_to_tile(max_lat, max_lon, zoom)
    
    # Ensure min/max ordering
    x_start, x_end = min(x_min, x_max), max(x_min, x_max)
    y_start, y_end = min(y_min, y_max), max(y_min, y_max)
    
    # Width and height of stitched image in tiles
    width_tiles = x_end - x_start + 1
    height_tiles = y_end - y_start + 1
    
    print(f"Downloading {width_tiles}x{height_tiles} = {width_tiles * height_tiles} OSM tiles...")
    
    # Create empty image for stitching
    tile_size = 256
    stitched_img = Image.new("RGB", (width_tiles * tile_size, height_tiles * tile_size))
    
    # Set headers to comply with OSM Tile Usage Policy
    headers = {'User-Agent': 'LondonTreesOutsideWoodlandCanopyHealthAnalysis/1.0 (cherrytian@example.com)'}
    
    for x in range(x_start, x_end + 1):
        for y in range(y_start, y_end + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req) as response:
                    tile_img = Image.open(response)
                    px = (x - x_start) * tile_size
                    py = (y - y_start) * tile_size
                    stitched_img.paste(tile_img, (px, py))
            except Exception as e:
                print(f"Failed to fetch tile {zoom}/{x}/{y}: {e}")
                # Paste white tile on error
                px = (x - x_start) * tile_size
                py = (y - y_start) * tile_size
                stitched_img.paste(Image.new("RGB", (tile_size, tile_size), (255, 255, 255)), (px, py))
                
    # Calculate geographical bounds of the stitched image
    img_max_lat, img_min_lon = tile_to_latlon(x_start, y_start, zoom)
    img_min_lat, img_max_lon = tile_to_latlon(x_end + 1, y_end + 1, zoom)
    
    return stitched_img, (img_min_lon, img_min_lat, img_max_lon, img_max_lat)

def crop_and_align_raster(raster_path, bbox_bng):
    """Crops a raster to a given BNG bounding box and returns the data and extent."""
    with rasterio.open(raster_path) as src:
        # Get pixel window
        from rasterio.windows import from_bounds
        window = from_bounds(bbox_bng[0], bbox_bng[1], bbox_bng[2], bbox_bng[3], src.transform)
        data = src.read(window=window, boundless=True, fill_value=-128)
        # Calculate window transform
        win_transform = src.window_transform(window)
        # Bounds of window
        # Bounds of window
        w_bounds = src.window_bounds(window)
        return data, w_bounds

def main():
    # Target Point (Easting/Northing)
    lat, lon = 51.54730768282329, -0.20424984819019443
    transformer_to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    
    qx, qy = transformer_to_bng.transform(lon, lat)
    
    # 200m x 200m bounding box in BNG
    half_size = 100.0
    xmin, xmax = qx - half_size, qx + half_size
    ymin, ymax = qy - half_size, qy + half_size
    bbox_bng = [xmin, ymin, xmax, ymax]
    
    # Convert corners of bounding box back to WGS84 for OSM
    wgs_min_lon, wgs_min_lat = transformer_to_wgs84.transform(xmin, ymin)
    wgs_max_lon, wgs_max_lat = transformer_to_wgs84.transform(xmax, ymax)
    wgs84_bounds = [wgs_min_lat, wgs_min_lon, wgs_max_lat, wgs_max_lon]
    
    # 1. Fetch OSM Basemap
    osm_img, osm_extent_wgs84 = get_osm_basemap(wgs84_bounds, zoom=18)
    
    # Project OSM image to BNG for correct overlay alignment
    # (Since zoom 18 is small and near 51N, we can do a simple affine/projective warp 
    # or let matplotlib handle it by converting coordinates. Even better, we can 
    # reproject OSM bounds to BNG and display the image with BNG extent).
    osm_xmin, osm_ymin = transformer_to_bng.transform(osm_extent_wgs84[0], osm_extent_wgs84[1])
    osm_xmax, osm_ymax = transformer_to_bng.transform(osm_extent_wgs84[2], osm_extent_wgs84[3])
    osm_extent_bng = [osm_xmin, osm_xmax, osm_ymin, osm_ymax]
    
    # 2. Load and crop TOW GDB canopy polygons
    print("Loading TOW GDB canopy polygons...")
    gdb_path = "FR_TOW_V1_London/FR_TOW_V1_London.gdb"
    gdf = gpd.read_file(gdb_path, bbox=(xmin, ymin, xmax, ymax))
    
    # 3. Load PCA components for False Color Representation
    print("Cropping PCA rasters...")
    pc1_path = "report_images/canopy_pca_pc1.tif"
    pc2_path = "report_images/canopy_pca_pc2.tif"
    pc3_path = "report_images/canopy_pca_pc3.tif"
    
    pc1, r_bounds = crop_and_align_raster(pc1_path, bbox_bng)
    pc2, _ = crop_and_align_raster(pc2_path, bbox_bng)
    pc3, _ = crop_and_align_raster(pc3_path, bbox_bng)
    
    # Squeeze to 2D
    pc1 = pc1[0]
    pc2 = pc2[0]
    pc3 = pc3[0]
    
    # Mask out NoData (-128)
    mask = (pc1 == -128) | (pc2 == -128) | (pc3 == -128)
    
    # Normalize to [0, 1] for visualization
    def normalize(band):
        b_min, b_max = band[band != -128].min(), band[band != -128].max()
        if b_max - b_min > 0:
            norm = (band - b_min) / (b_max - b_min)
        else:
            norm = np.zeros_like(band)
        norm[band == -128] = 0
        return norm
        
    pc1_norm = normalize(pc1)
    pc2_norm = normalize(pc2)
    pc3_norm = normalize(pc3)
    
    # Stack into RGB False Colour Image
    false_color_img = np.dstack((pc1_norm, pc2_norm, pc3_norm))
    
    # 4. Generate Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), facecolor='white')
    
    # Panel 1: OSM + Canopy Polygons
    ax = axes[0]
    ax.imshow(osm_img, extent=[osm_extent_bng[0], osm_extent_bng[1], osm_extent_bng[2], osm_extent_bng[3]])
    
    # Plot canopy polygons
    if len(gdf) > 0:
        gdf.plot(ax=ax, facecolor='none', edgecolor='#2ecc71', linewidth=2, label='TOW Canopy')
        
    # Plot target point
    ax.scatter([qx], [qy], color='#e74c3c', marker='*', s=200, zorder=5, label='Query Location')
    
    ax.set_title("OSM Base & Canopy Outline", fontsize=12, fontweight='bold')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Easting (BNG)")
    ax.set_ylabel("Northing (BNG)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    # Panel 2: False Colour Embedding Space + OSM Overlay
    ax2 = axes[1]
    ax2.imshow(osm_img, extent=[osm_extent_bng[0], osm_extent_bng[1], osm_extent_bng[2], osm_extent_bng[3]], alpha=0.6)
    
    # Display False Color Raster with Transparency
    extent_raster = [r_bounds[0], r_bounds[2], r_bounds[1], r_bounds[3]]
    # Apply alpha mask for no data values
    alpha = np.ones(pc1.shape) * 0.7  # 70% opacity for data
    alpha[mask] = 0.0                # 0% opacity for no-data
    false_color_rgba = np.dstack((false_color_img, alpha))
    
    ax2.imshow(false_color_rgba, extent=extent_raster)
    
    if len(gdf) > 0:
        gdf.plot(ax=ax2, facecolor='none', edgecolor='white', linewidth=1, linestyle='--', alpha=0.7)
        
    ax2.scatter([qx], [qy], color='white', marker='*', s=200, zorder=5, edgecolor='black', label='Query Location')
    
    ax2.set_title("AEF Embedding Space False Colour (PCA RGB)", fontsize=12, fontweight='bold')
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel("Easting (BNG)")
    ax2.set_ylabel("Northing (BNG)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    plt.suptitle("Local Canopy Map & Embedding Feature Analysis", fontsize=14, fontweight='bold', y=0.96)
    plt.tight_layout()
    
    output_path = "report_images/local_canopy_map.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Successfully generated local map at {output_path}")

    # 5. Append Map to Markdown Report
    print("Appending map to markdown report...")
    with open("local_tree_report.md", "r") as f:
        content = f.read()
        
    # Check if section already exists, if so overwrite/replace
    map_section = """
## 4. Spatial Map & Geographic Context

The maps below show the spatial layout of your location. The left panel shows the OpenStreetMap (OSM) basemap overlaid with the green boundaries of the **Trees Outside Woodland (TOW)** canopy. The right panel overlays the **AEF Embedding Space False Colour** (derived from the first three PCA components of the 64-band Sentinel-2 embeddings, mapped to Red, Green, and Blue), highlighting the local canopy characteristics and variations.

![Local Canopy Map](report_images/local_canopy_map.png)
"""
    if "## 4. Spatial Map" in content:
        # Strip out old section if it existed
        content = content.split("## 4. Spatial Map")[0]
        
    content += map_section
    
    with open("local_tree_report.md", "w") as f:
        f.write(content)
    print("Markdown report updated.")

if __name__ == "__main__":
    main()
