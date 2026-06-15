import os
import subprocess
import csv
from collections import defaultdict
import numpy as np
import rasterio

# Bounding box for London Trees Outside Woodland (TOW) in EPSG:27700
TOW_EXTENT = [503000, 155000, 562000, 201000]

def read_london_index(csv_path="london_index.csv"):
    """Reads the local index file and groups tile paths by year."""
    tiles_by_year = defaultdict(list)
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["year"])
            # Convert s3:// paths to HTTPS paths
            path = row["path"].replace(
                "s3://us-west-2.opendata.source.coop/",
                "https://data.source.coop/"
            )
            # Prepend vsicurl
            tiles_by_year[year].append(f"/vsicurl/{path}")
    return tiles_by_year

def main():
    gdb_path = "./FR_TOW_V1_London/FR_TOW_V1_London.gdb"
    mask_tif = "tow_mask_10m.tif"
    
    # Configure GDAL environment variables for high performance over HTTP (vsicurl)
    env = os.environ.copy()
    env["OGR_ORGANIZE_POLYGONS"] = "SKIP"  # Bypass expensive polygon ring nesting
    env["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
    env["GDAL_HTTP_MULTIPLEX"] = "YES"
    env["GDAL_CACHEMAX"] = "1024"
    env["GDAL_NUM_THREADS"] = "ALL_CPUS"
    env["VSI_CACHE"] = "TRUE"
    env["VSI_CACHE_SIZE"] = "104857600" # 100 MB cache
    env["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    env["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = "tif,tiff"

    # Step 1: Generate the static 10m binary mask once (highly optimized native GDAL tool!)
    if not os.path.exists(mask_tif):
        print(f"Creating static 10m TOW binary mask: {mask_tif}...")
        rasterize_cmd = [
            "gdal_rasterize",
            "-tr", "10", "10",
            "-te", str(TOW_EXTENT[0]), str(TOW_EXTENT[1]), str(TOW_EXTENT[2]), str(TOW_EXTENT[3]),
            "-ot", "Byte",
            "-init", "0",
            "-burn", "1",
            "-at",  # CUTLINE_ALL_TOUCHED=TRUE equivalent: "partially contains or contains"
            "-co", "COMPRESS=DEFLATE",
            "-l", "FR_TOW_V1_London",
            gdb_path,
            mask_tif
        ]
        subprocess.run(rasterize_cmd, env=env, check=True)
        print("TOW mask created successfully.")
    else:
        print("TOW mask already exists. Reusing it.")

    # Read the local index to find tiles for each year
    tiles_by_year = read_london_index()
    
    # Read the mask array once
    with rasterio.open(mask_tif) as src_mask:
        mask_arr = src_mask.read(1)

    # Loop over all available years
    for year in sorted(tiles_by_year.keys()):
        temp_tif = f"london_aef_10m_{year}_temp.tif"
        final_tif = f"london_aef_clipped_10m_{year}.tif"
        
        if os.path.exists(final_tif):
            print(f"Clipped GeoTIFF for year {year} already exists. Skipping...")
            continue
            
        print(f"\n--- Processing Year {year} ---")
        tiles = tiles_by_year[year]
        
        # Warp remote AEF COGs to temporary local 10m GeoTIFF (no cutline = fast!)
        print(f"Warping tiles for {year} to {temp_tif}...")
        warp_cmd = [
            "gdalwarp",
            "-t_srs", "EPSG:27700",
            "-te", str(TOW_EXTENT[0]), str(TOW_EXTENT[1]), str(TOW_EXTENT[2]), str(TOW_EXTENT[3]),
            "-tr", "10", "10",
            "-r", "near",
            "-co", "COMPRESS=DEFLATE",
            "-overwrite"
        ] + tiles + [temp_tif]
        
        subprocess.run(warp_cmd, env=env, check=True)
        print(f"Temporary GeoTIFF for {year} created.")

        # Apply the static binary mask using fast NumPy operations
        print(f"Applying mask and saving final clipped GeoTIFF: {final_tif}...")
        with rasterio.open(temp_tif) as src_aef:
            meta = src_aef.meta.copy()
            meta.update({
                "nodata": -128,
                "compress": "deflate"
            })
            
            with rasterio.open(final_tif, "w", **meta) as dest:
                for b in range(1, src_aef.count + 1):
                    band_data = src_aef.read(b)
                    band_data[mask_arr == 0] = -128
                    dest.write(band_data, b)
                    
                    if b % 16 == 0:
                        print(f"  Processed band {b} / {src_aef.count}...")

        # Clean up temporary warped file
        if os.path.exists(temp_tif):
            os.remove(temp_tif)
            
        print(f"Completed processing for year {year}!")

    # Step 4: Cleanup static mask
    if os.path.exists(mask_tif):
        os.remove(mask_tif)
        
    print("\nAll years processed successfully!")

if __name__ == "__main__":
    main()
