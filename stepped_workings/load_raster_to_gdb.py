import os
import sqlite3
import struct
import subprocess
import numpy as np
import rasterio

def dequantize_embeddings(val):
    """Dequantizes Int8 embeddings to Float32 using DeepMind's formula."""
    nodata_mask = (val == -128)
    val_clean = np.where(nodata_mask, 0, val)
    normalized = val_clean.astype(np.float32) / 127.5
    dequantized = (normalized ** 2) * np.sign(val_clean)
    dequantized[nodata_mask] = 0.0
    return dequantized

def load_raster_to_gdb(raster_path, gdb_path, layer_name):
    """Loads valid pixels from a clipped 64-band AEF raster into a File GDB BLOB field."""
    print(f"Opening raster: {raster_path}...")
    sqlite_db = "temp_pixel_embeddings.db"
    if os.path.exists(sqlite_db):
        os.remove(sqlite_db)

    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE pixel_embeddings (
            pixel_id INTEGER PRIMARY KEY AUTOINCREMENT,
            x REAL,
            y REAL,
            Embedding BLOB,
            wkt_geom TEXT
        )
    """)

    with rasterio.open(raster_path) as src:
        # Read the first band to identify valid (non-NoData) pixels
        print("Reading band 1 to identify valid canopy pixels...")
        band1 = src.read(1)
        valid_y, valid_x = np.where(band1 != -128)
        total_pixels = len(valid_y)
        print(f"Found {total_pixels} valid pixels to process.")

        # Read all 64 bands for the entire raster into memory (only ~120MB for 5900x4600 Int8)
        print("Reading all 64 bands...")
        all_bands = src.read()  # Shape: (64, height, width)

        print("Processing pixels in chunks...")
        insert_data = []
        chunk_size = 100000

        for i in range(total_pixels):
            y_idx = valid_y[i]
            x_idx = valid_x[i]

            # Get spatial coordinates of pixel center
            px, py = src.xy(y_idx, x_idx)

            # Get 64-band value for this pixel
            pixel_val = all_bands[:, y_idx, x_idx]
            
            # Dequantize
            dequant = dequantize_embeddings(pixel_val)
            
            # Pack as 64 float32 little-endian floats (256 bytes)
            packed = struct.pack("<64f", *dequant)
            
            wkt = f"POINT({px} {py})"
            insert_data.append((px, py, sqlite3.Binary(packed), wkt))

            if len(insert_data) >= chunk_size:
                cursor.executemany(
                    "INSERT INTO pixel_embeddings (x, y, Embedding, wkt_geom) VALUES (?, ?, ?, ?)",
                    insert_data
                )
                insert_data = []
                print(f"  Processed {i + 1} / {total_pixels} pixels...")

        # Insert remaining data
        if insert_data:
            cursor.executemany(
                "INSERT INTO pixel_embeddings (x, y, Embedding, wkt_geom) VALUES (?, ?, ?, ?)",
                insert_data
            )
            
        conn.commit()
        conn.close()

    print("Temporary SQLite database populated.")

    # Create VRT file
    vrt_name = "temp_pixel_embeddings.vrt"
    vrt_content = f"""<OGRVRTDataSource>
    <OGRVRTLayer name="pixel_embeddings">
        <SrcDataSource>{sqlite_db}</SrcDataSource>
        <SrcLayer>pixel_embeddings</SrcLayer>
        <GeometryType>wkbPoint</GeometryType>
        <LayerSRS>EPSG:27700</LayerSRS>
        <GeometryField encoding="WKT" field="wkt_geom"/>
    </OGRVRTLayer>
</OGRVRTDataSource>"""

    with open(vrt_name, "w") as f:
        f.write(vrt_content)

    # Convert to File Geodatabase
    if os.path.exists(gdb_path):
        # OpenFileGDB driver requires deleting folder to overwrite
        subprocess.run(["rm", "-rf", gdb_path])

    print(f"Converting SQLite table to File Geodatabase ({gdb_path})...")
    cmd = [
        "ogr2ogr",
        "-f", "OpenFileGDB",
        gdb_path,
        vrt_name,
        "-nln", layer_name,
        "-select", "Embedding"
    ]
    subprocess.run(cmd, check=True)

    # Cleanup temp files
    if os.path.exists(sqlite_db):
        os.remove(sqlite_db)
    if os.path.exists(vrt_name):
        os.remove(vrt_name)

    print(f"Success! Created {gdb_path} with layer {layer_name}")

if __name__ == "__main__":
    # Example usage for 2025
    load_raster_to_gdb(
        raster_path="london_aef_clipped_10m_2025.tif",
        gdb_path="london_tow_embeddings_2025.gdb",
        layer_name="London_TOW_Embeddings_2025"
    )
