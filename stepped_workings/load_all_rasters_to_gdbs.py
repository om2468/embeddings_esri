import os
import sqlite3
import struct
import subprocess
import numpy as np
import rasterio
from load_raster_to_gdb import load_raster_to_gdb

def main():
    years = range(2017, 2026)
    
    for year in years:
        raster_path = f"london_aef_clipped_10m_{year}.tif"
        gdb_path = f"london_tow_embeddings_{year}.gdb"
        layer_name = f"London_TOW_Embeddings_{year}"
        
        if not os.path.exists(raster_path):
            print(f"Warning: Raster for year {year} not found at {raster_path}. Skipping.")
            continue
            
        if os.path.exists(gdb_path):
            print(f"GDB for year {year} already exists at {gdb_path}. Skipping to prevent redundant computation. Delete the directory if you wish to overwrite.")
            continue
            
        print(f"\n==========================================")
        print(f"Loading embeddings for year {year} to GDB...")
        print(f"==========================================")
        try:
            load_raster_to_gdb(
                raster_path=raster_path,
                gdb_path=gdb_path,
                layer_name=layer_name
            )
            print(f"Successfully processed and saved GDB for year {year}!")
        except Exception as e:
            print(f"Error processing year {year}: {e}")

if __name__ == "__main__":
    main()
