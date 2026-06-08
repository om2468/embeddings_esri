import os
import sqlite3
import struct
import json
import urllib.request
import subprocess
import random
import math

# Sample dataset
DATASET = [
    {
        "place_name": "Redlands Head Office",
        "description": "The main corporate headquarters of Esri in Redlands, California. Innovation center for GIS technology.",
        "lon": -117.19567,
        "lat": 34.05609
    },
    {
        "place_name": "Redlands East Campus",
        "description": "Esri campus building focusing on developer technologies, support services, and spatial analysis libraries.",
        "lon": -117.19123,
        "lat": 34.05751
    },
    {
        "place_name": "Redlands City Hall",
        "description": "Redlands city administration building where local government decisions and civic planning take place.",
        "lon": -117.18182,
        "lat": 34.05556
    },
    {
        "place_name": "University of Redlands",
        "description": "A private liberal arts university in Redlands, known for its strong environmental studies and spatial programs.",
        "lon": -117.16333,
        "lat": 34.06278
    }
]

EMBEDDING_DIM = 768

def get_ollama_embeddings(texts, host="http://localhost:11434", model="nomic-embed-text"):
    """
    Fetches embeddings locally using Ollama's /api/embed API endpoint.
    """
    print(f"Requesting embeddings for {len(texts)} texts from local Ollama ({model})...")
    url = f"{host.rstrip('/')}/api/embed"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "input": texts
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["embeddings"]
    except Exception as e:
        print(f"Ollama local endpoint failed or not running: {e}")
        return None

def get_nomic_embeddings(texts, api_key):
    """
    Fetches real Nomic text embeddings using the hosted API.
    """
    print(f"Requesting embeddings for {len(texts)} texts from Nomic Cloud API...")
    url = "https://api-atlas.nomic.ai/v1/embedding/text"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "nomic-embed-text-v1.5",
        "texts": texts,
        "task_type": "search_document"
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["embeddings"]
    except Exception as e:
        print(f"Error fetching embeddings from Nomic Cloud: {e}")
        return None

def generate_mock_embedding():
    """Generates a random normalized float vector of EMBEDDING_DIM dimensions."""
    vec = [random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(x*x for x in vec))
    return [x / norm for x in vec]

def main():
    api_key = os.environ.get("NOMIC_API_KEY")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "nomic-embed-text")
    
    # Generate or fetch embeddings
    texts = [item["description"] for item in DATASET]
    embeddings = None
    
    # 1. Try local Ollama first
    embeddings = get_ollama_embeddings(texts, host=ollama_host, model=ollama_model)
    
    # 2. Try hosted Nomic API next
    if not embeddings and api_key:
        embeddings = get_nomic_embeddings(texts, api_key)
        
    # 3. Fallback to mock/simulated embeddings
    if not embeddings:
        print("Using mock/simulated Nomic Embeddings (dimension 768)...")
        embeddings = [generate_mock_embedding() for _ in range(len(DATASET))]

    # SQLite Temp Database Setup
    sqlite_db = "temp_embeddings.db"
    if os.path.exists(sqlite_db):
        os.remove(sqlite_db)
        
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE location_embeddings (
            place_name TEXT,
            description TEXT,
            embedding BLOB,
            wkt_geom TEXT
        )
    """)
    
    for i, item in enumerate(DATASET):
        packed_embedding = struct.pack(f"<{EMBEDDING_DIM}f", *embeddings[i])
        wkt = f"POINT({item['lon']} {item['lat']})"
        cursor.execute(
            "INSERT INTO location_embeddings (place_name, description, embedding, wkt_geom) VALUES (?, ?, ?, ?)",
            (item["place_name"], item["description"], sqlite3.Binary(packed_embedding), wkt)
        )
        
    conn.commit()
    conn.close()
    print("Temporary SQLite database populated.")

    # Create OGR VRT XML wrapper to map geometry from WKT column
    vrt_name = "temp_embeddings.vrt"
    vrt_content = f"""<OGRVRTDataSource>
    <OGRVRTLayer name="location_embeddings">
        <SrcDataSource>{sqlite_db}</SrcDataSource>
        <SrcLayer>location_embeddings</SrcLayer>
        <GeometryType>wkbPoint</GeometryType>
        <LayerSRS>EPSG:4326</LayerSRS>
        <GeometryField encoding="WKT" field="wkt_geom"/>
    </OGRVRTLayer>
</OGRVRTDataSource>"""

    with open(vrt_name, "w") as f:
        f.write(vrt_content)
    print("Temporary OGR VRT file created.")

    # Target Geodatabase name
    gdb_name = "nomic_embeddings_example.gdb"
    if os.path.exists(gdb_name):
        print(f"Removing existing {gdb_name}...")
        subprocess.run(["rm", "-rf", gdb_name])

    # Convert to File Geodatabase using ogr2ogr via VRT file
    print(f"Converting SQLite table to File Geodatabase ({gdb_name}) using OGR VRT...")
    cmd = [
        "ogr2ogr",
        "-f", "OpenFileGDB",
        gdb_name,
        vrt_name,
        "-select", "place_name,description,embedding"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Cleanup temp files
    if os.path.exists(sqlite_db):
        os.remove(sqlite_db)
    if os.path.exists(vrt_name):
        os.remove(vrt_name)
        
    if result.returncode == 0:
        print(f"Success! Created {gdb_name}")
        print("Schema verification using ogrinfo:")
        verify_cmd = ["ogrinfo", "-al", "-so", gdb_name]
        subprocess.run(verify_cmd)
    else:
        print("Error converting to GDB:")
        print(result.stderr)

if __name__ == "__main__":
    main()
