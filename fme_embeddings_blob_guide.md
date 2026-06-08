# Writing Nomic Embeddings to Esri File Geodatabase BLOB Fields using FME

This guide explains how to construct a Feature Manipulation Engine (FME) workspace to generate local/cloud Nomic embeddings, pack them into the correct binary structure, and write them directly into an Esri File Geodatabase BLOB (Binary Large Object) field.

---

## Workspace Overview

The FME workspace pipeline consists of the following transformers:

```
[ Reader (e.g. Creator / Creator + CSV) ] 
                 │
                 ▼
          [ HTTPCaller ]         <-- Calls Ollama (local) or Nomic (cloud) API
                 │
                 ▼
         [ JSONExtractor ]       <-- Parses the raw JSON array response
                 │
                 ▼
          [ PythonCaller ]       <-- Packs the float array into binary bytes
                 │
                 ▼
      [ OpenFileGDB Writer ]     <-- Writes to Geodatabase feature class (binary field)
```

---

## Detailed Step-by-Step Configuration

### Step 1: Querying the Embeddings API (`HTTPCaller`)
The `HTTPCaller` sends the text representation of each feature to the Nomic embedding model.

#### Local Ollama Configuration:
* **HTTP Method**: `POST`
* **Request URL**: `http://localhost:11434/api/embed`
* **Headers**:
  * `Content-Type`: `application/json`
* **Upload Body (JSON)**:
  ```json
  {
    "model": "nomic-embed-text",
    "input": ["@Value(description)"]
  }
  ```

#### Nomic Cloud API Configuration:
* **HTTP Method**: `POST`
* **Request URL**: `https://api-atlas.nomic.ai/v1/embedding/text`
* **Headers**:
  * `Content-Type`: `application/json`
  * `Authorization`: `Bearer YOUR_NOMIC_API_KEY`
* **Upload Body (JSON)**:
  ```json
  {
    "model": "nomic-embed-text-v1.5",
    "texts": ["@Value(description)"],
    "task_type": "search_document"
  }
  ```

---

### Step 2: Extracting the Float Array (`JSONExtractor`)
Ollama returns the embeddings array as a nested JSON structure inside `_response_body`.
* Set **JSON Document** to: `_response_body`
* Extract the query value to a new attribute (e.g. `_embeddings`):
  * **Ollama Path**: `$.embeddings[0]`
  * **Nomic Cloud Path**: `$.embeddings[0]`

This creates a string attribute containing a comma-separated list of float values (e.g. `[0.0123, -0.456, ...]`).

---

### Step 3: Packing Float List to Binary Bytes (`PythonCaller`)
ArcGIS Pro's embeddings tools expect the BLOB field to contain **raw 32-bit floating point numbers (float32)** in **little-endian order** (yielding a binary sequence of exactly 3072 bytes for a 768-dimensional model like Nomic). 

Use a `PythonCaller` to serialize the text representation into actual binary bytes:

```python
import fme
import fmeobjects
import struct
import json

def processFeature(feature):
    # Retrieve the JSON array string from Step 2
    embeddings_str = feature.getAttribute('_embeddings')
    
    if embeddings_str:
        try:
            # Parse the string back into a Python float list
            vector = json.loads(embeddings_str)
            
            # Pack the floats into a binary string using struct
            # '<' specifies little-endian byte order
            # 'f' specifies single-precision float (4 bytes each)
            packed_blob = struct.pack(f"<{len(vector)}f", *vector)
            
            # Set the packed byte buffer back onto the feature as an attribute.
            # FME automatically treats python 'bytes' objects as fme_binarybuffer
            feature.setAttribute('embedding', packed_blob)
            
        except Exception as e:
            # Log warning if formatting fails
            log = fmeobjects.FMELogFile()
            log.logMessageString(f"Failed to pack embedding: {str(e)}", fmeobjects.FME_WARN)
            
    return feature
```

---

### Step 4: Writing to File Geodatabase (`OpenFileGDB` Writer)
Add an **Esri Geodatabase (OpenFile Geodb)** or **Esri Geodatabase (File Geodb)** Writer.

In the Writer Feature Type schema parameters:
1. Under **User Attributes**, add your `embedding` attribute.
2. Set the **Data Type** of `embedding` to **`binary`** (FME translates this to `BLOB` inside the final File Geodatabase).
3. Connect the output geometry from your features to the Writer. FME will automatically generate the Esri `SHAPE` geometry field.

---

## Verifying Output File GDB
Once the FME workspace runs successfully, you can verify the results using the `ogrinfo` command-line utility:

```bash
ogrinfo -al -so nomic_embeddings_example.gdb
```
The output schema should display `embedding: Binary` and the geometry type as expected (e.g. `Geometry: Point`). In ArcGIS Pro 3.7, you can now run the `Extract Embeddings To Fields` geoprocessing tool directly on this output.
