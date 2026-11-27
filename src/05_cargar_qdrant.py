import json
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_EMBEDDINGS = BASE_DIR / "data" / "embeddings.npy"
RUTA_CHUNKS = BASE_DIR / "data" / "chunks" / "chunks_limpios.json"

client = QdrantClient(url="http://localhost:6333")
print("Cargando archivos desde data/...")
embeddings = np.load(RUTA_EMBEDDINGS)
with open(RUTA_CHUNKS, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Vectores encontrados: {embeddings.shape[0]} | Dimensión: {embeddings.shape[1]}")
print(f"Chunks encontrados: {len(chunks)}")

if len(embeddings) != len(chunks):
    raise ValueError(
        f"La cantidad no coincide ({len(embeddings)} embeddings vs {len(chunks)} chunks). "
        "Revisa si embeddings.npy se generó con chunks.json en lugar de chunks_limpios.json."
    )

nombre_coleccion = "proyecto_bdii"

if client.collection_exists(nombre_coleccion):
    client.delete_collection(nombre_coleccion)

client.create_collection(
    collection_name=nombre_coleccion,
    vectors_config=VectorParams(
        size=int(embeddings.shape[1]),
        distance=Distance.COSINE
    ),
)

puntos = []
for idx, (vector, chunk) in enumerate(zip(embeddings, chunks)):
    # Si el chunk es un string simple, lo envolvemos en un diccionario
    payload = chunk if isinstance(chunk, dict) else {"texto": str(chunk)}
    puntos.append(
        PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload=payload
        )
    )


batch_size = 500
for i in range(0, len(puntos), batch_size):
    client.upsert(
        collection_name=nombre_coleccion,
        points=puntos[i : i + batch_size]
    )
    print(f"Subidos {min(i + batch_size, len(puntos))} / {len(puntos)} puntos...")

print(f"\n¡Carga completada en la colección '{nombre_coleccion}'!")