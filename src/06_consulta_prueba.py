import time
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_SALIDA = BASE_DIR / "resultados_qdrant.txt"

# 1. Conectar con Qdrant local
client = QdrantClient(url="http://localhost:6333")
COLECCION = "proyecto_bdii"

# 2. Cargar el modelo multilingüe
print("Cargando modelo paraphrase-multilingual-MiniLM-L12-v2...")
modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def buscar_en_leyes(pregunta: str, archivo_salida, limite: int = 3):
    def registrar(texto: str = ""):
        print(texto)
        archivo_salida.write(texto + "\n")

    vector_pregunta = modelo.encode(pregunta).tolist()

    inicio = time.perf_counter()
    resultados = client.query_points(
        collection_name=COLECCION,
        query=vector_pregunta,
        limit=limite,
        with_payload=True
    ).points
    tiempo_ms = (time.perf_counter() - inicio) * 1000

    registrar(f"\n{'=' * 70}")
    registrar(f"PREGUNTA: {pregunta}")
    registrar(f"Tiempo de consulta en Qdrant: {tiempo_ms:.2f} ms")
    registrar(f"{'=' * 70}")

    for i, punto in enumerate(resultados, 1):
        p = punto.payload
        registrar(f"\n{'-' * 60}")
        registrar(f"RESULTADO #{i}")
        registrar(f"  id          : {punto.id}")
        registrar(f"  score       : {punto.score:.4f}")
        registrar(f"  payload     :")
        registrar(f"    documento : {p.get('documento')}")
        registrar(f"    categoria : {p.get('categoria')}")
        registrar(f"    fuente    : {p.get('fuente')}")
        registrar(f"    seccion   : {p.get('seccion')}")
        registrar(f"    articulo  : {p.get('articulo')}")
        registrar(f"    parte     : {p.get('parte')}")
        registrar(f"    pagina    : {p.get('pagina')}")
        registrar(f"    archivo   : {p.get('archivo')}")
        registrar(f"    n_palabras: {p.get('n_palabras')}")
        registrar(f"    id_chunk  : {p.get('id')}")
        registrar(f"    contenido : {p.get('contenido')}")


if __name__ == "__main__":
    preguntas_prueba = [
        "¿Cuántos días de vacaciones corresponden por año trabajado?",
        "¿Cuáles son las causas justas para despedir a un trabajador sin responsabilidad patronal?",
        "¿Cómo se calcula el salario mínimo y quién lo regula?",
        "¿Qué derechos cubre el seguro social en caso de enfermedad o maternidad?",
        "¿Cuáles son las obligaciones de seguridad e higiene en el lugar de trabajo?",
        "¿Cuáles son los requisitos para ingresar al servicio civil y qué derechos tienen los empleados públicos?"
    ]

    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        f.write("RESULTADOS DE CONSULTAS VECTORIALES - QDRANT\n")
        f.write("Modelo: paraphrase-multilingual-MiniLM-L12-v2 | Colección: proyecto_bdii\n")
        for p in preguntas_prueba:
            buscar_en_leyes(p, f, limite=3)

    print(f"\nResultados guardados correctamente en: {RUTA_SALIDA}")