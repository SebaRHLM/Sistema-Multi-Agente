import argparse
from typing import Any

from sentence_transformers import SentenceTransformer

from .rag_Util.setting_RAG import EMBED_MODEL
from .rag_Util.crear_y_llenar_dbv import preparar_dbv

_collection = None
_embedder = None


def inicializar_search_tool() -> None:
    """Inicializa una sola vez la colección y el modelo de embeddings."""
    global _collection, _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)

    if _collection is None:
        _collection = preparar_dbv(
            reset=False,
            embedder=_embedder,
        )

def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Busca evidencia clínica relevante en la base vectorial."""
    inicializar_search_tool()

    query_embedding = _embedder.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documentos = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved_chunks = []

    for document, metadata, distance in zip(documentos, metadatas, distances):
        retrieved_chunks.append(
            {
                "texto": document,
                "source": metadata.get("source"),
                "chunk_id": metadata.get("chunk_id"),
                "distancia": distance,
            }
        )

    return retrieved_chunks


def format_search_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No se encontró evidencia clínica relevante en la base vectorial."

    formatted_results = []

    for i, result in enumerate(results, start=1):
        formatted_results.append(
            f"[Contexto {i}]\n"
            f"Fuente: {result['source']}\n"
            f"Chunk ID: {result['chunk_id']}\n"
            f"Distancia: {result['distancia']}\n"
            f"Texto: {result['texto']}"
        )

    return "\n\n".join(formatted_results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Busca evidencia clínica en la base vectorial del RAG."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="hipertensión arterial diagnóstico y tratamiento",
        help="Consulta clínica para buscar en el RAG.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Cantidad de fragmentos a recuperar.",
    )
    args = parser.parse_args()

    print(format_search_results(search(args.query, top_k=args.top_k)))


if __name__ == "__main__":
    main()
