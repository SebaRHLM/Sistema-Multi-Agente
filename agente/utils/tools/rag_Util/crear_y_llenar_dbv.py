from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from .cargaArchivos import load_pdf
from .chunking import build_chunks
from .setting_RAG import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    PDF_RAG_PATH,
)


def crear_base_vectorial(reset: bool = True) -> tuple[chromadb.ClientAPI, Collection]:
    """
    Crea y retorna el cliente y la coleccion de ChromaDB.

    Args:
        reset (bool): Si es True, elimina la coleccion existente antes de crearla.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    existing = [c.name for c in client.list_collections()]
    if reset and COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        existing.remove(COLLECTION_NAME)

    if COLLECTION_NAME in existing:
        collection = client.get_collection(COLLECTION_NAME)
    else:
        collection = client.create_collection(COLLECTION_NAME)

    return client, collection


def cargar_documentos_pdf():
    """
    Carga todos los archivos PDF desde la carpeta configurada para el RAG.
    """
    pdf_path = Path(PDF_RAG_PATH)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de PDFs para el RAG: {pdf_path}"
        )

    docs = load_pdf([str(pdf_path)])

    if not docs:
        raise ValueError(
            f"No se encontraron archivos PDF válidos en: {pdf_path}"
        )

    return docs


def llenar_base_vectorial(
    collection: Collection,
    chunks: list[dict],
    embedder: SentenceTransformer,
) -> int:
    """
    Inserta los chunks procesados en la coleccion vectorial.
    """
    if not chunks:
        raise ValueError("No se generaron chunks para cargar en ChromaDB.")

    documents = [chunk["chunk"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]
    ids = [
        f"{chunk['source']}-{chunk['chunk_id']}"
        for chunk in chunks
    ]

    embeddings = embedder.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(documents)


def preparar_dbv(
    reset: bool = False,
    embedder: SentenceTransformer | None = None,
) -> Collection:
    """
    Prepara la base vectorial y retorna la colección lista para búsqueda.

    Si la colección ya contiene documentos y reset=False, solo la reutiliza.
    Si debe llenar la base, reutiliza el embedder recibido para evitar doble carga.
    """
    _, collection = crear_base_vectorial(reset=reset)

    if not reset and collection.count() > 0:
        return collection

    docs = cargar_documentos_pdf()
    chunks = build_chunks(docs)

    if embedder is None:
        embedder = SentenceTransformer(EMBED_MODEL)

    llenar_base_vectorial(collection, chunks, embedder)

    return collection
