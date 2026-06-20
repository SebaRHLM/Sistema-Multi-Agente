import os
import re

from .setting_RAG import SECTION_PATTERN

# =========================================================
# Paso 1 - Limpieza ligera del texto
# ===   ======================================================
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("￾", "")

    return text.strip()

# =========================================================
# Paso 2 - Chunking estructural
# =========================================================

# Funcion que aplica la división según las secciones declaradas en SECTION_PATTERN
def split_sections(text):

    sections = re.split(
        SECTION_PATTERN,
        text
    )

    clean_sections = []

    for section in sections:

        section = section.strip()

        if len(section.split()) > 40:
            clean_sections.append(section)

    return clean_sections


# =========================================================
# Paso 3 — Chunking Semantico
# =========================================================

def semantic_chunking(text, chunk_size=200):
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        temp_chunk = current_chunk + " " + sentence

        if len(temp_chunk.split()) <= chunk_size:

            current_chunk = temp_chunk

        else:

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            current_chunk = sentence

    if current_chunk.strip():

        chunks.append(current_chunk.strip())

    return chunks


# =========================================================
# Paso 4 — Sobre el Overlap
# =========================================================

def apply_overlap(chunks, overlap=20):

    if len(chunks) <= 1:
        return chunks

    overlapped_chunks = [chunks[0]]

    for i in range(1, len(chunks)):

        previous_chunk = chunks[i - 1]

        current_chunk = chunks[i]

        prev_words = previous_chunk.split()

        overlap_text = " ".join(prev_words[-overlap:])

        merged_chunk = (overlap_text + " " + current_chunk)

        overlapped_chunks.append(merged_chunk)

    return overlapped_chunks

# =========================================================
# Funcion maestra - Implenta los pasos 1, 2, 3 y 4
# =========================================================

def build_chunks(docs, chunk_size=200, overlap=20):
    """
    Genera chunks semánticos para RAG del SAD Clinico.

    Args:
        docs (list):
            Lista de documentos cargados.

        chunk_size (int):
            Máximo de palabras por chunk.

        overlap (int):
            Cantidad de palabras compartidas.

    Returns:
        list[dict]
    """

    all_chunks = []
    for doc in docs:
        source = doc["source"]
        text = doc["text"]

        # Paso 1
        text = clean_text(text)

        # Paso 2
        sections = split_sections(text)

        # Paso 3
        semantic_chunks = []
        for section in sections:

            chunks = semantic_chunking(
                section,
                chunk_size=chunk_size
            )

            semantic_chunks.extend(chunks)

        # Paso 4
        semantic_chunks = apply_overlap(semantic_chunks, overlap=overlap)

        # guardar chunks
        for idx, chunk in enumerate(semantic_chunks):

            if len(chunk.split()) < 40:
                continue
            chunk_data = {
                "source": os.path.basename(source),
                "chunk_id": idx,
                "chunk": chunk
            }
            all_chunks.append(chunk_data)
    return all_chunks 
