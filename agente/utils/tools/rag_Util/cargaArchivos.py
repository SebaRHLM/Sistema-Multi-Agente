import os

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

global docs # Variable global para almacenar los documentos que almacena el rag

def load_pdf(paths: list[str]):
    """
    Carga archivos PDF y devuelve documentos completos.

    Args:
        paths (list[str]):
            Lista de archivos o directorios.

    Returns:
        list[dict]
    """

    if fitz is None:
        raise ImportError(
            "No se pudo importar PyMuPDF. Instala la dependencia con: "
            "python -m pip install PyMuPDF"
        )

    all_documents = []
    def process_pdf(file_path):
        try:
            doc = fitz.open(file_path)
            full_text = ""
            for page in doc:

                text = page.get_text()
                full_text += "\n" + text

            document = {
                "source": file_path,
                "text": full_text
            }
            return document

        except Exception as e:

            print(f"Error al cargar '{file_path}': {e}")

            return None

    # RECORRER RUTA
    for path in paths:
        # Archivo individual
        if os.path.isfile(path):

            if path.lower().endswith(".pdf"):
                doc = process_pdf(path)

                if doc:
                    all_documents.append(doc)

        # Directorio
        elif os.path.isdir(path):

            for root, _, files in os.walk(path):

                for file_name in files:

                    if file_name.lower().endswith(".pdf"):
                        file_path = os.path.join(
                            root,
                            file_name
                        )
                        doc = process_pdf(file_path)
                        if doc:
                            all_documents.append(doc)

    return all_documents
