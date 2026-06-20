from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "sad_clinico_rag"
PDF_RAG_PATH = str(BASE_DIR / "archivos-pdf-rag")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SECTION_PATTERN_TEXT = (
    r'(?='
    r'(Introduction|'
    r'Definition of hypertension|'
    r'Classification of blood pressure|'
    r'Blood pressure measurement|'
    r'Evaluation|'
    r'History and Physical|'
    r'Etiology|'
    r'Epidemiology|'
    r'Pathophysiology|'
    r'Follow-up|'
    r'Treatment|'
    r'Risk Assessment)'
    r')'
)
SECTION_PATTERN = re.compile(SECTION_PATTERN_TEXT, flags=re.IGNORECASE)

# AGENTE UTILS
NAME_MODEL= "llama-3.1-8b-instant"