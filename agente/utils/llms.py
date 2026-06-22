import os
from dotenv import load_dotenv

load_dotenv()

# Para ahorrar dinero, por defectos en el desarrollo del proyecto, se usa modo simulado, con mocks.
# Cambia USE_REAL_LLM=true en .env cuando quieras llamar la API real.
USE_REAL_LLM = os.getenv("USE_REAL_LLM", "false").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

llm_openai = None

if USE_REAL_LLM:
    from langchain_openai import ChatOpenAI

    llm_openai = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
    )
