# Sistema Multiagente de Apoyo Clínico

## Disclaimer

Este sistema multiagente fue desarrollado como proyecto académico para la asignatura **LLM: Fundamentos y Práctica**, de la carrera de **Ingeniería en Informática** de la **Pontificia Universidad Católica de Valparaíso**.

El sistema tiene fines exclusivamente académicos y experimentales. No entrega diagnósticos médicos definitivos, no prescribe tratamientos y no reemplaza la evaluación de un profesional de la salud. Toda salida generada debe ser interpretada como apoyo preliminar al razonamiento clínico.

---

## Objetivo del sistema

El objetivo del sistema es apoyar la lectura y análisis preliminar de casos clínicos escritos en lenguaje natural, mediante una arquitectura multiagente capaz de:

* Extraer datos clínicos estructurados desde un caso clínico.
* Calcular métricas relevantes, como la presión arterial media o MAP.
* Consultar evidencia clínica mediante un sistema RAG basado en documentos PDF.
* Ejecutar una herramienta de clasificación cardiovascular basada en un modelo ML preentrenado.
* Generar un análisis clínico preliminar con limitaciones explícitas.
* Revisar la suficiencia de la respuesta mediante un nodo juez.

El sistema está orientado principalmente a casos relacionados con presión arterial, hipotensión, hipertensión y enfermedades cardiovasculares asociadas.

---

## Arquitectura del sistema

El sistema utiliza una arquitectura multiagente basada en **LangGraph**, donde cada nodo cumple una responsabilidad específica dentro del flujo de razonamiento.

Flujo general:

```text
START
  ↓
extractor_LLM
  ↓
gestor_clinico
  ├── calcular_map
  ├── clasificar_paciente_cardiovascular
  ├── search_rag
  │      ↓
  │   router_memoria_rag
  │      ├── resumir_evidencia_rag
  │      └── gestor_clinico
  └── analizar
          ↓
        juez
          ↓
    respuesta_final
          ↓
         END
```

### Nodos principales

| Nodo                                 | Función                                                                                      |
| ------------------------------------ | -------------------------------------------------------------------------------------------- |
| `extractor_LLM`                      | Extrae datos clínicos estructurados desde el caso clínico.                                   |
| `gestor_clinico`                     | Decide qué herramienta o nodo debe ejecutarse a continuación.                                |
| `calcular_map`                       | Calcula la presión arterial media a partir de presión sistólica y diastólica.                |
| `clasificar_paciente_cardiovascular` | Ejecuta un modelo ML XGBoost para estimar riesgo cardiovascular según variables del dataset. |
| `search_rag`                         | Busca evidencia clínica en documentos PDF usando una base vectorial.                         |
| `resumir_evidencia_rag`              | Compacta evidencia recuperada cuando supera cierto tamaño.                                   |
| `analizar`                           | Genera un análisis preliminar usando los datos disponibles.                                  |
| `juez`                               | Evalúa si la respuesta tiene información suficiente o si debe iterar.                        |
| `respuesta_final`                    | Construye el reporte final para el usuario.                                                  |

---

## Herramientas del sistema

### 1. `calcular_MAP`

Calcula la presión arterial media mediante:

```text
MAP = (PAS + 2 × PAD) / 3
```

Donde:

* `PAS`: presión arterial sistólica.
* `PAD`: presión arterial diastólica.

---

### 2. `search_clinical_evidence`

Busca evidencia clínica en una base vectorial construida desde archivos PDF. El RAG utiliza:

* Carga de PDFs.
* Limpieza ligera de texto.
* Chunking estructural y semántico.
* Overlap entre chunks.
* Embeddings con `SentenceTransformer`.
* Almacenamiento y consulta en ChromaDB.

---

### 3. `clasificar_paciente_cardiovascular`

Ejecuta un modelo ML preentrenado con variables del dataset cardiovascular.

Variables esperadas:

| Variable   | Descripción                                                                                                 |
| ---------- | ----------------------------------------------------------------------------------------------------------- |
| `age`      | Edad del paciente en años.                                                                                  |
| `sex`      | Sexo: `1 = male`, `0 = female`.                                                                             |
| `cp`       | Tipo de dolor torácico: `0 = typical angina`, `1 = atypical angina`, `2 = non-anginal`, `3 = asymptomatic`. |
| `trestbps` | Presión arterial en reposo o al ingreso hospitalario.                                                       |
| `chol`     | Colesterol sérico en mg/dl.                                                                                 |
| `fbs`      | Glucosa en ayunas mayor a 120 mg/dl: `1 = true`, `0 = false`.                                               |
| `restecg`  | ECG en reposo: `0 = normal`, `1 = ST-T abnormality`, `2 = left ventricular hypertrophy`.                    |
| `thalach`  | Frecuencia cardíaca máxima alcanzada.                                                                       |
| `exang`    | Angina inducida por ejercicio: `1 = true`, `0 = false`.                                                     |
| `oldpeak`  | Depresión ST inducida por ejercicio relativa al reposo.                                                     |
| `slope`    | Pendiente del segmento ST en ejercicio: valores `0`, `1`, `2`.                                              |
| `ca`       | Número de vasos principales coloreados por fluoroscopía: `0–3`.                                             |
| `thal`     | `1 = normal`, `2 = fixed defect`, `3 = reversible defect`.                                                  |

Las columnas `id`, `origin` y `num` no se utilizan como entrada del modelo. `num` corresponde al atributo objetivo.

---

## Requisitos previos

Antes de ejecutar el sistema, se recomienda contar con:

* Python 3.10 o superior.
* Entorno virtual de Python.
* Archivos PDF clínicos dentro de la carpeta configurada para el RAG.
* API Key de OpenAI, solo si se desea ejecutar con LLM reales.
* Token de Hugging Face, recomendado para descargar modelos de embeddings sin restricciones de solicitudes anónimas.

---

## Instalación

### 1. Clonar o descargar el proyecto

Ubicarse en la carpeta raíz del proyecto:

```bash
cd agente_com_modificado
```

---

### 2. Crear entorno virtual

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

---
### 3. Instalar dependencias

El proyecto incluye un archivo `requirements.txt` con las dependencias necesarias para ejecutar el sistema multiagente, el RAG, las herramientas clínicas y el modelo ML cardiovascular.

Con el entorno virtual activado, ejecutar:

```bash
pip install -r requirements.txt
```

En caso de problemas con versiones antiguas de paquetes, se recomienda actualizar `pip` antes de instalar:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Las dependencias principales incluidas son:

* `langgraph`, `langchain`, `langchain-core` y `langchain-openai`, utilizadas para la arquitectura multiagente y llamadas a modelos de lenguaje.
* `python-dotenv` y `openai`, utilizadas para cargar variables de entorno y conectarse a la API.
* `chromadb`, `sentence-transformers` y `huggingface-hub`, utilizadas para el sistema RAG.
* `PyMuPDF`, utilizada para cargar documentos PDF.
* `pandas`, `numpy`, `scikit-learn`, `xgboost` y `joblib`, utilizadas por la herramienta de clasificación cardiovascular.
* `pydantic`, utilizada para estructurar las salidas de los nodos LLM.

Si se ejecutará el sistema en modo real, también se debe configurar el archivo `.env` con las claves correspondientes antes de iniciar el sistema.
---

## Configuración de variables de entorno

Crear un archivo `.env` en la raíz del proyecto.

### Ejecución con mocks

Para ejecutar sin llamadas reales a modelos LLM:

```env
USE_REAL_LLM=false
OPENAI_MODEL=gpt-5.4-mini
```

Este modo es útil para depurar el flujo del grafo sin consumir API.

---

### Ejecución con modelos reales

Para ejecutar usando modelos reales de OpenAI:

```env
USE_REAL_LLM=true
OPENAI_API_KEY=tu_api_key_de_openai
OPENAI_MODEL=gpt-5.4-mini
```

---

### Token de Hugging Face

Para evitar advertencias y límites por solicitudes no autenticadas al cargar modelos de embeddings:

```env
HF_TOKEN=tu_token_de_huggingface
```

El token debe ser de tipo lectura. No es necesario usar permisos de escritura.

---

## Preparación del RAG

Los documentos PDF deben estar ubicados en la carpeta configurada en:

```python
setting_RAG.py
```

Variable relevante:

```python
PDF_RAG_PATH = str(BASE_DIR / "archivos-pdf-rag")
```

La estructura esperada es similar a:

```text
agente/
  utils/
    tools/
      archivos-pdf-rag/
        documento_1.pdf
        documento_2.pdf
        documento_3.pdf
```

El sistema construye o reutiliza una base vectorial persistente en ChromaDB.

Si la colección ya existe y contiene documentos, se reutiliza. Si está vacía, el sistema carga PDFs, genera chunks, calcula embeddings y llena la base vectorial.

---

## Ejecución del sistema

Desde la raíz del proyecto:

```bash
python agente\agent.py
```

En Windows, si aparecen problemas de codificación, ejecutar antes:

```bash
chcp 65001
set PYTHONIOENCODING=utf-8
python agente\agent.py
```

---

## Selección de casos clínicos

Al iniciar, el sistema muestra un selector de casos:

```text
CASOS CLÍNICOS DISPONIBLES
1. Caso 1: Dolor torácico + ECG compatible con evento coronario
2. Caso 2: Diálisis + CKD + diabetes + fiebre + disnea
3. Caso 3: Marcapasos + fibrilación auricular + insuficiencia cardíaca

Selecciona un caso clínico [1-3]:
```

El usuario debe ingresar `1`, `2` o `3`.

---

## Salida esperada

Durante la ejecución se imprimen estados intermedios del grafo, incluyendo:

* Datos extraídos por el nodo extractor.
* Decisiones del gestor clínico.
* Resultado del cálculo MAP.
* Resultado del clasificador cardiovascular.
* Evidencia recuperada por RAG.
* Resumen de evidencia.
* Análisis preliminar.
* Revisión del juez.
* Respuesta final.

Ejemplo de flujo esperado:

```text
[LLM Extractor] Extrayendo datos clínicos del caso
[LLM Investigador] Decidiendo si usar herramienta
[Tool] calcular_MAP
[LLM Investigador] Decidiendo si usar herramienta
[Tool] clasificar_paciente_cardiovascular
[LLM Investigador] Decidiendo si usar herramienta
[Tool] search_clinical_evidence
[Router Memoria RAG] Evidencia RAG grande. Se deriva al nodo resumidor.
[LLM Resumidor RAG] Compactando evidencia recuperada
[LLM Analizador] Generando análisis preliminar
[LLM Juez] Evaluando suficiencia de la información
[Nodo-Respuesta Final] Redactando respuesta final
```

---

## Diagrama del grafo

El sistema genera un diagrama Mermaid y un PNG del grafo en la carpeta:

```text
outputs/
```

Archivos esperados:

```text
outputs/diagrama_multiagente.mmd
outputs/diagrama_multiagente.png
```

---

## Modo mock vs modo real

| Modo | Variable             | Uso                                                              |
| ---- | -------------------- | ---------------------------------------------------------------- |
| Mock | `USE_REAL_LLM=false` | Ejecuta respuestas simuladas. Útil para depurar sin costo.       |
| Real | `USE_REAL_LLM=true`  | Ejecuta llamadas reales al modelo configurado en `OPENAI_MODEL`. |

Antes de ejecutar en modo real, se recomienda validar que el sistema funcione correctamente en modo mock.

---

## Problemas frecuentes

### 1. `llm_openai is None`

Causa probable:

* `USE_REAL_LLM=true`, pero no se cargó correctamente la API key.
* El archivo `.env` no está en la raíz correcta.
* Falta `OPENAI_API_KEY`.

Solución:

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(bool(os.getenv('OPENAI_API_KEY')))"
```

---

### 2. Warning de Hugging Face

Mensaje típico:

```text
You are sending unauthenticated requests to the HF Hub.
```

Solución:

Agregar en `.env`:

```env
HF_TOKEN=tu_token_de_huggingface
```

---

### 3. El RAG demora en la primera búsqueda

La primera llamada puede tardar porque se carga el modelo de embeddings o la base vectorial. Se recomienda precargar el RAG al inicio del sistema si se usará en una demostración.

---

### 4. Error por carpeta de PDFs no encontrada

Mensaje típico:

```text
No existe la carpeta de PDFs para el RAG
```

Solución:

Verificar que exista la carpeta configurada en `PDF_RAG_PATH` y que contenga archivos `.pdf`.

---

### 5. Caracteres extraños en consola

Ejemplo:

```text
presiÃ³n
clÃ­nico
anÃ¡lisis
```

Solución en Windows:

```bash
chcp 65001
set PYTHONIOENCODING=utf-8
python agente\agent.py
```

También se recomienda guardar los archivos `.py` como UTF-8.

---

## Limitaciones del sistema

* El sistema no entrega diagnóstico definitivo.
* El sistema no prescribe tratamientos.
* La herramienta ML es solo un apoyo y puede no reflejar correctamente casos clínicos agudos.
* La calidad del RAG depende de los documentos PDF cargados.
* El sistema puede requerir datos adicionales como troponinas, ECG completo, ecocardiograma, signos vitales seriados, medicamentos actuales y antecedentes relevantes.
* El análisis final debe ser revisado por un profesional de la salud.

---

## Recomendaciones para replicar una prueba

Para una prueba segura y reproducible:

1. Ejecutar primero con mocks:

```env
USE_REAL_LLM=false
```

2. Verificar que el grafo llegue a `respuesta_final`.

3. Activar modo real:

```env
USE_REAL_LLM=true
```

4. Confirmar que `OPENAI_API_KEY` esté configurada.

5. Confirmar que `HF_TOKEN` esté configurado si se usará RAG.

6. Ejecutar:

```bash
python agente\agent.py
```

7. Seleccionar un caso clínico del 1 al 3.

8. Revisar la salida final y los estados intermedios.

---

## Créditos

Proyecto académico desarrollado para la asignatura **LLM: Fundamentos y Práctica**, carrera de **Ingeniería en Informática**, Pontificia Universidad Católica de Valparaíso.
