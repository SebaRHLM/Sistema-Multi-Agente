## 1. Ingestion
En esta fase se realiza:
- La carga de los archivos (artículos .pdf preprocesados).
- Chunking.
- Vectorización de los chunks (embeddings).
- La carga de embeddings a la base de datos vectorial.
### 1.1 Cargar archivos
- Se realiza la carga de los artículos `.pdf` usando `PyMupdf`
- Se realiza una limpieza de acuerdo a la información que almacenan estos artículos. Los archivos que se entregan en la carpeta `artículos-pdf` son los archivos resultantes de la limpieza manual de los artículos `2017 Guideline for High Blood Pressure in Adults - American College of Cardiology`, `Guidelines for the management of arterial hipertension`, `Hypotension_An_Overview_of_Updated_Data_for_Health`.
- Se utiliza chatGPT para la identificación de secciones luego de la limpiza hecha a mano de los articulos. Lo anterior tiene el proposito de realizar un mejor chunking.
### 1.2 Chunking
El chunking realizado esta estructurado para fomentar la división de la información, de manera que esta sea relevante y de utilidad para el RAG, de forma que sea información útil la que se alamcena y no ruido. Por lo tanto el chunking se raliza de la siguiente manera:
- Paso 1 - Limpieza ligera del texto
    - A pesar de la limpieza realizada manualente, se realiza una pequeña lipieza con código que eliminae elementos que no aportan significado clínico al sistema RAG y que pueden afectar negativamente la representación semántica de los documentos (saltos de línea innecesarios, caracteres especiales, espacios repetidos, símbolos provenientes de la extracción PDF,  errores de codificación.)
    - En teória, esto mejora la creación de los embeddings generados por `sentence-transformer`.
- Paso 2 - Chunking estructural
    - Se propone un chinking que tome en consideración las secciones de los articulos (Definition of hypertension, Classification of blood pressure, Evaluation, Etiology, Pathophysiology) con el fin de no cortar definiciones a la mitad de su definición.
    - Esto nos beneficia, manteniendo secciones médicas detalladas dentro de los artículos científicos
- Paso 3 - Chinking semántico
    - Se realiza para dividir estas secciones indentificadas en el paso 2 en bloques más pequeños, pero manteniendo coherencia conceptual. En lugar de cortar arbitrariamente por caracteres como haciamos en un principio (tarea 1), el sistema agrupa oraciones relacionadas hasta alcanzar un límite razonable de palabras.
    - Esto nos ayuda teoricamente a que los embeddings representen conceptos médicos más específicos.
- Paso 4 - Sobre el Overlap
    - Se implementa overlap para evitar pérdida de contexto entre chunks consecutivos.
    - En casos como una oración dependa semánticamente de la anterior, una recomendación continúe en el siguiente chunk, una definición quede incompleta si se separa abruptamente, es donde el overlap funciona, haciendo que el sistema agrega la última oración del chunk anterior al siguiente chunk.
    - Esto nos permite mantener de cierta forma el contexto de una definición incompleta o una oración que detalle un trastorno de la presión arterial.
#### 1.2.1 Posibles mejoras
- Se puede añadir información sobre los tratamientos de estos trastornos a la presión arterial. Dicha información si o si debería seguir el procedimiento aqui realizado, primero una limpieza manual y segundo, aplicar la estructura de chunking propuesta
- También se pueden añadir más secciones hardcodeadas en `SECTION_PATTERN``con el fin de dividir exactamente por termino o tratamiento si es que en un futuro se añade esta información.
### 1.3 Vectorización - Creación de embbedings - Base de datos Vectorial
En esta sección:
- Se crea la base de datos vectorial `collection=client.create_collection(COLLECTION_NAME)`
- Se carga el modelo de embedding `embedder=SentenceTransformer(EMBED_MODEL)`