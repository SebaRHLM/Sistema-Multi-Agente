# Cambios implementados

Se transformó el flujo lineal en un grafo multiagente con herramientas opcionales decididas por un LLM investigador.

## Flujo nuevo

START → preprocesar → decidir_herramienta

Desde `decidir_herramienta`, el grafo usa un router condicional:

- `calcular_map` → vuelve a `decidir_herramienta`
- `search_rag` → vuelve a `decidir_herramienta`
- `analizar` → `juez`

Luego el juez decide:

- `volver_investigador` → `decidir_herramienta`
- `respuesta_final` → END

## Archivos nuevos

- `agente/utils/llms.py`
- `agente/utils/schemas.py`
- `agente/utils/routers.py`
- `requirements.txt`
- `.env.example`

## Modo de ejecución

```bash
pip install -r requirements.txt
python agente/agent.py
```

Por defecto usa modo simulado (`USE_REAL_LLM=false`). Para usar API real, configurar `.env`.
