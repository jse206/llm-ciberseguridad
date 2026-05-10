# LLM Ciberseguridad — Evaluación de Arquitecturas

Implementación y benchmark comparativo de cuatro arquitecturas LLM con recuperación de conocimiento externo aplicadas al análisis de vulnerabilidades y amenazas en ciberseguridad.

## Arquitecturas

| ID | Nombre | Descripción |
|---|---|---|
| A | Text2SQL | Consultas en lenguaje natural sobre base de datos SQLite NVD/CVE |
| B | API Calls | Consultas en tiempo real a NVD API v2.0 y AbuseIPDB |
| C | GraphRAG | Recuperación sobre grafo de conocimiento MITRE ATT&CK |
| D | Toolformer | Agente ReAct que selecciona autónomamente entre A, B y C |

## Requisitos

- Python 3.12+
- Claves de API:
  - `OPENAI_API_KEY` — [platform.openai.com](https://platform.openai.com)
  - `ABUSEIPDB_API_KEY` — [abuseipdb.com](https://www.abuseipdb.com) (opcional)

## Instalación rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/jse206/llm-ciberseguridad.git
cd llm-ciberseguridad

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

pip install -r requirements.txt

# 3. Configurar claves de API
cp .env.example .env

# 4. Descargar datos y construir el grafo MITRE ATT&CK
cd 03_implementacion
python setup.py
cd ..

# 5. Arrancar la API web
uvicorn api.main:app --reload --port 8001
# → http://localhost:8001
```

## Con Docker 

```bash
# 1. Copiar y configurar variables de entorno
cp .env.example .env

# 2. Arrancar
docker compose up --build

# → http://localhost:8001
```

> La primera vez que se levanta el contenedor, `setup.py` descarga automáticamente
> los datos NVD/CVE y construye el grafo MITRE ATT&CK. Los datos se
> persisten en un volumen Docker para ejecuciones posteriores.

## Interfaz web

La aplicación incluye una SPA con tres pestañas:

- **Chat** — consultas en lenguaje natural con cualquiera de las 4 arquitecturas
- **Benchmark** — visualización de resultados o relanzamiento del benchmark completo
- **Informe** — gráficas comparativas 

## Benchmark

Evaluación sobre 100 preguntas × 4 arquitecturas × 5 métricas:
accuracy, tasa de alucinación, trazabilidad de fuentes, manejo de errores y latencia.

```bash
# Ver resultados existentes
python 04_benchmark/benchmark_runner.py --results

# Relanzar benchmark completo
python 04_benchmark/benchmark_runner.py --all
```

## Stack tecnológico

Python · FastAPI · OpenAI GPT-4o-mini · SQLite · NetworkX · Chart.js

## Referencias

- Pourreza & Rafiei (2023). DIN-SQL. NeurIPS 2023.
- Edge et al. (2024). GraphRAG. arXiv:2404.16130.
- Schick et al. (2023). Toolformer. NeurIPS 2023.
- Yao et al. (2023). ReAct. ICLR 2023.
- MITRE Corporation (2024). ATT&CK® for Enterprise.
- NIST (2024). NVD API v2.0.
