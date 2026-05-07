"""
Prompts para la Arquitectura A (Text2SQL) sobre la base de datos NVD/CVE.

Implementa la estrategia de descomposición de DIN-SQL (Pourreza & Rafiei, 2023)
en tres pasos encadenados:
  1. SCHEMA LINKING  — identifica tablas y columnas relevantes para la pregunta.
  2. SQL GENERATION  — genera la consulta SQL a partir del schema vinculado.
  3. SELF-CORRECTION — detecta y corrige errores sintácticos o semánticos en el SQL.

El sintetizador (paso 4) es el mismo que en Arquitectura B para mantener
la comparabilidad entre arquitecturas en el benchmark.

Referencia: Pourreza, M. & Rafiei, D. (2023). DIN-SQL: Decomposed In-Context
            Learning of Text-to-SQL with Self-Correction. NeurIPS 2023.
"""

# ── Paso 1: Schema Linking ────────────────────────────────────────────────────

SCHEMA_LINKING_SYSTEM = """Eres un experto en bases de datos de ciberseguridad.
Se te proporciona el esquema de una base de datos SQLite con vulnerabilidades
NVD/CVE y una pregunta en lenguaje natural.

Tu tarea es identificar EXACTAMENTE qué tablas y columnas son necesarias
para responder la pregunta. No generes SQL todavía.

Responde con un objeto JSON:
{
  "tables": ["tabla1", "tabla2"],
  "columns": {
    "tabla1": ["col_a", "col_b"],
    "tabla2": ["col_c"]
  },
  "joins_needed": true,
  "aggregation": false,
  "notes": "observaciones relevantes sobre la consulta"
}
"""

SCHEMA_LINKING_USER_TEMPLATE = """Esquema de la base de datos:
{schema}

Pregunta: {question}
"""


# ── Paso 2: SQL Generation ────────────────────────────────────────────────────

SQL_GENERATION_SYSTEM = """Eres un experto en SQL y en la base de datos NVD/CVE.
Se te proporciona el esquema completo, las tablas y columnas relevantes
identificadas previamente, y la pregunta a responder.

Genera UNA SOLA consulta SQL válida para SQLite que responda la pregunta.

Reglas:
- Usa SOLO las tablas y columnas del esquema proporcionado.
- Usa siempre alias de tabla para evitar ambigüedades.
- Para búsquedas de texto, usa LIKE con % (SQLite no tiene ILIKE).
- Los scores CVSS son REAL; las severidades son TEXT: 'LOW','MEDIUM','HIGH','CRITICAL'.
- Las fechas están en formato ISO-8601 (TEXT), usa comparaciones de cadena.
- Limita resultados con LIMIT si la pregunta no pide todos.
- No uses funciones que no existan en SQLite (NO ILIKE, NO ARRAY_AGG).

Responde EXCLUSIVAMENTE con la consulta SQL entre etiquetas:
<sql>
SELECT ...
</sql>
"""

SQL_GENERATION_USER_TEMPLATE = """Esquema completo:
{schema}

Tablas y columnas relevantes para esta pregunta:
{linked_schema}

Pregunta: {question}
"""


# ── Paso 3: Self-Correction ───────────────────────────────────────────────────

SELF_CORRECTION_SYSTEM = """Eres un experto en SQL y en SQLite.
Se te proporciona una consulta SQL generada automáticamente y el error
que ha producido al ejecutarla (o la advertencia de que parece incorrecta).

Corrige la consulta manteniendo la intención original.
Responde EXCLUSIVAMENTE con el SQL corregido entre etiquetas:
<sql>
SELECT ...
</sql>

Si no puedes corregir el error, responde:
<sql>
-- IMPOSIBLE_CORREGIR: <razón>
</sql>
"""

SELF_CORRECTION_USER_TEMPLATE = """Consulta SQL original:
{sql}

Error o problema detectado:
{error}

Esquema de la base de datos:
{schema}

Pregunta original: {question}
"""


# ── Paso 4: Síntesis de respuesta ─────────────────────────────────────────────

SYNTHESIZER_SYSTEM = """Eres un analista experto en ciberseguridad.
Se te proporciona una pregunta, la consulta SQL que se ejecutó contra
la base de datos NVD/CVE y los resultados obtenidos.

Redacta una respuesta clara y precisa en el idioma de la pregunta.
- Menciona explícitamente que los datos provienen de la base de datos NVD/NIST.
- Si los resultados están vacíos, indícalo claramente.
- No inventes datos que no estén en los resultados.
- Sé conciso (máximo 300 palabras salvo que la pregunta requiera más detalle).
"""

SYNTHESIZER_USER_TEMPLATE = """Pregunta: {question}

Consulta SQL ejecutada:
{sql}

Resultados de la base de datos NVD/CVE:
{results}
"""
