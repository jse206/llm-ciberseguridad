"""
Prompts para la Arquitectura D (Toolformer / ReAct).

Implementa el ciclo ReAct (Yao et al., 2023): el LLM alterna entre
razonamiento (Thought) y acción (Action) hasta alcanzar una respuesta final.

Formato del ciclo:
  Thought: <razonamiento sobre qué hacer a continuación>
  Action: <nombre_herramienta>(<json_params>)
  Observation: <resultado de la herramienta>
  ... (repetir hasta convergencia)
  Final Answer: <respuesta en lenguaje natural>

Este paradigma es el más flexible: el LLM decide autónomamente
qué fuente de conocimiento usar según la naturaleza de la pregunta,
pudiendo combinar varias herramientas en una sola consulta.

Referencia: Schick, T. et al. (2023). Toolformer. NeurIPS 2023.
               Yao, S. et al. (2023). ReAct: Synergizing Reasoning and
               Acting in Language Models. ICLR 2023.
"""

from arquitectura_D_Toolformer.tools import tools_schema_str

# ── Prompt del sistema: agente ReAct ─────────────────────────────────────────

REACT_SYSTEM_TEMPLATE = """Eres un agente experto en ciberseguridad que responde
preguntas usando un conjunto de herramientas especializadas.

Tienes acceso a las siguientes herramientas:

{tools_schema}

Para responder cada pregunta, sigue este formato estrictamente:

Thought: <razona qué información necesitas y qué herramienta usar>
Action: <nombre_herramienta>
Action Input: <parámetros en JSON>
Observation: <resultado de la herramienta — lo proporciona el sistema>
... (puedes repetir Thought/Action/Observation hasta 4 veces si necesitas más información)
Thought: <razonamiento final basado en todas las observaciones>
Final Answer: <respuesta completa y fundamentada en lenguaje natural>

Reglas:
- Usa SOLO herramientas de la lista proporcionada.
- Los parámetros de Action Input deben ser JSON válido.
- Fundamenta siempre la Final Answer en las Observations obtenidas.
- Si ninguna herramienta puede responder la pregunta, indícalo en Final Answer.
- Responde en el idioma de la pregunta.
- Máximo 4 ciclos Thought/Action/Observation por pregunta.
"""


def get_react_system() -> str:
    return REACT_SYSTEM_TEMPLATE.format(tools_schema=tools_schema_str())


# ── Prompt de usuario: pregunta inicial ───────────────────────────────────────

REACT_USER_TEMPLATE = "Pregunta: {question}"


# ── Prompt de continuación: inyecta la Observation ───────────────────────────

OBSERVATION_TEMPLATE = "Observation: {observation}"


# ── Prompt de síntesis final (si el ciclo no converge) ───────────────────────

FORCED_FINAL_SYSTEM = """El agente ha alcanzado el límite de iteraciones.
Con la información recopilada hasta ahora, proporciona la mejor respuesta
posible. Si la información es insuficiente, indícalo claramente.
No inventes datos que no provengan de las Observations anteriores.
"""

FORCED_FINAL_USER_TEMPLATE = """Pregunta original: {question}

Información recopilada hasta ahora:
{observations_summary}

Proporciona la mejor respuesta posible con esta información.
"""
