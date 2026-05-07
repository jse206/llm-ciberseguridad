"""
Definición de las herramientas disponibles para la Arquitectura D (Toolformer).

Cada herramienta es un wrapper sobre una de las arquitecturas anteriores
o sobre una API directa. El LLM selecciona autónomamente qué herramienta
usar en cada turno del ciclo ReAct (Yao et al., 2023).

Herramientas disponibles:
  - query_nvd_local   : consulta Text2SQL sobre la BD NVD/CVE local (Arq. A)
  - query_nvd_api     : consulta a la NVD API v2.0 en tiempo real (Arq. B)
  - query_mitre_graph : consulta GraphRAG sobre MITRE ATT&CK (Arq. C)
  - check_ip          : reputación de IP en AbuseIPDB (Arq. B)

Referencia: Schick, T. et al. (2023). Toolformer. NeurIPS 2023.
               Yao, S. et al. (2023). ReAct. ICLR 2023.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Descriptor de una herramienta invocable por el LLM."""
    name: str
    description: str
    parameters: dict          # JSON Schema de los parámetros
    fn: Callable              # función que ejecuta la herramienta
    examples: list[str] = field(default_factory=list)  # ejemplos para el prompt


def _run_nvd_local(question: str) -> dict:
    """Delega en la Arquitectura A (Text2SQL)."""
    from arquitectura_A_Text2SQL.chain import ArchitectureAChain
    result = ArchitectureAChain().run(question)
    return {
        "answer": result["answer"],
        "sql": result.get("sql"),
        "rows": result.get("sql_rows", 0),
        "error": result.get("sql_error"),
        "source": "NVD/CVE local SQLite",
    }


def _run_nvd_api(cve_id: str | None = None, keyword: str | None = None,
                 severity: str | None = None, max_results: int = 5) -> dict:
    """Delega en los clientes de la Arquitectura B."""
    from arquitectura_B_API.api_client import get_cve_by_id, search_cves
    if cve_id:
        return get_cve_by_id(cve_id)
    return search_cves(keyword=keyword, severity=severity, max_results=max_results)


def _run_mitre_graph(question: str) -> dict:
    """Delega en la Arquitectura C (GraphRAG)."""
    from arquitectura_C_GraphRAG.graph_retriever import retrieve
    retrieval = retrieve(question)
    return {
        "context": retrieval["context"],
        "seed_nodes": retrieval["seed_nodes"],
        "nodes": len(retrieval["nodes"]),
        "edges": len(retrieval["edges"]),
        "source": "MITRE ATT&CK Enterprise (grafo local)",
    }


def _run_check_ip(ip_address: str) -> dict:
    """Delega en el cliente AbuseIPDB de la Arquitectura B."""
    from arquitectura_B_API.api_client import check_ip
    return check_ip(ip_address)


# ── Registro de herramientas ──────────────────────────────────────────────────

TOOLS: list[Tool] = [
    Tool(
        name="query_nvd_local",
        description=(
            "Consulta la base de datos NVD/CVE local mediante lenguaje natural. "
            "Ideal para preguntas sobre CVSS scores, severidades, conteos, "
            "agregaciones y búsquedas sobre los CVEs descargados (2022-2025). "
            "Devuelve respuesta sintetizada y la SQL ejecutada."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Pregunta sobre CVEs en lenguaje natural"}
            },
            "required": ["question"],
        },
        fn=_run_nvd_local,
        examples=[
            "¿Cuántos CVEs críticos hay en 2024?",
            "¿Cuál es el CVSS de CVE-2021-44228?",
            "Lista los 5 CVEs con mayor score de 2023",
        ],
    ),
    Tool(
        name="query_nvd_api",
        description=(
            "Consulta la NVD API v2.0 en tiempo real. "
            "Úsala cuando necesites datos actualizados que pueden no estar en la BD local, "
            "para buscar por ID de CVE específico, palabra clave o severidad. "
            "Devuelve los datos crudos de la API."
        ),
        parameters={
            "type": "object",
            "properties": {
                "cve_id": {"type": "string", "description": "ID del CVE (p. ej. CVE-2024-1234)"},
                "keyword": {"type": "string", "description": "Término de búsqueda"},
                "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                "max_results": {"type": "integer", "default": 5},
            },
        },
        fn=_run_nvd_api,
        examples=[
            "Obtener datos actualizados de CVE-2024-3400",
            "Buscar CVEs recientes de severity CRITICAL sobre Cisco",
        ],
    ),
    Tool(
        name="query_mitre_graph",
        description=(
            "Consulta el grafo de conocimiento MITRE ATT&CK Enterprise. "
            "Ideal para preguntas sobre técnicas de ataque (Txxxx), tácticas, "
            "grupos APT (Gxxxx), mitigaciones (Mxxxx), herramientas y sus relaciones. "
            "Devuelve el subgrafo relevante serializado como texto."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Pregunta sobre MITRE ATT&CK"}
            },
            "required": ["question"],
        },
        fn=_run_mitre_graph,
        examples=[
            "¿Qué técnicas usa el grupo APT28?",
            "¿Qué mitigaciones cubre T1566 Phishing?",
            "¿En qué táctica está T1059?",
        ],
    ),
    Tool(
        name="check_ip",
        description=(
            "Consulta la reputación de una dirección IP en AbuseIPDB. "
            "Devuelve abuse score (0-100), país, ISP y categorías de abuso. "
            "Úsala cuando la pregunta mencione una dirección IP específica."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ip_address": {"type": "string", "description": "Dirección IPv4 o IPv6"}
            },
            "required": ["ip_address"],
        },
        fn=_run_check_ip,
        examples=["¿Es 1.2.3.4 una IP maliciosa?", "Analiza la reputación de 45.33.32.156"],
    ),
]

TOOLS_BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}


def call_tool(name: str, params: dict) -> dict:
    """
    Invoca una herramienta por nombre y devuelve su resultado.
    Captura errores para que el ciclo ReAct pueda continuar.
    """
    tool = TOOLS_BY_NAME.get(name)
    if not tool:
        return {"error": f"Herramienta '{name}' no existe. Disponibles: {list(TOOLS_BY_NAME)}"}
    try:
        result = tool.fn(**params)
        logger.info("Tool '%s' ejecutada OK", name)
        return result if isinstance(result, dict) else {"result": result}
    except Exception as e:
        logger.warning("Tool '%s' error: %s", name, e)
        return {"error": str(e)}


def tools_schema_str() -> str:
    """Serializa las herramientas como texto para incluir en el prompt del LLM."""
    lines = []
    for t in TOOLS:
        lines.append(f"### {t.name}")
        lines.append(f"  {t.description}")
        lines.append(f"  Parámetros: {json.dumps(t.parameters['properties'], ensure_ascii=False)}")
        if t.examples:
            lines.append(f"  Ejemplos: {' | '.join(t.examples)}")
        lines.append("")
    return "\n".join(lines)
