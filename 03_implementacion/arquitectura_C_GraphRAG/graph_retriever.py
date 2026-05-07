"""
Recuperación de subgrafos relevantes del grafo MITRE ATT&CK.

Implementa la etapa de recuperación del paradigma GraphRAG
(Edge et al., 2024; Pan et al., 2024) adaptada al grafo de conocimiento
MITRE ATT&CK Enterprise en formato NetworkX.

Estrategia de recuperación en dos fases:
  1. Entity matching — localiza nodos cuyo nombre, mitre_id o descripción
     coincida con términos de la pregunta (búsqueda léxica).
  2. Neighborhood expansion — expande el subgrafo a N saltos de profundidad
     para capturar el contexto relacional (técnica ↔ táctica ↔ grupo ↔ mitigación).

Referencia: Edge, D. et al. (2024). From Local to Global: A Graph RAG
               Approach to Query-Focused Summarization. arXiv:2404.16130.
               Pan, S. et al. (2024). Unifying LLMs and Knowledge Graphs.
               IEEE TKDE.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import networkx as nx

from data.mitre_loader import load_graph

logger = logging.getLogger(__name__)

_MAX_SEED_NODES = 5       # nodos semilla máximos por pregunta
_MAX_SUBGRAPH_NODES = 40  # límite del subgrafo para no saturar el prompt

# Indicadores de pregunta compleja que requieren más saltos en el grafo
_COMPLEX_KEYWORDS = {
    "todos", "todas", "cuántos", "cuántas", "lista", "relacionado", "relacionadas",
    "grupos", "técnicas", "tácticas", "mitigaciones", "múltiples", "combina",
    "qué grupos", "cadena", "kill chain", "comparar", "compara", "diferencia",
}
_STOPWORDS_ES = {
    "cuál", "cuáles", "cómo", "qué", "está", "para", "según", "tiene", "una",
    "uno", "las", "los", "del", "con", "que", "son", "más", "este", "esta",
    "sobre", "entre", "desde", "hasta", "cuando",
}


@lru_cache(maxsize=1)
def _get_graph() -> nx.DiGraph:
    return load_graph()


def _compute_hops(question: str) -> int:
    """Estima el número de saltos necesarios según la complejidad de la pregunta."""
    q_lower = question.lower()
    hits = sum(1 for kw in _COMPLEX_KEYWORDS if kw in q_lower)
    if hits >= 3:
        return 3
    if hits >= 1:
        return 2
    return 1


def _entity_matching_fallback(G: nx.DiGraph, question: str) -> list[str]:
    """
    Fallback semántico: busca por tokens de la pregunta cuando la búsqueda
    exacta no encuentra nodos semilla. Evita falsos positivos usando un umbral
    mínimo y filtrando stopwords.
    """
    q_lower = question.lower()
    tokens = [
        w.strip("¿?.,;:()")
        for w in q_lower.split()
        if len(w) > 4 and w.strip("¿?.,;:()") not in _STOPWORDS_ES
    ]
    if not tokens:
        return []

    scored: list[tuple[float, str]] = []
    for node_key, attrs in G.nodes(data=True):
        name = (attrs.get("name") or "").lower()
        desc = (attrs.get("description") or "").lower()[:300]
        score = 0.0
        for token in tokens:
            if token in name:
                score += 2.0
            elif token in desc:
                score += 0.5
        if score >= 1.0:
            scored.append((score, node_key))

    scored.sort(reverse=True)
    logger.info("Fallback semántico: %d nodos candidatos para '%s'", len(scored), question[:60])
    return [key for _, key in scored[:_MAX_SEED_NODES]]


def retrieve(question: str, hops: int | None = None) -> dict:
    """
    Recupera el subgrafo más relevante para la pregunta dada.

    Args:
        question : pregunta en lenguaje natural.
        hops     : profundidad de expansión (auto-calculado si None).

    Returns:
        dict con:
          - "nodes"     : lista de nodos con sus atributos
          - "edges"     : lista de aristas (src, relation, tgt)
          - "seed_nodes": nodos semilla encontrados por entity matching
          - "context"   : representación textual del subgrafo para el prompt
    """
    if hops is None:
        hops = _compute_hops(question)

    G = _get_graph()
    seeds = _entity_matching(G, question)
    if not seeds:
        logger.info("Sin semillas exactas; activando fallback semántico para: %s", question[:60])
        seeds = _entity_matching_fallback(G, question)
    if not seeds:
        logger.warning("Sin nodos semilla (exacto ni fallback) para: %s", question)
        return {"nodes": [], "edges": [], "seed_nodes": [], "context": ""}

    subgraph_nodes = _expand_neighborhood(G, seeds, hops)
    subgraph = G.subgraph(subgraph_nodes)

    nodes = [{"id": n, **G.nodes[n]} for n in subgraph.nodes()]
    edges = [
        {"src": u, "relation": d.get("relation", ""), "tgt": v, "description": d.get("description", "")}
        for u, v, d in subgraph.edges(data=True)
    ]
    context = _build_context(nodes, edges, seeds)

    logger.info(
        "Recuperados %d nodos, %d aristas (seeds: %s)",
        len(nodes), len(edges), seeds,
    )
    return {"nodes": nodes, "edges": edges, "seed_nodes": seeds, "context": context}


def _entity_matching(G: nx.DiGraph, question: str) -> list[str]:
    """
    Localiza nodos cuyo nombre o ID MITRE aparece en la pregunta.
    Devuelve lista de node keys ordenados por relevancia.
    """
    q_lower = question.lower()
    # Extraer IDs MITRE explícitos (T1234, T1234.001, G0001, S0001, M1042…)
    mitre_ids = re.findall(r"\b[TGSM]\d{4}(?:\.\d{3})?\b", question.upper())

    scored: list[tuple[float, str]] = []
    for node_key, attrs in G.nodes(data=True):
        score = 0.0
        name = attrs.get("name", "").lower()
        mitre_id = attrs.get("mitre_id", "") or ""

        if mitre_id.upper() in mitre_ids:
            score += 10.0
        if name and name in q_lower:
            score += 5.0
        # Coincidencia parcial por palabras
        for word in q_lower.split():
            if len(word) > 3 and word in name:
                score += 1.0

        if score > 0:
            scored.append((score, node_key))

    scored.sort(reverse=True)
    return [key for _, key in scored[:_MAX_SEED_NODES]]


def _expand_neighborhood(G: nx.DiGraph, seeds: list[str], hops: int) -> set[str]:
    """BFS limitado a `hops` saltos desde los nodos semilla."""
    visited = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            neighbors = set(G.predecessors(node)) | set(G.successors(node))
            next_frontier |= neighbors - visited
        visited |= next_frontier
        frontier = next_frontier
        if len(visited) >= _MAX_SUBGRAPH_NODES:
            break
    return set(list(visited)[:_MAX_SUBGRAPH_NODES])


def _build_context(nodes: list[dict], edges: list[dict], seeds: list[str]) -> str:
    """Serializa el subgrafo a texto estructurado para el prompt LLM."""
    lines = ["=== SUBGRAFO MITRE ATT&CK ===\n"]

    lines.append("--- Entidades recuperadas ---")
    for n in nodes:
        marker = "★ " if n["id"] in seeds else "  "
        mitre_id = n.get("mitre_id") or ""
        node_type = n.get("type", "")
        name = n.get("name", "")
        desc = (n.get("description") or "")[:200]
        lines.append(f"{marker}[{node_type}] {mitre_id} — {name}")
        if desc:
            lines.append(f"     {desc}{'…' if len(n.get('description','')) > 200 else ''}")

        # Atributos específicos por tipo
        if node_type == "attack-pattern":
            tactics = n.get("kill_chain_phases", [])
            platforms = n.get("platforms", [])
            if tactics:
                lines.append(f"     Tácticas: {', '.join(tactics)}")
            if platforms:
                lines.append(f"     Plataformas: {', '.join(platforms)}")

    lines.append("\n--- Relaciones ---")
    for e in edges[:60]:  # limitar aristas en el contexto
        rel_desc = f" ({e['description'][:80]})" if e.get("description") else ""
        lines.append(f"  {e['src']} --[{e['relation']}]--> {e['tgt']}{rel_desc}")

    return "\n".join(lines)
