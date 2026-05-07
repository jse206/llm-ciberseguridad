"""
Carga del framework MITRE ATT&CK (Enterprise) desde STIX 2.0 a NetworkX.

Referencia: MITRE Corporation (2024). ATT&CK® for Enterprise.
               Edge et al. (2024). arXiv:2404.16130 (GraphRAG).

El grafo resultante es la base de la Arquitectura C (GraphRAG).
Cada nodo tiene atributos STIX completos; las aristas codifican
las relaciones semánticas del framework (usa, mitiga, subtécnica-de…).

Uso:
    python -m data.mitre_loader            # descarga + construye grafo
    python -m data.mitre_loader --stats    # muestra estadísticas del grafo
"""

import json
import logging
import pickle
import argparse
from pathlib import Path

import requests
import networkx as nx
from tqdm import tqdm

from config import MITRE_STIX_URL, MITRE_STIX_PATH, MITRE_GRAPH_PATH

logger = logging.getLogger(__name__)

# Tipos STIX relevantes
_STIX_TYPES = {
    "attack-pattern",    # Técnicas y sub-técnicas
    "intrusion-set",     # Grupos APT
    "course-of-action",  # Mitigaciones
    "malware",           # Software malicioso
    "tool",              # Herramientas legítimas usadas en ataques
    "x-mitre-tactic",   # Tácticas (fases del ciclo de ataque)
    "x-mitre-data-source",
    "relationship",
}


def download_stix(force: bool = False) -> Path:
    """Descarga el fichero STIX de MITRE ATT&CK si no existe."""
    if MITRE_STIX_PATH.exists() and not force:
        logger.info("STIX ya descargado en %s", MITRE_STIX_PATH)
        return MITRE_STIX_PATH

    logger.info("Descargando MITRE ATT&CK Enterprise STIX…")
    resp = requests.get(MITRE_STIX_URL, timeout=120)
    resp.raise_for_status()
    MITRE_STIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    MITRE_STIX_PATH.write_bytes(resp.content)
    logger.info("STIX guardado: %.1f MB", MITRE_STIX_PATH.stat().st_size / 1e6)
    return MITRE_STIX_PATH


def _extract_mitre_id(obj: dict) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def _node_attrs(obj: dict) -> dict:
    """Extrae los atributos relevantes de un objeto STIX para el nodo del grafo."""
    attrs = {
        "stix_id": obj["id"],
        "type": obj["type"],
        "name": obj.get("name", ""),
        "description": obj.get("description", ""),
        "mitre_id": _extract_mitre_id(obj),
        "created": obj.get("created", ""),
        "modified": obj.get("modified", ""),
        "revoked": obj.get("revoked", False),
        "deprecated": obj.get("x_mitre_deprecated", False),
    }
    # Atributos específicos por tipo
    if obj["type"] == "attack-pattern":
        attrs["is_subtechnique"] = obj.get("x_mitre_is_subtechnique", False)
        attrs["platforms"] = obj.get("x_mitre_platforms", [])
        attrs["detection"] = obj.get("x_mitre_detection", "")
        attrs["kill_chain_phases"] = [
            p["phase_name"] for p in obj.get("kill_chain_phases", [])
        ]
    elif obj["type"] == "x-mitre-tactic":
        attrs["shortname"] = obj.get("x_mitre_shortname", "")
    elif obj["type"] == "intrusion-set":
        attrs["aliases"] = obj.get("aliases", [])
    return attrs


def build_graph(stix_path: Path = MITRE_STIX_PATH) -> nx.DiGraph:
    """
    Construye un grafo dirigido NetworkX a partir del bundle STIX.

    Nodos: técnicas, tácticas, grupos, mitigaciones, herramientas, malware.
    Aristas: relaciones STIX (uses, mitigates, subtechnique-of, detects…).
    """
    logger.info("Cargando STIX desde %s…", stix_path)
    bundle = json.loads(stix_path.read_text(encoding="utf-8"))
    objects = bundle.get("objects", [])

    G = nx.DiGraph()
    id_map: dict[str, str] = {}  # stix_id → node_key (mitre_id o stix_id)

    # ── Paso 1: nodos ──────────────────────────────────────────────────────────
    logger.info("Construyendo nodos…")
    for obj in tqdm(objects, desc="Nodos STIX"):
        if obj["type"] not in _STIX_TYPES or obj["type"] == "relationship":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        attrs = _node_attrs(obj)
        key = attrs["mitre_id"] or obj["id"]
        id_map[obj["id"]] = key
        G.add_node(key, **attrs)

    # ── Paso 2: aristas ────────────────────────────────────────────────────────
    logger.info("Construyendo aristas…")
    for obj in tqdm(objects, desc="Relaciones STIX"):
        if obj["type"] != "relationship":
            continue
        src = id_map.get(obj.get("source_ref", ""))
        tgt = id_map.get(obj.get("target_ref", ""))
        if src and tgt:
            G.add_edge(
                src,
                tgt,
                relation=obj.get("relationship_type", ""),
                description=obj.get("description", ""),
            )

    logger.info(
        "Grafo construido: %d nodos, %d aristas",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def save_graph(G: nx.DiGraph, path: Path = MITRE_GRAPH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    logger.info("Grafo guardado en %s", path)


def load_graph(path: Path = MITRE_GRAPH_PATH) -> nx.DiGraph:
    if not path.exists():
        raise FileNotFoundError(
            f"Grafo no encontrado en {path}. Ejecuta: python -m data.mitre_loader"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def graph_stats(G: nx.DiGraph) -> dict:
    """Estadísticas del grafo MITRE ATT&CK."""
    by_type: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    by_relation: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "unknown")
        by_relation[r] = by_relation.get(r, 0) + 1

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "by_type": by_type,
        "by_relation": by_relation,
    }


def setup() -> nx.DiGraph:
    """Punto de entrada para el setup inicial: descarga + construye + guarda."""
    download_stix()
    G = build_graph()
    save_graph(G)
    return G


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Carga MITRE ATT&CK Enterprise")
    parser.add_argument("--force", action="store_true", help="Forzar re-descarga STIX")
    parser.add_argument("--stats", action="store_true", help="Solo mostrar estadísticas")
    args = parser.parse_args()

    if args.stats:
        G = load_graph()
        import pprint
        pprint.pprint(graph_stats(G))
    else:
        G = setup() if not args.force else (download_stix(force=True) and build_graph())
        if not isinstance(G, nx.DiGraph):
            G = build_graph()
            save_graph(G)
        pprint_stats = graph_stats(G)
        print(pprint_stats)
