"""
Script de configuración inicial del entorno.

Ejecutar UNA VEZ antes de comenzar los experimentos:
    cd 03_implementacion
    python setup.py

Acciones:
  1. Verifica variables de entorno obligatorias
  2. Inicializa la base de datos SQLite NVD/CVE
  3. Descarga los CVEs de los años configurados en NVD_YEARS
  4. Descarga y construye el grafo MITRE ATT&CK
  5. Imprime un resumen de los datos disponibles
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup")


def check_env() -> bool:
    from config import OPENAI_API_KEY, ABUSEIPDB_API_KEY
    ok = True
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY no está configurada en .env")
        ok = False
    if not ABUSEIPDB_API_KEY:
        logger.warning("ABUSEIPDB_API_KEY no configurada (Arquitectura B limitada)")
    return ok


def setup_nvd() -> dict:
    from data.nvd_downloader import download_all
    logger.info("=== Descargando base de datos NVD/CVE ===")
    return download_all()


def setup_mitre() -> dict:
    from data.mitre_loader import setup, graph_stats
    logger.info("=== Construyendo grafo MITRE ATT&CK ===")
    G = setup()
    return graph_stats(G)


def print_summary(nvd_result: dict, mitre_stats: dict) -> None:
    print("\n" + "=" * 60)
    print("RESUMEN DEL ENTORNO — LLM Ciberseguridad")
    print("=" * 60)
    print(f"\n[NVD/CVE] Base de datos:")
    print(f"   Total CVEs: {nvd_result['db_stats']['total_cves']:,}")
    print(f"   Por severidad: {nvd_result['db_stats']['by_severity']}")
    print(f"   Por año: {nvd_result['by_year']}")
    print(f"\n[MITRE] Grafo ATT&CK:")
    print(f"   Nodos: {mitre_stats['nodes']:,}")
    print(f"   Aristas: {mitre_stats['edges']:,}")
    print(f"   Por tipo: {mitre_stats['by_type']}")
    print("\n[OK] Entorno listo para ejecutar los experimentos.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logger.info("Iniciando setup…")

    if not check_env():
        logger.error("Configura el archivo .env antes de continuar.")
        sys.exit(1)

    nvd_result = setup_nvd()
    mitre_stats = setup_mitre()
    print_summary(nvd_result, mitre_stats)
