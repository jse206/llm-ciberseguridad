"""
Descarga de CVEs desde la NVD API v2.0 y carga en SQLite.

Documentación de la API: https://nvd.nist.gov/developers/vulnerabilities
Referencia: NIST (2024). National Vulnerability Database (NVD) API v2.0.

Uso:
    python -m data.nvd_downloader          # descarga los años en NVD_YEARS
    python -m data.nvd_downloader --year 2024
"""

import json
import logging
import time
import argparse
from datetime import datetime, timedelta, timezone

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from config import (
    NVD_API_BASE,
    NVD_API_KEY,
    NVD_DB_PATH,
    NVD_RESULTS_PER_PAGE,
    NVD_YEARS,
)
from data.nvd_db import init_db, insert_cve, get_conn, db_stats, count_cves_for_year

logger = logging.getLogger(__name__)

# La API pública sin clave permite 5 req/30s; con clave, 50 req/30s
_RATE_SLEEP = 6.5 if not NVD_API_KEY else 0.65


def _build_headers() -> dict:
    h = {"Accept": "application/json"}
    if NVD_API_KEY:
        h["apiKey"] = NVD_API_KEY
    return h


def parse_cve(item: dict) -> dict:
    """
    Normaliza un ítem de la respuesta NVD API v2.0 al esquema interno.
    Devuelve un dict listo para insert_cve().
    """
    cve = item["cve"]
    cve_id = cve["id"]

    # Descripción en inglés
    description = next(
        (d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"),
        "",
    )

    # CVSS v3
    cvss_v3_score, cvss_v3_severity, cvss_v3_vector = None, None, None
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        if key in metrics and metrics[key]:
            m = metrics[key][0]["cvssData"]
            cvss_v3_score = m.get("baseScore")
            cvss_v3_severity = m.get("baseSeverity")
            cvss_v3_vector = m.get("vectorString")
            break

    # CVSS v2
    cvss_v2_score = None
    if "cvssMetricV2" in metrics and metrics["cvssMetricV2"]:
        cvss_v2_score = metrics["cvssMetricV2"][0]["cvssData"].get("baseScore")

    # CWEs
    cwes = []
    for weakness in cve.get("weaknesses", []):
        for d in weakness.get("description", []):
            if d["lang"] == "en" and d["value"].startswith("CWE-"):
                cwes.append(d["value"])

    # CPEs
    cpes = []
    for config in cve.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                cpes.append(
                    {"uri": match["criteria"], "vulnerable": match.get("vulnerable", True)}
                )

    # Referencias
    references = [
        {
            "url": r["url"],
            "source": r.get("source", ""),
            "tags": json.dumps(r.get("tags", [])),
        }
        for r in cve.get("references", [])
    ]

    return {
        "cve_id": cve_id,
        "description": description,
        "cvss_v3_score": cvss_v3_score,
        "cvss_v3_severity": cvss_v3_severity,
        "cvss_v3_vector": cvss_v3_vector,
        "cvss_v2_score": cvss_v2_score,
        "published": cve.get("published", ""),
        "last_modified": cve.get("lastModified", ""),
        "vuln_status": cve.get("vulnStatus", ""),
        "cwes": cwes,
        "cpes": cpes,
        "references": references,
    }


@retry(
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    wait=wait_exponential(multiplier=10, min=10, max=120),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch_page(url: str) -> dict:
    """Descarga una página de la API NVD con retry automático ante timeouts."""
    resp = requests.get(url, headers=_build_headers(), timeout=120)
    resp.raise_for_status()
    return resp.json()


def download_year(year: int) -> int:
    """
    Descarga todos los CVEs publicados en `year` y los inserta en SQLite.
    Devuelve el número de CVEs insertados.
    """
    existing = count_cves_for_year(year)
    if existing > 0:
        logger.info("Año %d ya tiene %d CVEs en BD — omitiendo.", year, existing)
        return existing

    # La API NVD v2.0 limita el rango de fechas a ~120 días por petición.
    # Dividimos el año en bloques de 90 días para mantenernos dentro del límite.
    logger.info("Descargando CVEs del año %d…", year)
    total_inserted = 0

    chunk_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    chunk_days = timedelta(days=90)

    with get_conn() as conn:
        while chunk_start <= year_end:
            chunk_end = min(chunk_start + chunk_days - timedelta(seconds=1), year_end)
            start_str = chunk_start.strftime("%Y-%m-%dT%H:%M:%S.000")
            end_str = chunk_end.strftime("%Y-%m-%dT%H:%M:%S.999")

            start_index = 0
            first_page = True
            pbar = None

            while True:
                url = (
                    f"{NVD_API_BASE}?pubStartDate={start_str}&pubEndDate={end_str}"
                    f"&resultsPerPage={NVD_RESULTS_PER_PAGE}&startIndex={start_index}"
                )
                data = _fetch_page(url)

                total_results = data["totalResults"]
                items = data.get("vulnerabilities", [])

                if first_page:
                    pbar = tqdm(
                        total=total_results,
                        desc=f"NVD {year} {start_str[:10]}",
                        unit="CVE",
                    )
                    first_page = False

                for item in items:
                    entry = parse_cve(item)
                    insert_cve(conn, entry)
                    total_inserted += 1
                    pbar.update(1)

                start_index += len(items)
                if start_index >= total_results or not items:
                    break

                time.sleep(_RATE_SLEEP)

            if pbar:
                pbar.close()

            chunk_start = chunk_end + timedelta(seconds=1)
            time.sleep(_RATE_SLEEP)

    logger.info("Año %d: %d CVEs insertados.", year, total_inserted)
    return total_inserted


def download_all(years: list[int] | None = None) -> dict:
    """
    Descarga todos los años indicados y devuelve un resumen.
    """
    init_db()
    years = years or NVD_YEARS
    summary = {}
    for year in sorted(years):
        summary[year] = download_year(year)
        time.sleep(1)

    stats = db_stats()
    logger.info("Descarga completa. Estadísticas: %s", stats)
    return {"by_year": summary, "db_stats": stats}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Descarga CVEs desde NVD API v2.0")
    parser.add_argument("--year", type=int, help="Descargar solo este año")
    args = parser.parse_args()

    if args.year:
        init_db()
        n = download_year(args.year)
        print(f"Insertados {n} CVEs del año {args.year}.")
    else:
        result = download_all()
        print(result)
