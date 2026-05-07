"""
Clientes para las APIs externas de la Arquitectura B.

APIs integradas:
  - NVD API v2.0  (NIST, 2024) — consultas CVE en tiempo real
  - AbuseIPDB     (AbuseIPDB, 2024) — reputación de direcciones IP

Cada función devuelve siempre un dict con:
  - "data"    : resultado de la API (None si falla)
  - "source"  : URL canónica usada (para trazabilidad)
  - "error"   : mensaje de error o None
  - "latency_s": tiempo de respuesta en segundos
"""

import time
import logging
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    NVD_API_BASE,
    NVD_API_KEY,
    NVD_RESULTS_PER_PAGE,
    ABUSEIPDB_API_KEY,
    ABUSEIPDB_BASE,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 30


def _nvd_headers() -> dict:
    h = {"Accept": "application/json"}
    if NVD_API_KEY:
        h["apiKey"] = NVD_API_KEY
    return h


# ── NVD API ───────────────────────────────────────────────────────────────────

@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3))
def get_cve_by_id(cve_id: str) -> dict:
    """
    Recupera un CVE concreto por su identificador (p. ej. 'CVE-2021-44228').
    Endpoint: GET /rest/json/cves/2.0?cveId=<id>
    """
    url = NVD_API_BASE
    params = {"cveId": cve_id.upper()}
    source = f"{url}?cveId={cve_id.upper()}"

    t0 = time.perf_counter()
    try:
        resp = requests.get(url, headers=_nvd_headers(), params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        result = vulns[0]["cve"] if vulns else None
        return {"data": result, "source": source, "error": None,
                "latency_s": round(time.perf_counter() - t0, 3)}
    except Exception as e:
        logger.warning("NVD get_cve_by_id(%s) error: %s", cve_id, e)
        return {"data": None, "source": source, "error": str(e),
                "latency_s": round(time.perf_counter() - t0, 3)}


@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3))
def search_cves(
    keyword: str | None = None,
    severity: str | None = None,
    cwe_id: str | None = None,
    pub_start: str | None = None,
    pub_end: str | None = None,
    max_results: int = 10,
) -> dict:
    """
    Búsqueda de CVEs con filtros opcionales.

    Args:
        keyword   : término de búsqueda en descripción.
        severity  : LOW | MEDIUM | HIGH | CRITICAL.
        cwe_id    : p. ej. 'CWE-79'.
        pub_start : fecha inicio publicación (YYYY-MM-DD).
        pub_end   : fecha fin publicación (YYYY-MM-DD).
        max_results: máximo de CVEs a devolver.
    """
    # Parámetros sin fechas (requests puede URL-encodarlos sin problema)
    params: dict[str, Any] = {"resultsPerPage": min(max_results, NVD_RESULTS_PER_PAGE)}
    if keyword:
        params["keywordSearch"] = keyword
    if severity:
        params["cvssV3Severity"] = severity.upper()
    if cwe_id:
        params["cweId"] = cwe_id

    # Las fechas se añaden directamente a la URL para evitar que requests
    # codifique los dos puntos (:→%3A), lo que causa 404 en la API NVD.
    date_fragment = ""
    if pub_start:
        date_fragment += f"&pubStartDate={pub_start}T00:00:00.000"
    if pub_end:
        date_fragment += f"&pubEndDate={pub_end}T23:59:59.999"

    source = NVD_API_BASE + "?" + "&".join(f"{k}={v}" for k, v in params.items()) + date_fragment
    try:
        t0 = time.perf_counter()
        base_url = NVD_API_BASE + "?" + "&".join(f"{k}={v}" for k, v in params.items()) + date_fragment
        resp = requests.get(base_url, headers=_nvd_headers(), timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        cves = [v["cve"] for v in data.get("vulnerabilities", [])]
        return {
            "data": cves,
            "total_results": data.get("totalResults", 0),
            "source": source,
            "error": None,
            "latency_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as e:
        logger.warning("NVD search_cves error: %s", e)
        return {"data": [], "total_results": 0, "source": source, "error": str(e),
                "latency_s": 0}


@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3))
def get_cves_by_cpe(cpe_name: str, max_results: int = 10) -> dict:
    """
    Devuelve CVEs que afectan a un producto identificado por su CPE.
    Ejemplo: cpe_name='cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*'
    """
    params = {"cpeName": cpe_name, "resultsPerPage": min(max_results, NVD_RESULTS_PER_PAGE)}
    source = f"{NVD_API_BASE}?cpeName={cpe_name}"
    t0 = time.perf_counter()
    try:
        resp = requests.get(
            NVD_API_BASE, headers=_nvd_headers(), params=params, timeout=_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        cves = [v["cve"] for v in data.get("vulnerabilities", [])]
        return {"data": cves, "source": source, "error": None,
                "latency_s": round(time.perf_counter() - t0, 3)}
    except Exception as e:
        logger.warning("NVD get_cves_by_cpe error: %s", e)
        return {"data": [], "source": source, "error": str(e),
                "latency_s": round(time.perf_counter() - t0, 3)}


# ── AbuseIPDB API ─────────────────────────────────────────────────────────────

def check_ip(ip_address: str, max_age_days: int = 90) -> dict:
    """
    Consulta la reputación de una dirección IP en AbuseIPDB.

    Devuelve abusiveness score (0-100), categorías de abuso,
    país de origen, ISP y total de reportes.
    """
    if not ABUSEIPDB_API_KEY:
        return {
            "data": None,
            "source": f"{ABUSEIPDB_BASE}/check",
            "error": "ABUSEIPDB_API_KEY no configurada",
            "latency_s": 0,
        }

    url = f"{ABUSEIPDB_BASE}/check"
    headers = {"Key": ABUSEIPDB_API_KEY, "Accept": "application/json"}
    params = {"ipAddress": ip_address, "maxAgeInDays": max_age_days, "verbose": True}

    t0 = time.perf_counter()
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return {
            "data": resp.json().get("data"),
            "source": f"{url}?ipAddress={ip_address}",
            "error": None,
            "latency_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as e:
        logger.warning("AbuseIPDB check_ip(%s) error: %s", ip_address, e)
        return {
            "data": None,
            "source": f"{url}?ipAddress={ip_address}",
            "error": str(e),
            "latency_s": round(time.perf_counter() - t0, 3),
        }
