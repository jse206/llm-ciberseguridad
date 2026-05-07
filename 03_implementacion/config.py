"""Configuración centralizada del proyecto."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Rutas base ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
IMPL_DIR = Path(__file__).parent
DATA_DIR = IMPL_DIR / "data" / "storage"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0      # determinismo máximo para evaluación reproducible
LLM_MAX_TOKENS = 1024

# ── NVD / CVE ─────────────────────────────────────────────────────────────────
NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_API_KEY = os.getenv("NVD_API_KEY", "")  # opcional, aumenta rate-limit
NVD_DB_PATH = DATA_DIR / "nvd_cve.db"
NVD_YEARS = [int(y) for y in os.getenv("NVD_YEARS", "2018,2019,2020,2021,2022,2023,2024,2025").split(",")]
NVD_RESULTS_PER_PAGE = 2000   # máximo permitido por la API

# ── MITRE ATT&CK ──────────────────────────────────────────────────────────────
MITRE_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)
MITRE_STIX_PATH = DATA_DIR / "enterprise-attack.json"
MITRE_GRAPH_PATH = DATA_DIR / "mitre_graph.pkl"

# ── AbuseIPDB ─────────────────────────────────────────────────────────────────
ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY", "")
ABUSEIPDB_BASE = "https://api.abuseipdb.com/api/v2"

# ── Benchmark ─────────────────────────────────────────────────────────────────
BENCHMARK_DIR = ROOT_DIR / "04_benchmark"
RESULTS_DIR = BENCHMARK_DIR / "resultados"
