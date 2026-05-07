"""
Esquema SQLite y operaciones sobre la base de datos NVD/CVE.

Diseño orientado a la Arquitectura A (Text2SQL): las consultas que el pipeline
traducirá desde lenguaje natural corresponden exactamente a las tablas
y columnas aquí definidas.
"""

import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path

from config import NVD_DB_PATH

logger = logging.getLogger(__name__)

# ── DDL ───────────────────────────────────────────────────────────────────────

_SCHEMA = """
-- Tabla principal de vulnerabilidades
CREATE TABLE IF NOT EXISTS cve (
    cve_id          TEXT PRIMARY KEY,
    description     TEXT,
    cvss_v3_score   REAL,
    cvss_v3_severity TEXT,          -- LOW / MEDIUM / HIGH / CRITICAL
    cvss_v3_vector  TEXT,
    cvss_v2_score   REAL,
    published       TEXT,           -- ISO-8601
    last_modified   TEXT,
    vuln_status     TEXT            -- Analyzed / Modified / Awaiting Analysis
);

-- Debilidades asociadas (CWE)
CREATE TABLE IF NOT EXISTS cve_cwe (
    cve_id  TEXT REFERENCES cve(cve_id),
    cwe_id  TEXT,
    PRIMARY KEY (cve_id, cwe_id)
);

-- Productos afectados (CPE simplificado)
CREATE TABLE IF NOT EXISTS cve_cpe (
    cve_id      TEXT REFERENCES cve(cve_id),
    cpe_uri     TEXT,
    vulnerable  INTEGER DEFAULT 1,
    PRIMARY KEY (cve_id, cpe_uri)
);

-- Referencias externas (advisories, PoC, parches)
CREATE TABLE IF NOT EXISTS cve_reference (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    cve_id  TEXT REFERENCES cve(cve_id),
    url     TEXT,
    source  TEXT,
    tags    TEXT    -- JSON array de etiquetas
);

-- Índices para acelerar las consultas más habituales del benchmark
CREATE INDEX IF NOT EXISTS idx_cve_severity  ON cve(cvss_v3_severity);
CREATE INDEX IF NOT EXISTS idx_cve_score     ON cve(cvss_v3_score);
CREATE INDEX IF NOT EXISTS idx_cve_published ON cve(published);
CREATE INDEX IF NOT EXISTS idx_cwe_id        ON cve_cwe(cwe_id);
"""


@contextmanager
def get_conn(db_path: Path = NVD_DB_PATH):
    """Context manager que devuelve una conexión SQLite con row_factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = NVD_DB_PATH) -> None:
    """Crea las tablas si no existen."""
    with get_conn(db_path) as conn:
        conn.executescript(_SCHEMA)
    logger.info("Base de datos NVD inicializada en %s", db_path)


def insert_cve(conn: sqlite3.Connection, entry: dict) -> None:
    """
    Inserta o reemplaza un CVE y sus relaciones.
    `entry` es el objeto normalizado devuelto por nvd_downloader.parse_cve().
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO cve
            (cve_id, description, cvss_v3_score, cvss_v3_severity,
             cvss_v3_vector, cvss_v2_score, published, last_modified, vuln_status)
        VALUES
            (:cve_id, :description, :cvss_v3_score, :cvss_v3_severity,
             :cvss_v3_vector, :cvss_v2_score, :published, :last_modified, :vuln_status)
        """,
        entry,
    )
    for cwe in entry.get("cwes", []):
        conn.execute(
            "INSERT OR IGNORE INTO cve_cwe (cve_id, cwe_id) VALUES (?, ?)",
            (entry["cve_id"], cwe),
        )
    for cpe in entry.get("cpes", []):
        conn.execute(
            "INSERT OR IGNORE INTO cve_cpe (cve_id, cpe_uri, vulnerable) VALUES (?, ?, ?)",
            (entry["cve_id"], cpe["uri"], int(cpe.get("vulnerable", True))),
        )
    for ref in entry.get("references", []):
        conn.execute(
            "INSERT INTO cve_reference (cve_id, url, source, tags) VALUES (?, ?, ?, ?)",
            (entry["cve_id"], ref["url"], ref.get("source", ""), ref.get("tags", "[]")),
        )


# ── Consultas de utilidad ──────────────────────────────────────────────────────

def get_cve(cve_id: str, db_path: Path = NVD_DB_PATH) -> dict | None:
    """Devuelve un CVE completo como dict, o None si no existe."""
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM cve WHERE cve_id = ?", (cve_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["cwes"] = [
            r["cwe_id"]
            for r in conn.execute(
                "SELECT cwe_id FROM cve_cwe WHERE cve_id = ?", (cve_id,)
            ).fetchall()
        ]
        return result


def count_cves(db_path: Path = NVD_DB_PATH) -> int:
    with get_conn(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM cve").fetchone()[0]


def count_cves_for_year(year: int, db_path: Path = NVD_DB_PATH) -> int:
    """Cuenta los CVEs publicados en un año concreto (para skip en re-descarga)."""
    with get_conn(db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM cve WHERE published LIKE ?", (f"{year}-%",)
        ).fetchone()[0]


def db_stats(db_path: Path = NVD_DB_PATH) -> dict:
    """Estadísticas de la base de datos NVD/CVE."""
    with get_conn(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM cve").fetchone()[0]
        by_severity = {
            row["cvss_v3_severity"]: row["cnt"]
            for row in conn.execute(
                "SELECT cvss_v3_severity, COUNT(*) AS cnt FROM cve GROUP BY cvss_v3_severity"
            ).fetchall()
        }
        return {"total_cves": total, "by_severity": by_severity}
