#!/bin/sh
set -e

DATA_DIR="03_implementacion/data/storage"
DB_FILE="$DATA_DIR/nvd_cve.db"

if [ ! -f "$DB_FILE" ]; then
    echo "[setup] Primera ejecución — descargando datos NVD/CVE y grafo MITRE ATT&CK..."
    cd 03_implementacion
    python setup.py
    cd ..
    echo "[setup] Datos listos."
fi

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8001}"
