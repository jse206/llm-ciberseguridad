#!/bin/sh
set -e

DATA_DIR="03_implementacion/data/storage"
DB_FILE="$DATA_DIR/nvd_cve.db"

if [ ! -f "$DB_FILE" ]; then
    echo "[setup] Primera ejecución — descargando datos en background..."
    (cd 03_implementacion && python setup.py >> /tmp/setup.log 2>&1) &
fi

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8001}"
