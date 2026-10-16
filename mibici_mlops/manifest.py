import hashlib
import json
from pathlib import Path

import pandas as pd

from mibici_mlops.config import (
    MESES_URLS,
    NOMENCLATURA_URL,
    PROJ_ROOT,
)

def sha256(archivo):
    h = hashlib.sha256()
    with archivo.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def registrar(archivo, url=None):
    if not archivo.is_file():
        raise FileNotFoundError(archivo)

    resultado = {
        "ruta": str(archivo.relative_to(PROJ_ROOT)),
        "sha256": sha256(archivo),
        "bytes": archivo.stat().st_size,
    }

    if url:
        resultado["url_origen"] = url

    if archivo.suffix == ".parquet":
        resultado["registros"] = len(pd.read_parquet(archivo))

    return resultado


def main():
    archivos = []

    for mes, url in MESES_URLS.items():
        archivos.append(
            registrar(
                PROJ_ROOT / "data/raw" / f"viajes_{mes}.csv",
                url,
            )
        )
        archivos.append(
            registrar(
                PROJ_ROOT / "data/processed/viajes"
                / f"viajes_{mes}.parquet"
            )
        )

    archivos.append(
        registrar(
            PROJ_ROOT / "data/raw" / Path(NOMENCLATURA_URL).name,
            NOMENCLATURA_URL,
        )
    )
    archivos.append(
        registrar(
            PROJ_ROOT / "data/processed/estaciones.parquet"
        )
    )

    manifiesto = {
        "proyecto": "MiBici - Equipo 42",
        "periodo": {
            "inicio": "2025-07",
            "fin": "2026-09",
        },
        "archivos": archivos,
    }

    destino = PROJ_ROOT / "data_manifest.json"
    destino.write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Manifiesto generado: {destino}")
    print(f"Archivos registrados: {len(archivos)}")


if __name__ == "__main__":
    main()
