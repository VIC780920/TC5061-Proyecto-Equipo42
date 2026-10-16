
"""Ingesta inicial de viajes MiBici a formato Parquet."""

import subprocess

import pandas as pd
from loguru import logger

from mibici_mlops.config import MESES_URLS, RAW_DATA_DIR, VIAJES_PATH


COLUMNAS = {
    "Viaje_Id": "viaje_id",
    "Inicio_del_viaje": "inicio_viaje",
    "Fin_del_viaje": "fin_viaje",
    "Origen_Id": "origen_id",
    "Destino_Id": "destino_id",
}


def procesar_mes(mes: str) -> None:
    """Descarga, limpia y convierte los viajes de un mes."""

    url = MESES_URLS[mes]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIAJES_PATH.mkdir(parents=True, exist_ok=True)

    archivo_csv = RAW_DATA_DIR / f"viajes_{mes}.csv"
    archivo_parquet = VIAJES_PATH / f"viajes_{mes}.parquet"

    if not archivo_csv.exists():
        logger.info(f"Descargando {mes}")

        subprocess.run(
            ["curl", "-fL", "--retry", "3", url, "-o", str(archivo_csv)],
            check=True,
        )
    try:
        df = pd.read_csv(
            archivo_csv,
            usecols=list(COLUMNAS),
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        logger.warning(f"{mes}: UTF-8 no compatible; probando latin-1")

        df = pd.read_csv(
           archivo_csv,
           usecols=list(COLUMNAS),
           encoding="latin-1",
        )

    registros_originales = len(df)

    df = df.rename(columns=COLUMNAS)
    df = df.drop_duplicates()

    df["inicio_viaje"] = pd.to_datetime(
        df["inicio_viaje"], errors="coerce"
    )
    df["fin_viaje"] = pd.to_datetime(
        df["fin_viaje"], errors="coerce"
    )

    if df[["inicio_viaje", "fin_viaje"]].isna().any().any():
        raise ValueError(f"Fechas inválidas en {mes}")

    df.to_parquet(archivo_parquet, index=False)

    logger.success(
        f"{mes}: {registros_originales} originales, "
        f"{len(df)} procesados, "
        f"{registros_originales - len(df)} duplicados eliminados"
    )


def procesar_estaciones():
    import subprocess
    import pandas as pd

    from mibici_mlops.config import (
        NOMENCLATURA_URL,
        RAW_DATA_DIR,
        ESTACIONES_PATH,
    )

    archivo_csv = RAW_DATA_DIR / "nomenclatura_2026_09.csv"

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ESTACIONES_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not archivo_csv.exists():
        logger.info("Descargando catalogo de estaciones")
        subprocess.run(
            ["curl", "-fL", "--retry", "3",
             NOMENCLATURA_URL, "-o", str(archivo_csv)],
            check=True,
        )

    df = pd.read_csv(archivo_csv, encoding="utf-8-sig")

    df = df.rename(columns={
        "id": "estacion_id",
        "name": "nombre",
        "latitude": "latitud",
        "longitude": "longitud",
        "dpcapacity": "capacidad",
    })

    columnas = [
        "estacion_id", "nombre", "latitud",
        "longitud", "capacidad",
    ]
    df = df[columnas]

    if df[columnas].isna().any().any():
        raise ValueError("El catalogo contiene valores faltantes")

    if df["estacion_id"].duplicated().any():
        raise ValueError("El catalogo contiene estaciones duplicadas")

    df.to_parquet(ESTACIONES_PATH, index=False)

    logger.success(
        f"Catalogo procesado: {len(df)} estaciones, "
        f"archivo: {ESTACIONES_PATH}"
    )

if __name__ == "__main__":
    for mes in MESES_URLS:
        procesar_mes(mes)
