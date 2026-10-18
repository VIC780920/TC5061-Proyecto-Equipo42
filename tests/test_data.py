"""Pruebas unitarias para la ingesta de datos MiBici."""

import pandas as pd

from mibici_mlops import dataset


def test_procesar_mes_limpia_duplicados_y_guarda_parquet(tmp_path, monkeypatch):
    """Elimina duplicados, convierte fechas y genera un Parquet válido."""
    mes = "2025_07"
    archivo_csv = tmp_path / f"viajes_{mes}.csv"
    archivo_parquet = tmp_path / f"viajes_{mes}.parquet"

    datos = pd.DataFrame({
        "Viaje_Id": [1, 1, 2],
        "Inicio_del_viaje": [
            "2025-07-01 08:00:00",
            "2025-07-01 08:00:00",
            "2025-07-01 09:00:00",
        ],
        "Fin_del_viaje": [
            "2025-07-01 08:15:00",
            "2025-07-01 08:15:00",
            "2025-07-01 09:20:00",
        ],
        "Origen_Id": [10, 10, 11],
        "Destino_Id": [11, 11, 12],
    })
    datos.to_csv(archivo_csv, index=False)

    monkeypatch.setitem(dataset.MESES_URLS, mes, "https://example.invalid/viajes.csv")
    monkeypatch.setattr(dataset, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(dataset, "VIAJES_PATH", tmp_path)

    dataset.procesar_mes(mes)

    assert archivo_parquet.exists()

    resultado = pd.read_parquet(archivo_parquet)

    assert len(resultado) == 2
    assert resultado["viaje_id"].tolist() == [1, 2]
    assert pd.api.types.is_datetime64_any_dtype(resultado["inicio_viaje"])
    assert pd.api.types.is_datetime64_any_dtype(resultado["fin_viaje"])
    assert list(resultado.columns) == [
        "viaje_id",
        "inicio_viaje",
        "fin_viaje",
        "origen_id",
        "destino_id",
    ]
