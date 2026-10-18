"""Pruebas de las tasas históricas del modelo P2."""

import pandas as pd

from mibici_mlops.preparar_p2 import agregar_tasas_historicas_p2


def test_tasas_no_utilizan_etiquetas_de_val_y_test():
    """Las tasas fuera de train solo deben depender de train."""
    tabla = pd.DataFrame({
        "estacion_id": [1] * 6,
        "hora": pd.date_range("2026-01-01 08:00", periods=6, freq="D"),
        "hora_dia": [8] * 6,
        "conjunto": ["train", "train", "train", "train", "val", "test"],
        "riesgo_p2_t_1h": [1, 2, 1, 0, 1, 2],
    })

    original = agregar_tasas_historicas_p2(tabla)

    modificada = tabla.copy()
    modificada.loc[
        modificada["conjunto"].isin(["val", "test"]),
        "riesgo_p2_t_1h",
    ] = 0

    resultado = agregar_tasas_historicas_p2(modificada)

    columnas = ["tasa_desabasto_hist", "tasa_saturacion_hist"]
    fuera_train = tabla["conjunto"].isin(["val", "test"])

    pd.testing.assert_frame_equal(
        original.loc[fuera_train, columnas],
        resultado.loc[fuera_train, columnas],
    )

    assert original.loc[fuera_train, "tasa_desabasto_hist"].eq(0.5).all()
    assert original.loc[fuera_train, "tasa_saturacion_hist"].eq(0.25).all()
