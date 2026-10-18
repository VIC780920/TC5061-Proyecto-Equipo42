"""Pruebas unitarias para el modelo P2 de MiBici."""

import pandas as pd
from mibici_mlops.modelo_p2 import (
    OBJETIVO_P2,
    VARIABLES_P2,
    crear_modelo_p2,
    preparar_datos_modelo,
)


def test_variables_predictoras_p2():
    """Verifica las variables predictoras y la etiqueta."""
    assert len(VARIABLES_P2) == 26
    assert len(set(VARIABLES_P2)) == 26
    assert OBJETIVO_P2 == "riesgo_p2_t_1h"

    # Evitar fuga de información de la hora actual.
    assert "salidas" not in VARIABLES_P2
    assert "llegadas" not in VARIABLES_P2
    assert "flujo_neto" not in VARIABLES_P2
    assert "flujo_relativo" not in VARIABLES_P2


def test_configuracion_random_forest():
    """Verifica los parámetros seleccionados para P2."""
    modelo = crear_modelo_p2()

    assert modelo.n_estimators == 50
    assert modelo.max_depth == 12
    assert modelo.min_samples_leaf == 20
    assert modelo.class_weight == {0: 1, 1: 10, 2: 10}
    assert modelo.random_state == 42
    assert modelo.n_jobs == 2


def test_guardar_y_cargar_modelo_p2(tmp_path):
    """Comprueba el guardado y la recuperación de un modelo."""
    import joblib

    modelo = crear_modelo_p2()
    ruta = tmp_path / "modelo_p2.joblib"

    joblib.dump(modelo, ruta)
    recuperado = joblib.load(ruta)

    assert recuperado.n_estimators == 50
    assert recuperado.class_weight == {0: 1, 1: 10, 2: 10}


def test_preparar_datos_modelo_particiones_y_nulos():
    """Verifica las particiones y la exclusion de valores faltantes."""
    from mibici_mlops.modelo_p2 import preparar_datos_modelo

    filas = []
    for conjunto, etiqueta in [
        ("train", 0),
        ("val", 1),
        ("test", 2),
    ]:
        fila = {variable: 1.0 for variable in VARIABLES_P2}
        fila["conjunto"] = conjunto
        fila[OBJETIVO_P2] = etiqueta
        filas.append(fila)

    fila_nula = filas[0].copy()
    fila_nula["salidas_lag_1"] = float("nan")
    filas.append(fila_nula)

    tabla = pd.DataFrame(filas)
    X_train, y_train, X_val, y_val, X_test, y_test = preparar_datos_modelo(tabla)

    assert len(X_train) == len(X_val) == len(X_test) == 1
    assert len(y_train) == len(y_val) == len(y_test) == 1
    assert list(y_train) == [0]
    assert list(y_val) == [1]
    assert list(y_test) == [2]
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == 26
    assert not X_train.isna().any().any()
