"""Entrenamiento reproducible del modelo P2 de MiBici."""

import pandas as pd

from mibici_mlops.preparar_p2 import generar_dataset_p2
from mibici_mlops.modelo_p2 import (
    preparar_datos_modelo,
    crear_modelo_p2,
    guardar_modelo_p2,
)


def main():
    print("Preparando dataset P2...", flush=True)
    tabla = generar_dataset_p2()

    X_train, y_train, X_val, y_val, X_test, y_test = (
        preparar_datos_modelo(tabla)
    )

    print(f"Registros de entrenamiento: {len(X_train):,}")
    print(f"Registros de validación: {len(X_val):,}")
    print(f"Registros de prueba reservados: {len(X_test):,}")

    # Entrenar con train + validation, sin utilizar test.
    X_final = pd.concat([X_train, X_val], ignore_index=True)
    y_final = pd.concat([y_train, y_val], ignore_index=True)

    del tabla, X_train, y_train, X_val, y_val, X_test, y_test

    print("Entrenando modelo Random Forest P2...", flush=True)
    modelo = crear_modelo_p2()
    modelo.fit(X_final, y_final)

    ruta = guardar_modelo_p2(modelo)

    print("Entrenamiento completado correctamente.")
    print(f"Árboles: {len(modelo.estimators_)}")
    print(f"Modelo guardado en: {ruta}")


if __name__ == "__main__":
    main()
