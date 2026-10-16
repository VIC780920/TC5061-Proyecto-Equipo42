"""P1 – Baseline de regresión: viajes que inician por estación en la siguiente hora.

Baseline naive estacional: la predicción para la estación s a la hora h es el número de
salidas de esa estación a la misma hora una semana antes (rezago de 168 h).

Uso desde la raíz del repo:  python -m mibici_mlops.modeling.p1
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mibici_mlops.config import FIGURES_DIR, VIAJES_PATH
from mibici_mlops.features import agregar_estacion_hora, agregar_rezagos, particion_temporal
from mibici_mlops.tracking import registrar_run

REZAGO = 168
RUN_NAME = "p1-baseline-semana-anterior"
OBJETIVO = "salidas"
PRED = f"{OBJETIVO}_lag_{REZAGO}"


def cargar_viajes() -> pd.DataFrame:
    return pd.read_parquet(VIAJES_PATH, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"])


def evaluar(df: pd.DataFrame) -> dict:
    metricas = {}
    for conjunto in ["val", "test"]:
        d = df[df["conjunto"] == conjunto]
        metricas[f"{conjunto}_mae"] = mean_absolute_error(d[OBJETIVO], d[PRED])
        metricas[f"{conjunto}_rmse"] = root_mean_squared_error(d[OBJETIVO], d[PRED])
    return metricas


def graficar(df: pd.DataFrame, ruta) -> None:
    test = df[df["conjunto"] == "test"]
    total = test.groupby("hora")[[OBJETIVO, PRED]].sum().iloc[: 24 * 14]
    error_hora = (test[OBJETIVO] - test[PRED]).abs().groupby(test["hora"].dt.hour).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    total.plot(ax=axes[0])
    axes[0].set_title("Test: viajes totales por hora (primeras 2 semanas)")
    axes[0].legend(["real", "baseline (semana anterior)"])
    axes[0].set_xlabel("")
    error_hora.plot.bar(ax=axes[1])
    axes[1].set_title("Test: MAE por hora del día")
    axes[1].set_xlabel("hora")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def main(run_name: str = RUN_NAME) -> str:
    logger.info("P1: cargando viajes y construyendo la tabla estación-hora")
    tabla = agregar_estacion_hora(cargar_viajes())
    tabla = agregar_rezagos(tabla, OBJETIVO, (REZAGO,))
    tabla = particion_temporal(tabla)

    evaluable = tabla[tabla["conjunto"].isin(["val", "test"])].dropna(subset=[PRED])
    metricas = evaluar(evaluable)
    logger.info(f"P1 métricas: {metricas}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta_fig = FIGURES_DIR / "p1_baseline_real_vs_pred.png"
    graficar(evaluable, ruta_fig)

    params = {
        "modelo": "naive_semana_anterior",
        "rezago_horas": REZAGO,
        "objetivo": "salidas_siguiente_hora",
        "granularidad": "estacion-hora",
        "horizonte_horas": 1,
        "n_estaciones": tabla["estacion_id"].nunique(),
        "n_filas_val": int((evaluable["conjunto"] == "val").sum()),
        "n_filas_test": int((evaluable["conjunto"] == "test").sum()),
    }
    return registrar_run("P1", run_name, params, metricas, artefactos=[ruta_fig])


if __name__ == "__main__":
    main()