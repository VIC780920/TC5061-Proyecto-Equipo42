"""P1 – Versión challenger y A/B testing offline contra el champion (entrega 5).

Diseño acordado por el equipo:
- Champion (versión vigente): p1-hgb-final, entrenado con train + val (jul-2025 a jun-2026).
- Challenger: mismo HGB y mismos hiperparámetros que el champion, reentrenado con los datos
  disponibles hasta ago-2026 (incluye jul y ago del periodo actual).
- Evaluación: sep-2026, que ninguna de las dos versiones vio al entrenar.

    python -m mibici_mlops.monitoring.ab_test_p1 challenger
    python -m mibici_mlops.monitoring.ab_test_p1 ab-test
    python -m mibici_mlops.monitoring.ab_test_p1 ab-test --promover
"""

from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import typer

from mibici_mlops import config
from mibici_mlops.config import PROJ_ROOT, SEED, TEST_INICIO, VAL_INICIO
from mibici_mlops.modeling.p1 import (
    CARACTERISTICAS,
    HGB_DEFAULTS,
    OBJETIVO,
    TIPOS_CONFIABLES,
    _valor,
    calcular_metricas,
    construir_tabla,
    crear_modelo,
    graficar_resultados,
)
from mibici_mlops.registry import registrar_champion
from mibici_mlops.tracking import iniciar_run, registrar_resultados

CHALLENGER_FIN = "2026-09-01"
EVAL_INICIO = "2026-09-01"
EVAL_FIN = "2026-10-01"
N_BOOT = 2000
SALIDA = PROJ_ROOT / "reports" / "ab_test"
NOMBRE = config.MODELOS_REGISTRADOS["p1"]

app = typer.Typer()


def _cliente() -> MlflowClient:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return MlflowClient()


def _fin_entrenamiento(client: MlflowClient, run_id: str) -> str:
    """Fecha (exclusiva) hasta donde llegan los datos con que se entrenó la versión."""
    params = client.get_run(run_id).data.params
    if "entreno_fin" in params:
        return params["entreno_fin"]
    conjuntos = params.get("entrenado_con", "").split("+")
    if "test" in conjuntos:
        return "2026-10-01"
    return TEST_INICIO if "val" in conjuntos else VAL_INICIO


def periodo_evaluacion(tabla: pd.DataFrame) -> pd.DataFrame:
    return tabla[(tabla["hora"] >= EVAL_INICIO) & (tabla["hora"] < EVAL_FIN)].copy()


def estaciones_nuevas(tabla: pd.DataFrame) -> pd.Index:
    """Estaciones cuya primera hora observada en la tabla es del periodo actual (desde jul-2026)."""
    primera = tabla.groupby("estacion_id")["hora"].min()
    return primera[primera >= TEST_INICIO].index


@app.command()
def challenger():
    """Reentrena el HGB del champion con datos hasta ago-2026 y lo registra con alias challenger."""
    client = _cliente()
    champion = client.get_model_version_by_alias(NOMBRE, "champion")
    params_champion = client.get_run(champion.run_id).data.params
    params = {k: _valor(v) for k, v in params_champion.items() if k in HGB_DEFAULTS}
    logger.info(f"Champion versión {champion.version}; hiperparámetros: {params}")

    tabla = construir_tabla()
    entreno = tabla[tabla["hora"] < CHALLENGER_FIN]
    evaluacion = periodo_evaluacion(tabla)

    with iniciar_run(
        "P1",
        "p1-hgb-challenger",
        model_family="hgb",
        stage_of_work="candidate",
        params={
            "modelo": "hgb",
            "entrenado_con": "train+val+jul-ago-2026",
            "entreno_fin": CHALLENGER_FIN,
            **params,
        },
        tags={
            "rol": "challenger",
            "champion_version": champion.version,
            "champion_run_id": champion.run_id,
        },
    ) as run:
        modelo = crear_modelo("hgb", params)
        logger.info(f"P1 challenger: entrenando con {len(entreno):,} filas")
        modelo.fit(entreno[CARACTERISTICAS], entreno[OBJETIVO])
        pred = modelo.predict(evaluacion[CARACTERISTICAS])
        metricas = calcular_metricas(evaluacion[OBJETIVO], pred, "test_sep")
        logger.info(f"P1 challenger en sep-2026: {metricas}")

        with tempfile.TemporaryDirectory() as tmp:
            graficas = graficar_resultados(evaluacion, pred, "sep-2026", Path(tmp))
            registrar_resultados(
                metricas,
                modelo=modelo,
                X_ejemplo=entreno[CARACTERISTICAS].dropna(),
                graficas=graficas,
                config={
                    "pregunta": "P1",
                    "rol": "challenger",
                    "hiperparametros": params,
                    "caracteristicas": CARACTERISTICAS,
                    "entreno": f"{config.FECHA_INICIO} a {CHALLENGER_FIN} (exclusivo)",
                    "evaluado_en": f"{EVAL_INICIO} a {EVAL_FIN} (exclusivo)",
                    "champion_version": champion.version,
                },
                datos_entrenamiento=entreno,
                nombre_datos="p1-estacion-hora-hasta-2026-08",
                tipos_confiables=TIPOS_CONFIABLES["hgb"],
            )
        run_id = run.info.run_id

    version = registrar_champion("p1", run_id, alias="challenger")
    client.update_model_version(
        NOMBRE,
        version.version,
        description="Challenger: mismo HGB del champion reentrenado con datos hasta ago-2026.",
    )
    logger.success(f"{NOMBRE} versión {version.version} registrada con alias challenger")


def _metricas_ab(d: pd.DataFrame) -> dict:
    y = d[OBJETIVO]
    return {
        "filas": len(d),
        "mae_champion": float((y - d["pred_a"]).abs().mean()),
        "mae_challenger": float((y - d["pred_b"]).abs().mean()),
        "rmse_champion": float(np.sqrt(((y - d["pred_a"]) ** 2).mean())),
        "rmse_challenger": float(np.sqrt(((y - d["pred_b"]) ** 2).mean())),
    }


def bootstrap_pareado(d: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """IC 95 % de la diferencia (challenger − champion) remuestreando días completos."""
    err_a, err_b = (d[OBJETIVO] - d["pred_a"]), (d[OBJETIVO] - d["pred_b"])
    por_dia = (
        pd.DataFrame(
            {
                "dia": d["hora"].dt.floor("D"),
                "abs_a": err_a.abs(),
                "abs_b": err_b.abs(),
                "sq_a": err_a**2,
                "sq_b": err_b**2,
                "n": 1,
            }
        )
        .groupby("dia")
        .sum()
        .to_numpy()
    )
    rng = np.random.default_rng(SEED)
    muestras = por_dia[rng.integers(0, len(por_dia), (n_boot, len(por_dia)))].sum(axis=1)
    abs_a, abs_b, sq_a, sq_b, n = muestras.T
    delta_mae = (abs_b - abs_a) / n
    delta_rmse = np.sqrt(sq_b / n) - np.sqrt(sq_a / n)
    return {
        "dias": len(por_dia),
        "delta_mae_ic_inf": float(np.percentile(delta_mae, 2.5)),
        "delta_mae_ic_sup": float(np.percentile(delta_mae, 97.5)),
        "delta_rmse_ic_inf": float(np.percentile(delta_rmse, 2.5)),
        "delta_rmse_ic_sup": float(np.percentile(delta_rmse, 97.5)),
    }


def split_trafico(d: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Simula un A/B en línea: cada estación se asigna 50/50 a una versión (como repartir tráfico)."""
    rng = np.random.default_rng(SEED)
    estaciones = d["estacion_id"].unique()
    grupo_b = set(rng.permutation(estaciones)[: len(estaciones) // 2])
    en_b = d["estacion_id"].isin(grupo_b)
    error = np.where(en_b, d["pred_b"], d["pred_a"]) - d[OBJETIVO]
    por_estacion = (
        pd.DataFrame({"estacion_id": d["estacion_id"], "b": en_b, "abs": np.abs(error), "n": 1})
        .groupby(["estacion_id", "b"])
        .sum()
        .reset_index()
    )
    a, b = por_estacion[~por_estacion["b"]], por_estacion[por_estacion["b"]]
    mae_a, mae_b = a["abs"].sum() / a["n"].sum(), b["abs"].sum() / b["n"].sum()
    deltas = []
    for _ in range(n_boot):
        ma = a.iloc[rng.integers(0, len(a), len(a))]
        mb = b.iloc[rng.integers(0, len(b), len(b))]
        deltas.append(mb["abs"].sum() / mb["n"].sum() - ma["abs"].sum() / ma["n"].sum())
    return {
        "estaciones_grupo_champion": len(a),
        "estaciones_grupo_challenger": len(b),
        "mae_grupo_champion": float(mae_a),
        "mae_grupo_challenger": float(mae_b),
        "delta_mae": float(mae_b - mae_a),
        "delta_mae_ic_inf": float(np.percentile(deltas, 2.5)),
        "delta_mae_ic_sup": float(np.percentile(deltas, 97.5)),
    }


def graficar_ab(d: pd.DataFrame, segmentos: pd.DataFrame, ruta: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    por_hora = (
        pd.DataFrame(
            {
                "champion": (d[OBJETIVO] - d["pred_a"]).abs(),
                "challenger": (d[OBJETIVO] - d["pred_b"]).abs(),
            }
        )
        .groupby(d["hora_dia"])
        .mean()
    )
    por_hora.plot(ax=axes[0], marker="o")
    axes[0].set_title("MAE por hora del día (sep-2026)")
    axes[0].set_xlabel("hora")
    segmentos.set_index("segmento")[["mae_champion", "mae_challenger"]].plot.bar(ax=axes[1], rot=0)
    axes[1].set_title("MAE por segmento (sep-2026)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].legend(["champion", "challenger"], loc="upper center")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    return ruta


@app.command()
def ab_test(promover: bool = False):
    """Compara champion y challenger sobre sep-2026 y registra el resultado en MLflow."""
    client = _cliente()
    version_a = client.get_model_version_by_alias(NOMBRE, "champion")
    version_b = client.get_model_version_by_alias(NOMBRE, "challenger")
    if version_a.version == version_b.version:
        raise ValueError("champion y challenger apuntan a la misma versión")
    for v in (version_a, version_b):
        fin = _fin_entrenamiento(client, v.run_id)
        if fin > EVAL_INICIO:
            raise ValueError(f"La versión {v.version} se entrenó con datos hasta {fin}: hay fuga")

    tabla = construir_tabla()
    nuevas = estaciones_nuevas(tabla)
    d = periodo_evaluacion(tabla)
    d["pred_a"] = mlflow.pyfunc.load_model(f"models:/{NOMBRE}@champion").predict(
        d[CARACTERISTICAS]
    )
    d["pred_b"] = mlflow.pyfunc.load_model(f"models:/{NOMBRE}@challenger").predict(
        d[CARACTERISTICAS]
    )

    global_ = _metricas_ab(d)
    global_["delta_mae"] = global_["mae_challenger"] - global_["mae_champion"]
    global_["delta_rmse"] = global_["rmse_challenger"] - global_["rmse_champion"]
    global_.update(bootstrap_pareado(d))
    err_a, err_b = (d[OBJETIVO] - d["pred_a"]).abs(), (d[OBJETIVO] - d["pred_b"]).abs()
    global_["pct_gana_challenger"] = float((err_b < err_a).mean() * 100)
    global_["pct_gana_champion"] = float((err_a < err_b).mean() * 100)
    global_["pct_empate"] = float((err_a == err_b).mean() * 100)

    mascaras = {
        "todas": pd.Series(True, index=d.index),
        "estaciones nuevas": d["estacion_id"].isin(nuevas),
        "estaciones existentes": ~d["estacion_id"].isin(nuevas),
        "entre semana": d["es_fin_semana"] == 0,
        "fin de semana": d["es_fin_semana"] == 1,
    }
    segmentos = pd.DataFrame(
        [
            {"segmento": k, "estaciones": d.loc[m, "estacion_id"].nunique(), **_metricas_ab(d[m])}
            for k, m in mascaras.items()
        ]
    )
    segmentos["delta_mae"] = segmentos["mae_challenger"] - segmentos["mae_champion"]
    segmentos["delta_mae_pct"] = segmentos["delta_mae"] / segmentos["mae_champion"] * 100
    trafico = split_trafico(d)

    gana = global_["delta_mae_ic_sup"] < 0 and global_["delta_rmse"] <= 0
    decision = "promover challenger" if gana else "mantener champion"

    SALIDA.mkdir(parents=True, exist_ok=True)
    resumen = pd.DataFrame(
        {
            "metrica": ["MAE", "RMSE"],
            f"champion (v{version_a.version})": [
                global_["mae_champion"],
                global_["rmse_champion"],
            ],
            f"challenger (v{version_b.version})": [
                global_["mae_challenger"],
                global_["rmse_challenger"],
            ],
            "diferencia (challenger - champion)": [global_["delta_mae"], global_["delta_rmse"]],
            "IC 95% inf": [global_["delta_mae_ic_inf"], global_["delta_rmse_ic_inf"]],
            "IC 95% sup": [global_["delta_mae_ic_sup"], global_["delta_rmse_ic_sup"]],
        }
    )
    resumen.to_csv(SALIDA / "ab_resumen_p1.csv", index=False)
    segmentos.to_csv(SALIDA / "ab_segmentos_p1.csv", index=False)
    pd.DataFrame([trafico]).to_csv(SALIDA / "ab_split_trafico_p1.csv", index=False)
    grafica = graficar_ab(d, segmentos, SALIDA / "ab_test_p1.png")

    with iniciar_run(
        "P1",
        "p1-ab-test",
        model_family="hgb",
        stage_of_work="ab_test",
        params={
            "version_champion": version_a.version,
            "version_challenger": version_b.version,
            "evaluacion": f"{EVAL_INICIO} a {EVAL_FIN} (exclusivo)",
            "n_boot": N_BOOT,
            "regla": "promover si IC95 sup de delta_mae < 0 y delta_rmse <= 0",
        },
        tags={"decision": decision},
    ):
        mlflow.log_metrics({f"test_sep_{k}": v for k, v in global_.items()})
        mlflow.log_metrics({f"test_sep_trafico_{k}": v for k, v in trafico.items()})
        for archivo in ("ab_resumen_p1.csv", "ab_segmentos_p1.csv", "ab_split_trafico_p1.csv"):
            mlflow.log_artifact(str(SALIDA / archivo), artifact_path="ab_test")
        mlflow.log_artifact(str(grafica), artifact_path="plots")

    pd.set_option("display.width", 200)
    print(
        f"\nA/B offline P1 | evaluación {EVAL_INICIO} a {EVAL_FIN} | {global_['filas']:,} filas, "
        f"{global_['dias']} días | {len(nuevas)} estaciones nuevas en la tabla"
    )
    print("\nResumen:\n" + resumen.round(4).to_string(index=False))
    print(
        f"\nGana challenger: {global_['pct_gana_challenger']:.2f}% | gana champion: "
        f"{global_['pct_gana_champion']:.2f}% | empate: {global_['pct_empate']:.2f}%"
    )
    print("\nSegmentos:\n" + segmentos.round(4).to_string(index=False))
    print("\nSplit de tráfico simulado por estación:\n" + pd.Series(trafico).round(4).to_string())
    print(f"\nDecisión: {decision}")

    if promover and gana:
        client.set_registered_model_alias(NOMBRE, "champion_anterior", version_a.version)
        client.set_registered_model_alias(NOMBRE, "champion", version_b.version)
        logger.success(f"Alias champion movido a la versión {version_b.version}")
    elif promover:
        logger.info("No se promueve: el challenger no cumplió la regla")


if __name__ == "__main__":
    app()
