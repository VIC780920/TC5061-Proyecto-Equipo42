"""Registra el modelo final de una pregunta en el Model Registry y le asigna el alias champion.

Reemplaza el registro manual desde la interfaz, para que cualquiera deje el registry listo
desde un clon limpio después de la corrida oficial:

    python -m mibici_mlops.registry --question p1
    python -m mibici_mlops.registry --question p2 --run-id <run_id>

Sin --run-id busca la última corrida terminada con el nombre de RUNS_FINALES. Copia a la versión
el linaje (run_id, git_commit, data_version) y no duplica versiones si el run ya estaba registrado.
"""

from loguru import logger
import mlflow
from mlflow import MlflowClient
import typer

from mibici_mlops import config

RUNS_FINALES = {"p1": "p1-hgb-final"}
DESCRIPCIONES = {
    "p1": "P1 - Viajes que inician por estación en la siguiente hora (regresión). "
    "HistGradientBoosting con pérdida Poisson reentrenado con train + val.",
    "p2": "P2 - Riesgo de desabasto o saturación por estación en la siguiente hora "
    "(clasificación de 3 clases).",
    "p3": "P3 - Grupos de estaciones con patrones de uso parecidos (clustering).",
}

app = typer.Typer()


def buscar_run_final(client: MlflowClient, question: str):
    nombre_run = RUNS_FINALES.get(question)
    if nombre_run is None:
        raise ValueError(f"Para {question} indica el run con --run-id")
    experimento = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if experimento is None:
        raise ValueError(f"No existe el experimento {config.MLFLOW_EXPERIMENT}")
    runs = client.search_runs(
        [experimento.experiment_id],
        filter_string=f"attributes.run_name = '{nombre_run}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"No hay una corrida terminada '{nombre_run}' en {config.MLFLOW_EXPERIMENT}"
        )
    return runs[0]


def registrar_champion(question: str, run_id: str | None = None, alias: str = "champion"):
    """Registra el modelo del run (o reutiliza su versión) y le asigna el alias. Devuelve la versión."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    nombre = config.MODELOS_REGISTRADOS[question]
    run = client.get_run(run_id) if run_id else buscar_run_final(client, question)
    model_uri = run.data.tags.get("model_uri")
    if not model_uri:
        raise ValueError(f"El run {run.info.run_id} no tiene el tag model_uri (¿registró modelo?)")

    existentes = client.search_model_versions(
        f"name = '{nombre}' and run_id = '{run.info.run_id}'"
    )
    if existentes:
        version = max(existentes, key=lambda v: int(v.version))
        logger.info(f"{nombre}: el run ya estaba registrado como versión {version.version}")
    else:
        version = mlflow.register_model(
            model_uri,
            nombre,
            tags={
                "question": question,
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "git_commit": run.data.tags.get("git_commit", ""),
                "data_version": run.data.params.get("data_version", ""),
            },
        )
    client.update_registered_model(nombre, description=DESCRIPCIONES[question])
    client.set_registered_model_alias(nombre, alias, version.version)
    logger.success(
        f"{nombre} versión {version.version} con alias '{alias}' (run {run.info.run_id})"
    )
    return version


@app.command()
def main(question: str = "p1", run_id: str = "", alias: str = "champion"):
    registrar_champion(question, run_id or None, alias)


if __name__ == "__main__":
    app()
