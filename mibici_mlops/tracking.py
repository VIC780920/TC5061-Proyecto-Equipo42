"""Registro común de runs en MLflow para P1, P2 y P3 (Equipo 42).

Aplica en un solo lugar las convenciones del equipo: un experimento por proyecto,
etiquetas de equipo, pregunta y commit, parámetros de reproducción, métricas con
prefijo del conjunto y nada de datos personales.
"""

import hashlib
import subprocess

import mlflow
import mlflow.sklearn
from loguru import logger

from mibici_mlops.config import (
    EQUIPO,
    FECHA_FIN,
    FECHA_INICIO,
    MANIFIESTO_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
    SEED,
    TEST_INICIO,
    VAL_INICIO,
)

PREFIJOS_METRICAS = ("train_", "val_", "test_")
CAMPOS_PERSONALES = ("usuario", "genero", "nacimiento")


def obtener_git_commit() -> tuple[str, bool]:
    """Devuelve el hash del commit actual y si hay cambios sin commitear en archivos versionados."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True).strip()
        cambios = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJ_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido", True
    return commit, bool(cambios)


def obtener_data_version() -> str:
    """Versión de los datos: primeros 12 caracteres del SHA-256 del manifiesto versionado en Git."""
    if not MANIFIESTO_PATH.exists():
        logger.warning(f"No existe {MANIFIESTO_PATH.name}; se registra data_version='sin-manifiesto'")
        return "sin-manifiesto"
    return hashlib.sha256(MANIFIESTO_PATH.read_bytes()).hexdigest()[:12]


def registrar_run(
    pregunta: str,
    run_name: str,
    params: dict,
    metricas: dict,
    artefactos: list | tuple = (),
    modelo=None,
    tags: dict | None = None,
) -> str:
    """Registra un run en el experimento del equipo y devuelve su run_id.

    pregunta: "P1", "P2" o "P3". run_name: nombre descriptivo (ej. "p1-baseline-semana-anterior").
    params: hiperparámetros y detalles del modelo; se agregan solos fecha_inicio, fecha_fin,
    val_inicio, test_inicio, data_version y seed. metricas: nombres con prefijo train_, val_ o
    test_. artefactos: rutas de archivos (gráficas, mapas). modelo: estimador de scikit-learn opcional.
    """
    malas = [m for m in metricas if not m.startswith(PREFIJOS_METRICAS)]
    if malas:
        raise ValueError(f"Métricas sin prefijo de conjunto (train_/val_/test_): {malas}")
    personales = [p for p in params if any(c in p.lower() for c in CAMPOS_PERSONALES)]
    if personales:
        raise ValueError(f"No se registran datos personales como parámetros: {personales}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    commit, con_cambios = obtener_git_commit()

    params_comunes = {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "val_inicio": VAL_INICIO,
        "test_inicio": TEST_INICIO,
        "data_version": obtener_data_version(),
        "seed": SEED,
    }
    etiquetas = {
        "equipo": EQUIPO,
        "pregunta": pregunta,
        "git_commit": commit,
        "git_cambios_sin_commit": str(con_cambios),
        **(tags or {}),
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(etiquetas)
        mlflow.log_params({**params_comunes, **params})
        mlflow.log_metrics(metricas)
        for ruta in artefactos:
            mlflow.log_artifact(str(ruta))
        if modelo is not None:
            mlflow.sklearn.log_model(modelo, name="modelo")

    logger.success(f"Run '{run_name}' registrado en '{MLFLOW_EXPERIMENT}': {run.info.run_id}")
    return run.info.run_id