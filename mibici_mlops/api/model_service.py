"""Capa de inferencia: carga desde el Model Registry de MLflow el modelo que sirve la API.

La ubicación del registry sale de MLFLOW_TRACKING_URI y el modelo de MODEL_NAME + MODEL_ALIAS
(o MODEL_VERSION), con valores por defecto en config.py. El modelo se carga una sola vez y se
reutiliza en cada petición; si el registry o el modelo no están disponibles se responde 503.
"""

import os
import threading

from fastapi import HTTPException
from loguru import logger
import mlflow
from mlflow import MlflowClient

from mibici_mlops import config

# Sin esto, si el servidor de MLflow está apagado cada petición espera varios minutos de
# reintentos antes de responder 503. Se pueden cambiar con las mismas variables de entorno.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")

_modelo = None
_info: dict = {}
_lock = threading.Lock()


class ModeloNoDisponible(HTTPException):
    """El registry o el modelo no están disponibles; FastAPI lo devuelve como 503."""

    def __init__(self, detalle: str):
        super().__init__(status_code=503, detail=detalle)


def model_uri() -> str:
    """URI del modelo en el registry: por versión si MODEL_VERSION existe, si no por alias."""
    if config.API_MODEL_VERSION:
        return f"models:/{config.API_MODEL_NAME}/{config.API_MODEL_VERSION}"
    return f"models:/{config.API_MODEL_NAME}@{config.API_MODEL_ALIAS}"


def _resolver_version(client: MlflowClient):
    if config.API_MODEL_VERSION:
        return client.get_model_version(config.API_MODEL_NAME, config.API_MODEL_VERSION)
    return client.get_model_version_by_alias(config.API_MODEL_NAME, config.API_MODEL_ALIAS)


def get_model():
    """Devuelve el modelo pyfunc (con su firma) listo para predict(DataFrame)."""
    global _modelo, _info
    if _modelo is not None:
        return _modelo
    with _lock:
        if _modelo is None:
            uri = model_uri()
            try:
                mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
                version = _resolver_version(MlflowClient())
                modelo = mlflow.pyfunc.load_model(uri)
            except Exception as exc:
                logger.error(f"No se pudo cargar {uri} desde {config.MLFLOW_TRACKING_URI}: {exc}")
                raise ModeloNoDisponible(
                    f"Modelo no disponible: {uri}. Revisa que el servidor de MLflow esté "
                    "activo y que MLFLOW_TRACKING_URI apunte a él."
                ) from exc
            _info = {
                "nombre": config.API_MODEL_NAME,
                "alias": None if config.API_MODEL_VERSION else config.API_MODEL_ALIAS,
                "version": str(version.version),
                "run_id": version.run_id,
            }
            _modelo = modelo
            logger.success(f"Modelo cargado: {uri} (versión {version.version})")
    return _modelo


def info_modelo() -> dict:
    """Nombre, alias, versión y run de origen del modelo cargado (vacío si aún no se carga)."""
    return dict(_info)


def reiniciar() -> None:
    """Olvida el modelo cargado; la siguiente petición lo vuelve a buscar en el registry."""
    global _modelo, _info
    with _lock:
        _modelo = None
        _info = {}
