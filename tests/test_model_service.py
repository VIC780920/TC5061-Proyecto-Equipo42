"""Pruebas de la capa de inferencia (model_service) y del registro del champion (registry).

Usan un MLflow temporal (SQLite en tmp_path) con un modelo HGB chico guardado igual que en P1
(float32, firma y skops), así que no dependen del servidor de tracking ni de los datos.
"""

import mlflow
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici_mlops import config, registry
from mibici_mlops.api import model_service
from mibici_mlops.modeling.p1 import CARACTERISTICAS, TIPOS_CONFIABLES
from mibici_mlops.tracking import registrar_resultados

NOMBRE = config.MODELOS_REGISTRADOS["p1"]


def _datos(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.uniform(0, 10, size=(n, len(CARACTERISTICAS))), columns=CARACTERISTICAS)
    X["hora_dia"] = rng.integers(0, 24, n)
    X["dia_semana"] = rng.integers(0, 7, n)
    X["es_fin_semana"] = (X["dia_semana"] >= 5).astype(int)
    X["mes"] = rng.integers(1, 13, n)
    X = X.astype("float32")
    y = rng.poisson(1 + X["salidas_lag_168"] / 5)
    return X, y


def _registrar_run_final(run_name="p1-hgb-final"):
    X, y = _datos()
    modelo = HistGradientBoostingRegressor(loss="poisson", max_iter=20, random_state=42)
    modelo.fit(X, y)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("git_commit", "abc1234")
        mlflow.log_param("data_version", "2a196cfa2823")
        registrar_resultados(
            {"test_mae": 0.77},
            modelo=modelo,
            X_ejemplo=X,
            registrar_manifiesto=False,
            tipos_confiables=TIPOS_CONFIABLES["hgb"],
        )
    return run.info.run_id, modelo, X


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "API_MODEL_NAME", NOMBRE)
    monkeypatch.setattr(config, "API_MODEL_ALIAS", "champion")
    monkeypatch.setattr(config, "API_MODEL_VERSION", None)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    model_service.reiniciar()
    yield
    model_service.reiniciar()


def test_registrar_champion_asigna_alias_y_linaje(mlflow_temporal):
    run_id, _, _ = _registrar_run_final()

    version = registry.registrar_champion("p1")

    client = mlflow.MlflowClient()
    champion = client.get_model_version_by_alias(NOMBRE, "champion")
    assert champion.version == version.version
    assert champion.run_id == run_id
    assert champion.tags["git_commit"] == "abc1234"
    assert champion.tags["data_version"] == "2a196cfa2823"
    assert client.get_registered_model(NOMBRE).description


def test_registrar_champion_no_duplica_versiones(mlflow_temporal):
    _registrar_run_final()

    registry.registrar_champion("p1")
    registry.registrar_champion("p1")

    assert len(mlflow.MlflowClient().search_model_versions(f"name = '{NOMBRE}'")) == 1


def test_registrar_champion_usa_el_run_mas_reciente(mlflow_temporal):
    _registrar_run_final()
    run_nuevo, _, _ = _registrar_run_final()

    version = registry.registrar_champion("p1")

    assert version.run_id == run_nuevo


def test_registrar_champion_sin_run_final_falla(mlflow_temporal):
    with pytest.raises(ValueError, match="p1-hgb-final"):
        registry.registrar_champion("p1")


def test_get_model_carga_por_alias_y_predice_igual(mlflow_temporal):
    run_id, modelo, X = _registrar_run_final()
    registry.registrar_champion("p1")

    cargado = model_service.get_model()
    pred = cargado.predict(X.head(10))

    np.testing.assert_allclose(pred, modelo.predict(X.head(10)), rtol=1e-6)
    assert (pred >= 0).all()
    assert model_service.info_modelo() == {
        "nombre": NOMBRE,
        "alias": "champion",
        "version": "1",
        "run_id": run_id,
    }


def test_get_model_reutiliza_el_modelo_cargado(mlflow_temporal):
    _registrar_run_final()
    registry.registrar_champion("p1")

    assert model_service.get_model() is model_service.get_model()


def test_get_model_por_version_tiene_prioridad(mlflow_temporal, monkeypatch):
    _registrar_run_final()
    registry.registrar_champion("p1")
    _registrar_run_final()
    registry.registrar_champion("p1")
    monkeypatch.setattr(config, "API_MODEL_VERSION", "1")

    model_service.get_model()

    assert model_service.model_uri() == f"models:/{NOMBRE}/1"
    assert model_service.info_modelo()["version"] == "1"
    assert model_service.info_modelo()["alias"] is None


def test_get_model_sin_modelo_registrado_responde_503(mlflow_temporal):
    with pytest.raises(model_service.ModeloNoDisponible) as error:
        model_service.get_model()

    assert error.value.status_code == 503


def test_get_model_reintenta_despues_de_un_fallo(mlflow_temporal):
    with pytest.raises(model_service.ModeloNoDisponible):
        model_service.get_model()
    _registrar_run_final()
    registry.registrar_champion("p1")

    assert model_service.get_model() is not None


def test_get_model_rechaza_tipos_que_no_coinciden_con_la_firma(mlflow_temporal):
    _, _, X = _registrar_run_final()
    registry.registrar_champion("p1")
    sin_columna = X.drop(columns=["capacidad"]).head(2)

    with pytest.raises(MlflowException):
        model_service.get_model().predict(sin_columna)


def test_api_predict_con_modelo_del_registry(mlflow_temporal, monkeypatch):
    main = pytest.importorskip("mibici_mlops.api.main")
    from fastapi.testclient import TestClient

    _, modelo, X = _registrar_run_final()
    registry.registrar_champion("p1")
    monkeypatch.setattr(main.app, "dependency_overrides", {})
    fila = X.iloc[0]
    payload = {c: float(fila[c]) for c in CARACTERISTICAS}
    for c in ("hora_dia", "dia_semana", "es_fin_semana", "mes"):
        payload[c] = int(fila[c])

    respuesta = TestClient(main.app).post("/predict", json=payload)

    assert respuesta.status_code == 200
    esperado = float(modelo.predict(X.head(1))[0])
    assert respuesta.json()["prediction"] == pytest.approx(esperado, rel=1e-6)


def test_api_responde_503_sin_registry(mlflow_temporal, monkeypatch):
    main = pytest.importorskip("mibici_mlops.api.main")
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main.app, "dependency_overrides", {})
    payload = {c: 1 for c in CARACTERISTICAS}

    respuesta = TestClient(main.app).post("/predict", json=payload)

    assert respuesta.status_code == 503
