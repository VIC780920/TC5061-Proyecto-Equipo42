"""Preparación de datos para el modelo P2 de MiBici."""

from pathlib import Path

import pandas as pd

from mibici_mlops.config import (
    ESTACIONES_PATH,
    FECHA_FIN,
    PROJ_ROOT,
    TEST_INICIO,
    VAL_INICIO,
    VIAJES_PATH,
)
from mibici_mlops.features import (
    agregar_calendario,
    agregar_capacidad,
    agregar_estacion_hora,
    agregar_promedios,
    agregar_rezagos,
    crear_etiqueta_p2,
    particion_temporal,
)

def consolidar_flujos_horarios() -> pd.DataFrame:
    """Consolida los viajes por mes y completa la cuadrícula horaria."""

    archivos = sorted(VIAJES_PATH.glob("viajes_*.parquet"))

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos en {VIAJES_PATH}"
        )

    acumulados = []

    for archivo in archivos:
        print(f"Procesando: {archivo.name}")

        viajes = pd.read_parquet(
            archivo,
            columns=[
                "inicio_viaje",
                "fin_viaje",
                "origen_id",
                "destino_id",
            ],
        )

        salidas = viajes.groupby(
            ["origen_id", viajes["inicio_viaje"].dt.floor("h")]
        ).size().rename("salidas")
        salidas.index.names = ["estacion_id", "hora"]

        llegadas = viajes.groupby(
            ["destino_id", viajes["fin_viaje"].dt.floor("h")]
        ).size().rename("llegadas")
        llegadas.index.names = ["estacion_id", "hora"]

        mensual = pd.concat(
            [salidas, llegadas], axis=1
        ).fillna(0)

        acumulados.append(mensual.astype("int32"))
        del viajes

    conteos = pd.concat(acumulados)
    conteos = (
        conteos.groupby(level=["estacion_id", "hora"])
        .sum()
        .reset_index()
    )

    tabla = agregar_estacion_hora(conteos)

    print(f"Meses procesados: {len(archivos)}")
    print(f"Registros estación-hora: {len(tabla):,}")

    return tabla


def preparar_base_p2() -> pd.DataFrame:
    """Prepara los flujos horarios con capacidad válida."""

    tabla = consolidar_flujos_horarios()

    estaciones = pd.read_parquet(ESTACIONES_PATH)
    estaciones = estaciones.loc[estaciones["capacidad"] > 0].copy()

    tabla = tabla.loc[tabla["estacion_id"].isin(estaciones["estacion_id"])].copy()
    tabla = agregar_capacidad(tabla, estaciones)

    # Excluir registros posteriores al periodo del proyecto
    fecha_fin_exclusiva = (
        pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)
    )
    tabla = tabla.loc[tabla["hora"] < fecha_fin_exclusiva].copy()

    return tabla


def generar_dataset_p2() -> pd.DataFrame:
    """Genera el dataset P2 con variables históricas y etiqueta."""

    print("Preparando base P2...")
    tabla = preparar_base_p2()

    print("Agregando variables de calendario...")
    tabla = agregar_calendario(tabla)

    print("Calculando rezagos históricos...")
    tabla = agregar_rezagos(
        tabla,
        columna="salidas",
        rezagos=(1, 2, 24, 168),
    )

    tabla = agregar_rezagos(
        tabla,
        columna="llegadas",
        rezagos=(1, 2, 24, 168),
    )

    tabla = agregar_rezagos(
        tabla,
        columna="flujo_neto",
        rezagos=(1, 2, 24, 168),
    )

    print("Calculando promedios históricos...")
    tabla = agregar_promedios(
        tabla,
        columna="salidas",
    )

    tabla = agregar_promedios(
        tabla,
        columna="llegadas",
    )

    tabla = agregar_promedios(
        tabla,
        columna="flujo_neto",
    )

    # Flujo neto acumulado de las tres horas anteriores,
    # relativo a la capacidad de la estación.
    tabla["flujo_neto_3h_rel"] = (
        tabla["flujo_neto_lag_1"]
        + tabla["flujo_neto_lag_2"]
        + tabla.groupby("estacion_id")["flujo_neto"].shift(3)
    ) / tabla["capacidad"].replace(0, float("nan"))

    print("Generando etiquetas P2...")
    tabla = crear_etiqueta_p2(
        tabla,
        umbral=0.30,
    )

    # Riesgo de desabasto o saturación en la siguiente hora
    tabla["riesgo_p2_t_1h"] = tabla.groupby("estacion_id")["riesgo_p2"].shift(-1)

    print("Calculando baseline estacional P2 (167 horas)...")
    tabla = agregar_rezagos(
        tabla,
        columna="riesgo_p2",
        rezagos=(167,),
    )

    print("Asignando particiones temporales...")
    tabla = particion_temporal(
        tabla,
        columna="hora",
    )
    # Evitar que las etiquetas t+1 crucen los límites temporales.
    for fecha_corte in [VAL_INICIO, TEST_INICIO]:
        ultima_hora = pd.Timestamp(fecha_corte) - pd.Timedelta(hours=1)
        tabla.loc[
            tabla["hora"] == ultima_hora,
            "riesgo_p2_t_1h",
        ] = float("nan")
    tabla = agregar_tasas_historicas_p2(tabla)
    return tabla

def agregar_tasas_historicas_p2(tabla: pd.DataFrame) -> pd.DataFrame:
    """Calcula tasas históricas sin utilizar etiquetas futuras."""
    tabla = tabla.copy()
    tabla = tabla.sort_values(["hora", "estacion_id"])

    claves = ["estacion_id", "hora_dia"]
    es_train = tabla["conjunto"].eq("train")
    train = tabla.loc[es_train].copy()

    # Para predecir t+1, la etiqueta de una fila anterior
    # solo es utilizable cuando ya se observó su resultado.
    for clase, nombre in [
        (1, "tasa_desabasto_hist"),
        (2, "tasa_saturacion_hist"),
    ]:
        indicador = train["riesgo_p2_t_1h"].map(lambda x: float(x == clase) if pd.notna(x) else float("nan"))
        acumulado = indicador.fillna(0).groupby(
            [train[c] for c in claves]
        ).cumsum()

        # Desplazamiento adicional para excluir la etiqueta
        # de la fila anterior, que corresponde a la hora actual.
        suma_previa = acumulado.groupby(
            [train[c] for c in claves]
        ).shift(1).fillna(0)
        conteo_previo = indicador.notna().astype(int).groupby([train[c] for c in claves]).cumsum().groupby([train[c] for c in claves]).shift(1).fillna(0)
        train[nombre] = (
            suma_previa / conteo_previo.replace(0, float("nan"))
        ).fillna(0)

        tasas = (
            train.assign(indicador=indicador)
            .groupby(claves)["indicador"]
            .mean()
            .rename(nombre)
        )

        tabla[nombre] = 0.0
        tabla.loc[train.index, nombre] = train[nombre]
        fuera_train = ~es_train
        tabla.loc[fuera_train, nombre] = (
            tabla.loc[fuera_train, claves]
            .merge(tasas.reset_index(), on=claves, how="left")[nombre]
            .fillna(0)
            .to_numpy()
        )

    return tabla
