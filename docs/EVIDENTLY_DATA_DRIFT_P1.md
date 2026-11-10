# Monitoreo de Data Drift y calidad de datos — MiBici P1

## 1. Objetivo

Evaluar cambios en las distribuciones y en la completitud de las variables predictoras del modelo P1, que estima las salidas de bicicletas por estación y hora.

El análisis utiliza Evidently 0.7.23 y corresponde a una evaluación offline sobre datos históricos.

## 2. Periodos analizados

- Referencia: enero a marzo de 2026.
- Actual: julio a septiembre de 2026.
- Tamaño de muestra: 50,000 registros por periodo.
- Muestreo: aleatorio reproducible, con semilla 42.
- Variables predictoras: 11.

Las muestras se almacenan en `reports/evidently/`.

## 3. Metodología

Se utiliza `DataDriftPreset` de Evidently para comparar las distribuciones de las variables predictoras.

La configuración aplicada utiliza un umbral individual de 0.1 y un umbral global de proporción de variables con drift de 0.5.

Evidently aplica distancia de Wasserstein normalizada a las variables numéricas y Jensen-Shannon a `es_fin_semana`.

También se utiliza `DataSummaryPreset` para analizar características generales y completitud de los datos.

Se realiza un análisis complementario excluyendo `mes` del monitoreo, sin modificar las variables del modelo original.

## 4. Resultados de Data Drift

| Indicador | Análisis original | Sin mes |
|---|---:|---:|
| Variables evaluadas | 11 | 10 |
| Variables con drift | 1 | 0 |
| Porcentaje con drift | 9.09% | 0% |
| Drift global detectado | No | No |

La única variable que supera el umbral individual es `mes`, con un score de 7.264529.

Este resultado es consistente con la comparación de trimestres diferentes y debe interpretarse considerando la estacionalidad.

Las demás variables no superaron sus respectivos umbrales de drift.

## 5. Resultados de calidad de datos

| Variable | Faltantes referencia | Faltantes actual |
|---|---:|---:|
| capacidad | 0 (0%) | 188 (0.376%) |
| salidas_media_4sem | 5 (0.01%) | 1,836 (3.672%) |
| Total | 5 | 2,024 |

El aumento de valores faltantes constituye un hallazgo de calidad de datos que requiere investigación.

Se recomienda revisar las causas de ausencia de información, especialmente para `salidas_media_4sem`, antes de tomar decisiones sobre el tratamiento de los datos.

## 6. Limitaciones

- Se comparan periodos estacionales diferentes.
- El análisis se realiza sobre muestras, no sobre la totalidad de los registros.
- Las pruebas evalúan distribuciones marginales y no garantizan estabilidad predictiva.
- No se demuestra degradación del modelo ni se justifica automáticamente un reentrenamiento.
- El análisis complementario sin `mes` es diagnóstico; no modifica el modelo.
- Los resultados dependen de los métodos estadísticos, umbrales y tipos de variables configurados.

## 7. Reproducción

El análisis se ejecuta offline desde la raíz del repositorio con Python 3.12.

### 7.1. Preparar las muestras

Se requiere acceso a los datos fuente del proyecto MiBici y al entorno principal del proyecto.

Ejecutar:

    .venv/bin/python scripts/preparar_muestras_evidently.py

El script utiliza la función `construir_tabla()` del modelo P1, reutiliza el caché cuando corresponde y genera dos muestras reproducibles de 50,000 registros y 11 variables, con semilla 42.

Los periodos son enero-marzo de 2026 (referencia) y julio-septiembre de 2026 (actual).

### 7.2. Instalar Evidently

Crear el entorno independiente con Python 3.12:

    python3.12 -m venv .venv-evidently

Instalar las dependencias:

    .venv-evidently/bin/python -m pip install -r requirements-evidently.txt

### 7.3. Ejecutar el monitoreo

Ejecutar:

    .venv-evidently/bin/python scripts/evidently_drift_p1.py

El script genera los reportes HTML y archivos CSV de Data Drift y completitud de datos en `reports/evidently/`.

### 7.4. Control de versiones

Se verificó que las muestras regeneradas son idénticas a las originales y que los resultados del monitoreo se reproducen.

El directorio `reports/evidently/` está excluido mediante `.gitignore`. Los archivos Parquet, HTML y CSV se conservan localmente y no se incorporan al repositorio.

## 8. Evidencias

- `reports/evidently/reporte_drift_p1.html`
- `reports/evidently/reporte_drift_p1_sin_mes.html`
- `reports/evidently/resultados_drift_p1.csv`
- `reports/evidently/resultados_calidad_p1.csv`

## 9. Conclusión

En las muestras analizadas no se detectó drift global bajo los umbrales establecidos. El único cambio individual identificado corresponde a `mes`, mientras que las demás variables no superaron sus umbrales.

Se identificó un incremento de valores faltantes en el periodo actual, principalmente en `salidas_media_4sem`.

Se recomienda complementar estos resultados con la evaluación del desempeño predictivo y las pruebas offline A/B realizadas por el equipo antes de decidir sobre un eventual reentrenamiento.
