# ML Canvas — Proyecto MiBici (Equipo 42)

TC5061 MLOps · Etapa 1 · Versión 1 (con cifras reales de la exploración y del baseline de P1). Lo de P2 y P3 es una propuesta: Víctor (P2) y Andrea (P3) la ajustan con sus resultados.

Formato basado en el Machine Learning Canvas de Louis Dorard.

## Resumen de las preguntas

| | P1 | P2 | P3 |
|---|---|---|---|
| Tarea | Regresión | Clasificación binaria | Clustering |
| Qué se predice o agrupa | Viajes que inician en la estación | Si la estación estará en riesgo de quedarse sin bicis o sin anclajes | Grupos de estaciones con perfil de uso parecido |
| Granularidad | Estación × hora | Estación × hora de corte | Estación |
| Horizonte | Siguiente hora | Siguientes 3 horas | No predictivo; se recalcula cada trimestre |
| Usuario | Operador del sistema | Equipo de rebalanceo | Planeación urbana y expansión |

---

## 1. Propuesta de valor
- **Operador (P1):** anticipar la demanda de la siguiente hora en cada estación para planear la flota en lugar de reaccionar a quejas.
- **Rebalanceo (P2):** una lista priorizada de estaciones en riesgo para mover bicis antes de que la estación se vacíe o se llene.
- **Planeación (P3):** identificar tipos de estaciones (por ejemplo, de origen matutino, de destino laboral o de uso recreativo de fin de semana) y ver dónde se ubican para decidir expansión y capacidad.

## 2. Tarea de ML
- P1: regresión de un conteo (viajes por estación y hora).
- P2: clasificación binaria con una etiqueta construida a partir del flujo neto y la capacidad, porque los datos no incluyen el inventario.
- P3: clustering no supervisado sobre el perfil de uso de cada estación.

## 3. Decisiones
- P1: cuántas bicis conviene tener en cada estación o zona en la siguiente hora.
- P2: a qué estaciones mandar camionetas en las próximas horas y en qué orden.
- P3: en qué zonas abrir o ampliar estaciones y qué capacidad darles según el tipo de uso.

## 4. Cómo se hacen las predicciones
- Etapa 1: solo evaluación offline sobre validación y test.
- Etapas siguientes: P1 y P2 en batch al cierre de cada hora (servicio con FastAPI + Docker); P3 recalculado cada trimestre.

## 5. Evaluación offline
- Partición temporal: entrenamiento jul-2025 → mar-2026 (9 meses), validación abr → jun-2026 (3 meses) y test jul → sep-2026 (3 meses). Validación y test siempre son posteriores al entrenamiento.
- Los hiperparámetros se eligen con validación; el modelo final se reentrena con entrenamiento + validación (12 meses) y se reporta una sola vez en test.
- Cada modelo se compara contra su baseline con la misma partición.
- En P2 el costo es asimétrico: no avisar de una estación que se vacía (usuario sin servicio) cuesta más que una falsa alarma (un viaje de camioneta de más).

## 6. Fuentes de datos
- Viajes mensuales del portal de datos abiertos de MiBici, jul-2025 a sep-2026: 15 archivos `datos_abiertos_AAAA_MM.csv` (enlaces en `config.py`), **5,616,821 viajes** después de la limpieza.
- Nomenclatura de estaciones `nomenclatura_2026_09.csv`: **484 estaciones** con id, nombre, latitud, longitud y capacidad; 4 tienen capacidad 0.
- El sistema crece dentro del rango: de 368,186 viajes y 367 estaciones con viajes en jul-2025 a 490,056 viajes y 468 estaciones en sep-2026.
- No se usan datos externos en esta etapa (clima y días festivos quedan como mejora).

## 7. Recolección y versionado de datos
- `dataset.py` descarga los meses y la nomenclatura, normaliza columnas, codificación y fechas, quita duplicados y guarda Parquet en `data/processed/`.
- Manifiesto versionado en Git con número de registros y hash SHA-256 de cada archivo; MLflow registra como `data_version` el hash del manifiesto.
- Las columnas de usuario (`usuario_id`, `genero`, `anio_nacimiento`) no se usan en ningún modelo ni se registran en MLflow; se propone no guardarlas en los datos procesados.

## 8. Variables
Todas se calculan con información disponible al cierre de la hora en que se hace la predicción (rezagos con `shift` sobre la cuadrícula completa y estadísticas históricas solo del entrenamiento).
- **Base común:** tabla estación × hora con salidas, llegadas, flujo neto (llegadas − salidas) y capacidad; las horas sin viajes van en 0 y cada estación entra desde su primer viaje.
- **P1:** hora del día, día de la semana, fin de semana, mes; viajes de la estación hace 1, 2, 24 y 168 horas; promedio de las últimas 24 horas; promedio de la misma hora en las 4 semanas anteriores; capacidad.
- **P2:** las de P1 aplicadas a salidas, llegadas y flujo neto; flujo neto de las últimas 3 horas relativo a la capacidad; tasa histórica de riesgo de la estación en ese bloque horario (calculada en entrenamiento).
- **P3:** proporción de salidas y de llegadas por hora (24 h) en días entre semana y en fin de semana, y flujo neto promedio relativo a la capacidad. La ubicación no entra al clustering; se usa para el mapa.

## 9. Construcción de modelos
- Un baseline sencillo y al menos un candidato por pregunta (detalle en la tabla de modelos y métricas).
- Todo se entrena desde el código del repositorio y se registra en MLflow (experimento `mibici-equipo42`) con parámetros, métricas, etiquetas y artefactos, mediante la función común `registrar_run`.
- Reentrenamiento previsto: mensual, al publicarse un mes nuevo, o al detectar drift.

## 10. Monitoreo (etapas siguientes)
- Drift de las variables de entrada con Evidently. Ya se observa un cambio entre jul-2025 y sep-2026 (+33 % de viajes y 101 estaciones más con viajes), así que el monitoreo es necesario desde el inicio.
- Seguimiento de `test_mae` (P1) y `test_recall` (P2) contra el baseline en cada mes nuevo; si el modelo deja de superarlo, se reentrena.

---

## Formulación detallada

### P1 — Viajes por estación en la siguiente hora
- **Qué se predice:** al cierre de cada hora *t*, el número de viajes que iniciarán en la estación *s* durante la hora siguiente [*t*, *t* + 1 h).
- **Granularidad y horizonte:** estación × hora; siguiente hora.
- **Relevancia:** la demanda cambia mucho por hora y por estación (entre semana hay picos a las 8 h y de 17 a 19 h; en fin de semana el uso se concentra a mediodía). Saberlo con una hora de anticipación permite mover bicis antes de que falten.
- **Quién y qué decide:** el operador decide dónde concentrar la flota y qué estaciones priorizar en la siguiente hora.

### P2 — Riesgo de quedarse sin bicis o sin anclajes
- **Qué se predice:** al cierre de cada hora *t*, si la estación *s* (capacidad *C*) estará en riesgo durante las siguientes *W* = 3 horas.
- **Etiqueta:** sea *F(k)* el flujo neto acumulado (llegadas − salidas) de *t* a *t* + *k* horas, *k* = 1…3.
  - Riesgo de vaciarse si el mínimo de *F(k)* ≤ −0.3·*C*.
  - Riesgo de llenarse si el máximo de *F(k)* ≥ 0.6·*C*.
  - *y* = 1 si se cumple cualquiera.
- **Calibración:** en jul-2025 esta etiqueta deja 7.34 % de estación-horas positivas (6.17 % de vaciarse y 1.18 % de llenarse), un desbalance manejable que no queda ni en ~1 % ni en ~50 %.
- **Justificación de la aproximación:** sin inventario, una salida neta grande respecto a la capacidad en pocas horas vacía la estación salvo que empezara casi llena o que la rebalanceen; lo mismo con las llegadas netas y el llenado. Se usa el mínimo y el máximo dentro de la ventana, no solo el valor final, para no perder picos intermedios. El umbral de vaciarse es más bajo (más sensible) porque es el caso de mayor costo: el usuario sin bici se queda sin servicio, mientras que el que trae bici puede dejarla en otra estación.
- **Limitaciones:** se desconoce el inventario inicial y el rebalanceo con camionetas no aparece en los datos, así que la etiqueta mide presión de demanda, no disponibilidad real. Mejora futura: 3 clases (vaciarse / llenarse / normal).
- **Quién y qué decide:** el equipo de rebalanceo decide a qué estaciones mandar camionetas y en qué orden.

### P3 — Grupos de estaciones por patrón de uso
- **Qué se agrupa:** cada estación con historia suficiente en el periodo de entrenamiento, según su perfil de uso.
- **Granularidad y horizonte:** estación; descriptivo, se recalcula cada trimestre. El periodo de test se usa para revisar que los grupos sean estables.
- **Relevancia:** muestra qué zonas generan viajes en la mañana, cuáles los reciben y cuáles se usan sobre todo en fin de semana; eso orienta dónde crecer y con qué capacidad.
- **Quién y qué decide:** planeación urbana decide zonas de expansión y capacidad de estaciones nuevas, comparándolas con grupos de estaciones parecidas.

---

## Modelos y métricas

| | Baseline | Candidato(s) | Métricas | Por qué esas métricas |
|---|---|---|---|---|
| **P1** | Naive estacional: viajes de la misma estación a la misma hora una semana antes | `HistGradientBoostingRegressor` con `loss="poisson"`; `Ridge` como alternativa lineal | **`test_mae`**, `test_rmse` (y sus versiones `val_`) | MAE queda en viajes por hora y se explica fácil al operador; RMSE castiga los errores grandes, que se concentran en las horas pico. MAPE se descarta porque el 53.1 % de las estación-horas tiene 0 salidas. La pérdida Poisson se usa al entrenar el candidato porque el objetivo es un conteo. |
| **P2** | Naive estacional: la etiqueta de la misma estación y bloque horario una semana antes | `LogisticRegression(class_weight="balanced")`; `HistGradientBoostingClassifier` | **`test_pr_auc`** (average precision), `test_recall`, `test_precision`, `test_f1` | Con 7.34 % de positivos, accuracy engaña. Recall porque no avisar de una estación por vaciarse es lo más caro; precision para no mandar camionetas de más; PR-AUC resume ambas sin fijar un umbral. |
| **P3** | `KMeans` con *k* elegido por silhouette (*k* = 3…8) | PCA + `KMeans`; `AgglomerativeClustering` (Ward) | **`train_silhouette`**, `train_davies_bouldin`, `test_estabilidad_ari` | Sin etiquetas, se usan métricas internas de separación y compactación. La estabilidad (ARI entre los grupos de entrenamiento y la asignación con perfiles del test) revisa que los grupos no sean ruido de un solo periodo. |

**Resultado del baseline de P1** (run `p1-baseline-semana-anterior`): `val_mae` 0.993, `val_rmse` 2.004, `test_mae` 0.942, `test_rmse` 1.921 viajes por estación-hora. El error se concentra en las horas pico (cerca de 1.8 viajes a las 18 h y 1.4 a las 8 h) y es casi 0 de 1 a 4 h.

---

## Decisiones y justificación

| Decisión | Justificación |
|---|---|
| Rango de 15 meses (jul-2025 → sep-2026), no 18 ni 24 | La estacionalidad anual es débil (con 24 meses, unos 382 mil viajes al mes estables) y lo que domina es la hora y el día de la semana. Además el sistema creció mucho dentro del propio rango (de 367 a 468 estaciones con viajes), así que los datos más viejos representan peor el servicio actual. Con 15 meses el modelo final se entrena con 12 y cada mes del test tiene su equivalente del año anterior. |
| Partición 9 / 3 / 3 por fecha | Hace falta un periodo de validación para elegir hiperparámetros sin tocar el test. Al ser temporal, el modelo nunca ve información posterior a lo que evalúa. |
| P1 por hora y no por día | El operador decide hora a hora. Los muchos ceros se manejan con MAE/RMSE en la evaluación y con pérdida Poisson en el candidato. |
| P2 con ventana de 3 h y no con el flujo del día | En el acumulado diario se cancelan la salida de la mañana y la llegada de la tarde; con 3 h se detectan esos picos y el resultado sirve para rebalancear durante el día. |
| Cada estación entra desde su primer viaje | Entre jul-2025 y sep-2026 aparecieron 101 estaciones con viajes; llenar con ceros el periodo antes de que existieran sesgaría rezagos y métricas. |
| Sin datos personales en datos procesados ni en MLflow | Ninguna pregunta usa `usuario_id`, `genero` ni `anio_nacimiento`; quitarlos aplica minimización de datos y cumple la buena práctica de no registrar datos personales. `registrar_run` rechaza parámetros con esos nombres. |
