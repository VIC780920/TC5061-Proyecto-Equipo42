# A/B testing offline — MiBici P1

## 1. Objetivo

Comparar la versión vigente del modelo P1 (champion) contra una versión reentrenada con datos del periodo actual (challenger) y decidir con evidencia cuál debe quedar como versión vigente en el Model Registry de MLflow.

## 2. Diseño

| Pieza | Detalle |
|---|---|
| Champion | `mibici-equipo42-p1-demanda@champion` (`p1-hgb-final`), entrenado con train + val: jul-2025 a jun-2026 |
| Challenger | Mismo HistGradientBoosting y mismos hiperparámetros que el champion, reentrenado con datos de jul-2025 a ago-2026 (`p1-hgb-challenger`), registrado como nueva versión del mismo modelo con alias `challenger` |
| Conjunto de evaluación | sep-2026, el mismo para las dos versiones. Ninguna lo vio al entrenar |
| Métricas | MAE (principal) y RMSE, las mismas que se usaron para elegir el modelo P1 |

- Solo cambian los datos de entrenamiento: así la diferencia mide el efecto de reentrenar con datos recientes, que es la pregunta del Continuous Training.
- El script verifica que ninguna versión se haya entrenado con datos de sep-2026, para evitar fuga de datos.

## 3. Evidencia estadística

- **Comparación pareada (principal):** las dos versiones predicen exactamente las mismas filas. La diferencia de MAE y RMSE (challenger − champion) lleva un intervalo de confianza del 95 % por bootstrap, remuestreando días completos (2,000 repeticiones), porque las horas de un mismo día están correlacionadas.
- **Por segmento:** estaciones nuevas (primera hora observada desde jul-2026) vs existentes, y entre semana vs fin de semana. Esto conecta con el hallazgo del monitoreo de drift sobre las estaciones sin historial.
- **Split de tráfico simulado (complemento):** reparte las estaciones 50/50 entre las versiones, como haría un A/B en producción. Tiene más varianza porque cada grupo ve estaciones distintas, por eso la comparación pareada es la principal.

**Regla de decisión** (definida antes de correr): se promueve el challenger si el límite superior del IC 95 % de la diferencia de MAE es menor que 0 y el RMSE no empeora. Si no, se mantiene el champion.

## 4. Reproducción (PowerShell, entorno `mibici`)

Terminal 1, en la raíz del repo:

    python -m mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

Terminal 2 (requiere el champion ya registrado: `python -m mibici_mlops.registry --question p1`):

    $env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
    python -m mibici_mlops.monitoring.ab_test_p1 challenger
    python -m mibici_mlops.monitoring.ab_test_p1 ab-test

- `challenger` entrena, registra el run `p1-hgb-challenger` y crea la nueva versión con alias `challenger`.
- `ab-test` registra el run `p1-ab-test` (métricas, tabla y gráfica) y escribe en `reports/ab_test/`: `ab_resumen_p1.csv`, `ab_segmentos_p1.csv`, `ab_split_trafico_p1.csv` y `ab_test_p1.png`.
- `ab-test --promover` además mueve el alias `champion` a la nueva versión (y deja `champion_anterior` en la versión previa), solo si el challenger cumple la regla.

## 5. Resultados (corrida del 14 nov 2026)

- Champion: versión 1 (`learning_rate` 0.05, `max_leaf_nodes` 63). Challenger: versión 2, mismos hiperparámetros, entrenada con 3,843,549 filas (run `p1-hgb-challenger`).
- Evaluación: sep-2026, 338,690 filas (estación × hora), 30 días, 471 estaciones; 58 son estaciones nuevas (primera hora observada desde jul-2026), las mismas que identificó el monitoreo de drift.

### 5.1. Comparación pareada

| Métrica | Champion (v1) | Challenger (v2) | Diferencia (v2 − v1) | IC 95 % |
|---|---:|---:|---:|---|
| MAE | 0.8371 | 0.8356 | −0.0016 | [−0.0022, −0.0008] |
| RMSE | 1.5267 | 1.5212 | −0.0055 | [−0.0104, −0.0009] |

Fila por fila, el challenger tiene menor error en el 49.52 % de los casos y el champion en el 50.48 %.

### 5.2. Por segmento

| Segmento | Estaciones | Filas | MAE v1 | MAE v2 | RMSE v1 | RMSE v2 | Δ MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Todas | 471 | 338,690 | 0.8371 | 0.8356 | 1.5267 | 1.5212 | −0.19 % |
| Estaciones nuevas | 58 | 41,330 | 0.4642 | 0.4622 | 0.7790 | 0.7782 | −0.43 % |
| Estaciones existentes | 413 | 297,360 | 0.8890 | 0.8875 | 1.6033 | 1.5974 | −0.17 % |
| Entre semana | 471 | 248,354 | 0.8622 | 0.8606 | 1.5217 | 1.5180 | −0.19 % |
| Fin de semana | 471 | 90,336 | 0.7683 | 0.7670 | 1.5404 | 1.5302 | −0.17 % |

### 5.3. Split de tráfico simulado

236 estaciones con el champion y 235 con el challenger: MAE 0.8476 vs 0.8255, diferencia −0.0222 con IC 95 % [−0.1108, 0.0669]. El intervalo incluye 0 y es mucho más ancho que el de la comparación pareada: la diferencia entre los grupos de estaciones pesa más que la diferencia entre modelos.

### 5.4. Interpretación y decisión

- El challenger cumple la regla: el IC 95 % de la diferencia de MAE queda completo debajo de 0 y el RMSE también baja. La mejora no es por azar.
- La mejora es muy pequeña (−0.19 % de MAE) y fila por fila los modelos están prácticamente empatados; el challenger gana un poco más en RMSE, o sea, reduce algunos errores grandes.
- Mejora en todos los segmentos y no empeora en ninguno. La mejora relativa más grande está en las estaciones nuevas, que el champion nunca vio y el challenger ya vio en jul–ago.
- Esto es consistente con el drift leve del monitoreo: reentrenar con datos recientes ayuda, pero poco.
- Con un diseño por split de tráfico, una diferencia de este tamaño no se habría detectado; por eso la comparación pareada sobre las mismas filas es la evidencia principal.
- **Decisión:** promover la versión 2 a `champion` (la versión 1 queda como `champion_anterior`), con `ab-test --promover`.
