# Despliegue de MiBici con Docker

## Construcción
```bash
docker build -f Containerfile -t mibici-equipo42-api:dev .
```

## Ejecución
```bash
docker run --rm --name mibici-api -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 mibici-equipo42-api:dev
```

## Arquitectura
La API FastAPI se ejecuta en un contenedor Docker y consulta los modelos registrados en un servidor MLflow externo mediante MLFLOW_TRACKING_URI.

## Estado de validación
Imagen Docker construida correctamente. Importaciones de FastAPI, Uvicorn, MLflow y mibici_mlops verificadas.
Pendiente: integración de la API, arranque del servidor y pruebas HTTP.

## Configuración del servidor MLflow

Para permitir conexiones desde el contenedor Docker, iniciar MLflow
en la máquina anfitriona con los parámetros de red adecuados:

    python -m mlflow server \
      --backend-store-uri sqlite:///mlflow.db \
      --host 0.0.0.0 \
      --port 5000 \
      --allowed-hosts "localhost;127.0.0.1;host.docker.internal;host.containers.internal"

El contenedor utiliza la variable de entorno:

    MLFLOW_TRACKING_URI=http://host.docker.internal:5000

La configuración de allowed-hosts permite que MLflow acepte
solicitudes desde Docker y evita errores HTTP 403 por validación
del encabezado Host.

Nota de seguridad: utilizar esta configuración únicamente en un
entorno de desarrollo confiable. No exponer el puerto 5000 a Internet.

Para registrar el modelo champion de P1, una vez integrado el PR #30:

    python -m mibici_mlops.registry --question p1

Las pruebas HTTP y de inferencia se realizarán después de integrar
los PR #29 (FastAPI) y #30 (servicio de inferencia).
