"""Genera muestras reproducibles para el monitoreo offline de MiBici P1."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mibici_mlops.modeling.p1 import CARACTERISTICAS, construir_tabla


OUTPUT = ROOT / "reports" / "evidently"
SAMPLE_SIZE = 50_000
RANDOM_STATE = 42


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    tabla = construir_tabla()
    fechas = tabla["hora"]

    referencia = tabla.loc[
        (fechas >= "2026-01-01") & (fechas < "2026-04-01"),
        CARACTERISTICAS,
    ]

    actual = tabla.loc[
        (fechas >= "2026-07-01") & (fechas < "2026-10-01"),
        CARACTERISTICAS,
    ]

    if len(referencia) < SAMPLE_SIZE or len(actual) < SAMPLE_SIZE:
        raise ValueError(
            "No existen suficientes registros para generar las muestras."
        )

    referencia = referencia.sample(
        n=SAMPLE_SIZE, random_state=RANDOM_STATE
    )
    actual = actual.sample(
        n=SAMPLE_SIZE, random_state=RANDOM_STATE
    )

    referencia.to_parquet(OUTPUT / "referencia_p1.parquet", index=False)
    actual.to_parquet(OUTPUT / "actual_p1.parquet", index=False)

    print("Muestras generadas correctamente:")
    print(f"Referencia: {referencia.shape}")
    print(f"Actual: {actual.shape}")
    print(f"Directorio: {OUTPUT}")


if __name__ == "__main__":
    main()
