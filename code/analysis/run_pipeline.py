from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PIPELINE_VERSION = "3.1"

# Configuración predeterminada para ejecutar directamente con F5 en Spyder.
# La carpeta "resultados" debe estar junto a este archivo.
INPUT_FOLDER_NAME = "resultados"
OUTPUT_FOLDER_NAME = "salidas"
EXPECTED_REPLICATES = 3


def run(command: list[str]) -> None:
    print("\n> " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline 3.1 compatible con salidas v8."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Carpeta con los JSON finales. Si se omite, se usa la carpeta "
            f"'{INPUT_FOLDER_NAME}' situada junto a este archivo."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Carpeta de salida. Si se omite, se usa la carpeta "
            f"'{OUTPUT_FOLDER_NAME}' situada junto a este archivo."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs=2,
        metavar=("CONDICION_1", "CONDICION_2"),
    )
    parser.add_argument(
        "--expected-replicates",
        type=int,
        default=EXPECTED_REPLICATES,
    )
    args = parser.parse_args()

    py = sys.executable
    input_dir = (
        args.input_dir.resolve()
        if args.input_dir is not None
        else root / INPUT_FOLDER_NAME
    )
    out = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / OUTPUT_FOLDER_NAME
    )

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"No se encuentra la carpeta de resultados: {input_dir}\n"
            f"Crea una carpeta llamada '{INPUT_FOLDER_NAME}' junto a "
            "run_pipeline.py y coloca dentro todos los JSON."
        )

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No se encontraron archivos JSON directamente dentro de: {input_dir}"
        )

    required_scripts = [
        "01_auditoria_datos.py",
        "02_dataset_maestro.py",
        "03_estadistica_descriptiva.py",
        "04_inferencia.py",
        "05_figuras_publicacion.py",
        "06_tablas_publicacion.py",
    ]
    missing_scripts = [name for name in required_scripts if not (root / name).is_file()]
    if missing_scripts:
        raise FileNotFoundError(
            "Faltan scripts del pipeline junto a run_pipeline.py:\n- "
            + "\n- ".join(missing_scripts)
        )

    out.mkdir(parents=True, exist_ok=True)

    print(f"Pipeline {PIPELINE_VERSION}")
    print(f"Carpeta de entrada: {input_dir}")
    print(f"JSON detectados: {len(json_files)}")
    print(f"Carpeta de salida: {out}")

    run(
        [
            py,
            str(root / "01_auditoria_datos.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(out / "01_auditoria"),
        ]
    )

    master_cmd = [
        py,
        str(root / "02_dataset_maestro.py"),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(out / "02_dataset_maestro"),
        "--expected-replicates",
        str(args.expected_replicates),
    ]
    if args.conditions:
        master_cmd += ["--conditions", *args.conditions]
    run(master_cmd)

    run(
        [
            py,
            str(root / "03_estadistica_descriptiva.py"),
            "--input-dir",
            str(out / "02_dataset_maestro"),
            "--output-dir",
            str(out / "03_descriptiva"),
        ]
    )
    run(
        [
            py,
            str(root / "04_inferencia.py"),
            "--input-dir",
            str(out / "02_dataset_maestro"),
            "--output-dir",
            str(out / "04_inferencia"),
        ]
    )
    run(
        [
            py,
            str(root / "05_figuras_publicacion.py"),
            "--master-dir",
            str(out / "02_dataset_maestro"),
            "--descriptive-dir",
            str(out / "03_descriptiva"),
            "--inference-dir",
            str(out / "04_inferencia"),
            "--output-dir",
            str(out / "05_figuras_articulo"),
        ]
    )
    run(
        [
            py,
            str(root / "06_tablas_publicacion.py"),
            "--master-dir",
            str(out / "02_dataset_maestro"),
            "--descriptive-dir",
            str(out / "03_descriptiva"),
            "--inference-dir",
            str(out / "04_inferencia"),
            "--output-dir",
            str(out / "06_tablas_articulo"),
        ]
    )

    print(f"\nPipeline {PIPELINE_VERSION} completado correctamente. Salidas: {out}")


if __name__ == "__main__":
    main()
