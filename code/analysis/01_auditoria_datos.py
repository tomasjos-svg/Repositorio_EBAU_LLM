from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PIPELINE_VERSION = "3.1"

from common import (
    EXPECTED_COLUMNS,
    METRICS,
    deduplicate_rows,
    ensure_dir,
    load_json_folder,
    write_excel_sheets,
)

# Rutas predeterminadas relativas a la carpeta que contiene este script.
PIPELINE_ROOT = Path(__file__).resolve().parent
SPYDER_INPUT_DIR = PIPELINE_ROOT / "resultados"
SPYDER_OUTPUT_DIR = PIPELINE_ROOT / "salidas" / "01_auditoria"


def correct_audit_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige las banderas derivadas antes de construir la auditoría.

    La función ``normalize_dataframe`` de ``common.py`` considera débil una
    referencia vacía. Esto es razonable en una ejecución válida, pero los
    registros con error técnico también tienen normalmente la referencia
    vacía. Por tanto, un error técnico no debe contabilizarse además como
    referencia débil.

    Esta corrección se aplica aquí para que la auditoría separe correctamente:
    - errores técnicos;
    - referencias débiles entre las ejecuciones que sí llegaron a evaluarse.
    """
    out = df.copy()

    out["valid_execution"] = ~out["technical_error"].fillna(False).astype(bool)

    # Una referencia solo puede clasificarse como débil si la ejecución fue válida.
    out["weak_reference_auto_raw"] = out["weak_reference_auto"].fillna(False).astype(bool)
    out["weak_reference_auto"] = (
        out["valid_execution"] & out["weak_reference_auto_raw"]
    )

    out["valid_for_primary_analysis"] = (
        out["valid_execution"] & ~out["weak_reference_auto"]
    )
    out["valid_for_sensitivity_analysis"] = out["valid_execution"]

    return out


def build_audit(
    df: pd.DataFrame,
    inventory: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    missing_expected = pd.DataFrame(
        {
            "column": EXPECTED_COLUMNS,
            "missing_count": [
                int(df[col].isna().sum()) if col in df.columns else len(df)
                for col in EXPECTED_COLUMNS
            ],
            "missing_pct": [
                float(df[col].isna().mean() * 100)
                if col in df.columns
                else 100.0
                for col in EXPECTED_COLUMNS
            ],
        }
    )

    conditions = (
        df.groupby(["condition", "model", "api"], dropna=False)
        .agg(
            records=("row_id", "size"),
            unique_problems=("problem_id", "nunique"),
            valid_executions=("valid_execution", "sum"),
            technical_errors=("technical_error", "sum"),
            weak_references=("weak_reference_auto", "sum"),
            primary_valid=("valid_for_primary_analysis", "sum"),
        )
        .reset_index()
    )

    datasets = (
        df.groupby(["dataset_name", "domain", "education_level"], dropna=False)
        .agg(
            records=("row_id", "size"),
            questions=("question_number", "nunique"),
            runs=("run", "nunique"),
            conditions=("condition", "nunique"),
            valid_executions=("valid_execution", "sum"),
            technical_errors=("technical_error", "sum"),
            weak_references=("weak_reference_auto", "sum"),
            primary_valid=("valid_for_primary_analysis", "sum"),
        )
        .reset_index()
    )

    invalid_metric_range_rows: list[dict] = []
    for metric in METRICS:
        bad = df[metric].notna() & ~df[metric].between(0, 100)
        columns = [
            "row_id",
            "dataset_name",
            "question_number",
            "run",
            "condition",
            "valid_execution",
            "technical_error",
            metric,
        ]
        for _, row in df.loc[bad, columns].iterrows():
            invalid_metric_range_rows.append(
                {
                    "row_id": row["row_id"],
                    "dataset_name": row["dataset_name"],
                    "question_number": row["question_number"],
                    "run": row["run"],
                    "condition": row["condition"],
                    "valid_execution": row["valid_execution"],
                    "technical_error": row["technical_error"],
                    "metric": metric,
                    "value": row[metric],
                }
            )
    invalid_metric_ranges = pd.DataFrame(
        invalid_metric_range_rows,
        columns=[
            "row_id",
            "dataset_name",
            "question_number",
            "run",
            "condition",
            "valid_execution",
            "technical_error",
            "metric",
            "value",
        ],
    )

    technical_errors = df.loc[
        df["technical_error"],
        [
            "dataset_name",
            "question_number",
            "run",
            "condition",
            "error_type",
            "error",
            "source_file",
        ],
    ].copy()

    # Solo ejecuciones válidas con una referencia realmente débil.
    weak_references = df.loc[
        df["weak_reference_auto"],
        [
            "dataset_name",
            "question_number",
            "run",
            "condition",
            "error_type",
            "evaluation_comment",
            "reference_solution",
            "source_file",
        ],
    ].copy()

    experiment_hashes = (
        df.groupby(["experiment_hash"], dropna=False)
        .agg(
            records=("row_id", "size"),
            datasets=("dataset_name", "nunique"),
            files=("source_file", "nunique"),
        )
        .reset_index()
    )

    summary = pd.DataFrame(
        [
            {"indicator": "source_files", "value": int(len(inventory))},
            {
                "indicator": "unique_file_contents",
                "value": int((~inventory["duplicate_file_content"]).sum()),
            },
            {"indicator": "records_after_file_dedup", "value": int(len(df))},
            {"indicator": "unique_rows", "value": int(df["row_id"].nunique())},
            {
                "indicator": "duplicate_row_occurrences",
                "value": int(len(duplicates)),
            },
            {"indicator": "datasets", "value": int(df["dataset_name"].nunique())},
            {
                "indicator": "unique_problems",
                "value": int(df["problem_id"].nunique()),
            },
            {"indicator": "conditions", "value": int(df["condition"].nunique())},
            {
                "indicator": "valid_execution_records",
                "value": int(df["valid_execution"].sum()),
            },
            {
                "indicator": "technical_error_records",
                "value": int(df["technical_error"].sum()),
            },
            {
                "indicator": "weak_reference_records_auto_valid_only",
                "value": int(df["weak_reference_auto"].sum()),
            },
            {
                "indicator": "primary_analysis_records",
                "value": int(df["valid_for_primary_analysis"].sum()),
            },
            {
                "indicator": "sensitivity_analysis_records",
                "value": int(df["valid_for_sensitivity_analysis"].sum()),
            },
            {
                "indicator": "invalid_metric_values",
                "value": int(len(invalid_metric_ranges)),
            },
        ]
    )

    dataset_name_mismatches = df.loc[
        df["dataset_name_mismatch"],
        [
            "source_file",
            "dataset_name_reported",
            "dataset_name_from_file",
            "dataset_name_canonical",
        ],
    ].drop_duplicates()

    return {
        "summary": summary,
        "file_inventory": inventory,
        "datasets": datasets,
        "conditions": conditions,
        "experiment_hashes": experiment_hashes,
        "missing_values": missing_expected,
        "duplicate_rows": duplicates,
        "technical_errors": technical_errors,
        "weak_references": weak_references,
        "invalid_metric_ranges": invalid_metric_ranges,
        "dataset_name_mismatches": dataset_name_mismatches,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita los JSON del experimento LLM matemático."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("salidas/01_auditoria")
    )

    # Spyder suele ejecutar el archivo sin argumentos de línea de comandos.
    if len(sys.argv) == 1:
        return argparse.Namespace(
            input_dir=SPYDER_INPUT_DIR,
            output_dir=SPYDER_OUTPUT_DIR,
        )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de entrada: {args.input_dir}"
        )

    output_dir = ensure_dir(args.output_dir)
    df, inventory = load_json_folder(args.input_dir)
    clean, duplicates = deduplicate_rows(df)
    clean = correct_audit_flags(clean)
    sheets = build_audit(clean, inventory, duplicates)

    write_excel_sheets(output_dir / "auditoria_datos.xlsx", sheets)
    clean.to_csv(
        output_dir / "registros_auditados.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sheets["summary"].to_csv(
        output_dir / "resumen_auditoria.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Auditoría completada")
    print(sheets["summary"].to_string(index=False))
    print(f"Salidas: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
