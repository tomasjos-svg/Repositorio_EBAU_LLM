from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    METRICS,
    TEXT_COLUMNS,
    deduplicate_rows,
    ensure_dir,
    load_json_folder,
    load_v8_companion_files,
    write_excel_sheets,
)

PIPELINE_VERSION = "3.1"
DEFAULT_EXPECTED_REPLICATES = 3

# Rutas predeterminadas relativas a la carpeta que contiene este script.
PIPELINE_ROOT = Path(__file__).resolve().parent
SPYDER_INPUT_DIR = PIPELINE_ROOT / "resultados"
SPYDER_OUTPUT_DIR = PIPELINE_ROOT / "salidas" / "02_dataset_maestro"

# Estas columnas se conservan completas en CSV, pero no se exportan al Excel
# para evitar el límite de 32.767 caracteres por celda de Excel.
EXCEL_EXCLUDED_LONG_COLUMNS = set(TEXT_COLUMNS)


def _as_bool(series: pd.Series) -> pd.Series:
    """Convierte una serie booleana o textual en booleanos de forma robusta."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "si", "sí"}
    )


def _slug_condition(value: object) -> str:
    """Crea un identificador estable y legible para una condición experimental."""
    text = str(value).strip().lower()
    replacements = {
        " ": "_",
        "+": "plus",
        "-": "_",
        "/": "_",
        "\\": "_",
        ".": "_",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def add_analysis_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Añade identificadores explícitos que simplifican los análisis posteriores."""
    out = df.copy()
    question = (
        out.get("question_id", out["question_number"])
        .astype(str)
        .str.strip()
        .str.upper()
    )
    out["problem_uid"] = (
        out["dataset_name"].astype(str)
        + "_Q"
        + question
    )
    out["condition_uid"] = out["condition"].map(_slug_condition)
    out["analysis_uid"] = out["problem_uid"] + "__" + out["condition_uid"]
    return out


def dominant_error(group: pd.DataFrame) -> pd.Series:
    """Obtiene el error modal dentro de las réplicas, conservando empates."""
    errors = group["error_type"].dropna().astype(str).str.strip()
    errors = errors[errors.ne("")]
    if errors.empty:
        return pd.Series(
            {
                "dominant_error_type": pd.NA,
                "dominant_error_count": 0,
                "dominant_error_tie": False,
            }
        )

    counts = errors.value_counts()
    maximum = int(counts.max())
    modes = sorted(counts[counts.eq(maximum)].index.tolist())
    return pd.Series(
        {
            "dominant_error_type": "|".join(modes),
            "dominant_error_count": maximum,
            "dominant_error_tie": len(modes) > 1,
        }
    )


def aggregate_replicates(
    df: pd.DataFrame,
    include_weak_references: bool = False,
    expected_replicates: int = DEFAULT_EXPECTED_REPLICATES,
) -> pd.DataFrame:
    """Agrega las réplicas por problema y condición experimental."""
    valid_execution = _as_bool(df["valid_execution"])
    weak_reference = _as_bool(df["weak_reference_auto"])

    mask = valid_execution.copy()
    if not include_weak_references:
        mask &= ~weak_reference

    valid = add_analysis_identifiers(df.loc[mask].copy())

    group_cols = [
        "dataset_name",
        "domain",
        "education_level",
        "problem_id",
        "problem_uid",
        "question_number",
        "condition",
        "condition_uid",
        "analysis_uid",
        "model",
        "api",
        "judge_model",
    ]

    agg_map: dict[str, tuple[str, str]] = {
        "replicate_count": ("run", "nunique"),
        "replicate_rows": ("row_id", "size"),
        "execution_time_mean": ("execution_time_seconds", "mean"),
        "execution_time_sd": ("execution_time_seconds", "std"),
        "execution_time_min": ("execution_time_seconds", "min"),
        "execution_time_max": ("execution_time_seconds", "max"),
        "execution_time_sum": ("execution_time_seconds", "sum"),
        "tokens_mean": ("total_tokens", "mean"),
        "tokens_sd": ("total_tokens", "std"),
        "tokens_min": ("total_tokens", "min"),
        "tokens_max": ("total_tokens", "max"),
        "tokens_sum": ("total_tokens", "sum"),
        "cost_mean_usd": ("estimated_cost_usd", "mean"),
        "cost_sd_usd": ("estimated_cost_usd", "std"),
        "cost_min_usd": ("estimated_cost_usd", "min"),
        "cost_max_usd": ("estimated_cost_usd", "max"),
        "cost_sum_usd": ("estimated_cost_usd", "sum"),
    }

    for metric in METRICS:
        agg_map[f"{metric}_mean"] = (metric, "mean")
        agg_map[f"{metric}_sd"] = (metric, "std")
        agg_map[f"{metric}_min"] = (metric, "min")
        agg_map[f"{metric}_max"] = (metric, "max")

    result = (
        valid.groupby(group_cols, dropna=False)
        .agg(**agg_map)
        .reset_index()
    )

    dominant = (
        valid.groupby(group_cols, dropna=False, group_keys=False)
        .apply(dominant_error, include_groups=False)
        .reset_index()
    )
    result = result.merge(
        dominant,
        on=group_cols,
        how="left",
        validate="one_to_one",
    )

    for metric in METRICS:
        result[f"{metric}_range"] = (
            result[f"{metric}_max"] - result[f"{metric}_min"]
        )

    result["execution_time_range"] = (
        result["execution_time_max"] - result["execution_time_min"]
    )
    result["tokens_range"] = result["tokens_max"] - result["tokens_min"]
    result["cost_range_usd"] = (
        result["cost_max_usd"] - result["cost_min_usd"]
    )

    result["expected_replicates"] = int(expected_replicates)
    result["missing_replicates"] = (
        result["expected_replicates"] - result["replicate_count"]
    ).clip(lower=0)
    result["extra_replicates"] = (
        result["replicate_count"] - result["expected_replicates"]
    ).clip(lower=0)
    result["complete_replicates"] = result["replicate_count"].eq(
        result["expected_replicates"]
    )
    result["usable_for_statistics"] = result["complete_replicates"]

    return result.sort_values(
        ["dataset_name", "question_number", "condition"],
        kind="stable",
    ).reset_index(drop=True)


def _runs_as_text(values: pd.Series) -> str:
    runs = sorted(pd.to_numeric(values, errors="coerce").dropna().astype(int).unique())
    return ",".join(str(value) for value in runs)


def build_completeness_diagnostics(
    clean: pd.DataFrame,
    by_problem_primary: pd.DataFrame,
    by_problem_sensitivity: pd.DataFrame,
    expected_replicates: int,
) -> pd.DataFrame:
    """
    Diagnostica por qué una combinación problema-condición no conserva las
    réplicas esperadas en el análisis principal.

    La cuadrícula esperada se toma del análisis de sensibilidad, que incluye
    todas las ejecuciones técnicamente válidas, aunque la referencia sea débil.
    """
    key_cols = [
        "dataset_name",
        "domain",
        "education_level",
        "problem_id",
        "problem_uid",
        "question_number",
        "condition",
        "condition_uid",
        "analysis_uid",
        "model",
        "api",
        "judge_model",
    ]

    base = by_problem_sensitivity[
        key_cols + ["replicate_count"]
    ].rename(columns={"replicate_count": "sensitivity_replicate_count"})

    primary_counts = by_problem_primary[
        key_cols + ["replicate_count"]
    ].rename(columns={"replicate_count": "primary_replicate_count"})

    diagnostics = base.merge(
        primary_counts,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )
    diagnostics["primary_replicate_count"] = (
        diagnostics["primary_replicate_count"].fillna(0).astype(int)
    )

    detail = clean.groupby(
        ["problem_id", "condition"],
        dropna=False,
    ).agg(
        total_registered_rows=("row_id", "size"),
        technical_error_count=("technical_error", "sum"),
        weak_reference_count=("weak_reference_auto", "sum"),
        valid_execution_count=("valid_execution", "sum"),
        runs_registered=("run", _runs_as_text),
    ).reset_index()

    primary_runs = (
        clean.loc[clean["valid_for_primary_analysis"]]
        .groupby(["problem_id", "condition"], dropna=False)["run"]
        .agg(_runs_as_text)
        .rename("runs_primary")
        .reset_index()
    )
    sensitivity_runs = (
        clean.loc[clean["valid_for_sensitivity_analysis"]]
        .groupby(["problem_id", "condition"], dropna=False)["run"]
        .agg(_runs_as_text)
        .rename("runs_sensitivity")
        .reset_index()
    )

    diagnostics = diagnostics.merge(
        detail,
        on=["problem_id", "condition"],
        how="left",
        validate="one_to_one",
    )
    diagnostics = diagnostics.merge(
        primary_runs,
        on=["problem_id", "condition"],
        how="left",
        validate="one_to_one",
    )
    diagnostics = diagnostics.merge(
        sensitivity_runs,
        on=["problem_id", "condition"],
        how="left",
        validate="one_to_one",
    )

    diagnostics["expected_replicates"] = int(expected_replicates)
    diagnostics["missing_primary_replicates"] = (
        diagnostics["expected_replicates"]
        - diagnostics["primary_replicate_count"]
    ).clip(lower=0)
    diagnostics["complete_in_primary"] = diagnostics[
        "primary_replicate_count"
    ].eq(diagnostics["expected_replicates"])

    def diagnose(row: pd.Series) -> str:
        missing = int(row["missing_primary_replicates"])
        if missing == 0:
            return "Completo"

        total = int(row.get("total_registered_rows", 0) or 0)
        technical = int(row.get("technical_error_count", 0) or 0)
        weak = int(row.get("weak_reference_count", 0) or 0)

        if total < expected_replicates:
            return "Registro realmente ausente"
        if technical > 0 and weak > 0:
            return "Réplicas excluidas por error técnico y referencia débil"
        if technical > 0:
            return "Réplica excluida por error técnico"
        if weak > 0:
            return "Réplica excluida por referencia débil"
        return "Menos de tres réplicas válidas por causa no identificada"

    diagnostics["causa_incompletitud"] = diagnostics.apply(diagnose, axis=1)
    diagnostics["replicas_excluidas_por_weak_reference"] = diagnostics[
        "weak_reference_count"
    ]
    diagnostics["replicas_excluidas_por_error_tecnico"] = diagnostics[
        "technical_error_count"
    ]

    return diagnostics.sort_values(
        ["complete_in_primary", "dataset_name", "question_number", "condition"],
        kind="stable",
    ).reset_index(drop=True)


def paired_problem_completeness(
    by_problem: pd.DataFrame,
    expected_conditions: int,
) -> pd.DataFrame:
    """Resume si cada problema dispone de todas las condiciones y réplicas."""
    summary = (
        by_problem.groupby(
            [
                "dataset_name",
                "domain",
                "education_level",
                "problem_id",
                "problem_uid",
            ],
            dropna=False,
        )
        .agg(
            available_conditions=("condition", "nunique"),
            all_conditions_complete=("complete_replicates", "all"),
            total_missing_replicates=("missing_replicates", "sum"),
        )
        .reset_index()
    )
    summary["expected_conditions"] = int(expected_conditions)
    summary["complete_for_paired_analysis"] = (
        summary["available_conditions"].eq(expected_conditions)
        & summary["all_conditions_complete"]
    )
    summary["paired_with_available_replicates"] = summary[
        "available_conditions"
    ].eq(expected_conditions)
    return summary


def filter_problem_rows(
    by_problem: pd.DataFrame,
    paired_summary: pd.DataFrame,
    flag_column: str,
) -> pd.DataFrame:
    """Conserva las filas problema-condición cuyos problemas cumplen una bandera."""
    valid_problem_ids = paired_summary.loc[
        paired_summary[flag_column], "problem_id"
    ]
    return by_problem.loc[
        by_problem["problem_id"].isin(valid_problem_ids)
    ].copy()


def excel_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Crea una vista apta para Excel sin columnas de texto excepcionalmente largas."""
    columns_to_drop = [
        col for col in EXCEL_EXCLUDED_LONG_COLUMNS if col in frame.columns
    ]
    return frame.drop(columns=columns_to_drop, errors="ignore").copy()


def build_metadata(
    clean: pd.DataFrame,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    by_problem_primary: pd.DataFrame,
    by_problem_sensitivity: pd.DataFrame,
    strict_problem_rows: pd.DataFrame,
    paired_primary_rows: pd.DataFrame,
    paired_sensitivity_rows: pd.DataFrame,
    diagnostics: pd.DataFrame,
    inventory: pd.DataFrame,
    duplicates: pd.DataFrame,
    expected_replicates: int,
    valid_conditions: int,
) -> dict[str, Any]:
    """Genera metadatos legibles por personas y por otros scripts."""
    technical = _as_bool(clean["technical_error"])
    weak = _as_bool(clean["weak_reference_auto"]) & ~technical
    valid = _as_bool(clean["valid_execution"])

    return {
        "pipeline_version": PIPELINE_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": int(len(inventory)),
        "unique_file_contents": int((~inventory["duplicate_file_content"]).sum()),
        "records_total_unique": int(len(clean)),
        "duplicate_row_occurrences": int(len(duplicates)),
        "datasets": int(clean["dataset_name"].nunique()),
        "unique_problems": int(clean["problem_id"].nunique()),
        "conditions_total_registered": int(clean["condition"].nunique()),
        "conditions_valid_for_analysis": int(valid_conditions),
        "valid_executions": int(valid.sum()),
        "technical_error_records": int(technical.sum()),
        "weak_reference_records_valid_only": int(weak.sum()),
        "primary_records": int(len(primary)),
        "sensitivity_records": int(len(sensitivity)),
        "problem_condition_rows_primary": int(len(by_problem_primary)),
        "problem_condition_rows_sensitivity": int(len(by_problem_sensitivity)),
        "complete_problem_condition_rows_primary": int(
            by_problem_primary["complete_replicates"].sum()
        ),
        "incomplete_problem_condition_rows_expected_grid": int(
            (~diagnostics["complete_in_primary"]).sum()
        ),
        "strict_paired_problems": int(strict_problem_rows["problem_id"].nunique()),
        "strict_problem_condition_rows": int(len(strict_problem_rows)),
        "paired_primary_problems": int(
            paired_primary_rows["problem_id"].nunique()
        ),
        "paired_primary_problem_condition_rows": int(len(paired_primary_rows)),
        "paired_sensitivity_problems": int(
            paired_sensitivity_rows["problem_id"].nunique()
        ),
        "paired_sensitivity_problem_condition_rows": int(
            len(paired_sensitivity_rows)
        ),
        "expected_replicates_per_problem_condition": int(expected_replicates),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye el dataset maestro a partir de los JSON."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--conditions",
        nargs=2,
        metavar=("CONDICION_1", "CONDICION_2"),
        help="Dos nombres canónicos model + API para el análisis. Si se omite, se usa la pareja histórica del artículo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("salidas/02_dataset_maestro"),
    )
    parser.add_argument(
        "--expected-replicates",
        type=int,
        default=DEFAULT_EXPECTED_REPLICATES,
        help="Número esperado de réplicas por problema y condición (por defecto: 3).",
    )

    if len(sys.argv) == 1:
        return argparse.Namespace(
            input_dir=SPYDER_INPUT_DIR,
            output_dir=SPYDER_OUTPUT_DIR,
            expected_replicates=DEFAULT_EXPECTED_REPLICATES,
            conditions=None,
        )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.expected_replicates < 1:
        raise ValueError("--expected-replicates debe ser un entero mayor o igual que 1.")
    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de entrada: {args.input_dir.resolve()}"
        )

    output_dir = ensure_dir(args.output_dir)
    df, inventory = load_json_folder(args.input_dir)
    clean, duplicates = deduplicate_rows(df)
    clean = add_analysis_identifiers(clean)
    config_inventory, common_references = load_v8_companion_files(args.input_dir)

    registered_conditions = sorted(clean["condition"].dropna().astype(str).unique())
    preferred = ["gpt-4-turbo + Assistants API", "gpt-5.4 + Responses API"]
    if args.conditions:
        selected_conditions = list(args.conditions)
    elif all(item in registered_conditions for item in preferred):
        selected_conditions = preferred
    elif len(registered_conditions) == 2:
        selected_conditions = registered_conditions
    else:
        raise ValueError(
            "La v8 contiene más de dos condiciones. Indica --conditions con dos nombres exactos. "
            f"Disponibles: {registered_conditions}"
        )
    missing_selected = [item for item in selected_conditions if item not in registered_conditions]
    if missing_selected:
        raise ValueError(f"Condiciones solicitadas no encontradas: {missing_selected}")
    clean["selected_for_analysis"] = clean["condition"].isin(selected_conditions)

    # Se recalculan las banderas para impedir que un error técnico sea tratado
    # como referencia débil en los resúmenes y datasets derivados.
    valid_execution = _as_bool(clean["valid_execution"])
    weak_reference = _as_bool(clean["weak_reference_auto"]) & valid_execution
    clean["weak_reference_auto"] = weak_reference
    clean["valid_for_primary_analysis"] = valid_execution & ~weak_reference
    clean["valid_for_sensitivity_analysis"] = valid_execution

    primary = clean.loc[clean["valid_for_primary_analysis"] & clean["selected_for_analysis"]].copy()
    sensitivity = clean.loc[clean["valid_for_sensitivity_analysis"] & clean["selected_for_analysis"]].copy()

    analytical_clean = clean.loc[clean["selected_for_analysis"]].copy()

    by_problem_primary = aggregate_replicates(
        analytical_clean,
        include_weak_references=False,
        expected_replicates=args.expected_replicates,
    )
    by_problem_sensitivity = aggregate_replicates(
        analytical_clean,
        include_weak_references=True,
        expected_replicates=args.expected_replicates,
    )

    valid_conditions = int(sensitivity["condition"].nunique())

    paired_primary_summary = paired_problem_completeness(
        by_problem_primary,
        expected_conditions=valid_conditions,
    )
    paired_sensitivity_summary = paired_problem_completeness(
        by_problem_sensitivity,
        expected_conditions=valid_conditions,
    )

    # 1) Análisis estricto: las dos condiciones y las tres réplicas no débiles.
    strict_problem_rows = filter_problem_rows(
        by_problem_primary,
        paired_primary_summary,
        "complete_for_paired_analysis",
    )

    # 2) Principal pareado: las dos condiciones, usando las réplicas no débiles disponibles.
    paired_primary_rows = filter_problem_rows(
        by_problem_primary,
        paired_primary_summary,
        "paired_with_available_replicates",
    )

    # 3) Sensibilidad: todas las referencias, con ambas condiciones completas.
    paired_sensitivity_rows = filter_problem_rows(
        by_problem_sensitivity,
        paired_sensitivity_summary,
        "complete_for_paired_analysis",
    )

    diagnostics = build_completeness_diagnostics(
        clean=analytical_clean,
        by_problem_primary=by_problem_primary,
        by_problem_sensitivity=by_problem_sensitivity,
        expected_replicates=args.expected_replicates,
    )
    problemas_incompletos = diagnostics.loc[
        ~diagnostics["complete_in_primary"]
    ].copy()

    # Datasets a nivel de réplica asociados a cada política analítica.
    strict_problem_ids = set(strict_problem_rows["problem_id"])
    paired_primary_problem_ids = set(paired_primary_rows["problem_id"])
    paired_sensitivity_problem_ids = set(paired_sensitivity_rows["problem_id"])

    strict_replicates = primary.loc[
        primary["problem_id"].isin(strict_problem_ids)
    ].copy()
    paired_primary_replicates = primary.loc[
        primary["problem_id"].isin(paired_primary_problem_ids)
    ].copy()
    paired_sensitivity_replicates = sensitivity.loc[
        sensitivity["problem_id"].isin(paired_sensitivity_problem_ids)
    ].copy()

    metadata = build_metadata(
        clean=clean,
        primary=primary,
        sensitivity=sensitivity,
        by_problem_primary=by_problem_primary,
        by_problem_sensitivity=by_problem_sensitivity,
        strict_problem_rows=strict_problem_rows,
        paired_primary_rows=paired_primary_rows,
        paired_sensitivity_rows=paired_sensitivity_rows,
        diagnostics=diagnostics,
        inventory=inventory,
        duplicates=duplicates,
        expected_replicates=args.expected_replicates,
        valid_conditions=valid_conditions,
    )

    metadata["pipeline_version"] = PIPELINE_VERSION
    metadata["registered_conditions"] = registered_conditions
    metadata["selected_conditions"] = selected_conditions
    metadata["v8_config_files"] = int(len(config_inventory))
    metadata["v8_common_reference_rows"] = int(len(common_references))

    # CSV de nivel réplica. Conservan todos los textos completos.
    csv_outputs = {
        "dataset_maestro_todos.csv": clean,
        "dataset_maestro_analisis_principal.csv": primary,
        "dataset_maestro_sensibilidad.csv": sensitivity,
        "dataset_replica_estricto.csv": strict_replicates,
        "dataset_replica_pareado_principal.csv": paired_primary_replicates,
        "dataset_replica_pareado_sensibilidad.csv": paired_sensitivity_replicates,
    }
    for filename, frame in csv_outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    # CSV agregados por problema y condición.
    problem_outputs = {
        "dataset_por_problema_principal.csv": by_problem_primary,
        "dataset_por_problema_sensibilidad.csv": by_problem_sensitivity,
        "dataset_por_problema_estricto.csv": strict_problem_rows,
        "dataset_por_problema_pareado_principal.csv": paired_primary_rows,
        "dataset_por_problema_pareado_sensibilidad.csv": paired_sensitivity_rows,
        "completitud_pares_principal.csv": paired_primary_summary,
        "completitud_pares_sensibilidad.csv": paired_sensitivity_summary,
        "diagnostico_completitud.csv": diagnostics,
        "problemas_incompletos.csv": problemas_incompletos,
    }
    for filename, frame in problem_outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    config_inventory.to_csv(output_dir / "inventario_config_v8.csv", index=False, encoding="utf-8-sig")
    common_references.to_csv(output_dir / "referencias_comunes_v8.csv", index=False, encoding="utf-8-sig")

    metadata_path = output_dir / "metadata_dataset_maestro.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata_frame = pd.DataFrame(
        [{"indicator": key, "value": value} for key, value in metadata.items()]
    )

    # Excel de inspección: sin prompts, soluciones ni evaluaciones extensas.
    write_excel_sheets(
        output_dir / "dataset_maestro.xlsx",
        {
            "metadata": metadata_frame,
            "todos_sin_textos": excel_view(clean),
            "principal_sin_textos": excel_view(primary),
            "sensibilidad_sin_textos": excel_view(sensitivity),
            "por_problema_principal": by_problem_primary,
            "por_problema_sens": by_problem_sensitivity,
            "estricto": strict_problem_rows,
            "pareado_principal": paired_primary_rows,
            "pareado_sensibilidad": paired_sensitivity_rows,
            "pares_principal": paired_primary_summary,
            "pares_sensibilidad": paired_sensitivity_summary,
            "diagnostico_completitud": diagnostics,
            "problemas_incompletos": problemas_incompletos,
            "inventario_archivos": inventory,
            "duplicados": duplicates,
            "config_v8": config_inventory,
            "referencias_v8": common_references,
        },
    )

    # Registro de ejecución legible.
    log_lines = [
        f"Pipeline version: {PIPELINE_VERSION}",
        f"Generated UTC: {metadata['generated_utc']}",
        f"Input directory: {args.input_dir.resolve()}",
        f"Output directory: {output_dir.resolve()}",
        "",
    ]
    log_lines.extend(f"{key}: {value}" for key, value in metadata.items())
    (output_dir / "pipeline_dataset_maestro.log").write_text(
        "\n".join(log_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print(f"DATASET MAESTRO - VERSION {PIPELINE_VERSION}")
    print("=" * 65)
    labels = {
        "source_files": "Archivos JSON",
        "datasets": "Datasets",
        "unique_problems": "Problemas únicos",
        "conditions_total_registered": "Condiciones registradas",
        "conditions_valid_for_analysis": "Condiciones válidas para análisis",
        "records_total_unique": "Registros únicos totales",
        "valid_executions": "Ejecuciones válidas",
        "technical_error_records": "Errores técnicos",
        "weak_reference_records_valid_only": "Referencias débiles válidas",
        "primary_records": "Registros análisis principal",
        "sensitivity_records": "Registros sensibilidad",
        "strict_paired_problems": "Problemas análisis estricto",
        "paired_primary_problems": "Problemas principal pareado",
        "paired_sensitivity_problems": "Problemas sensibilidad pareada",
        "incomplete_problem_condition_rows_expected_grid": (
            "Problema-condición incompletos"
        ),
    }
    for key, label in labels.items():
        print(f"{label:.<48}{metadata[key]:>15}")
    print("=" * 65)
    print(f"Salidas: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
