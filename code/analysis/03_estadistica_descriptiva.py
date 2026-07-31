from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import (
    METRICS,
    bootstrap_ci_mean,
    confidence_interval_mean,
    ensure_dir,
    write_excel_sheets,
)

PIPELINE_VERSION = "3.1"

# Rutas predeterminadas relativas a la carpeta que contiene este script.
PIPELINE_ROOT = Path(__file__).resolve().parent
SPYDER_INPUT_DIR = PIPELINE_ROOT / "salidas" / "02_dataset_maestro"
SPYDER_OUTPUT_DIR = PIPELINE_ROOT / "salidas" / "03_descriptiva"

SCENARIOS: dict[str, dict[str, str]] = {
    "principal": {
        "problem_file": "dataset_por_problema_pareado_principal.csv",
        "replicate_file": "dataset_replica_pareado_principal.csv",
        "description": (
            "Análisis principal: problemas emparejados con las réplicas no débiles "
            "disponibles en ambas condiciones."
        ),
    },
    "estricto": {
        "problem_file": "dataset_por_problema_estricto.csv",
        "replicate_file": "dataset_replica_estricto.csv",
        "description": (
            "Análisis estricto: problemas con las dos condiciones y tres réplicas "
            "no débiles completas por condición."
        ),
    },
    "sensibilidad": {
        "problem_file": "dataset_por_problema_pareado_sensibilidad.csv",
        "replicate_file": "dataset_replica_pareado_sensibilidad.csv",
        "description": (
            "Análisis de sensibilidad: problemas emparejados incluyendo referencias "
            "potencialmente débiles, pero excluyendo errores técnicos."
        ),
    },
}

RESOURCE_COLUMNS = {
    "execution_time_seconds": "execution_time_seconds",
    "total_tokens": "total_tokens",
    "estimated_cost_usd": "estimated_cost_usd",
}


def _parse_bool(series: pd.Series) -> pd.Series:
    """Convierte valores booleanos o textuales en booleanos de forma robusta."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "si", "sí"}
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera estadística descriptiva para los escenarios principal, "
            "estricto y de sensibilidad."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("salidas/02_dataset_maestro"),
        help="Carpeta generada por 02_dataset_maestro.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("salidas/03_descriptiva"),
    )

    # Spyder suele ejecutar el archivo sin argumentos de línea de comandos.
    if len(sys.argv) == 1:
        return argparse.Namespace(
            input_dir=SPYDER_INPUT_DIR,
            output_dir=SPYDER_OUTPUT_DIR,
        )
    return parser.parse_args()


def validate_input_files(input_dir: Path) -> None:
    """Comprueba que existen todos los CSV requeridos por los tres escenarios."""
    missing: list[str] = []
    for config in SCENARIOS.values():
        for key in ("problem_file", "replicate_file"):
            filename = config[key]
            if not (input_dir / filename).exists():
                missing.append(filename)
    if missing:
        formatted = "\n - ".join(sorted(set(missing)))
        raise FileNotFoundError(
            "Faltan archivos generados por 02_dataset_maestro.py:\n - " + formatted
        )


def load_scenario(input_dir: Path, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los datos a nivel problema y a nivel réplica de un escenario."""
    config = SCENARIOS[scenario]
    problems = pd.read_csv(input_dir / config["problem_file"])
    replicates = pd.read_csv(input_dir / config["replicate_file"])

    required_problem_cols = {
        "problem_id",
        "condition",
        "dataset_name",
        "domain",
        "education_level",
        *{f"{metric}_mean" for metric in METRICS},
    }
    missing_problem = required_problem_cols.difference(problems.columns)
    if missing_problem:
        raise ValueError(
            f"{config['problem_file']} no contiene las columnas requeridas: "
            f"{sorted(missing_problem)}"
        )

    required_replicate_cols = {
        "problem_id",
        "condition",
        "dataset_name",
        "domain",
        "education_level",
        "error_type",
        *METRICS,
        *RESOURCE_COLUMNS.keys(),
    }
    missing_replicate = required_replicate_cols.difference(replicates.columns)
    if missing_replicate:
        raise ValueError(
            f"{config['replicate_file']} no contiene las columnas requeridas: "
            f"{sorted(missing_replicate)}"
        )

    problems = problems.copy()
    replicates = replicates.copy()
    problems["scenario"] = scenario
    replicates["scenario"] = scenario
    return problems, replicates


def _problem_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra las medias por problema para trabajar con los nombres base."""
    out = df.copy()
    rename_map = {f"{metric}_mean": metric for metric in METRICS}
    out = out.rename(columns=rename_map)
    for metric in METRICS:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out


def describe_grouped(
    df: pd.DataFrame,
    group_cols: list[str],
    unit_label: str,
) -> pd.DataFrame:
    """Calcula descriptivos e intervalos de confianza por grupos."""
    rows: list[dict[str, Any]] = []

    grouped = df.groupby(group_cols, dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))

        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue

            ci_low, ci_high = confidence_interval_mean(values)
            boot_low, boot_high = bootstrap_ci_mean(values)
            rows.append(
                {
                    **base,
                    "unit_of_analysis": unit_label,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "sd": (
                        float(values.std(ddof=1)) if values.size > 1 else np.nan
                    ),
                    "sem": (
                        float(values.sem(ddof=1)) if values.size > 1 else np.nan
                    ),
                    "median": float(values.median()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "iqr": float(
                        values.quantile(0.75) - values.quantile(0.25)
                    ),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "ci95_low_t": ci_low,
                    "ci95_high_t": ci_high,
                    "ci95_low_boot": boot_low,
                    "ci95_high_boot": boot_high,
                    "pct_zero": float((values == 0).mean() * 100),
                    "pct_hundred": float((values == 100).mean() * 100),
                }
            )

    return pd.DataFrame(rows)


def _variability_row(
    condition: object,
    metric: str,
    group: pd.DataFrame,
    basis: str,
    expected_replicates: int,
) -> dict[str, Any]:
    """Construye una fila de variabilidad sin tratar n=1 como estabilidad cero."""
    sd_col = f"{metric}_sd"
    range_col = f"{metric}_range"
    sd_values = pd.to_numeric(group[sd_col], errors="coerce").dropna()
    range_values = pd.to_numeric(group[range_col], errors="coerce").dropna()
    return {
        "condition": condition,
        "metric": metric,
        "analysis_basis": basis,
        "expected_replicates": expected_replicates,
        "n_problem_conditions": int(len(group)),
        "n_with_estimable_sd": int(sd_values.size),
        "n_with_estimable_range": int(range_values.size),
        "mean_within_problem_sd": float(sd_values.mean()) if not sd_values.empty else np.nan,
        "median_within_problem_sd": float(sd_values.median()) if not sd_values.empty else np.nan,
        "mean_range": float(range_values.mean()) if not range_values.empty else np.nan,
        "median_range": float(range_values.median()) if not range_values.empty else np.nan,
        "pct_range_gt_15": float((range_values > 15).mean() * 100) if not range_values.empty else np.nan,
        "pct_range_gt_20": float((range_values > 20).mean() * 100) if not range_values.empty else np.nan,
        "pct_range_gt_25": float((range_values > 25).mean() * 100) if not range_values.empty else np.nan,
        "pct_range_gt_30": float((range_values > 30).mean() * 100) if not range_values.empty else np.nan,
    }


def describe_replicate_variability(
    problems: pd.DataFrame,
    expected_replicates: int = 3,
) -> pd.DataFrame:
    """
    Resume la variabilidad entre réplicas.

    Se generan dos estimaciones por condición y métrica:
    1) grupos con al menos dos réplicas, donde la variabilidad es estimable;
    2) grupos completos con exactamente el número esperado de réplicas.

    Los grupos con una sola réplica no se interpretan como estables.
    """
    rows: list[dict[str, Any]] = []
    work = problems.copy()
    work["replicate_count"] = pd.to_numeric(
        work["replicate_count"], errors="coerce"
    )

    for condition, group in work.groupby("condition", dropna=False):
        estimable = group.loc[group["replicate_count"].ge(2)].copy()
        complete = group.loc[
            group["replicate_count"].eq(expected_replicates)
        ].copy()
        for metric in METRICS:
            rows.append(
                _variability_row(
                    condition, metric, estimable,
                    "available_with_at_least_2_replicates",
                    expected_replicates,
                )
            )
            rows.append(
                _variability_row(
                    condition, metric, complete,
                    f"complete_{expected_replicates}_replicates",
                    expected_replicates,
                )
            )
    return pd.DataFrame(rows)


def replicate_count_distribution(problems: pd.DataFrame) -> pd.DataFrame:
    """Distribución explícita de grupos con 1, 2, 3 o más réplicas."""
    work = problems.copy()
    work["replicate_count"] = pd.to_numeric(
        work["replicate_count"], errors="coerce"
    ).astype("Int64")
    counts = (
        work.groupby(["condition", "replicate_count"], dropna=False)
        .size()
        .rename("n_problem_conditions")
        .reset_index()
    )
    totals = counts.groupby("condition", dropna=False)[
        "n_problem_conditions"
    ].transform("sum")
    counts["pct_problem_conditions"] = (
        counts["n_problem_conditions"] / totals * 100
    )
    counts["variability_estimable"] = counts["replicate_count"].ge(2)
    return counts

def error_taxonomy(
    replicates: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Calcula frecuencias y porcentajes de los tipos de error."""
    work = replicates.copy()
    work["error_type"] = (
        work["error_type"].fillna("missing_error_type").astype(str).str.strip()
    )
    counts = (
        work.groupby(group_cols + ["error_type"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = counts.groupby(group_cols, dropna=False)["n"].transform("sum")
    counts["pct"] = counts["n"] / totals * 100
    return counts


def dominant_error_taxonomy(problems: pd.DataFrame) -> pd.DataFrame:
    """Resume el error dominante de cada problema-condición."""
    work = problems.copy()
    work["dominant_error_type"] = (
        work["dominant_error_type"]
        .fillna("missing_error_type")
        .astype(str)
        .str.strip()
    )
    counts = (
        work.groupby(["condition", "dominant_error_type"], dropna=False)
        .size()
        .rename("n_problem_conditions")
        .reset_index()
    )
    totals = counts.groupby("condition", dropna=False)[
        "n_problem_conditions"
    ].transform("sum")
    counts["pct_problem_conditions"] = (
        counts["n_problem_conditions"] / totals * 100
    )
    return counts


def resource_summary(replicates: pd.DataFrame) -> pd.DataFrame:
    """Resume tiempo, tokens y coste a nivel de ejecución."""
    rows: list[dict[str, Any]] = []
    for condition, group in replicates.groupby("condition", dropna=False):
        for source_col, label in RESOURCE_COLUMNS.items():
            values = pd.to_numeric(group[source_col], errors="coerce").dropna()
            if values.empty:
                continue
            ci_low, ci_high = confidence_interval_mean(values)
            boot_low, boot_high = bootstrap_ci_mean(values)
            rows.append(
                {
                    "condition": condition,
                    "resource": label,
                    "n_executions": int(values.size),
                    "mean": float(values.mean()),
                    "sd": (
                        float(values.std(ddof=1)) if values.size > 1 else np.nan
                    ),
                    "median": float(values.median()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "sum": float(values.sum()),
                    "ci95_low_t": ci_low,
                    "ci95_high_t": ci_high,
                    "ci95_low_boot": boot_low,
                    "ci95_high_boot": boot_high,
                }
            )
    return pd.DataFrame(rows)


def resource_summary_paired(problems: pd.DataFrame) -> pd.DataFrame:
    """
    Resume recursos a nivel problema-condición.

    Cada condición aporta exactamente los mismos problemas del escenario, por lo
    que esta tabla evita que números desiguales de réplicas distorsionen la
    comparación de eficiencia. Se emplea la media de recursos de las réplicas
    disponibles dentro de cada problema-condición.
    """
    mapping = {
        "execution_time_mean": "execution_time_seconds",
        "tokens_mean": "total_tokens",
        "cost_mean_usd": "estimated_cost_usd",
    }
    rows: list[dict[str, Any]] = []
    for condition, group in problems.groupby("condition", dropna=False):
        replicate_counts = pd.to_numeric(
            group["replicate_count"], errors="coerce"
        ).dropna()
        for source_col, label in mapping.items():
            values = pd.to_numeric(group[source_col], errors="coerce").dropna()
            if values.empty:
                continue
            ci_low, ci_high = confidence_interval_mean(values)
            boot_low, boot_high = bootstrap_ci_mean(values)
            rows.append({
                "condition": condition,
                "resource": label,
                "unit_of_analysis": "problem_condition_mean",
                "n_problem_conditions": int(values.size),
                "mean_replicate_count": float(replicate_counts.mean()) if not replicate_counts.empty else np.nan,
                "min_replicate_count": float(replicate_counts.min()) if not replicate_counts.empty else np.nan,
                "max_replicate_count": float(replicate_counts.max()) if not replicate_counts.empty else np.nan,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "median": float(values.median()),
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
                "min": float(values.min()),
                "max": float(values.max()),
                "ci95_low_t": ci_low,
                "ci95_high_t": ci_high,
                "ci95_low_boot": boot_low,
                "ci95_high_boot": boot_high,
            })
    return pd.DataFrame(rows)


def scenario_summary(
    scenario: str,
    problems: pd.DataFrame,
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    """Genera indicadores básicos del tamaño y completitud de cada escenario."""
    conditions = sorted(problems["condition"].dropna().astype(str).unique())
    return pd.DataFrame(
        [
            {"indicator": "scenario", "value": scenario},
            {"indicator": "description", "value": SCENARIOS[scenario]["description"]},
            {"indicator": "problem_condition_rows", "value": int(len(problems))},
            {"indicator": "unique_problems", "value": int(problems["problem_id"].nunique())},
            {"indicator": "conditions", "value": int(problems["condition"].nunique())},
            {"indicator": "condition_names", "value": " | ".join(conditions)},
            {"indicator": "replicate_rows", "value": int(len(replicates))},
            {"indicator": "datasets", "value": int(problems["dataset_name"].nunique())},
            {"indicator": "domains", "value": int(problems["domain"].nunique())},
            {"indicator": "education_levels", "value": int(problems["education_level"].nunique())},
        ]
    )


def build_scenario_outputs(
    scenario: str,
    problems_raw: pd.DataFrame,
    replicates: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Construye todas las tablas descriptivas de un escenario."""
    problems = _problem_metric_frame(problems_raw)

    outputs = {
        "resumen": scenario_summary(scenario, problems, replicates),
        "general": describe_grouped(problems, ["condition"], "problem"),
        "por_dataset": describe_grouped(
            problems, ["dataset_name", "condition"], "problem"
        ),
        "por_dominio": describe_grouped(
            problems, ["domain", "condition"], "problem"
        ),
        "por_nivel": describe_grouped(
            problems, ["education_level", "condition"], "problem"
        ),
        "variabilidad_replicas": describe_replicate_variability(problems_raw),
        "distribucion_replicas": replicate_count_distribution(problems_raw),
        "taxonomia_replica": error_taxonomy(replicates, ["condition"]),
        "taxonomia_dataset": error_taxonomy(
            replicates, ["dataset_name", "condition"]
        ),
        "taxonomia_dominante": dominant_error_taxonomy(problems_raw),
        "recursos_replica": resource_summary(replicates),
        "recursos_pareados": resource_summary_paired(problems_raw),
    }
    return outputs


def write_scenario_outputs(
    output_dir: Path,
    scenario: str,
    outputs: dict[str, pd.DataFrame],
) -> None:
    """Exporta un escenario a Excel y CSV."""
    scenario_dir = ensure_dir(output_dir / scenario)
    write_excel_sheets(
        scenario_dir / f"estadistica_descriptiva_{scenario}.xlsx",
        outputs,
    )
    for name, frame in outputs.items():
        frame.to_csv(
            scenario_dir / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )


def build_cross_scenario_comparison(
    scenario_outputs: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Fusiona tablas equivalentes para comparar escenarios."""
    tables: dict[str, pd.DataFrame] = {}
    for table_name in [
        "general",
        "variabilidad_replicas",
        "taxonomia_replica",
        "taxonomia_dominante",
        "distribucion_replicas",
        "recursos_replica",
        "recursos_pareados",
    ]:
        frames: list[pd.DataFrame] = []
        for scenario, outputs in scenario_outputs.items():
            frame = outputs[table_name].copy()
            frame.insert(0, "scenario", scenario)
            frames.append(frame)
        tables[table_name] = pd.concat(frames, ignore_index=True)
    return tables


def build_metadata(
    input_dir: Path,
    scenario_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir.resolve()),
        "unit_primary_descriptives": "problem",
        "resource_and_error_unit": "replicate",
    }
    for scenario, (problems, replicates) in scenario_data.items():
        metadata[f"{scenario}_unique_problems"] = int(
            problems["problem_id"].nunique()
        )
        metadata[f"{scenario}_problem_condition_rows"] = int(len(problems))
        metadata[f"{scenario}_replicate_rows"] = int(len(replicates))
    return metadata


def main() -> None:
    args = parse_arguments()
    input_dir = args.input_dir.resolve()
    output_dir = ensure_dir(args.output_dir.resolve())

    if not input_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {input_dir}")
    validate_input_files(input_dir)

    scenario_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    scenario_outputs: dict[str, dict[str, pd.DataFrame]] = {}

    for scenario in SCENARIOS:
        problems, replicates = load_scenario(input_dir, scenario)
        scenario_data[scenario] = (problems, replicates)
        outputs = build_scenario_outputs(scenario, problems, replicates)
        scenario_outputs[scenario] = outputs
        write_scenario_outputs(output_dir, scenario, outputs)

    comparison = build_cross_scenario_comparison(scenario_outputs)
    write_excel_sheets(
        output_dir / "comparacion_escenarios_descriptiva.xlsx",
        comparison,
    )
    for name, frame in comparison.items():
        frame.to_csv(
            output_dir / f"comparacion_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metadata = build_metadata(input_dir, scenario_data)
    (output_dir / "metadata_estadistica_descriptiva.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 67)
    print("ESTADISTICA DESCRIPTIVA - VERSION 3.0")
    print("=" * 67)
    for scenario, (problems, replicates) in scenario_data.items():
        print(
            f"{scenario.capitalize():.<30}"
            f"{problems['problem_id'].nunique():>6} problemas | "
            f"{len(problems):>6} problema-condición | "
            f"{len(replicates):>6} réplicas"
        )
    print("=" * 67)
    print("Unidad de los descriptivos de calidad: problema.")
    print("Unidad de taxonomía y recursos brutos: réplica.")
    print("Comparación principal de recursos: media por problema-condición pareada.")
    print(f"Salidas: {output_dir}")


if __name__ == "__main__":
    main()
