from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from common import ensure_dir, holm_adjust, write_excel_sheets

PIPELINE_VERSION = "3.1"
RANDOM_SEED = 20260716
N_BOOT = 10000

QUALITY_METRICS = [
    "score",
    "completeness",
    "symbolic_correctness",
    "pedagogical_quality",
    "faithfulness_to_problem",
    "subpart_accuracy",
]

RESOURCE_METRICS = {
    "execution_time_mean": "Tiempo medio por problema-condición (s)",
    "tokens_mean": "Tokens medios por problema-condición",
    "cost_mean_usd": "Coste medio por problema-condición (USD)",
}

SCENARIO_FILES = {
    "principal": "dataset_por_problema_pareado_principal.csv",
    "estricto": "dataset_por_problema_estricto.csv",
    "sensibilidad": "dataset_por_problema_pareado_sensibilidad.csv",
}

REPLICA_FILES = {
    "principal": "dataset_replica_pareado_principal.csv",
    "estricto": "dataset_replica_estricto.csv",
    "sensibilidad": "dataset_replica_pareado_sensibilidad.csv",
}

PIPELINE_ROOT = Path(__file__).resolve().parent
SPYDER_INPUT_DIR = PIPELINE_ROOT / "salidas" / "02_dataset_maestro"
SPYDER_OUTPUT_DIR = PIPELINE_ROOT / "salidas" / "04_inferencia"


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "si", "sí"}
    )


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo requerido: {path}")
    return pd.read_csv(path)


def condition_order(df: pd.DataFrame) -> tuple[str, str]:
    conditions = sorted(df["condition"].dropna().astype(str).unique())
    if len(conditions) != 2:
        raise ValueError(
            "Se necesitan exactamente dos condiciones analíticas; "
            f"se encontraron {len(conditions)}: {conditions}"
        )
    return conditions[0], conditions[1]


def paired_bootstrap_ci(
    differences: Iterable[float],
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """IC percentil del 95 % re-muestreando problemas emparejados."""
    arr = np.asarray(pd.Series(differences, dtype="float64").dropna(), dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[indexes].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def t_ci_difference(diff: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(diff, errors="coerce").dropna().astype(float)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values.iloc[0]), float(values.iloc[0])
    mean = float(values.mean())
    sem = float(stats.sem(values))
    margin = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    return mean - margin, mean + margin


def rank_biserial_from_differences(diff: pd.Series) -> float:
    """Correlación biserial de rangos para diferencias pareadas, excluyendo ceros."""
    values = pd.to_numeric(diff, errors="coerce").dropna().astype(float)
    values = values[values != 0]
    if values.empty:
        return 0.0
    ranks = stats.rankdata(np.abs(values))
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def hedges_gz(dz: float, n: int) -> float:
    """Corrección de pequeña muestra aplicada a d_z."""
    if not np.isfinite(dz) or n <= 2:
        return np.nan
    correction = 1.0 - (3.0 / (4.0 * (n - 1) - 1.0))
    return float(correction * dz)


def normality_diagnostics(diff: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(diff, errors="coerce").dropna().astype(float)
    result = {
        "difference_skewness": np.nan,
        "difference_kurtosis_excess": np.nan,
        "shapiro_w": np.nan,
        "p_shapiro": np.nan,
    }
    if len(values) >= 3:
        result["difference_skewness"] = float(stats.skew(values, bias=False))
        result["difference_kurtosis_excess"] = float(
            stats.kurtosis(values, fisher=True, bias=False)
        )
    # Shapiro se informa como diagnóstico, no como criterio automático de selección.
    if 3 <= len(values) <= 5000:
        shapiro = stats.shapiro(values)
        result["shapiro_w"] = float(shapiro.statistic)
        result["p_shapiro"] = float(shapiro.pvalue)
    return result


def build_pair(
    data: pd.DataFrame,
    value_col: str,
    index_col: str = "problem_id",
) -> tuple[pd.DataFrame, str, str]:
    c1, c2 = condition_order(data)
    pivot = data.pivot_table(
        index=index_col,
        columns="condition",
        values=value_col,
        aggfunc="mean",
    )
    pair = pivot.reindex(columns=[c1, c2]).dropna()
    return pair, c1, c2


def paired_test_one(
    data: pd.DataFrame,
    value_col: str,
    metric_label: str,
    scenario: str,
    family: str,
) -> dict[str, object] | None:
    pair, c1, c2 = build_pair(data, value_col)
    if len(pair) < 2:
        return None

    x = pair[c1].astype(float)
    y = pair[c2].astype(float)
    diff = y - x
    n = int(len(diff))

    t_result = stats.ttest_rel(y, x, nan_policy="omit")
    try:
        w_result = stats.wilcoxon(
            y,
            x,
            zero_method="pratt",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        w_stat = float(w_result.statistic)
        w_p = float(w_result.pvalue)
    except ValueError:
        w_stat, w_p = np.nan, 1.0

    sd_diff = float(diff.std(ddof=1))
    dz = float(diff.mean() / sd_diff) if sd_diff > 0 else np.nan
    ci_t_low, ci_t_high = t_ci_difference(diff)
    ci_boot_low, ci_boot_high = paired_bootstrap_ci(diff)
    diagnostics = normality_diagnostics(diff)

    return {
        "scenario": scenario,
        "family": family,
        "metric": metric_label,
        "value_column": value_col,
        "condition_1": c1,
        "condition_2": c2,
        "n_paired_problems": n,
        "mean_condition_1": float(x.mean()),
        "sd_condition_1": float(x.std(ddof=1)),
        "median_condition_1": float(x.median()),
        "mean_condition_2": float(y.mean()),
        "sd_condition_2": float(y.std(ddof=1)),
        "median_condition_2": float(y.median()),
        "mean_difference_2_minus_1": float(diff.mean()),
        "sd_difference": sd_diff,
        "median_difference": float(diff.median()),
        "ci95_difference_t_low": ci_t_low,
        "ci95_difference_t_high": ci_t_high,
        "ci95_difference_boot_low": ci_boot_low,
        "ci95_difference_boot_high": ci_boot_high,
        "t_paired": float(t_result.statistic),
        "p_t": float(t_result.pvalue),
        "wilcoxon_w": w_stat,
        "p_wilcoxon": w_p,
        "cohen_dz": dz,
        "hedges_gz": hedges_gz(dz, n),
        "rank_biserial": rank_biserial_from_differences(diff),
        "positive_differences": int((diff > 0).sum()),
        "negative_differences": int((diff < 0).sum()),
        "zero_differences": int((diff == 0).sum()),
        **diagnostics,
    }


def paired_tests_quality(data: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in QUALITY_METRICS:
        value_col = f"{metric}_mean"
        row = paired_test_one(data, value_col, metric, scenario, "quality")
        if row is not None:
            rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_t_holm"] = holm_adjust(result["p_t"])
        result["p_wilcoxon_holm"] = holm_adjust(result["p_wilcoxon"])
    return result


def paired_tests_resources(data: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value_col, label in RESOURCE_METRICS.items():
        row = paired_test_one(data, value_col, label, scenario, "resources")
        if row is not None:
            pair, c1, c2 = build_pair(data, value_col)
            denominator = pair[c1].replace(0, np.nan)
            ratios = pair[c2] / denominator
            row["mean_ratio_2_over_1"] = float(ratios.mean())
            row["median_ratio_2_over_1"] = float(ratios.median())
            rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_t_holm"] = holm_adjust(result["p_t"])
        result["p_wilcoxon_holm"] = holm_adjust(result["p_wilcoxon"])
    return result


def stratified_quality_tests(
    data: pd.DataFrame,
    scenario: str,
    strata_col: str,
    minimum_pairs: int = 5,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for stratum, group in data.groupby(strata_col, dropna=False):
        frame = paired_tests_quality(group, scenario)
        if frame.empty:
            continue
        frame.insert(0, strata_col, stratum)
        frame["minimum_pairs_required"] = minimum_pairs
        frame["sufficient_pairs"] = frame["n_paired_problems"] >= minimum_pairs
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    # Además de Holm dentro de cada estrato, se ofrece ajuste global exploratorio.
    result["p_t_holm_global"] = holm_adjust(result["p_t"])
    result["p_wilcoxon_holm_global"] = holm_adjust(result["p_wilcoxon"])
    return result


def mixed_models(replica_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Ejecuta modelos lineales mixtos a nivel de réplica cuando ``statsmodels``
    está disponible. Si la librería no está instalada, devuelve una fila de
    estado y permite que el resto del análisis continúe con normalidad.
    """
    try:
        import statsmodels.formula.api as smf
    except ModuleNotFoundError:
        return pd.DataFrame([
            {
                "scenario": scenario,
                "metric": pd.NA,
                "reference_condition": pd.NA,
                "comparison_condition": pd.NA,
                "converged": False,
                "model_status": "omitted_missing_statsmodels",
                "error": (
                    "Modelos mixtos omitidos: statsmodels no está instalado. "
                    "El resto de la inferencia se ejecutó correctamente."
                ),
            }
        ])

    data = replica_df.copy()
    if "valid_execution" in data.columns:
        data = data.loc[_as_bool(data["valid_execution"])].copy()
    if data.empty:
        return pd.DataFrame([
            {
                "scenario": scenario,
                "metric": pd.NA,
                "reference_condition": pd.NA,
                "comparison_condition": pd.NA,
                "converged": False,
                "model_status": "omitted_empty_dataset",
                "error": "Modelos mixtos omitidos: el escenario no contiene registros.",
            }
        ])

    c1, c2 = condition_order(data)
    data["condition"] = pd.Categorical(data["condition"], categories=[c1, c2])
    rows: list[dict[str, object]] = []

    for metric in QUALITY_METRICS:
        formula = (
            f"{metric} ~ C(condition, Treatment(reference={c1!r})) "
            "+ C(dataset_name)"
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    formula,
                    data=data,
                    groups=data["problem_id"],
                    re_formula="1",
                )
                fit = model.fit(
                    reml=False,
                    method="lbfgs",
                    maxiter=1000,
                    disp=False,
                )
            terms = [
                name for name in fit.params.index
                if name.startswith("C(condition")
            ]
            if not terms:
                raise RuntimeError("No se encontró el coeficiente de condición.")
            term = terms[0]
            ci = fit.conf_int().loc[term]
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "reference_condition": c1,
                    "comparison_condition": c2,
                    "coefficient": float(fit.params[term]),
                    "std_error": float(fit.bse[term]),
                    "z": float(fit.tvalues[term]),
                    "p": float(fit.pvalues[term]),
                    "ci95_low": float(ci.iloc[0]),
                    "ci95_high": float(ci.iloc[1]),
                    "n_records": int(fit.nobs),
                    "n_problems": int(data["problem_id"].nunique()),
                    "n_datasets": int(data["dataset_name"].nunique()),
                    "converged": bool(fit.converged),
                    "random_intercept_variance": float(fit.cov_re.iloc[0, 0]),
                    "residual_variance": float(fit.scale),
                    "aic": float(fit.aic),
                    "bic": float(fit.bic),
                    "model_status": "executed",
                    "error": pd.NA,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "reference_condition": c1,
                    "comparison_condition": c2,
                    "converged": False,
                    "model_status": "failed",
                    "error": str(exc),
                }
            )

    result = pd.DataFrame(rows)
    if "p" in result.columns:
        result["p_holm"] = holm_adjust(result["p"])
    return result


def comparison_across_scenarios(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    selected = []
    for scenario, frame in frames.items():
        if frame.empty:
            continue
        cols = [
            "scenario",
            "metric",
            "n_paired_problems",
            "mean_condition_1",
            "mean_condition_2",
            "mean_difference_2_minus_1",
            "ci95_difference_boot_low",
            "ci95_difference_boot_high",
            "p_t_holm",
            "p_wilcoxon_holm",
            "cohen_dz",
            "hedges_gz",
            "rank_biserial",
        ]
        selected.append(frame[cols])
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia pareada, análisis de sensibilidad y modelos mixtos."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("salidas/02_dataset_maestro"),
        help="Carpeta generada por 02_dataset_maestro.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("salidas/04_inferencia"),
    )
    parser.add_argument(
        "--minimum-stratum-pairs",
        type=int,
        default=5,
        help="Mínimo de problemas pareados para marcar un estrato como interpretable.",
    )
    if len(sys.argv) == 1:
        return argparse.Namespace(
            input_dir=SPYDER_INPUT_DIR,
            output_dir=SPYDER_OUTPUT_DIR,
            minimum_stratum_pairs=5,
        )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {args.input_dir}")
    output_dir = ensure_dir(args.output_dir)

    problem_data: dict[str, pd.DataFrame] = {}
    replica_data: dict[str, pd.DataFrame] = {}
    for scenario, filename in SCENARIO_FILES.items():
        problem_data[scenario] = read_csv_required(args.input_dir / filename)
    for scenario, filename in REPLICA_FILES.items():
        replica_data[scenario] = read_csv_required(args.input_dir / filename)

    quality_results: dict[str, pd.DataFrame] = {}
    resource_results: dict[str, pd.DataFrame] = {}
    mixed_results: dict[str, pd.DataFrame] = {}
    stratified_outputs: dict[str, pd.DataFrame] = {}

    for scenario, data in problem_data.items():
        quality_results[scenario] = paired_tests_quality(data, scenario)
        resource_results[scenario] = paired_tests_resources(data, scenario)

        for strata_col, label in [
            ("dataset_name", "dataset"),
            ("domain", "dominio"),
            ("education_level", "nivel"),
        ]:
            stratified_outputs[f"{label}_{scenario}"] = stratified_quality_tests(
                data,
                scenario,
                strata_col,
                minimum_pairs=args.minimum_stratum_pairs,
            )

    # El modelo mixto principal usa el escenario pareado principal. Los escenarios
    # estricto y sensibilidad se conservan como comprobaciones de robustez.
    for scenario, data in replica_data.items():
        mixed_results[scenario] = mixed_models(data, scenario)

    comparison = comparison_across_scenarios(quality_results)
    quality_all = pd.concat(quality_results.values(), ignore_index=True)
    resources_all = pd.concat(resource_results.values(), ignore_index=True)
    mixed_all = pd.concat(mixed_results.values(), ignore_index=True)

    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_iterations": N_BOOT,
        "bootstrap_seed": RANDOM_SEED,
        "primary_unit": "problem",
        "mixed_model_unit": "replica_nested_in_problem",
        "multiple_testing": "Holm within each family and scenario",
        "minimum_stratum_pairs": int(args.minimum_stratum_pairs),
        "primary_scenario": "principal",
        "strict_scenario": "estricto",
        "sensitivity_scenario": "sensibilidad",
        "statsmodels_optional": True,
        "mixed_models_status": sorted(
            set(
                str(value)
                for frame in mixed_results.values()
                if "model_status" in frame.columns
                for value in frame["model_status"].dropna().unique()
            )
        ),
    }
    (output_dir / "metadata_inferencia.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sheets: dict[str, pd.DataFrame] = {
        "comparacion_escenarios": comparison,
        "calidad_todos": quality_all,
        "recursos_todos": resources_all,
        "modelos_mixtos": mixed_all,
    }
    for scenario, frame in quality_results.items():
        sheets[f"calidad_{scenario}"] = frame
    for scenario, frame in resource_results.items():
        sheets[f"recursos_{scenario}"] = frame
    for name, frame in stratified_outputs.items():
        sheets[name] = frame
    for scenario, frame in mixed_results.items():
        sheets[f"mixto_{scenario}"] = frame

    write_excel_sheets(output_dir / "inferencia_estadistica.xlsx", sheets)
    for name, frame in sheets.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 75)
    print(f"INFERENCIA ESTADISTICA - VERSION {PIPELINE_VERSION}")
    print("=" * 75)
    for scenario in ["principal", "estricto", "sensibilidad"]:
        frame = quality_results[scenario]
        n = int(frame["n_paired_problems"].max()) if not frame.empty else 0
        sig_t = int((frame["p_t_holm"] < 0.05).sum()) if not frame.empty else 0
        sig_w = int((frame["p_wilcoxon_holm"] < 0.05).sum()) if not frame.empty else 0
        print(
            f"{scenario.capitalize():.<28}{n:>5} problemas | "
            f"t-Holm significativas: {sig_t}/6 | Wilcoxon-Holm: {sig_w}/6"
        )
    print("=" * 75)
    print("Análisis principal: comparación pareada a nivel de problema.")
    mixed_statuses = sorted(
        set(
            str(value)
            for frame in mixed_results.values()
            if "model_status" in frame.columns
            for value in frame["model_status"].dropna().unique()
        )
    )
    if "omitted_missing_statsmodels" in mixed_statuses:
        print("Modelos mixtos: OMITIDOS porque statsmodels no está instalado.")
        print("El resto de las pruebas se completó correctamente.")
    else:
        print("Modelos mixtos: análisis complementario a nivel de réplica, con")
        print("intercepto aleatorio por problema y control fijo por dataset.")
    print(f"Salidas: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
