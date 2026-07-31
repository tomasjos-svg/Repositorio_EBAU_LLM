# -*- coding: utf-8 -*-
"""
VALIDACION HUMANA DEL JUEZ LLM
==============================

Lee:
  1) validacion_humana_2023.json
  2) Validacion_humana_2023.xlsx (hoja CLAVE_CIEGA)
  3) todos los JSON de la subcarpeta resultados/

Empareja cada respuesta por:
  dataset_name + question_number + condition + run

Calcula, para cada métrica:
  - Pearson y Spearman
  - ICC(A,1), acuerdo absoluto
  - kappa de Cohen ponderada tras discretización
  - diferencia de medias, MAE, RMSE
  - t pareada y Wilcoxon, con corrección de Holm
  - Bland-Altman

Genera:
  - libro Excel con tablas
  - CSV de datos emparejados y resultados
  - figuras PNG
  - texto automático para la sección del artículo
  - diagnóstico de emparejamiento

Dependencias:
    pip install pandas numpy scipy matplotlib openpyxl scikit-learn

Ejecución:
    python validacion_humana_juez_llm.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

ROOT = Path(__file__).resolve().parent
HUMAN_JSON = ROOT / "validacion_humana_2023.json"
BLIND_KEY_XLSX = ROOT / "Validacion_humana_2023.xlsx"
RESULTS_DIR = ROOT / "resultados"
OUTPUT_DIR = ROOT / "salida_validacion_humana"

BLIND_KEY_SHEET = "CLAVE_CIEGA"

METRICS: Dict[str, str] = {
    "score": "Score global",
    "completeness": "Completitud",
    "symbolic_correctness": "Corrección simbólica",
    "pedagogical_quality": "Calidad pedagógica",
    "faithfulness_to_problem": "Fidelidad al problema",
    "subpart_accuracy": "Precisión por apartados",
}

# Categorías ordinales para kappa. Se pueden modificar antes de ejecutar.
KAPPA_BINS = [-0.001, 49.999, 69.999, 89.999, 100.001]
KAPPA_LABELS = ["0-49", "50-69", "70-89", "90-100"]

ID_ALIASES = {
    "dataset_name": [
        "dataset_name", "dataset", "corpus", "dataset_id",
    ],
    "question_number": [
        "question_number", "question", "pregunta", "ejercicio",
        "question_id", "numero_pregunta", "n_pregunta",
    ],
    "condition": [
        "experiment_label", "condition", "condicion", "condición",
        "condition_label", "experiment_condition", "pipeline_condition",
    ],
    "run": [
        "run", "replica", "réplica", "replicate", "run_index",
        "iteration", "repetition",
    ],
}

METRIC_ALIASES = {
    "score": ["score", "judge_score", "llm_judge_score", "puntuacion"],
    "completeness": [
        "completeness", "judge_completeness", "llm_judge_completeness",
        "completitud",
    ],
    "symbolic_correctness": [
        "symbolic_correctness", "judge_symbolic_correctness",
        "llm_judge_symbolic_correctness", "correccion_simbolica",
    ],
    "pedagogical_quality": [
        "pedagogical_quality", "judge_pedagogical_quality",
        "llm_judge_pedagogical_quality", "calidad_pedagogica",
    ],
    "faithfulness_to_problem": [
        "faithfulness_to_problem", "judge_faithfulness_to_problem",
        "llm_judge_faithfulness_to_problem", "fidelidad",
    ],
    "subpart_accuracy": [
        "subpart_accuracy", "judge_subpart_accuracy",
        "llm_judge_subpart_accuracy", "precision_por_apartados",
    ],
}

ERROR_ALIASES = [
    "error_type", "judge_error_type", "llm_judge_error_type", "tipo_error"
]


# =============================================================================
# UTILIDADES
# =============================================================================

def norm_key(value: Any) -> str:
    text = str(value).strip().lower()
    table = str.maketrans(
        "áéíóúüñç",
        "aeiouunc",
    )
    text = text.translate(table)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def norm_id(value: Any) -> Any:
    if value is None:
        return np.nan
    if isinstance(value, float) and math.isnan(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    return norm_key(text)


def to_number(value: Any) -> float:
    if value is None or value == "":
        return np.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.number)):
        number = float(value)
    else:
        match = re.search(r"-?\d+(?:[.,]\d+)?", str(value).strip())
        if not match:
            return np.nan
        number = float(match.group().replace(",", "."))
    return max(0.0, min(100.0, number))


def first_value(record: Dict[str, Any], aliases: Sequence[str]) -> Any:
    normalized = {norm_key(k): v for k, v in record.items()}
    for alias in aliases:
        key = norm_key(alias)
        if key in normalized and normalized[key] not in (None, ""):
            return normalized[key]
    return None


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON no válido: {path.name}, línea {exc.lineno}, "
            f"columna {exc.colno}: {exc.msg}"
        ) from exc


def recursive_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    """Devuelve todos los diccionarios anidados de una estructura JSON."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from recursive_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from recursive_dicts(item)


def flatten_shallow(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aplana únicamente subobjetos probables de evaluación.
    Evita convertir textos largos o listas enteras.
    """
    out = dict(record)
    likely = {
        "evaluation", "evaluacion", "judge", "llm_judge", "metrics",
        "scores", "result", "assessment", "grading", "valoracion",
    }
    for key, value in record.items():
        if isinstance(value, dict) and norm_key(key) in likely:
            for subkey, subvalue in value.items():
                out.setdefault(subkey, subvalue)
    return out


def normalize_frame_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("dataset_name", "question_number", "condition"):
        if col in out.columns:
            out[col] = out[col].map(norm_id)
    if "run" in out.columns:
        out["run"] = pd.to_numeric(out["run"], errors="coerce").astype("Int64")
    return out


# =============================================================================
# EVALUACIÓN HUMANA
# =============================================================================

def load_human_evaluation() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not HUMAN_JSON.exists():
        raise FileNotFoundError(
            f"No se encuentra {HUMAN_JSON.name} en {ROOT}"
        )

    data = read_json(HUMAN_JSON)
    if not isinstance(data, dict):
        raise ValueError("El JSON humano debe contener un objeto raíz.")

    metadata = data.get("metadata", {})
    evaluations = data.get("evaluations")

    if not isinstance(evaluations, list):
        raise ValueError(
            "No se encontró una lista 'evaluations' en el JSON humano."
        )

    default_run = metadata.get("run", 1)
    rows: List[Dict[str, Any]] = []

    for index, item in enumerate(evaluations, start=1):
        if not isinstance(item, dict):
            continue

        completed = str(item.get("completed", "")).strip().lower()
        completed_ok = completed in {"sí", "si", "yes", "true", "1"}

        row: Dict[str, Any] = {
            "human_row": index,
            "id_problem": item.get("id_problem"),
            "dataset_name": item.get("dataset_name"),
            "question_number": item.get("question_number"),
            "blind_label": str(item.get("blind_label", "")).strip().upper(),
            "run": item.get("run", default_run),
            "human_error_type": item.get("error_type"),
            "human_comment": item.get("human_comment"),
            "preference": item.get("preference"),
            "pair_comment": item.get("pair_comment"),
            "completed": completed_ok,
        }

        for metric in METRICS:
            row[f"human_{metric}"] = to_number(item.get(metric))

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[df["completed"]].copy()
    return normalize_frame_ids(df), metadata


# =============================================================================
# CLAVE CIEGA
# =============================================================================

def load_blind_key() -> pd.DataFrame:
    if not BLIND_KEY_XLSX.exists():
        raise FileNotFoundError(
            f"No se encuentra {BLIND_KEY_XLSX.name} en {ROOT}"
        )

    key = pd.read_excel(
        BLIND_KEY_XLSX,
        sheet_name=BLIND_KEY_SHEET,
        dtype=str,
    )

    required = {
        "dataset_name", "question_number",
        "A_condition", "B_condition", "run",
    }
    missing = required - set(key.columns)
    if missing:
        raise ValueError(
            "Faltan columnas en CLAVE_CIEGA: " + ", ".join(sorted(missing))
        )

    long_rows: List[Dict[str, Any]] = []
    for _, row in key.iterrows():
        for blind_label in ("A", "B"):
            long_rows.append({
                "id_problem": row.get("ID_PROBLEMA"),
                "dataset_name": row.get("dataset_name"),
                "question_number": row.get("question_number"),
                "blind_label": blind_label,
                "condition": row.get(f"{blind_label}_condition"),
                "model_real": row.get(f"{blind_label}_model"),
                "run": row.get("run"),
                "source_json": row.get(f"source_{blind_label}"),
            })

    long_key = pd.DataFrame(long_rows)
    return normalize_frame_ids(long_key)


def apply_blind_key(human: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["dataset_name", "question_number", "blind_label", "run"]
    merged = human.merge(
        key,
        on=join_cols,
        how="left",
        validate="many_to_one",
        suffixes=("", "_key"),
        indicator=True,
    )
    merged["blind_key_matched"] = merged["_merge"].eq("both")
    return merged.drop(columns="_merge")


# =============================================================================
# RESULTADOS DEL PIPELINE
# =============================================================================

def looks_like_result_record(record: Dict[str, Any]) -> bool:
    flat = flatten_shallow(record)

    has_metric = any(
        first_value(flat, aliases) is not None
        for aliases in METRIC_ALIASES.values()
    )
    has_dataset = first_value(flat, ID_ALIASES["dataset_name"]) is not None
    has_question = first_value(flat, ID_ALIASES["question_number"]) is not None
    has_condition = first_value(flat, ID_ALIASES["condition"]) is not None

    return has_metric and has_dataset and has_question and has_condition


def extract_pipeline_row(
    record: Dict[str, Any],
    source_file: str,
) -> Dict[str, Any]:
    flat = flatten_shallow(record)

    row: Dict[str, Any] = {
        "pipeline_source_file": source_file,
        "dataset_name": first_value(flat, ID_ALIASES["dataset_name"]),
        "question_number": first_value(flat, ID_ALIASES["question_number"]),
        "condition": first_value(flat, ID_ALIASES["condition"]),
        "run": first_value(flat, ID_ALIASES["run"]),
        "judge_error_type": first_value(flat, ERROR_ALIASES),
    }

    for metric, aliases in METRIC_ALIASES.items():
        row[f"judge_{metric}"] = to_number(first_value(flat, aliases))

    return row


def load_pipeline_results() -> Tuple[pd.DataFrame, List[str]]:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"No existe la subcarpeta de resultados: {RESULTS_DIR}"
        )

    files = sorted(RESULTS_DIR.rglob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No hay archivos JSON dentro de {RESULTS_DIR}"
        )

    rows: List[Dict[str, Any]] = []
    used: List[str] = []

    for path in files:
        try:
            data = read_json(path)
        except Exception as exc:
            print(f"AVISO: se omite {path.name}: {exc}")
            continue

        count_before = len(rows)
        seen_signatures = set()

        for record in recursive_dicts(data):
            if not looks_like_result_record(record):
                continue

            row = extract_pipeline_row(record, path.name)
            signature = (
                str(row.get("dataset_name")),
                str(row.get("question_number")),
                str(row.get("condition")),
                str(row.get("run")),
                tuple(row.get(f"judge_{m}") for m in METRICS),
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            rows.append(row)

        if len(rows) > count_before:
            used.append(path.name)

    if not rows:
        raise ValueError(
            "Se encontraron JSON, pero no se reconoció ningún registro "
            "con dataset, pregunta, condición y métricas del juez."
        )

    df = normalize_frame_ids(pd.DataFrame(rows))

    # Si run no está en algún registro, se asume 1 únicamente cuando todo el
    # estudio de validación usa la réplica 1.
    if df["run"].isna().any():
        df.loc[df["run"].isna(), "run"] = 1

    return df, used


def collapse_pipeline_duplicates(
    pipeline: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["dataset_name", "question_number", "condition", "run"]

    counts = (
        pipeline.groupby(keys, dropna=False)
        .size()
        .reset_index(name="n_duplicates")
    )
    duplicate_report = counts[counts["n_duplicates"] > 1].copy()

    aggregations: Dict[str, Any] = {}
    for col in pipeline.columns:
        if col in keys:
            continue
        if col.startswith("judge_") and col != "judge_error_type":
            aggregations[col] = "mean"
        else:
            aggregations[col] = lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan

    collapsed = (
        pipeline.groupby(keys, dropna=False, as_index=False)
        .agg(aggregations)
    )
    return collapsed, duplicate_report


def match_human_and_pipeline(
    human_with_key: pd.DataFrame,
    pipeline: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["dataset_name", "question_number", "condition", "run"]
    pipeline_unique, _ = collapse_pipeline_duplicates(pipeline)

    merged = human_with_key.merge(
        pipeline_unique,
        on=keys,
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("", "_pipeline"),
    )
    merged["pipeline_matched"] = merged["_merge"].eq("both")
    return merged.drop(columns="_merge")


# =============================================================================
# ESTADÍSTICA
# =============================================================================

def icc_a1(human: np.ndarray, judge: np.ndarray) -> float:
    """
    ICC(A,1): modelo bidireccional, acuerdo absoluto, medida individual.
    Fórmula de McGraw y Wong.
    """
    x = np.column_stack([human, judge]).astype(float)
    n, k = x.shape
    if n < 2:
        return np.nan

    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_err = np.sum(
        (x - row_means[:, None] - col_means[None, :] + grand) ** 2
    )

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denominator = (
        ms_rows
        + (k - 1) * ms_err
        + k * (ms_cols - ms_err) / n
    )
    if denominator == 0:
        return np.nan
    return (ms_rows - ms_err) / denominator


def bootstrap_ci(
    human: np.ndarray,
    judge: np.ndarray,
    statistic,
    repetitions: int = 3000,
    seed: int = 20260727,
) -> Tuple[float, float]:
    if len(human) < 3:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    values: List[float] = []
    n = len(human)

    for _ in range(repetitions):
        idx = rng.integers(0, n, n)
        try:
            value = float(statistic(human[idx], judge[idx]))
            if np.isfinite(value):
                values.append(value)
        except Exception:
            pass

    if not values:
        return np.nan, np.nan
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ordered = p.dropna().sort_values()
    m = len(ordered)
    running_max = 0.0

    for rank, (idx, value) in enumerate(ordered.items(), start=1):
        adjusted = min(1.0, (m - rank + 1) * float(value))
        running_max = max(running_max, adjusted)
        out.loc[idx] = running_max

    return out


def discretize(values: Sequence[float]) -> pd.Categorical:
    return pd.cut(
        pd.Series(values),
        bins=KAPPA_BINS,
        labels=KAPPA_LABELS,
        include_lowest=True,
        ordered=True,
    )


def analyze_metrics(matched: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for metric, label in METRICS.items():
        human_col = f"human_{metric}"
        judge_col = f"judge_{metric}"

        pairs = matched[[human_col, judge_col]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        if len(pairs) < 3:
            continue

        human = pairs[human_col].to_numpy(float)
        judge = pairs[judge_col].to_numpy(float)
        diff = judge - human
        diff_sd = diff.std(ddof=1)

        pearson = stats.pearsonr(human, judge)
        spearman = stats.spearmanr(human, judge)

        pearson_ci = bootstrap_ci(
            human, judge,
            lambda x, y: stats.pearsonr(x, y).statistic,
        )
        icc_value = icc_a1(human, judge)
        icc_ci = bootstrap_ci(human, judge, icc_a1)

        try:
            t_test = stats.ttest_rel(judge, human, nan_policy="omit")
            t_stat, t_p = float(t_test.statistic), float(t_test.pvalue)
        except Exception:
            t_stat, t_p = np.nan, np.nan

        try:
            wilcoxon = stats.wilcoxon(
                judge, human, zero_method="wilcox",
                correction=False, alternative="two-sided",
            )
            w_stat, w_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
        except Exception:
            w_stat, w_p = np.nan, np.nan

        human_cat = discretize(human)
        judge_cat = discretize(judge)

        try:
            kappa = cohen_kappa_score(
                human_cat, judge_cat, weights="quadratic"
            )
        except Exception:
            kappa = np.nan

        rows.append({
            "metric": metric,
            "metric_label": label,
            "n_pairs": len(pairs),
            "human_mean": human.mean(),
            "human_sd": human.std(ddof=1),
            "judge_mean": judge.mean(),
            "judge_sd": judge.std(ddof=1),
            "mean_difference_judge_minus_human": diff.mean(),
            "difference_sd": diff_sd,
            "mae": np.abs(diff).mean(),
            "rmse": np.sqrt(np.mean(diff ** 2)),
            "pearson_r": pearson.statistic,
            "pearson_p": pearson.pvalue,
            "pearson_ci95_low": pearson_ci[0],
            "pearson_ci95_high": pearson_ci[1],
            "spearman_rho": spearman.statistic,
            "spearman_p": spearman.pvalue,
            "icc_a1": icc_value,
            "icc_ci95_low": icc_ci[0],
            "icc_ci95_high": icc_ci[1],
            "quadratic_weighted_kappa": kappa,
            "paired_t": t_stat,
            "paired_t_p": t_p,
            "wilcoxon_w": w_stat,
            "wilcoxon_p": w_p,
            "bland_altman_bias": diff.mean(),
            "bland_altman_lower_limit": diff.mean() - 1.96 * diff_sd,
            "bland_altman_upper_limit": diff.mean() + 1.96 * diff_sd,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["paired_t_p_holm"] = holm_adjust(result["paired_t_p"])
        result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p"])
    return result


def analyze_error_types(
    matched: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not {"human_error_type", "judge_error_type"} <= set(matched.columns):
        return pd.DataFrame(), pd.DataFrame()

    pairs = matched[["human_error_type", "judge_error_type"]].dropna().copy()
    for col in pairs.columns:
        pairs[col] = pairs[col].map(norm_key)
    pairs = pairs[(pairs != "").all(axis=1)]

    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()

    labels = sorted(
        set(pairs["human_error_type"]) | set(pairs["judge_error_type"])
    )
    matrix = confusion_matrix(
        pairs["human_error_type"],
        pairs["judge_error_type"],
        labels=labels,
    )
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"humano_{x}" for x in labels],
        columns=[f"juez_{x}" for x in labels],
    )
    summary = pd.DataFrame([{
        "n_pairs": len(pairs),
        "exact_agreement": (
            pairs["human_error_type"] == pairs["judge_error_type"]
        ).mean(),
        "cohen_kappa": cohen_kappa_score(
            pairs["human_error_type"], pairs["judge_error_type"]
        ),
    }])
    return summary, matrix_df


# =============================================================================
# FIGURAS
# =============================================================================

def generate_figures(matched: pd.DataFrame) -> List[Path]:
    figure_dir = OUTPUT_DIR / "figuras"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for metric, label in METRICS.items():
        human_col = f"human_{metric}"
        judge_col = f"judge_{metric}"
        pairs = matched[[human_col, judge_col]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        if len(pairs) < 3:
            continue

        human = pairs[human_col].to_numpy(float)
        judge = pairs[judge_col].to_numpy(float)

        # Dispersión
        path_scatter = figure_dir / f"dispersion_{metric}.png"
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(human, judge, alpha=0.65)
        ax.plot([0, 100], [0, 100], linestyle="--")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Evaluación humana")
        ax.set_ylabel("Evaluación del juez LLM")
        ax.set_title(f"Acuerdo humano-juez: {label}")
        fig.tight_layout()
        fig.savefig(path_scatter, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path_scatter)

        # Bland-Altman
        means = (human + judge) / 2
        differences = judge - human
        bias = differences.mean()
        sd = differences.std(ddof=1)

        path_ba = figure_dir / f"bland_altman_{metric}.png"
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(means, differences, alpha=0.65)
        ax.axhline(bias)
        ax.axhline(bias + 1.96 * sd, linestyle="--")
        ax.axhline(bias - 1.96 * sd, linestyle="--")
        ax.set_xlabel("Media de ambas evaluaciones")
        ax.set_ylabel("Juez LLM - evaluación humana")
        ax.set_title(f"Bland-Altman: {label}")
        fig.tight_layout()
        fig.savefig(path_ba, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path_ba)

    return paths


# =============================================================================
# TABLAS Y TEXTO PARA EL ARTÍCULO
# =============================================================================

def format_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "< .001"
    return f"{value:.3f}"


def article_table(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "Métrica": results["metric_label"],
        "n": results["n_pairs"].astype(int),
        "Humano M (DE)": results.apply(
            lambda r: f"{r['human_mean']:.2f} ({r['human_sd']:.2f})", axis=1
        ),
        "Juez M (DE)": results.apply(
            lambda r: f"{r['judge_mean']:.2f} ({r['judge_sd']:.2f})", axis=1
        ),
        "Diferencia J-H": results[
            "mean_difference_judge_minus_human"
        ].map(lambda x: f"{x:.2f}"),
        "MAE": results["mae"].map(lambda x: f"{x:.2f}"),
        "Pearson r": results["pearson_r"].map(lambda x: f"{x:.3f}"),
        "Spearman ρ": results["spearman_rho"].map(lambda x: f"{x:.3f}"),
        "ICC(A,1)": results["icc_a1"].map(lambda x: f"{x:.3f}"),
        "κ ponderada": results[
            "quadratic_weighted_kappa"
        ].map(lambda x: f"{x:.3f}"),
        "p Wilcoxon-Holm": results["wilcoxon_p_holm"].map(format_p),
    })


def interpret_icc(value: float) -> str:
    if pd.isna(value):
        return "no calculable"
    if value < 0.50:
        return "pobre"
    if value < 0.75:
        return "moderado"
    if value < 0.90:
        return "bueno"
    return "excelente"


def interpret_corr(value: float) -> str:
    if pd.isna(value):
        return "no calculable"
    magnitude = abs(value)
    if magnitude < 0.30:
        return "débil"
    if magnitude < 0.50:
        return "moderada"
    if magnitude < 0.70:
        return "fuerte"
    return "muy fuerte"


def generate_manuscript_text(
    results: pd.DataFrame,
    matched_count: int,
    total_human: int,
    error_summary: pd.DataFrame,
) -> str:
    lines = [
        "VALIDACIÓN HUMANA DEL JUEZ LLM",
        "",
        "Método",
        "",
        (
            "Se realizó una validación humana ciega de las respuestas "
            "correspondientes a la réplica 1 del conjunto de 2023. "
            "La identidad de las condiciones experimentales se reconstruyó "
            "mediante la clave de cegamiento y cada valoración se emparejó "
            "con el resultado original mediante dataset, número de pregunta, "
            "condición experimental y réplica."
        ),
        "",
        (
            "Para cada una de las seis métricas se calcularon las "
            "correlaciones de Pearson y Spearman. El acuerdo absoluto de las "
            "puntuaciones continuas se estimó mediante ICC(A,1), modelo "
            "bidireccional de acuerdo absoluto y medida individual. Las "
            f"puntuaciones se discretizaron en {', '.join(KAPPA_LABELS)} "
            "para calcular la kappa de Cohen con ponderación cuadrática. "
            "También se calcularon la diferencia media juez-humano, el error "
            "absoluto medio, la raíz del error cuadrático medio, los límites "
            "de acuerdo de Bland-Altman y las pruebas pareadas t y Wilcoxon, "
            "con corrección de Holm entre métricas."
        ),
        "",
        "Resultados",
        "",
        (
            f"Se emparejaron {matched_count} de las {total_human} "
            "valoraciones humanas disponibles "
            f"({100 * matched_count / max(total_human, 1):.1f}%)."
        ),
    ]

    if results.empty:
        lines.extend([
            "",
            (
                "No fue posible calcular los índices de acuerdo porque no "
                "se obtuvieron suficientes pares válidos. Debe revisarse el "
                "diagnóstico de emparejamiento."
            ),
        ])
        return "\n".join(lines)

    for _, r in results.iterrows():
        lines.extend([
            "",
            (
                f"En {r['metric_label'].lower()} (n = {int(r['n_pairs'])}), "
                f"la media humana fue {r['human_mean']:.2f} "
                f"(DE = {r['human_sd']:.2f}) y la media del juez fue "
                f"{r['judge_mean']:.2f} (DE = {r['judge_sd']:.2f}). "
                f"La diferencia media juez-humano fue "
                f"{r['mean_difference_judge_minus_human']:.2f} puntos y "
                f"el MAE fue {r['mae']:.2f}. La asociación fue "
                f"{interpret_corr(r['pearson_r'])} "
                f"(Pearson r = {r['pearson_r']:.3f}, "
                f"p {format_p(r['pearson_p'])}; "
                f"Spearman ρ = {r['spearman_rho']:.3f}, "
                f"p {format_p(r['spearman_p'])}). "
                f"El acuerdo absoluto fue {interpret_icc(r['icc_a1'])} "
                f"(ICC(A,1) = {r['icc_a1']:.3f}; IC95% bootstrap "
                f"[{r['icc_ci95_low']:.3f}, {r['icc_ci95_high']:.3f}]) "
                f"y la kappa ponderada fue "
                f"κ = {r['quadratic_weighted_kappa']:.3f}. "
                f"La prueba de Wilcoxon, corregida mediante Holm, produjo "
                f"p {format_p(r['wilcoxon_p_holm'])}."
            ),
        ])

    if not error_summary.empty:
        e = error_summary.iloc[0]
        lines.extend([
            "",
            (
                f"Para el tipo de error se dispuso de {int(e['n_pairs'])} "
                f"pares no vacíos. El acuerdo exacto fue del "
                f"{100 * e['exact_agreement']:.1f}% y la kappa de Cohen "
                f"fue κ = {e['cohen_kappa']:.3f}."
            ),
        ])

    lines.extend([
        "",
        (
            "La correlación se interpretó como asociación, no como acuerdo. "
            "Por ello, la valoración principal de la validez operativa del "
            "juez debe basarse conjuntamente en ICC, kappa, sesgo medio, MAE "
            "y límites de Bland-Altman."
        ),
    ])
    return "\n".join(lines)


# =============================================================================
# DIAGNÓSTICO
# =============================================================================

def build_diagnostics(
    human_keyed: pd.DataFrame,
    pipeline: pd.DataFrame,
    matched: pd.DataFrame,
    used_files: List[str],
    duplicate_report: pd.DataFrame,
) -> str:
    lines = [
        "DIAGNÓSTICO DE VALIDACIÓN HUMANA",
        "=" * 76,
        f"JSON humano: {HUMAN_JSON}",
        f"Excel de clave ciega: {BLIND_KEY_XLSX}",
        f"Carpeta de resultados: {RESULTS_DIR}",
        f"Valoraciones humanas completas: {len(human_keyed)}",
        f"Registros del pipeline detectados: {len(pipeline)}",
        f"JSON del pipeline utilizados: {len(used_files)}",
        f"Clave ciega emparejada: {int(human_keyed['blind_key_matched'].sum())}"
        f"/{len(human_keyed)}",
        f"Pipeline emparejado: {int(matched['pipeline_matched'].sum())}"
        f"/{len(matched)}",
        "",
        "Archivos utilizados:",
    ]
    lines.extend(f"- {name}" for name in used_files)

    if not duplicate_report.empty:
        lines.extend([
            "",
            (
                "AVISO: se detectaron claves duplicadas en los JSON. "
                "El programa promedió las métricas duplicadas:"
            ),
            duplicate_report.to_string(index=False),
        ])

    unmatched_key = human_keyed[~human_keyed["blind_key_matched"]]
    if not unmatched_key.empty:
        lines.extend([
            "",
            "Registros sin correspondencia en CLAVE_CIEGA:",
            unmatched_key[
                ["dataset_name", "question_number", "blind_label", "run"]
            ].to_string(index=False),
        ])

    unmatched_pipeline = matched[~matched["pipeline_matched"]]
    if not unmatched_pipeline.empty:
        lines.extend([
            "",
            "Registros humanos sin correspondencia en los JSON del pipeline:",
            unmatched_pipeline[
                [
                    "dataset_name", "question_number", "blind_label",
                    "condition", "run", "model_real",
                ]
            ].to_string(index=False),
        ])

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1/7 Leyendo evaluación humana...")
    human, metadata = load_human_evaluation()

    print("2/7 Leyendo clave ciega...")
    blind_key = load_blind_key()
    human_keyed = apply_blind_key(human, blind_key)

    print("3/7 Leyendo JSON de la carpeta resultados/...")
    pipeline, used_files = load_pipeline_results()
    pipeline_unique, duplicate_report = collapse_pipeline_duplicates(pipeline)

    print("4/7 Emparejando humano y juez...")
    matched = match_human_and_pipeline(human_keyed, pipeline_unique)

    print("5/7 Calculando estadísticos...")
    metric_results = analyze_metrics(matched)
    error_summary, error_matrix = analyze_error_types(matched)
    table_article = article_table(metric_results)

    print("6/7 Generando figuras...")
    figures = generate_figures(matched)

    print("7/7 Exportando resultados...")
    matched_count = int(matched["pipeline_matched"].sum())
    text = generate_manuscript_text(
        metric_results,
        matched_count=matched_count,
        total_human=len(matched),
        error_summary=error_summary,
    )
    diagnostics = build_diagnostics(
        human_keyed,
        pipeline,
        matched,
        used_files,
        duplicate_report,
    )

    # CSV
    matched.to_csv(
        OUTPUT_DIR / "datos_emparejados.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_results.to_csv(
        OUTPUT_DIR / "resultados_acuerdo_metricas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    table_article.to_csv(
        OUTPUT_DIR / "tabla_articulo_validacion.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Excel
    excel_path = OUTPUT_DIR / "validacion_humana_resultados.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        human.to_excel(writer, sheet_name="humano", index=False)
        blind_key.to_excel(writer, sheet_name="clave_ciega_larga", index=False)
        pipeline.to_excel(writer, sheet_name="pipeline_detectado", index=False)
        matched.to_excel(writer, sheet_name="datos_emparejados", index=False)
        metric_results.to_excel(
            writer, sheet_name="acuerdo_metricas", index=False
        )
        table_article.to_excel(
            writer, sheet_name="tabla_articulo", index=False
        )
        duplicate_report.to_excel(
            writer, sheet_name="duplicados_pipeline", index=False
        )
        error_summary.to_excel(
            writer, sheet_name="acuerdo_tipo_error", index=False
        )
        error_matrix.to_excel(writer, sheet_name="matriz_tipo_error")

    # Texto
    (OUTPUT_DIR / "seccion_validacion_humana.txt").write_text(
        text, encoding="utf-8"
    )
    (OUTPUT_DIR / "diagnostico_emparejamiento.txt").write_text(
        diagnostics, encoding="utf-8"
    )
    (OUTPUT_DIR / "metadatos_validacion.json").write_text(
        json.dumps(
            {
                "metadata_human_json": metadata,
                "human_complete_rows": len(human),
                "blind_key_matches": int(
                    human_keyed["blind_key_matched"].sum()
                ),
                "pipeline_records_detected": len(pipeline),
                "pipeline_matches": matched_count,
                "pipeline_json_files_used": used_files,
                "figures_generated": [str(p.name) for p in figures],
                "kappa_bins": KAPPA_BINS,
                "kappa_labels": KAPPA_LABELS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("ANÁLISIS FINALIZADO")
    print("=" * 60)
    print(f"Valoraciones humanas completas: {len(human)}")
    print(
        f"Clave ciega emparejada: "
        f"{int(human_keyed['blind_key_matched'].sum())}/{len(human_keyed)}"
    )
    print(f"Pipeline emparejado: {matched_count}/{len(matched)}")
    print(f"Métricas analizadas: {len(metric_results)}")
    print(f"Figuras generadas: {len(figures)}")
    print(f"Salida: {OUTPUT_DIR}")
    print()
    print("Archivos principales:")
    print("- validacion_humana_resultados.xlsx")
    print("- tabla_articulo_validacion.csv")
    print("- resultados_acuerdo_metricas.csv")
    print("- datos_emparejados.csv")
    print("- seccion_validacion_humana.txt")
    print("- diagnostico_emparejamiento.txt")

    if matched_count < len(matched):
        print()
        print(
            "AVISO: hay registros sin emparejar. Revise "
            "diagnostico_emparejamiento.txt."
        )

    return 0 if not metric_results.empty else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
