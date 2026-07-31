from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINE_VERSION = "3.1"
DPI = 400

PIPELINE_ROOT = Path(__file__).resolve().parent
SPYDER_ROOT = PIPELINE_ROOT / "salidas"
SPYDER_MASTER_DIR = SPYDER_ROOT / "02_dataset_maestro"
SPYDER_DESCRIPTIVE_DIR = SPYDER_ROOT / "03_descriptiva"
SPYDER_INFERENCE_DIR = SPYDER_ROOT / "04_inferencia"
SPYDER_OUTPUT_DIR = SPYDER_ROOT / "05_figuras_articulo"

METRIC_LABELS = {
    "score": "Puntuación global",
    "completeness": "Completitud",
    "symbolic_correctness": "Corrección simbólica",
    "pedagogical_quality": "Calidad pedagógica",
    "faithfulness_to_problem": "Fidelidad al problema",
    "subpart_accuracy": "Precisión por apartados",
}

ERROR_LABELS = {
    "none": "Sin error",
    "algebraic_error": "Error algebraico",
    "reasoning_error": "Error de razonamiento",
    "hallucination": "Alucinación",
    "incomplete_solution": "Solución incompleta",
    "notation_error": "Error de notación",
    "wrong_interpretation": "Interpretación incorrecta",
    "calculation_error": "Error de cálculo",
    "formatting_error": "Error de formato",
    "no_answer": "Sin respuesta",
    "technical_error": "Error técnico",
    "weak_reference": "Referencia débil",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera figuras finales listas para el artículo."
    )
    parser.add_argument("--master-dir", type=Path, required=False)
    parser.add_argument("--descriptive-dir", type=Path, required=False)
    parser.add_argument("--inference-dir", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=False)
    if len(sys.argv) == 1:
        return argparse.Namespace(
            master_dir=SPYDER_MASTER_DIR,
            descriptive_dir=SPYDER_DESCRIPTIVE_DIR,
            inference_dir=SPYDER_INFERENCE_DIR,
            output_dir=SPYDER_OUTPUT_DIR,
        )
    args = parser.parse_args()
    root = Path("salidas")
    args.master_dir = args.master_dir or root / "02_dataset_maestro"
    args.descriptive_dir = args.descriptive_dir or root / "03_descriptiva"
    args.inference_dir = args.inference_dir or root / "04_inferencia"
    args.output_dir = args.output_dir or root / "05_figuras_articulo"
    return args


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo requerido: {path}")
    return path


def condition_short(condition: str) -> str:
    low = condition.lower()
    if "gpt-4" in low:
        return "GPT-4 Turbo + Assistants"
    if "gpt-5" in low:
        return "GPT-5.4 + Responses"
    return condition


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def load_inputs(args: argparse.Namespace) -> dict[str, pd.DataFrame | dict]:
    metadata_path = require_file(args.master_dir / "metadata_dataset_maestro.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "metadata": metadata,
        "problem_principal": pd.read_csv(require_file(args.master_dir / "dataset_por_problema_pareado_principal.csv")),
        "problem_strict": pd.read_csv(require_file(args.master_dir / "dataset_por_problema_estricto.csv")),
        "replica_principal": pd.read_csv(require_file(args.master_dir / "dataset_replica_pareado_principal.csv")),
        "desc_general": pd.read_csv(require_file(args.descriptive_dir / "comparacion_general.csv")),
        "desc_errors": pd.read_csv(require_file(args.descriptive_dir / "comparacion_taxonomia_replica.csv")),
        "desc_resources": pd.read_csv(require_file(args.descriptive_dir / "comparacion_recursos_pareados.csv")),
        "desc_variability": pd.read_csv(require_file(args.descriptive_dir / "comparacion_variabilidad_replicas.csv")),
        "quality_principal": pd.read_csv(require_file(args.inference_dir / "calidad_principal.csv")),
        "scenario_comparison": pd.read_csv(require_file(args.inference_dir / "comparacion_escenarios.csv")),
        "resources_principal": pd.read_csv(require_file(args.inference_dir / "recursos_principal.csv")),
    }


def figure_1_pipeline(metadata: dict, output_dir: Path) -> None:
    required = [
        "datasets",
        "unique_problems",
        "conditions_valid_for_analysis",
        "expected_replicates_per_problem_condition",
        "valid_executions",
        "paired_primary_problems",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise KeyError(
            "Faltan claves obligatorias en metadata_dataset_maestro.json: "
            + ", ".join(missing)
        )
    values = [
        f"{int(metadata['datasets'])} conjuntos\nde ejercicios",
        f"{int(metadata['unique_problems'])} problemas\núnicos",
        f"{int(metadata['conditions_valid_for_analysis'])} configuraciones\nválidas",
        f"{int(metadata['expected_replicates_per_problem_condition'])} réplicas por\nproblema-condición",
        f"{int(metadata['valid_executions'])} ejecuciones\nválidas",
        f"{int(metadata['paired_primary_problems'])} problemas\npareados principales",
    ]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")
    xs = np.linspace(0.07, 0.93, len(values))
    for i, label in enumerate(values):
        ax.text(
            xs[i], 0.52, label,
            ha="center", va="center", fontsize=10,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": "black"},
            transform=ax.transAxes,
        )
        if i < len(values) - 1:
            ax.annotate(
                "", xy=(xs[i + 1] - 0.065, 0.52), xytext=(xs[i] + 0.065, 0.52),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "linewidth": 1.4},
            )
    ax.set_title("Diseño experimental y flujo analítico", fontsize=14, pad=18)
    save_figure(fig, output_dir, "Figura_1_diseno_experimental")


def problem_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = df.pivot_table(index="problem_id", columns="condition", values=value_col, aggfunc="mean")
    if pivot.shape[1] != 2:
        raise ValueError(f"Se esperaban dos condiciones para {value_col}; encontradas {pivot.shape[1]}")
    return pivot.dropna()


def figure_2_distribution(problem_df: pd.DataFrame, output_dir: Path) -> None:
    pivot = problem_pivot(problem_df, "score_mean")
    conditions = list(pivot.columns)
    data = [pivot[c].to_numpy() for c in conditions]
    labels = [condition_short(c) for c in conditions]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    parts = ax.violinplot(data, positions=[1, 2], showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.35)
    ax.boxplot(data, positions=[1, 2], widths=0.25, showmeans=True, patch_artist=False)
    rng = np.random.default_rng(20260716)
    for pos, values in zip([1, 2], data):
        jitter = rng.normal(0, 0.035, len(values))
        ax.scatter(np.full(len(values), pos) + jitter, values, s=12, alpha=0.25)
    ax.set_xticks([1, 2], labels)
    ax.set_ylabel("Puntuación media por problema")
    ax.set_ylim(-3, 103)
    ax.set_title("Distribución de la puntuación global")
    save_figure(fig, output_dir, "Figura_2_distribucion_score")


def figure_3_paired(problem_df: pd.DataFrame, output_dir: Path) -> None:
    pivot = problem_pivot(problem_df, "score_mean")
    conditions = list(pivot.columns)
    labels = [condition_short(c) for c in conditions]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for _, row in pivot.iterrows():
        ax.plot([0, 1], [row[conditions[0]], row[conditions[1]]], marker="o", markersize=2.5, alpha=0.18)
    means = [pivot[c].mean() for c in conditions]
    ax.plot([0, 1], means, marker="o", markersize=8, linewidth=3, label="Media")
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("Puntuación media por problema")
    ax.set_ylim(-3, 103)
    ax.set_title("Cambio pareado de la puntuación por problema")
    ax.legend()
    save_figure(fig, output_dir, "Figura_3_comparacion_pareada_score")


def figure_4_forest(inference: pd.DataFrame, output_dir: Path) -> None:
    df = inference.copy()
    df["label"] = df["metric"].map(METRIC_LABELS).fillna(df["metric"])
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    mean = df["mean_difference_2_minus_1"].astype(float)
    low = df["ci95_difference_boot_low"].astype(float)
    high = df["ci95_difference_boot_high"].astype(float)
    xerr = np.vstack([mean - low, high - mean])
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.errorbar(mean, y, xerr=xerr, fmt="o", capsize=4)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y, df["label"])
    ax.set_xlabel("Diferencia media (GPT-5.4 − GPT-4 Turbo)")
    ax.set_title("Diferencias de rendimiento e intervalos bootstrap del 95 %")
    save_figure(fig, output_dir, "Figura_4_forest_diferencias")


def figure_5_quality_cost(desc_general: pd.DataFrame, desc_resources: pd.DataFrame, output_dir: Path) -> None:
    scores = desc_general[(desc_general["scenario"] == "principal") & (desc_general["metric"] == "score")][["condition", "mean"]].rename(columns={"mean": "score_mean"})
    res = desc_resources[(desc_resources["scenario"] == "principal")].copy()
    wide = res.pivot_table(index="condition", columns="resource", values="mean", aggfunc="first").reset_index()
    df = scores.merge(wide, on="condition", how="inner")
    required = ["estimated_cost_usd", "total_tokens"]
    if any(c not in df.columns for c in required):
        raise ValueError("No se encuentran coste y tokens en comparacion_recursos_pareados.csv")
    sizes = 250 + 1250 * (df["total_tokens"] / df["total_tokens"].max())
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.scatter(df["estimated_cost_usd"], df["score_mean"], s=sizes, alpha=0.55)
    for _, row in df.iterrows():
        ax.annotate(condition_short(row["condition"]), (row["estimated_cost_usd"], row["score_mean"]), xytext=(7, 7), textcoords="offset points")
    ax.set_xlabel("Coste medio por problema-condición (USD)")
    ax.set_ylabel("Puntuación global media")
    ax.set_title("Compromiso entre calidad, coste y consumo de tokens")
    ax.text(0.02, 0.02, "El tamaño de la burbuja representa los tokens medios", transform=ax.transAxes, fontsize=9)
    save_figure(fig, output_dir, "Figura_5_calidad_coste")


def figure_6_errors(errors: pd.DataFrame, output_dir: Path) -> None:
    df = errors[errors["scenario"] == "principal"].copy()
    df["error_label"] = df["error_type"].map(ERROR_LABELS).fillna(df["error_type"])
    pivot = df.pivot_table(index="error_label", columns="condition", values="pct", aggfunc="sum", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True).drop(columns="total")
    pivot.columns = [condition_short(c) for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(10, 7))
    pivot.plot(kind="barh", ax=ax)
    ax.set_xlabel("Porcentaje de ejecuciones")
    ax.set_ylabel("")
    ax.set_title("Taxonomía de errores por configuración")
    ax.legend(title="")
    save_figure(fig, output_dir, "Figura_6_taxonomia_errores")


def figure_7_variability(strict_df: pd.DataFrame, output_dir: Path) -> None:
    conditions = list(strict_df["condition"].dropna().unique())
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for condition in conditions:
        values = pd.to_numeric(strict_df.loc[strict_df["condition"] == condition, "score_range"], errors="coerce").dropna()
        ax.hist(values, bins=np.arange(0, 105, 5), alpha=0.45, label=condition_short(condition))
    ax.set_xlabel("Rango entre réplicas en la puntuación global")
    ax.set_ylabel("Número de problema-condición")
    ax.set_title("Variabilidad entre réplicas (escenario estricto)")
    ax.legend()
    save_figure(fig, output_dir, "Figura_7_variabilidad_replicas")


def figure_8_robustness(scenarios: pd.DataFrame, output_dir: Path) -> None:
    df = scenarios.copy()
    order = list(METRIC_LABELS)
    scenario_order = ["principal", "estricto", "sensibilidad"]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(order))
    for scenario in scenario_order:
        sub = df[df["scenario"] == scenario].set_index("metric").reindex(order)
        ax.plot(x, sub["mean_difference_2_minus_1"], marker="o", label=scenario.capitalize())
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x, [METRIC_LABELS[m] for m in order], rotation=25, ha="right")
    ax.set_ylabel("Diferencia media (GPT-5.4 − GPT-4 Turbo)")
    ax.set_title("Robustez de las diferencias entre escenarios analíticos")
    ax.legend()
    save_figure(fig, output_dir, "Figura_8_robustez_escenarios")


def write_captions(output_dir: Path, metadata: dict) -> None:
    captions = {
        "Figura 1": "Diseño experimental y flujo analítico del estudio.",
        "Figura 2": "Distribución de la puntuación global media por problema en el escenario principal.",
        "Figura 3": f"Comparación pareada de la puntuación global media para los {int(metadata['paired_primary_problems'])} problemas del escenario principal.",
        "Figura 4": "Diferencias medias entre configuraciones e intervalos bootstrap del 95 % para las seis métricas.",
        "Figura 5": "Compromiso entre puntuación global, coste medio y tokens consumidos por problema-condición.",
        "Figura 6": "Distribución porcentual de las categorías de error por configuración en el escenario principal.",
        "Figura 7": "Distribución del rango entre las tres réplicas en el escenario estricto.",
        "Figura 8": "Comparación de las diferencias medias en los escenarios principal, estricto y de sensibilidad.",
    }
    text = "\n\n".join(f"{key}. {value}" for key, value in captions.items())
    (output_dir / "pies_de_figura.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args)

    figure_1_pipeline(data["metadata"], args.output_dir)
    figure_2_distribution(data["problem_principal"], args.output_dir)
    figure_3_paired(data["problem_principal"], args.output_dir)
    figure_4_forest(data["quality_principal"], args.output_dir)
    figure_5_quality_cost(data["desc_general"], data["desc_resources"], args.output_dir)
    figure_6_errors(data["desc_errors"], args.output_dir)
    figure_7_variability(data["problem_strict"], args.output_dir)
    figure_8_robustness(data["scenario_comparison"], args.output_dir)
    write_captions(args.output_dir, data["metadata"])

    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "figures_generated": 8,
        "formats": ["png", "pdf", "svg"],
        "dpi_png": DPI,
    }
    (args.output_dir / "metadata_figuras.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 67)
    print(f"FIGURAS PARA PUBLICACION - VERSION {PIPELINE_VERSION}")
    print("=" * 67)
    print("Figuras generadas................................. 8")
    print("Formatos................................ PNG, PDF y SVG")
    print(f"Salidas: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
