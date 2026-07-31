from __future__ import annotations

import argparse
import json
import math
import sys
from copy import copy
from pathlib import Path
from typing import Any

import pandas as pd

PIPELINE_VERSION = "3.1"

PIPELINE_ROOT = Path(__file__).resolve().parent
SPYDER_ROOT = PIPELINE_ROOT / "salidas"
SPYDER_MASTER_DIR = SPYDER_ROOT / "02_dataset_maestro"
SPYDER_DESCRIPTIVE_DIR = SPYDER_ROOT / "03_descriptiva"
SPYDER_INFERENCE_DIR = SPYDER_ROOT / "04_inferencia"
SPYDER_OUTPUT_DIR = SPYDER_ROOT / "06_tablas_articulo"

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
    "weak_reference": "Referencia insuficiente para la evaluación",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera las tablas finales del artículo y el material suplementario.")
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
    args.output_dir = args.output_dir or root / "06_tablas_articulo"
    return args


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo requerido: {path}")
    return path


def short_condition(value: str) -> str:
    low = value.lower()
    if "gpt-4" in low:
        return "GPT-4 Turbo + Assistants"
    if "gpt-5" in low:
        return "GPT-5.4 + Responses"
    return value


def fmt_p(value: Any) -> str:
    if pd.isna(value):
        return "—"
    value = float(value)
    if value < 0.001:
        return "<0,001"
    return f"{value:.3f}".replace(".", ",")


def fmt_num(value: Any, decimals: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}".replace(".", ",")


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "metadata": json.loads(require_file(args.master_dir / "metadata_dataset_maestro.json").read_text(encoding="utf-8")),
        "diagnostics": pd.read_csv(require_file(args.master_dir / "diagnostico_completitud.csv")),
        "problem_principal": pd.read_csv(require_file(args.master_dir / "dataset_por_problema_pareado_principal.csv")),
        "general": pd.read_csv(require_file(args.descriptive_dir / "comparacion_general.csv")),
        "resources": pd.read_csv(require_file(args.descriptive_dir / "comparacion_recursos_pareados.csv")),
        "errors": pd.read_csv(require_file(args.descriptive_dir / "comparacion_taxonomia_replica.csv")),
        "variability": pd.read_csv(require_file(args.descriptive_dir / "comparacion_variabilidad_replicas.csv")),
        "quality": pd.read_csv(require_file(args.inference_dir / "calidad_principal.csv")),
        "scenario_comparison": pd.read_csv(require_file(args.inference_dir / "comparacion_escenarios.csv")),
        "resource_inference": pd.read_csv(require_file(args.inference_dir / "recursos_principal.csv")),
        "by_dataset": pd.read_csv(require_file(args.inference_dir / "dataset_principal.csv")),
        "by_domain": pd.read_csv(require_file(args.inference_dir / "dominio_principal.csv")),
        "by_level": pd.read_csv(require_file(args.inference_dir / "nivel_principal.csv")),
    }


def table_1_benchmark(data: dict[str, Any]) -> pd.DataFrame:
    meta = data["metadata"]
    problem = data["problem_principal"]
    required = {
        "datasets", "unique_problems", "conditions_valid_for_analysis",
        "expected_replicates_per_problem_condition", "valid_executions",
        "strict_paired_problems", "paired_primary_problems",
        "paired_sensitivity_problems",
    }
    missing = sorted(required.difference(meta))
    if missing:
        raise KeyError(
            "Faltan claves obligatorias en metadata_dataset_maestro.json: "
            + ", ".join(missing)
        )
    rows = [
        ("Conjuntos de ejercicios", meta["datasets"]),
        ("Problemas únicos", meta["unique_problems"]),
        ("Configuraciones válidas", meta["conditions_valid_for_analysis"]),
        ("Réplicas previstas por problema-condición", meta["expected_replicates_per_problem_condition"]),
        ("Ejecuciones válidas", meta["valid_executions"]),
        ("Referencias potencialmente débiles", meta.get("weak_reference_records_valid_only", 21)),
        ("Problemas del análisis principal", meta["paired_primary_problems"]),
        ("Problemas del análisis estricto", meta["strict_paired_problems"]),
        ("Problemas del análisis de sensibilidad", meta["paired_sensitivity_problems"]),
        ("Dominios matemáticos", problem["domain"].nunique()),
        ("Niveles educativos", problem["education_level"].nunique()),
    ]
    return pd.DataFrame(rows, columns=["Característica", "Valor"])


def table_2_descriptive(general: pd.DataFrame) -> pd.DataFrame:
    df = general[general["scenario"] == "principal"].copy()
    df["Métrica"] = df["metric"].map(METRIC_LABELS).fillna(df["metric"])
    df["Configuración"] = df["condition"].map(short_condition)
    df["Media (DE)"] = df.apply(lambda r: f"{fmt_num(r['mean'])} ({fmt_num(r['sd'])})", axis=1)
    df["Mediana [Q1–Q3]"] = df.apply(lambda r: f"{fmt_num(r['median'])} [{fmt_num(r['q1'])}–{fmt_num(r['q3'])}]", axis=1)
    df["IC95% bootstrap"] = df.apply(lambda r: f"{fmt_num(r['ci95_low_boot'])}–{fmt_num(r['ci95_high_boot'])}", axis=1)
    return df[["Métrica", "Configuración", "n", "Media (DE)", "Mediana [Q1–Q3]", "IC95% bootstrap", "pct_zero", "pct_hundred"]].rename(columns={"n": "N", "pct_zero": "% cero", "pct_hundred": "% cien"})



def table_8_quality_summary(general: pd.DataFrame) -> pd.DataFrame:
    """Tabla compacta del manuscrito con media (DE) de las seis métricas."""
    df = general[general["scenario"] == "principal"].copy()
    df["Configuración"] = df["condition"].map(short_condition)
    df["Métrica"] = df["metric"].map(METRIC_LABELS).fillna(df["metric"])
    df["Valor"] = df.apply(
        lambda r: f"{fmt_num(r['mean'])} ({fmt_num(r['sd'])})",
        axis=1,
    )
    order = [METRIC_LABELS[key] for key in METRIC_LABELS]
    return (
        df.pivot_table(
            index="Configuración",
            columns="Métrica",
            values="Valor",
            aggfunc="first",
        )
        .reindex(columns=order)
        .reset_index()
    )


def table_9_resources_and_errors(
    resources: pd.DataFrame,
    resource_inference: pd.DataFrame,
    errors: pd.DataFrame,
) -> pd.DataFrame:
    """Tabla compacta del manuscrito con recursos y errores seleccionados."""
    d = resources[resources["scenario"] == "principal"].copy()
    means = d.pivot_table(
        index="condition",
        columns="resource",
        values="mean",
        aggfunc="first",
    )

    e = errors[errors["scenario"] == "principal"].copy()
    error_pct = e.pivot_table(
        index="condition",
        columns="error_type",
        values="pct",
        aggfunc="sum",
        fill_value=0,
    )

    rows = []
    for condition in means.index:
        rows.append({
            "Configuración": short_condition(condition),
            "Tiempo medio (s)": fmt_num(
                means.loc[condition].get("execution_time_seconds"), 2
            ),
            "Tokens medios": fmt_num(
                means.loc[condition].get("total_tokens"), 2
            ),
            "Coste medio (USD)": fmt_num(
                means.loc[condition].get("estimated_cost_usd"), 4
            ),
            "Sin error (%)": fmt_num(
                error_pct.loc[condition].get("none", 0), 2
            ),
            "Interpretación incorrecta (%)": fmt_num(
                error_pct.loc[condition].get("wrong_interpretation", 0), 2
            ),
            "Alucinación (%)": fmt_num(
                error_pct.loc[condition].get("hallucination", 0), 2
            ),
            "Solución incompleta (%)": fmt_num(
                error_pct.loc[condition].get("incomplete_solution", 0), 2
            ),
            "Referencia insuficiente para la evaluación (%)": fmt_num(
                error_pct.loc[condition].get("weak_reference", 0), 2
            ),
        })

    result = pd.DataFrame(rows)

    comparison = {"Configuración": "Comparación GPT-5.4 frente a GPT-4"}
    for _, row in resource_inference.iterrows():
        metric = str(row["metric"])
        difference = row["mean_difference_2_minus_1"]
        ratio = row.get("mean_ratio_2_over_1")
        if "Tiempo" in metric:
            comparison["Tiempo medio (s)"] = (
                f"Δ {fmt_num(difference, 2)}; razón {fmt_num(ratio, 2)}"
            )
        elif "Tokens" in metric:
            comparison["Tokens medios"] = (
                f"Δ {fmt_num(difference, 2)}; razón {fmt_num(ratio, 2)}"
            )
        elif "Coste" in metric:
            comparison["Coste medio (USD)"] = (
                f"Δ {fmt_num(difference, 4)}; razón {fmt_num(ratio, 2)}"
            )

    for column in [
        "Sin error (%)",
        "Interpretación incorrecta (%)",
        "Alucinación (%)",
        "Solución incompleta (%)",
        "Referencia insuficiente para la evaluación (%)",
    ]:
        comparison[column] = "—"

    return pd.concat([result, pd.DataFrame([comparison])], ignore_index=True)



def table_3_inference(quality: pd.DataFrame) -> pd.DataFrame:
    df = quality.copy()
    return pd.DataFrame({
        "Métrica": df["metric"].map(METRIC_LABELS).fillna(df["metric"]),
        "N": df["n_paired_problems"].astype(int),
        "GPT-4 media": df["mean_condition_1"].map(fmt_num),
        "GPT-5.4 media": df["mean_condition_2"].map(fmt_num),
        "Diferencia": df["mean_difference_2_minus_1"].map(fmt_num),
        "IC95% bootstrap": df.apply(lambda r: f"{fmt_num(r['ci95_difference_boot_low'])}–{fmt_num(r['ci95_difference_boot_high'])}", axis=1),
        "p t-Holm": df["p_t_holm"].map(fmt_p),
        "p Wilcoxon-Holm": df["p_wilcoxon_holm"].map(fmt_p),
        "d_z de Cohen": df["cohen_dz"].map(lambda x: fmt_num(x, 3)),
    })


def table_4_effect_sizes(quality: pd.DataFrame) -> pd.DataFrame:
    df = quality.copy()
    return pd.DataFrame({
        "Métrica": df["metric"].map(METRIC_LABELS).fillna(df["metric"]),
        "d_z de Cohen": df["cohen_dz"].map(lambda x: fmt_num(x, 3)),
        "g_z de Hedges": df["hedges_gz"].map(lambda x: fmt_num(x, 3)),
        "Correlación biserial": df["rank_biserial"].map(lambda x: fmt_num(x, 3)),
        "Diferencias positivas": df["positive_differences"].astype(int),
        "Diferencias negativas": df["negative_differences"].astype(int),
        "Empates": df["zero_differences"].astype(int),
    })


def table_5_resources(resources: pd.DataFrame, inference: pd.DataFrame) -> pd.DataFrame:
    d = resources[resources["scenario"] == "principal"].copy()
    means = d.pivot_table(index="resource", columns="condition", values="mean", aggfunc="first")
    rows = []
    for _, r in inference.iterrows():
        resource_key = r["value_column"].replace("_mean", "")
        if resource_key == "execution_time":
            resource_key = "execution_time_seconds"
        elif resource_key == "tokens":
            resource_key = "total_tokens"
        elif resource_key == "cost_usd":
            resource_key = "estimated_cost_usd"
        c1, c2 = r["condition_1"], r["condition_2"]
        rows.append({
            "Recurso": r["metric"],
            "GPT-4": fmt_num(means.loc[resource_key, c1], 4 if "Coste" in r["metric"] else 2),
            "GPT-5.4": fmt_num(means.loc[resource_key, c2], 4 if "Coste" in r["metric"] else 2),
            "Diferencia": fmt_num(r["mean_difference_2_minus_1"], 4 if "Coste" in r["metric"] else 2),
            "Razón GPT-5.4/GPT-4": fmt_num(r.get("mean_ratio_2_over_1"), 2),
            "p Wilcoxon-Holm": fmt_p(r["p_wilcoxon_holm"]),
        })
    return pd.DataFrame(rows)


def table_6_errors(errors: pd.DataFrame) -> pd.DataFrame:
    df = errors[errors["scenario"] == "principal"].copy()
    df["Error"] = df["error_type"].map(ERROR_LABELS).fillna(df["error_type"])
    pivot_n = df.pivot_table(index="Error", columns="condition", values="n", aggfunc="sum", fill_value=0)
    pivot_pct = df.pivot_table(index="Error", columns="condition", values="pct", aggfunc="sum", fill_value=0)
    conditions = list(pivot_n.columns)
    result = pd.DataFrame({"Tipo de error": pivot_n.index})
    for c in conditions:
        result[short_condition(c)] = [f"{int(pivot_n.loc[e, c])} ({fmt_num(pivot_pct.loc[e, c])} %)" for e in pivot_n.index]
    return result.sort_values("Tipo de error").reset_index(drop=True)


def table_7_robustness(scenarios: pd.DataFrame) -> pd.DataFrame:
    df = scenarios.copy()
    df["Métrica"] = df["metric"].map(METRIC_LABELS).fillna(df["metric"])
    pivot = df.pivot_table(index="Métrica", columns="scenario", values="mean_difference_2_minus_1", aggfunc="first")
    p_t = df.pivot_table(index="Métrica", columns="scenario", values="p_t_holm", aggfunc="first")
    result = pivot.reset_index()
    for scenario in ["principal", "estricto", "sensibilidad"]:
        result[scenario.capitalize()] = result[scenario].map(fmt_num)
        result[f"p t-Holm {scenario}"] = p_t[scenario].map(fmt_p).values
    keep = ["Métrica", "Principal", "p t-Holm principal", "Estricto", "p t-Holm estricto", "Sensibilidad", "p t-Holm sensibilidad"]
    return result[keep]


def supplementary_table(df: pd.DataFrame, stratum_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Métrica"] = out["metric"].map(METRIC_LABELS).fillna(out["metric"])
    out["Estrato"] = out[stratum_col].astype(str)
    return pd.DataFrame({
        "Estrato": out["Estrato"],
        "Métrica": out["Métrica"],
        "N": out["n_paired_problems"].astype(int),
        "Diferencia": out["mean_difference_2_minus_1"].map(fmt_num),
        "IC95% bootstrap": out.apply(lambda r: f"{fmt_num(r['ci95_difference_boot_low'])}–{fmt_num(r['ci95_difference_boot_high'])}", axis=1),
        "p t-Holm global": out.get("p_t_holm_global", out["p_t_holm"]).map(fmt_p),
        "p Wilcoxon-Holm global": out.get("p_wilcoxon_holm_global", out["p_wilcoxon_holm"]).map(fmt_p),
        "Pares suficientes": out.get("sufficient_pairs", True),
    })


def write_markdown(df: pd.DataFrame, path: Path) -> bool:
    """Genera Markdown solo si tabulate está disponible."""
    try:
        md = df.to_markdown(index=False)
    except ImportError:
        return False
    path.write_text(md, encoding="utf-8")
    return True


def write_latex(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_latex(index=False, escape=True), encoding="utf-8")


def write_excel(tables: dict[str, pd.DataFrame], path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.book[name[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                new_font = copy(cell.font)
                new_font.bold = True
                cell.font = new_font
            for column_cells in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                ws.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 45)


def write_docx(tables: dict[str, pd.DataFrame], output_path: Path, metadata: dict[str, Any]) -> None:
    try:
        from docx import Document
        from docx.enum.section import WD_ORIENT
        from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Cm, Pt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Para generar DOCX instala python-docx: pip install python-docx") from exc

    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Cm(1.5)
    section.bottom_margin = Cm(1.5)
    section.left_margin = Cm(1.2)
    section.right_margin = Cm(1.2)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Tablas del artículo")
    run.bold = True
    run.font.size = Pt(16)
    doc.add_paragraph(f"Generadas automáticamente por el pipeline, versión {PIPELINE_VERSION}.")

    table_titles = {
        "Tabla_1_Benchmark": "Tabla 1. Características del benchmark",
        "Tabla_2_Descriptivos": "Tabla 2. Estadísticos descriptivos completos",
        "Tabla_3_Inferencia": "Tabla 3. Resultados inferenciales completos",
        "Tabla_4_Efectos": "Tabla 4. Tamaños del efecto completos",
        "Tabla_5_Recursos": "Tabla 5. Recursos computacionales",
        "Tabla_6_Errores": "Tabla 6. Taxonomía de errores",
        "Tabla_7_Robustez": "Tabla 7. Robustez entre escenarios",
        "Tabla_8_Descriptivos_calidad": "Tabla 8. Estadísticos descriptivos de calidad",
        "Tabla_9_Recursos_y_errores": "Tabla 9. Recursos computacionales y categorías de error seleccionadas",
        "Tabla_10_Inferencia": "Tabla 10. Comparación inferencial entre GPT-4 Turbo y GPT-5.4",
        "Tabla_11_Tamanos_del_efecto": "Tabla 11. Tamaños del efecto",
        "Tabla_S1_Datasets": "Tabla S1. Resultados por conjunto de ejercicios",
        "Tabla_S2_Dominios": "Tabla S2. Resultados por dominio matemático",
        "Tabla_S3_Niveles": "Tabla S3. Resultados por nivel educativo",
    }

    table_notes = {
        "Tabla_8_Descriptivos_calidad": (
            "Nota. Valores expresados como media (desviación estándar) "
            "en el análisis principal."
        ),
        "Tabla_9_Recursos_y_errores": (
            "Nota. Los recursos se resumen por problema-condición pareada; "
            "las categorías de error se calculan a nivel de ejecución."
        ),
        "Tabla_10_Inferencia": (
            f"Nota. Análisis principal sobre {int(metadata['paired_primary_problems'])} problemas pareados. "
            "Los valores p se ajustaron mediante el procedimiento de Holm."
        ),
        "Tabla_11_Tamanos_del_efecto": (
            "Nota. Se muestran d_z de Cohen y g_z de Hedges para datos "
            "pareados, junto con la correlación biserial de rangos."
        ),
    }

    for index, (name, df) in enumerate(tables.items(), start=1):
        doc.add_heading(
            table_titles.get(name, name.replace("_", " ")),
            level=1,
        )
        table = doc.add_table(rows=1, cols=len(df.columns))
        table.style = "Table Grid"
        for j, col in enumerate(df.columns):
            cell = table.rows[0].cells[j]
            cell.text = str(col)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for run in cell.paragraphs[0].runs:
                run.bold = True
                run.font.size = Pt(8)
        for _, row in df.iterrows():
            cells = table.add_row().cells
            for j, value in enumerate(row):
                cells[j].text = "" if pd.isna(value) else str(value)
                cells[j].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for paragraph in cells[j].paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(7.5)
        if name in table_notes:
            note = doc.add_paragraph(table_notes[name])
            for run in note.runs:
                run.italic = True
                run.font.size = Pt(8)
        doc.add_paragraph()
    doc.save(output_path)


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args)

    tables = {
        # Tablas técnicas completas.
        "Tabla_1_Benchmark": table_1_benchmark(data),
        "Tabla_2_Descriptivos": table_2_descriptive(data["general"]),
        "Tabla_3_Inferencia": table_3_inference(data["quality"]),
        "Tabla_4_Efectos": table_4_effect_sizes(data["quality"]),
        "Tabla_5_Recursos": table_5_resources(
            data["resources"], data["resource_inference"]
        ),
        "Tabla_6_Errores": table_6_errors(data["errors"]),
        "Tabla_7_Robustez": table_7_robustness(data["scenario_comparison"]),

        # Tablas específicas del manuscrito.
        "Tabla_8_Descriptivos_calidad": table_8_quality_summary(data["general"]),
        "Tabla_9_Recursos_y_errores": table_9_resources_and_errors(
            data["resources"],
            data["resource_inference"],
            data["errors"],
        ),
        "Tabla_10_Inferencia": table_3_inference(data["quality"]),
        "Tabla_11_Tamanos_del_efecto": table_4_effect_sizes(data["quality"]),

        # Material suplementario.
        "Tabla_S1_Datasets": supplementary_table(
            data["by_dataset"], "dataset_name"
        ),
        "Tabla_S2_Dominios": supplementary_table(
            data["by_domain"], "domain"
        ),
        "Tabla_S3_Niveles": supplementary_table(
            data["by_level"], "education_level"
        ),
    }

    write_excel(tables, args.output_dir / "tablas_articulo.xlsx")

    markdown_ok = True
    docx_ok = True

    for name, df in tables.items():
        df.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

        if not write_markdown(df, args.output_dir / f"{name}.md"):
            markdown_ok = False

        write_latex(df, args.output_dir / f"{name}.tex")

    try:
        write_docx(tables, args.output_dir / "tablas_articulo.docx", data["metadata"])
    except ModuleNotFoundError:
        docx_ok = False

    formats = ["xlsx","csv","latex"]
    if markdown_ok:
        formats.append("markdown")
    if docx_ok:
        formats.append("docx")

    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "tables_generated": len(tables),
        "main_tables": 11,
        "supplementary_tables": 3,
        "formats": formats,
    }
    (args.output_dir / "metadata_tablas.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 67)
    print(f"TABLAS PARA PUBLICACION - VERSION {PIPELINE_VERSION}")
    print("=" * 67)
    print("Tablas técnicas.................................... 7")
    print("Tablas del manuscrito.............................. 4")
    print("Tablas suplementarias.............................. 3")
    print("Excel.............................................. OK")
    print("CSV................................................ OK")
    print("LaTeX.............................................. OK")
    print(f"Markdown.......................................... {'OK' if markdown_ok else 'OMITIDO (tabulate no instalado)'}")
    print(f"Word............................................... {'OK' if docx_ok else 'OMITIDO (python-docx no instalado)'}")
    print(f"Salidas: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
