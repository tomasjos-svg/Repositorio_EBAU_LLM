from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

METRICS = [
    "score",
    "completeness",
    "symbolic_correctness",
    "pedagogical_quality",
    "faithfulness_to_problem",
    "subpart_accuracy",
]

TEXT_COLUMNS = [
    "evaluation_comment",
    "raw_evaluation",
    "model_solution",
    "reference_solution",
    "solver_prompt",
    "reference_prompt",
    "evaluation_prompt",
    "error",
]

EXPECTED_COLUMNS = [
    "run_id", "timestamp", "experiment_hash", "dataset_name", "domain",
    "education_level", "experiment_label", "api", "model", "judge_model",
    "run", *METRICS, "error_type", *TEXT_COLUMNS,
    "execution_time_seconds", "solver_input_tokens", "solver_output_tokens",
    "solver_total_tokens", "reference_input_tokens", "reference_output_tokens",
    "reference_total_tokens", "judge_input_tokens", "judge_output_tokens",
    "judge_total_tokens", "total_input_tokens", "total_output_tokens",
    "total_tokens", "estimated_cost_usd", "technical_error",
]

WEAK_REFERENCE_PATTERNS = [
    r"no se encuentra(?: una)? soluci[oó]n",
    r"no (?:se )?incluye(?: la)? soluci[oó]n",
    r"no contiene(?: una)? soluci[oó]n",
    r"no aparece(?: una)? soluci[oó]n",
    r"no es posible reconstruir",
    r"no puedo reconstruir",
    r"lamentablemente",
    r"soluci[oó]n espec[ií]fica .* no se incluye",
    r"ser[ií]a necesario buscar",
    r"informaci[oó]n (?:más|mas) detallada .* ser[ií]a necesaria",
    r"referencia oficial no contiene",
    r"referencia .* d[eé]bil",
    r"no parece contener informaci[oó]n detallada o legible",
]


def read_json_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path.name}: se esperaba una lista JSON, no {type(data).__name__}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{path.name}: todos los elementos deben ser objetos JSON")
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_condition(df: pd.DataFrame) -> pd.Series:
    api = df.get("api", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    model = df.get("model", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    return model + " + " + api


def problem_id(df: pd.DataFrame) -> pd.Series:
    dataset_col = "dataset_name_canonical" if "dataset_name_canonical" in df.columns else "dataset_name"
    return (
        df[dataset_col].astype(str)
        + "::q"
        + df.get("question_id", df["question_number"].astype(str).str.upper().str.strip())
    )


def row_id(df: pd.DataFrame) -> pd.Series:
    return (
        problem_id(df)
        + "::run"
        + df["run"].astype("Int64").astype(str)
        + "::"
        + canonical_condition(df)
    )


def detect_weak_reference(text: object) -> bool:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return True
    normalized = str(text).lower().strip()
    if len(normalized) < 40:
        return True
    return any(re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL) for pattern in WEAK_REFERENCE_PATTERNS)


def infer_dataset_name_from_source(source_file: object) -> str | None:
    if source_file is None or (isinstance(source_file, float) and np.isnan(source_file)):
        return None
    name = Path(str(source_file)).name
    match = re.match(r"resultados_(.+?)_\d{8}_\d{6}_[0-9a-fA-F]+(?:\(\d+\))?\.json$", name)
    return match.group(1) if match else None


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EXPECTED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    numeric_cols = [
        "run", *METRICS, "execution_time_seconds",
        "solver_input_tokens", "solver_output_tokens", "solver_total_tokens",
        "reference_input_tokens", "reference_output_tokens", "reference_total_tokens",
        "judge_input_tokens", "judge_output_tokens", "judge_total_tokens",
        "total_input_tokens", "total_output_tokens", "total_tokens",
        "estimated_cost_usd",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["question_number"] = (
        out["question_number"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    out["question_id"] = (
        out["question_number"]
        .str.upper()
    )

    out["dataset_name_reported"] = out["dataset_name"]
    inferred = out.get("source_file", pd.Series(pd.NA, index=out.index)).map(infer_dataset_name_from_source)
    out["dataset_name_from_file"] = inferred
    out["dataset_name_mismatch"] = inferred.notna() & out["dataset_name"].notna() & (inferred != out["dataset_name"].astype(str))
    out["dataset_name_canonical"] = inferred.fillna(out["dataset_name"].astype(str))
    out["dataset_name"] = out["dataset_name_canonical"]

    out["technical_error"] = (
        out["technical_error"]
        .fillna(False)
        .map(lambda x: str(x).strip().lower() in {"true", "1", "yes", "si", "sí"})
    )
    out["condition"] = canonical_condition(out)
    out["problem_id"] = problem_id(out)
    out["row_id"] = row_id(out)
    out["weak_reference_auto"] = out["reference_solution"].map(detect_weak_reference)
    out["valid_execution"] = ~out["technical_error"]
    out["valid_for_primary_analysis"] = out["valid_execution"] & ~out["weak_reference_auto"]
    out["valid_for_sensitivity_analysis"] = out["valid_execution"]
    return out


FINAL_RESULTS_RE = re.compile(
    r"^resultados_(?!.*_incremental\.json$).+_\d{8}_\d{6}_[0-9a-fA-F]+(?:\(\d+\))?\.json$"
)


def discover_final_result_files(input_dir: Path) -> list[Path]:
    """Localiza solo JSON finales de v8 y excluye incrementales y auxiliares."""
    candidates = sorted(input_dir.glob("resultados_*.json"))
    return [path for path in candidates if FINAL_RESULTS_RE.match(path.name)]


def load_v8_companion_files(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga inventarios resumidos de config_*.json y referencias_comunes_*.json."""
    configs: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("config_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            configs.append({
                "source_file": path.name,
                "experiment_hash": data.get("experiment_hash"),
                "dataset_name": data.get("dataset_name"),
                "base_name": data.get("base_name"),
                "timestamp": data.get("timestamp"),
                "generated_at_utc": data.get("generated_at_utc"),
                "problem_pdf_sha256": data.get("problem_pdf_sha256"),
                "solution_pdf_sha256": data.get("solution_pdf_sha256"),
                "environment_json": json.dumps(data.get("environment", {}), ensure_ascii=False),
                "experiment_config_json": json.dumps(data.get("experiment_config", {}), ensure_ascii=False),
                "inference_parameters_json": json.dumps(data.get("inference_parameters", {}), ensure_ascii=False),
            })
        except Exception as exc:
            configs.append({"source_file": path.name, "load_error": str(exc)})
    for path in sorted(input_dir.glob("referencias_comunes_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                for key, item in data.items():
                    if isinstance(item, dict):
                        references.append({"source_file": path.name, "reference_key": key, **item})
            else:
                references.append({"source_file": path.name, "load_error": "Formato no dict"})
        except Exception as exc:
            references.append({"source_file": path.name, "load_error": str(exc)})
    return pd.DataFrame(configs), pd.DataFrame(references)


def load_json_folder(input_dir: Path, pattern: str = "resultados_*.json") -> tuple[pd.DataFrame, pd.DataFrame]:
    files = discover_final_result_files(input_dir) if pattern == "resultados_*.json" else sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron JSON finales de resultados compatibles con v8 en {input_dir}")

    file_rows: list[dict] = []
    frames: list[pd.DataFrame] = []
    seen_hashes: set[str] = set()

    for path in files:
        sha = file_sha256(path)
        duplicated_file = sha in seen_hashes
        records = read_json_records(path)
        file_rows.append({
            "source_file": path.name,
            "sha256": sha,
            "file_size_bytes": path.stat().st_size,
            "record_count": len(records),
            "duplicate_file_content": duplicated_file,
        })
        if duplicated_file:
            continue
        seen_hashes.add(sha)
        frame = pd.DataFrame(records)
        frame["source_file"] = path.name
        frame["source_file_sha256"] = sha
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True, sort=False)
    normalized = normalize_dataframe(raw)
    file_inventory = pd.DataFrame(file_rows)
    return normalized, file_inventory


def deduplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    duplicate_mask = df.duplicated(subset=["row_id"], keep=False)
    duplicates = df.loc[duplicate_mask].sort_values(["row_id", "source_file"]).copy()
    clean = df.drop_duplicates(subset=["row_id"], keep="first").copy()
    return clean, duplicates


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_excel_value(value: object) -> object:
    if isinstance(value, str):
        # Excel/OpenXML no admite caracteres de control salvo tabulador, LF y CR.
        return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", value)
    return value


def write_excel_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            safe_name = name[:31]
            export = frame.copy()
            object_cols = export.select_dtypes(include=["object", "string"]).columns
            for col in object_cols:
                export[col] = export[col].map(_sanitize_excel_value)
            export.to_excel(writer, sheet_name=safe_name, index=False)


def confidence_interval_mean(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float]:
    from scipy import stats

    series = pd.Series(values, dtype="float64").dropna()
    n = len(series)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        value = float(series.iloc[0])
        return value, value
    mean = float(series.mean())
    sem = float(stats.sem(series))
    margin = float(stats.t.ppf((1 + confidence) / 2, n - 1) * sem)
    return mean - margin, mean + margin


def bootstrap_ci_mean(values: Iterable[float], n_boot: int = 5000, seed: int = 20260716) -> tuple[float, float]:
    arr = np.asarray(pd.Series(values, dtype="float64").dropna(), dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """
    Ajusta p-valores mediante el procedimiento secuencial de Holm-Bonferroni.

    La implementación es autónoma y no depende de ``statsmodels``. Conserva
    los valores no finitos como ``NaN`` y devuelve los resultados en el mismo
    orden que la secuencia de entrada.
    """
    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(p)
    valid_p = p[valid_mask]

    if valid_p.size == 0:
        return adjusted

    # Holm: ordenar de menor a mayor y multiplicar cada p por el número de
    # hipótesis que permanecen. Después se aplica el máximo acumulado para
    # asegurar monotonía y se limita el resultado al intervalo [0, 1].
    order = np.argsort(valid_p, kind="stable")
    ordered = valid_p[order]
    m = ordered.size
    factors = np.arange(m, 0, -1, dtype=float)
    ordered_adjusted = np.maximum.accumulate(ordered * factors)
    ordered_adjusted = np.clip(ordered_adjusted, 0.0, 1.0)

    restored = np.empty_like(ordered_adjusted)
    restored[order] = ordered_adjusted
    adjusted[valid_mask] = restored
    return adjusted
