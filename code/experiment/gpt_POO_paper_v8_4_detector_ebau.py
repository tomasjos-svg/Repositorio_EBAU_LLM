# -*- coding: utf-8 -*-
"""
gpt_POO_paper_v8_4_detector_ebau.py

Aplicacion experimental para comparar modelos GPT en la resolucion y evaluacion
educativa de problemas matematicos.

Version 8.4 con detector universal reforzado para formatos EBAU, seleccion de modelo dentro de PDFs recopilatorios, validacion cruzada entre enunciados y soluciones, confirmacion previa, diagnostico de deteccion, soporte para PDF sin capa de texto y control de resultados vacios.

Incorpora las recomendaciones metodologicas:
1. Diseno factorial API x modelo para desacoplar efecto de API y efecto de modelo.
2. Evaluador comun externo, fijo y auditado.
3. Metricas y contrastes prerregistrados: media, mediana, desviacion tipica,
   IC95, >=80, perfectos, ceros, tiempo, errores tecnicos, t pareada,
   Wilcoxon, Welch, Levene y Bonferroni.
4. Trazabilidad completa de prompts, configuracion, hashes y logs.
5. Metricas educativas: completitud, correccion simbolica, fidelidad al enunciado,
   calidad explicativa, exactitud por apartados y tipo de error.
6. Estratificacion por corpus, dominio y nivel educativo.
7. Coste y eficiencia: tokens, coste estimado, latencia media, mediana y p95.

No incluye revision manual ni automatica de calidad de PDFs como paso previo.

IMPORTANTE:
- No escribas la API key dentro de este archivo.
- Define OPENAI_API_KEY como variable de entorno.
- En Windows PowerShell:
    setx OPENAI_API_KEY "tu_clave"
- Luego abre una terminal nueva.
"""

import csv
import hashlib
import json
import logging
import math
import os
import subprocess
import importlib.metadata as importlib_metadata
import openai
os.environ["OPENAI_API_KEY"] = "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"

import platform
import queue
import re
import statistics
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from openai import OpenAI
from pypdf import PdfReader


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "No se encontro OPENAI_API_KEY. Define la variable de entorno antes de ejecutar."
    )

OUTPUT_DIR = "resultados_experimento"
LOG_DIR = "logs_experimento"

N_RUNS = 3
POLL_SECONDS = 2
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
MAX_QUESTIONS_FOR_TEST = None

# Si un PDF contiene varios modelos de examen completos, selecciona cual usar.
# 1 = primer modelo, 2 = segundo modelo, etc. En el PDF UCLM aportado,
# el modelo 2023/2024 es el primero y el 2022/2023 es el segundo.
EXAM_SECTION_INDEX = 1

UNSTABLE_STD_THRESHOLD = 25.0
ALPHA = 0.05

# Modelo comun externo para evaluar todas las ramas.
# Puedes cambiarlo por otro juez fijo, pero debe ser el mismo durante todo el experimento.
JUDGE_MODEL = "gpt-5.5"

# Esfuerzo de razonamiento por tipo de tarea en Responses API.
REASONING_EFFORT_SOLVER = "high"
REASONING_EFFORT_REFERENCE = "low"
REASONING_EFFORT_JUDGE = "low"
TEXT_VERBOSITY = "low"

# Diseno factorial. Si una combinacion no esta disponible en tu cuenta/API,
# se registrara el error y el experimento continuara con las demas condiciones.
EXPERIMENT_MATRIX = [
    {"api": "assistants", "model": "gpt-4-turbo", "enabled": True},
    {"api": "responses", "model": "gpt-4-turbo", "enabled": True},
    {"api": "assistants", "model": "gpt-5.4", "enabled": True},
    {"api": "responses", "model": "gpt-5.4", "enabled": True},
]

# Precios aproximados por token. Sustituye estos valores por los precios oficiales
# aplicables en la fecha del experimento. Se guardan para trazabilidad.
MODEL_PRICING_USD_PER_TOKEN = {
    "gpt-4-turbo": {"input": 0.000010, "output": 0.000030},
    "gpt-5.5": {"input": 0.000002, "output": 0.000008},
    "gpt-5.4": {"input": 0.000005, "output": 0.000015},
}

ERROR_TYPES = [
    "none",
    "algebraic_error",
    "reasoning_error",
    "hallucination",
    "incomplete_solution",
    "notation_error",
    "wrong_interpretation",
    "calculation_error",
    "formatting_error",
    "no_answer",
    "technical_error",
    "weak_reference",
]

DATASET_METADATA = {
    "demidovich_i": {"domain": "calculus", "education_level": "university"},
    "demidovich_ii": {"domain": "calculus", "education_level": "university"},
    "algebra_lineal_i": {"domain": "linear_algebra", "education_level": "university"},
    "algebra_lineal_iii": {"domain": "linear_algebra", "education_level": "university"},
    "expresiones_algebraicas": {"domain": "elementary_algebra", "education_level": "secondary"},
    "examen_2009_01_20": {"domain": "official_exam", "education_level": "university"},
    "examen_2009_02_06": {"domain": "official_exam", "education_level": "university"},
    "asturias_2023": {"domain": "official_exam", "education_level": "secondary"},
    "andalucia_2023": {"domain": "official_exam", "education_level": "secondary"},
    "unknown": {"domain": "unknown", "education_level": "unknown"},
}

EXPERIMENT_CONFIG = {
    "n_runs": N_RUNS,
    "poll_seconds": POLL_SECONDS,
    "max_retries": MAX_RETRIES,
    "retry_delay_seconds": RETRY_DELAY_SECONDS,
    "max_questions_for_test": MAX_QUESTIONS_FOR_TEST,
    "unstable_std_threshold": UNSTABLE_STD_THRESHOLD,
    "alpha": ALPHA,
    "judge_model": JUDGE_MODEL,
    "reasoning_effort_solver": REASONING_EFFORT_SOLVER,
    "reasoning_effort_reference": REASONING_EFFORT_REFERENCE,
    "reasoning_effort_judge": REASONING_EFFORT_JUDGE,
    "text_verbosity": TEXT_VERBOSITY,
    "experiment_matrix": EXPERIMENT_MATRIX,
    "model_pricing_usd_per_token": MODEL_PRICING_USD_PER_TOKEN,
    "error_types": ERROR_TYPES,
}

EXPERIMENT_HASH = hashlib.sha256(
    json.dumps(EXPERIMENT_CONFIG, sort_keys=True, ensure_ascii=False).encode("utf-8")
).hexdigest()


# ============================================================
# LOGGING
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

logging.info("Experiment hash: %s", EXPERIMENT_HASH)
logging.info("Experiment config: %s", json.dumps(EXPERIMENT_CONFIG, ensure_ascii=False))


# ============================================================
# FUNCIONES DE UTILIDAD
# ============================================================


def slugify(value: str) -> str:
    value = value.lower()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"



def infer_dataset_name(problem_pdf_path: str) -> str:
    name = slugify(os.path.splitext(os.path.basename(problem_pdf_path))[0])

    if "demidovich" in name and ("_i" in name or "1" in name):
        return "demidovich_i"
    if "demidovich" in name and ("_ii" in name or "2" in name):
        return "demidovich_ii"
    if "lineal" in name and "iii" in name:
        return "algebra_lineal_iii"
    if "lineal" in name and "i" in name:
        return "algebra_lineal_i"
    if "expresiones" in name or "algebraicas" in name:
        return "expresiones_algebraicas"
    if "20_01_2009" in name or "20_1_2009" in name:
        return "examen_2009_01_20"
    if "6_02_2009" in name or "06_02_2009" in name or "6_2_2009" in name:
        return "examen_2009_02_06"
    if "asturias" in name:
        return "asturias_2023"
    if "andalucia" in name:
        return "andalucia_2023"

    return name if name in DATASET_METADATA else "unknown"



def clean_pdf_text(text: Optional[str]) -> str:
    """Limpieza determinista minima del texto extraido del PDF.

    No es una auditoria de calidad del PDF: solo normaliza caracteres conflictivos.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)

    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "": "pi",
        "π": "pi",
        "": "=>",
        "": "infinito",
        "≤": "<=",
        "≥": ">=",
        "×": "x",
        "·": "*",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = clean_pdf_text(text)
        pages.append(f"\n\n===== PAGINA {i} =====\n\n{text}")

    return "\n".join(pages)


def has_useful_text(text: str, minimum_alnum: int = 80) -> bool:
    """Indica si la extraccion local contiene texto util suficiente."""
    if not text:
        return False
    return sum(ch.isalnum() for ch in text) >= minimum_alnum


def select_exam_section(text: str, section_index: int = 1) -> Tuple[str, int, int]:
    """Aisla un modelo de examen cuando el PDF contiene varios examenes completos.

    Busca encabezados del tipo "Evaluacion para el Acceso a la Universidad" y
    conserva solamente el bloque elegido. Devuelve (texto, indice_elegido,
    numero_total_de_bloques). Si no hay varios bloques, devuelve el texto entero.
    """
    if not text:
        return text, 1, 1

    # La normalizacion NFKD de algunos PDF inserta espacios en palabras
    # acentuadas (por ejemplo, "Evaluacio n"). Por eso usamos como ancla
    # la parte estable del encabezado.
    header_pattern = re.compile(
        r"(?im)^.*para\s+el\s+Acceso\s+a\s+la\s+Universidad\s*$"
    )
    starts = [m.start() for m in header_pattern.finditer(text)]

    if len(starts) <= 1:
        return text, 1, 1

    blocks: List[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[start:end].strip()
        # Solo aceptar bloques que parecen examenes reales, no menciones aisladas.
        top_level = re.findall(r"(?im)^\s*([1-9]\d?)\s*[.]\s+", block)
        if len(set(top_level)) >= 4:
            blocks.append(block)

    if len(blocks) <= 1:
        return (blocks[0] if blocks else text), 1, max(1, len(blocks))

    chosen = max(1, min(int(section_index), len(blocks)))
    return blocks[chosen - 1], chosen, len(blocks)


def _normalize_question_id(raw: str) -> str:
    """Normaliza un identificador principal sin alterar su familia."""
    raw = re.sub(r"\s+", "", str(raw).upper())
    raw = raw.replace("EX-", "EX").replace("E-", "E").replace("P-", "P").replace("Q-", "Q")
    return raw


def _roman_to_int(value: str) -> Optional[int]:
    values = {"I": 1, "V": 5, "X": 10, "L": 50}
    value = value.upper()
    total = 0
    previous = 0
    for char in reversed(value):
        current = values.get(char)
        if current is None:
            return None
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total if 1 <= total <= 50 else None


def detect_question_ids_detailed(text: str) -> Tuple[List[str], Dict[str, Any]]:
    """Detector universal y trazable de ejercicios principales.

    Reconoce, entre otros, los formatos habituales de EBAU:
    ``Ejercicio A1``, ``A1``, ``A1:``, ``Ejercicio 1``, numeracion simple y
    numeros romanos. Los subapartados a), b), a.1), b.2), etc. no se convierten
    en ejercicios independientes.
    """
    diagnostics: Dict[str, Any] = {
        "candidates": [],
        "accepted": [],
        "discarded": [],
        "pattern_used": None,
        "warnings": [],
    }

    if not text:
        return [], diagnostics

    lines = text.splitlines()
    families = [
        # Ejercicio A1 / Problema B2 / Pregunta C3
        (
            "named_prefixed",
            re.compile(
                r"^\s*(?:ejercicio|problema|pregunta)\s*"
                r"((?:EX|E|P|Q|A|B|C)\s*-?\s*\d{1,3})\b\s*(?:[.):\-])?\s*(.*)$",
                re.I,
            ),
        ),
        # Ejercicio 1 / Problema 2 / Pregunta 3
        (
            "named_numeric",
            re.compile(
                r"^\s*(?:ejercicio|problema|pregunta)\s*(\d{1,3})\b\s*(?:[.):\-])?\s*(.*)$",
                re.I,
            ),
        ),
        # Ej. A1 / Prob. 2 / Preg. B3
        (
            "abbrev_prefixed",
            re.compile(
                r"^\s*(?:ej\.?|prob\.?|preg\.?)\s*"
                r"((?:EX|E|P|Q|A|B|C)\s*-?\s*\d{1,3})\b\s*(?:[.):\-])?\s*(.*)$",
                re.I,
            ),
        ),
        (
            "abbrev_numeric",
            re.compile(
                r"^\s*(?:ej\.?|prob\.?|preg\.?)\s*(\d{1,3})\b\s*(?:[.):\-])?\s*(.*)$",
                re.I,
            ),
        ),
        # A1 / A1: / A1. / A1) / EX-2
        (
            "prefixed",
            re.compile(
                r"^\s*((?:EX|E|P|Q|A|B|C)\s*-?\s*\d{1,3})"
                r"\s*(?:[.):\-]|$)\s*(.*)$",
                re.I,
            ),
        ),
        # 1. / 1) / 1: / 1.-
        (
            "numeric",
            re.compile(r"^\s*(\d{1,2})\s*(?:\.-|[.):\-])\s*(.*)$", re.I),
        ),
        # I. / II) / IV:
        (
            "roman",
            re.compile(
                r"^\s*((?:X{0,3})(?:IX|IV|V?I{1,3}))\s*(?:\.-|[.):\-])\s*(.*)$",
                re.I,
            ),
        ),
    ]

    by_family: Dict[str, List[Tuple[int, str, str]]] = {
        name: [] for name, _ in families
    }

    for line_index, line in enumerate(lines):
        lineno = line_index + 1
        stripped = line.strip()
        if not stripped:
            continue

        # Exclusiones fuertes: subapartados y filas de tablas numericas.
        if re.match(r"^[a-z]\s*[.)]", stripped, re.I) or re.match(
            r"^[a-z]\.\d+\s*[)]?", stripped, re.I
        ):
            continue
        if re.match(r"^\d+[,.]\d+\s+\d+[,.]\d+", stripped):
            continue

        for family, pattern in families:
            match = pattern.match(line)
            if not match:
                continue

            raw_id = match.group(1)
            tail = (match.group(2) or "").strip()
            norm = _normalize_question_id(raw_id)

            if family == "roman":
                number = _roman_to_int(norm)
                if number is None:
                    diagnostics["discarded"].append(
                        {"line": lineno, "text": line, "reason": "romano_invalido"}
                    )
                    continue
                norm = str(number)

            # Es frecuente que el encabezado este solo y el enunciado empiece en
            # la linea siguiente. Solo se descarta si tampoco hay contenido util
            # inmediatamente despues.
            if not tail:
                next_nonempty = ""
                for following in lines[line_index + 1 : line_index + 4]:
                    if following.strip():
                        next_nonempty = following.strip()
                        break
                if len(next_nonempty) < 3:
                    diagnostics["discarded"].append(
                        {"line": lineno, "text": line, "reason": "sin_contenido"}
                    )
                    continue

            by_family[family].append((lineno, norm, line.strip()))
            diagnostics["candidates"].append(
                {"family": family, "line": lineno, "id": norm, "text": line.strip()}
            )
            break

    # Elegimos la familia mas coherente: mas IDs unicos, mejor continuidad y
    # prioridad para encabezados explicitos frente a menciones incidentales.
    explicit_families = {
        "named_prefixed",
        "named_numeric",
        "abbrev_prefixed",
        "abbrev_numeric",
        "prefixed",
    }
    scored = []
    for family, items in by_family.items():
        ordered: List[str] = []
        seen = set()
        nums: List[int] = []
        for _, qid, _ in items:
            if qid not in seen:
                seen.add(qid)
                ordered.append(qid)
                number_match = re.search(r"(\d+)$", qid)
                if number_match:
                    nums.append(int(number_match.group(1)))
        if not ordered:
            continue

        continuity = 0
        if len(nums) >= 2:
            sorted_nums = sorted(set(nums))
            continuity = sum(
                1 for a, b in zip(sorted_nums, sorted_nums[1:]) if b - a == 1
            )

        score = (
            len(ordered),
            continuity,
            family in explicit_families,
            family.startswith("named"),
        )
        scored.append((score, family, ordered))

    if not scored:
        diagnostics["warnings"].append("No se encontraron encabezados fiables.")
        return [], diagnostics

    scored.sort(reverse=True)
    _, chosen_family, ordered = scored[0]
    diagnostics["pattern_used"] = chosen_family
    diagnostics["accepted"] = ordered

    if len(ordered) == 1:
        all_mentions = sorted(
            set(
                re.findall(
                    r"(?im)\b(?:EX|E|P|Q|A|B|C)\s*-?\s*(\d{1,2})\b", text
                )
            )
        )
        if len(all_mentions) > 1:
            diagnostics["warnings"].append(
                "Solo se acepto un ejercicio, pero aparecen referencias a varios "
                "identificadores: " + ", ".join(all_mentions)
            )

    nums = []
    for qid in ordered:
        number_match = re.search(r"(\d+)$", qid)
        if number_match:
            nums.append(int(number_match.group(1)))
    if len(nums) >= 3:
        sorted_nums = sorted(set(nums))
        missing = [
            n for n in range(sorted_nums[0], sorted_nums[-1] + 1)
            if n not in sorted_nums
        ]
        if missing:
            diagnostics["warnings"].append(
                "Numeracion no continua; faltan: " + ", ".join(map(str, missing))
            )

    return ordered, diagnostics


def detect_question_ids(text: str) -> List[str]:
    ids, _ = detect_question_ids_detailed(text)
    return ids


def write_detection_diagnostic(path: str, source_name: str, diagnostics: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"DIAGNOSTICO DEL DETECTOR - {source_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Patron elegido: {diagnostics.get('pattern_used')}\n")
        f.write("Ejercicios aceptados: " + ", ".join(diagnostics.get("accepted", [])) + "\n\n")
        if diagnostics.get("warnings"):
            f.write("ADVERTENCIAS\n")
            for w in diagnostics["warnings"]:
                f.write(f"- {w}\n")
            f.write("\n")
        f.write("CANDIDATOS\n")
        for item in diagnostics.get("candidates", []):
            f.write(f"L{item['line']:04d} | {item['family']} | {item['id']} | {item['text']}\n")
        if diagnostics.get("discarded"):
            f.write("\nDESCARTADOS\n")
            for item in diagnostics["discarded"]:
                f.write(f"L{item['line']:04d} | {item['reason']} | {item['text']}\n")



def extract_score(response_text: Any) -> int:
    if response_text is None:
        return 0

    match = re.search(r"-?\d+", str(response_text))
    if not match:
        return 0

    score = int(match.group())
    return max(0, min(100, score))



def strip_code_fences(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()



def parse_evaluation_json(text: Any) -> Dict[str, Any]:
    """Parsea la evaluacion educativa. Si falla, aplica fallback robusto."""

    raw = "" if text is None else str(text)
    cleaned = strip_code_fences(raw)

    default = {
        "score": extract_score(raw),
        "reason": raw,
        "completeness": None,
        "symbolic_correctness": None,
        "pedagogical_quality": None,
        "faithfulness_to_problem": None,
        "subpart_accuracy": None,
        "error_type": "formatting_error" if raw else "no_answer",
        "raw_evaluation": raw,
    }

    try:
        data = json.loads(cleaned)
    except Exception:
        return default

    def clamp_metric(value: Any, default_value: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default_value
        try:
            number = float(value)
            return max(0.0, min(100.0, number))
        except Exception:
            return default_value

    error_type = str(data.get("error_type", "none"))
    if error_type not in ERROR_TYPES:
        error_type = "formatting_error"

    return {
        "score": int(clamp_metric(data.get("score"), 0)),
        "reason": str(data.get("reason", "")),
        "completeness": clamp_metric(data.get("completeness")),
        "symbolic_correctness": clamp_metric(data.get("symbolic_correctness")),
        "pedagogical_quality": clamp_metric(data.get("pedagogical_quality")),
        "faithfulness_to_problem": clamp_metric(data.get("faithfulness_to_problem")),
        "subpart_accuracy": clamp_metric(data.get("subpart_accuracy")),
        "error_type": error_type,
        "raw_evaluation": raw,
    }



def safe_ci95(values: List[float]) -> Tuple[float, float]:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]

    mean_value = float(np.mean(values))
    sem = stats.sem(values)

    if np.isnan(sem) or sem == 0:
        return mean_value, mean_value

    low, high = stats.t.interval(0.95, len(values) - 1, loc=mean_value, scale=sem)
    return float(low), float(high)



def safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))



def safe_percentile(values: List[float], percentile: float) -> float:
    values = [float(v) for v in values if v is not None]
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))



def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING_USD_PER_TOKEN.get(model)
    if not pricing:
        return 0.0
    return float(input_tokens * pricing["input"] + output_tokens * pricing["output"])



def get_usage_dict(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }



def sha256_file(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def get_package_version(package_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def get_environment_metadata() -> Dict[str, Any]:
    script_path = os.path.abspath(__file__)
    return {
        "captured_at_utc": utc_now_iso(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "script_file": os.path.basename(script_path),
        "script_path": script_path,
        "script_sha256": sha256_file(script_path),
        "openai_python_sdk": getattr(openai, "__version__", None),
        "numpy_version": get_package_version("numpy"),
        "pandas_version": get_package_version("pandas"),
        "scipy_version": get_package_version("scipy"),
        "pypdf_version": get_package_version("pypdf"),
        "openpyxl_version": get_package_version("openpyxl"),
        "experiment_hash": EXPERIMENT_HASH,
        "log_path": LOG_PATH,
    }


# ============================================================
# PROMPTS
# ============================================================


def build_solver_prompt(question_number: int, source_text: Optional[str] = None) -> str:
    if source_text is None:
        source_block = "Usa exclusivamente el documento adjunto."
    else:
        source_block = f"A continuacion tienes el texto completo del documento de enunciados.\n\nDOCUMENTO:\n{source_text}"

    return f"""
{source_block}

Resuelve el ejercicio {question_number}.

Instrucciones:
- Da una solucion matematica completa.
- Explica los pasos esenciales.
- No consultes la solucion oficial.
- No inventes datos no presentes en el enunciado.
- Si el enunciado contiene apartados o subapartados anidados (a.1, a.2, b.1, b.2, etc.), resuelvelos todos.
""".strip()



def build_reference_prompt(question_number: int, source_text: Optional[str] = None) -> str:
    if source_text is None:
        source_block = "Usa exclusivamente el documento adjunto."
    else:
        source_block = f"A continuacion tienes el texto completo del documento de soluciones.\n\nDOCUMENTO DE SOLUCIONES:\n{source_text}"

    return f"""
{source_block}

Extrae la solucion oficial correspondiente al ejercicio {question_number}.

Instrucciones:
- No resuelvas el ejercicio por tu cuenta.
- Limitate a extraer o reconstruir fielmente la solucion de referencia.
- Si hay apartados o subapartados anidados (a.1, a.2, b.1, b.2, etc.), conserva toda la estructura.
- Si no encuentras la solucion, dilo claramente.
""".strip()



def build_question_prompt(question_number: int, source_text: Optional[str] = None) -> str:
    if source_text is None:
        source_block = "Usa exclusivamente el documento adjunto."
    else:
        source_block = f"A continuacion tienes el texto completo del documento de enunciados.\n\nDOCUMENTO DE ENUNCIADOS:\n{source_text}"

    return f"""
{source_block}

Extrae literalmente, o reconstruye con la maxima fidelidad posible, el enunciado completo del ejercicio {question_number}.

Instrucciones:
- No resuelvas el ejercicio.
- Conserva todos los datos, simbolos, condiciones, apartados y subapartados anidados (a.1, a.2, b.1, b.2, etc.).
- No anadas informacion que no aparezca en el documento.
- Si no encuentras el enunciado, indicalo claramente.
""".strip()


def build_evaluation_prompt(question_text: str, model_solution: str, reference_solution: str) -> str:
    return f"""
Actuas exclusivamente como evaluador externo de una respuesta matematica.

ENUNCIADO ORIGINAL:
{question_text}

RESPUESTA CANDIDATA:
{model_solution}

SOLUCION DE REFERENCIA COMUN:
{reference_solution}

Evalua la respuesta candidata respecto al enunciado original y a la solucion de referencia.

Instrucciones metodologicas:
- No resuelvas un problema distinto.
- No corrijas ni reescribas la respuesta candidata.
- No favorezcas un estilo de redaccion concreto ni el estilo de ningun modelo.
- Acepta procedimientos alternativos matematicamente equivalentes.
- Penaliza errores algebraicos, numericos, simbolicos o de razonamiento.
- Penaliza apartados omitidos, datos inventados y conclusiones incompatibles.
- Usa la solucion de referencia como apoyo, pero verifica la coherencia con el enunciado.

Devuelve exclusivamente JSON valido, sin Markdown, con esta estructura exacta:

{{
  "score": 0,
  "reason": "justificacion concreta y verificable",
  "completeness": 0,
  "symbolic_correctness": 0,
  "pedagogical_quality": 0,
  "faithfulness_to_problem": 0,
  "subpart_accuracy": 0,
  "error_type": "none"
}}

Criterios:
- score: correccion matematica global entre 0 y 100.
- completeness: proporcion del problema correctamente resuelta entre 0 y 100.
- symbolic_correctness: correccion formal, algebraica, numerica y simbolica entre 0 y 100.
- pedagogical_quality: claridad, coherencia y utilidad educativa entre 0 y 100.
- faithfulness_to_problem: fidelidad al enunciado original entre 0 y 100.
- subpart_accuracy: exactitud considerando separadamente los apartados entre 0 y 100.
- error_type debe ser uno de: {', '.join(ERROR_TYPES)}.

Reglas orientativas de puntuacion:
- 100: solucion completa y matematicamente equivalente.
- 80-99: esencialmente correcta, con defectos menores.
- 50-79: avance sustancial, pero con errores u omisiones relevantes.
- 1-49: respuesta parcialmente relacionada, pero mayoritariamente incorrecta.
- 0: no responde, resuelve otro problema o inventa el enunciado.
""".strip()


# ============================================================
# CLASES AUXILIARES
# ============================================================

@dataclass
class BackendResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    endpoint: Optional[str] = None
    response_id: Optional[str] = None
    response_model: Optional[str] = None
    response_created: Optional[Any] = None
    system_fingerprint: Optional[str] = None
    request_parameters: Optional[Dict[str, Any]] = None
    retried_without_reasoning: bool = False
    request_started_at_utc: Optional[str] = None
    request_finished_at_utc: Optional[str] = None
    assistant_id: Optional[str] = None
    thread_id: Optional[str] = None
    run_id: Optional[str] = None


# ============================================================
# GUI
# ============================================================

class GUIManager:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Benchmark GPT matematicas - version metodologica")
        self.root.geometry("1050x700")

        self.message_queue = queue.Queue()
        self.task_manager = TaskManager()

        self.create_widgets()
        self.process_queue()

    def create_widgets(self):
        text_frame = tk.Frame(self.root)
        text_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.text_scroll = tk.Scrollbar(text_frame)
        self.text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_area = tk.Text(
            text_frame,
            height=32,
            width=125,
            yscrollcommand=self.text_scroll.set,
        )
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text_scroll.config(command=self.text_area.yview)

        button_frame = tk.Frame(self.root)
        button_frame.pack(padx=10, pady=10, fill=tk.X)

        self.run_button = tk.Button(
            button_frame,
            text="Ejecutar benchmark metodologico",
            command=self.start_experiment,
            height=2,
        )
        self.run_button.pack(side=tk.TOP, fill=tk.X)

        tk.Button(
            button_frame,
            text="Salir",
            command=self.salir,
            height=2,
        ).pack(side=tk.TOP, fill=tk.X)

    def start_experiment(self):
        problem_pdf = filedialog.askopenfilename(
            parent=self.root,
            title="Selecciona PDF de enunciados",
            filetypes=[("PDF files", "*.pdf")],
        )
        if not problem_pdf:
            self.display_message_threadsafe("No se selecciono PDF de enunciados.")
            return

        solution_pdf = filedialog.askopenfilename(
            parent=self.root,
            title="Selecciona PDF de soluciones",
            filetypes=[("PDF files", "*.pdf")],
        )
        if not solution_pdf:
            self.display_message_threadsafe("No se selecciono PDF de soluciones.")
            return

        inferred_dataset = infer_dataset_name(problem_pdf)
        dataset_name = simpledialog.askstring(
            "Nombre del dataset",
            "Nombre corto del dataset:",
            initialvalue=inferred_dataset,
            parent=self.root,
        ) or inferred_dataset
        dataset_name = slugify(dataset_name)

        self.run_button.config(state=tk.DISABLED)
        self.task_manager.run_async(
            self.display_message_threadsafe,
            problem_pdf,
            solution_pdf,
            dataset_name,
            self.on_experiment_finished,
        )

    def on_experiment_finished(self, error_message: Optional[str] = None):
        def finish_on_main_thread():
            self.run_button.config(state=tk.NORMAL)
            if error_message:
                messagebox.showerror("Error", error_message, parent=self.root)

        self.root.after(0, finish_on_main_thread)

    def display_message_threadsafe(self, message: str):
        self.message_queue.put(message)

    def process_queue(self):
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.text_area.insert(tk.END, message + "\n")
                self.text_area.see(tk.END)
        except queue.Empty:
            pass

        self.root.after(200, self.process_queue)

    def salir(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ============================================================
# TASK MANAGER
# ============================================================

class TaskManager:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY)
        self.assistants_backend = AssistantsMathBackend(self.client)
        self.responses_backend = ResponsesMathBackend(self.client)
        self.judge_backend = ResponsesMathBackend(self.client)

    def safe_call(self, func, *args, **kwargs):
        last_exception = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logging.exception("Attempt %s/%s failed in %s", attempt, MAX_RETRIES, getattr(func, "__name__", str(func)))

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)

        raise last_exception

    def run_async(self, display_func, problem_pdf: str, solution_pdf: str, dataset_name: str, finished_callback):
        thread = threading.Thread(
            target=self.run_experiment,
            args=(display_func, problem_pdf, solution_pdf, dataset_name, finished_callback),
            daemon=True,
        )
        thread.start()

    def run_experiment(self, display, problem_pdf: str, solution_pdf: str, dataset_name: str, finished_callback):
        error_message = None
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            problem_pdf_sha256 = sha256_file(problem_pdf)
            solution_pdf_sha256 = sha256_file(solution_pdf)

            metadata = DATASET_METADATA.get(dataset_name, DATASET_METADATA["unknown"])
            domain = metadata["domain"]
            education_level = metadata["education_level"]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = slugify(os.path.splitext(os.path.basename(problem_pdf))[0])

            run_id = f"{dataset_name}_{timestamp}_{EXPERIMENT_HASH[:10]}"

            incremental_json_path = os.path.join(OUTPUT_DIR, f"resultados_{run_id}_incremental.json")
            incremental_csv_path = os.path.join(OUTPUT_DIR, f"resultados_{run_id}_incremental.csv")

            display(f"Dataset: {dataset_name} | Dominio: {domain} | Nivel: {education_level}")
            display(f"Hash experimental: {EXPERIMENT_HASH}")

            display("Subiendo PDF de enunciados para Assistants API...")
            problem_file_id = self.safe_call(self.assistants_backend.upload_file, problem_pdf)

            display("Subiendo PDF de soluciones para Assistants API...")
            solution_file_id = self.safe_call(self.assistants_backend.upload_file, solution_pdf)

            display("Extrayendo texto local de enunciados para Responses API...")
            problem_text = extract_pdf_text(problem_pdf)

            display("Extrayendo texto local de soluciones para Responses API...")
            solution_text = extract_pdf_text(solution_pdf)

            # Algunos documentos recopilan varios modelos completos. Si se procesa
            # todo el PDF, "ejercicio 1" puede referirse a mas de un examen y el
            # extractor puede mezclar enunciados. Aislamos el modelo seleccionado.
            problem_text, problem_section, problem_section_count = select_exam_section(
                problem_text, EXAM_SECTION_INDEX
            )
            solution_text, solution_section, solution_section_count = select_exam_section(
                solution_text, EXAM_SECTION_INDEX
            )

            if problem_section_count > 1:
                display(
                    f"PDF de enunciados: {problem_section_count} modelos completos detectados; "
                    f"se usara el modelo {problem_section}."
                )
            if solution_section_count > 1:
                display(
                    f"PDF de soluciones: {solution_section_count} modelos completos detectados; "
                    f"se usara el modelo {solution_section}."
                )

            display("Detectando ejercicios...")

            # Muchos examenes oficiales son PDF escaneados y pypdf no puede extraer
            # su texto. En ese caso detectamos los identificadores en el documento
            # de soluciones, que normalmente reproduce los enunciados completos.
            detection_text = problem_text if has_useful_text(problem_text) else solution_text
            if not has_useful_text(problem_text):
                display(
                    "AVISO: el PDF de enunciados no contiene una capa de texto util. "
                    "Se usara el documento de soluciones solo para identificar y extraer "
                    "los enunciados comunes; las soluciones candidatas no recibiran el "
                    "contenido de las soluciones oficiales."
                )

            question_ids, problem_detection = detect_question_ids_detailed(detection_text)
            solution_ids, solution_detection = detect_question_ids_detailed(solution_text)

            diagnostic_problem_path = os.path.join(OUTPUT_DIR, f"diagnostico_detector_enunciados_{run_id}.txt")
            diagnostic_solution_path = os.path.join(OUTPUT_DIR, f"diagnostico_detector_soluciones_{run_id}.txt")
            write_detection_diagnostic(diagnostic_problem_path, "ENUNCIADOS", problem_detection)
            write_detection_diagnostic(diagnostic_solution_path, "SOLUCIONES", solution_detection)

            # Validacion cruzada. Si el documento de enunciados no es util, se adopta
            # la deteccion de soluciones; si ambos son utiles y discrepan, se avisa.
            if not question_ids and solution_ids:
                question_ids = solution_ids
                display("La numeracion se ha tomado del PDF de soluciones.")
            elif question_ids and solution_ids and question_ids != solution_ids:
                display(
                    "ADVERTENCIA: la deteccion de enunciados y soluciones no coincide. "
                    f"Enunciados={question_ids}; soluciones={solution_ids}."
                )

            # Fallback semantico solo cuando el detector estructural no encuentra nada.
            if not question_ids:
                detected_count = self.safe_call(
                    self.responses_backend.get_num_questions, detection_text
                )
                if detected_count > 0:
                    question_ids = [str(i) for i in range(1, detected_count + 1)]

            if MAX_QUESTIONS_FOR_TEST is not None:
                question_ids = question_ids[:MAX_QUESTIONS_FOR_TEST]

            if not question_ids:
                raise RuntimeError(
                    "No se detecto ningun ejercicio. Comprueba que el PDF de enunciados "
                    "o el de soluciones contenga texto legible o encabezados identificables."
                )

            display(
                f"Ejercicios detectados ({len(question_ids)}): "
                + ", ".join(question_ids)
            )
            display(f"Patron principal: {problem_detection.get('pattern_used') or solution_detection.get('pattern_used')}")
            for warning in problem_detection.get("warnings", []) + solution_detection.get("warnings", []):
                display("ADVERTENCIA DEL DETECTOR: " + warning)
            display(f"Diagnostico: {diagnostic_problem_path}")
            display(f"Diagnostico: {diagnostic_solution_path}")

            confirmation = messagebox.askyesno(
                "Confirmar ejercicios",
                "Se han detectado los siguientes ejercicios:\n\n"
                + ", ".join(question_ids)
                + "\n\nTotal: " + str(len(question_ids))
                + "\n\n¿Deseas continuar con el benchmark?",
                parent=None,
            )
            if not confirmation:
                raise RuntimeError("Ejecucion cancelada por el usuario tras revisar la deteccion.")

            display("Generando enunciados y soluciones de referencia comunes...")
            common_references = self.generate_common_references(
                problem_text=(problem_text if has_useful_text(problem_text) else solution_text),
                solution_text=solution_text,
                question_ids=question_ids,
                display=display,
            )

            references_path = os.path.join(OUTPUT_DIR, f"referencias_comunes_{run_id}.json")
            self.save_json(common_references, references_path)
            display(f"Referencias comunes guardadas en: {references_path}")

            experiments = self.build_experiments(problem_file_id, solution_file_id, problem_text, solution_text)

            all_results = []

            for experiment in experiments:
                label = experiment["label"]
                backend = experiment["backend"]
                model = experiment["model"]
                api_name = experiment["api"]

                display("")
                display("=" * 80)
                display(f"Ejecutando condicion: {label}")
                display("=" * 80)
                logging.info("Starting condition %s", label)

                for run_index in range(1, N_RUNS + 1):
                    display(f"RUN {run_index}/{N_RUNS} - {label}")

                    for question_number in question_ids:
                        start_time = time.time()
                        display(f"Ejercicio {question_number}: resolviendo...")

                        common_reference = common_references[str(question_number)]
                        question_text = common_reference["question_text"]

                        # Todos los solucionadores reciben exactamente el mismo enunciado
                        # comun ya extraido. Asi se evita que una API vea un PDF escaneado
                        # y otra una transcripcion distinta.
                        solver_prompt = build_solver_prompt(
                            question_number,
                            question_text,
                        )
                        reference_solution = common_reference["reference_solution"]
                        reference_prompt = common_reference["reference_prompt"]

                        model_solution = ""
                        evaluation_data = {}

                        solver_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost": 0.0}
                        reference_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost": 0.0}
                        judge_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost": 0.0}
                        solver_result = BackendResult(text="")
                        reference_result = BackendResult(text="")
                        judge_result = BackendResult(text="")

                        try:
                            solver_result = self.safe_call(
                                backend.solve_problem,
                                source=experiment["problem_source"],
                                question_number=question_number,
                                model=model,
                                prompt=solver_prompt,
                            )
                            model_solution = solver_result.text
                            solver_usage = self.usage_with_cost(model, solver_result)

                            display(f"Ejercicio {question_number}: evaluando con juez comun {JUDGE_MODEL}...")
                            evaluation_prompt = build_evaluation_prompt(
                                question_text=question_text,
                                model_solution=model_solution,
                                reference_solution=reference_solution,
                            )
                            judge_result = self.safe_call(
                                self.evaluate_with_common_judge,
                                evaluation_prompt=evaluation_prompt,
                            )
                            evaluation_data = parse_evaluation_json(judge_result.text)
                            judge_usage = self.usage_with_cost(JUDGE_MODEL, judge_result)

                            elapsed = time.time() - start_time

                            result = self.build_result_row(
                                run_id=run_id,
                                timestamp=timestamp,
                                dataset_name=dataset_name,
                                domain=domain,
                                education_level=education_level,
                                api_name=api_name,
                                model=model,
                                label=label,
                                run_index=run_index,
                                question_number=question_number,
                                question_text=question_text,
                                solver_prompt=solver_prompt,
                                reference_prompt=reference_prompt,
                                evaluation_prompt=evaluation_prompt,
                                model_solution=model_solution,
                                reference_solution=reference_solution,
                                evaluation_data=evaluation_data,
                                elapsed=elapsed,
                                solver_usage=solver_usage,
                                reference_usage=reference_usage,
                                judge_usage=judge_usage,
                                solver_result=solver_result,
                                reference_result=reference_result,
                                judge_result=judge_result,
                                problem_pdf_sha256=problem_pdf_sha256,
                                solution_pdf_sha256=solution_pdf_sha256,
                                error=None,
                            )

                        except Exception as e:
                            elapsed = time.time() - start_time
                            logging.exception("Technical error in %s run %s question %s", label, run_index, question_number)

                            evaluation_prompt = build_evaluation_prompt(question_text, model_solution, reference_solution)
                            evaluation_data = {
                                "score": 0,
                                "reason": str(e),
                                "completeness": 0,
                                "symbolic_correctness": 0,
                                "pedagogical_quality": 0,
                                "faithfulness_to_problem": 0,
                                "subpart_accuracy": 0,
                                "error_type": "technical_error",
                                "raw_evaluation": str(e),
                            }

                            result = self.build_result_row(
                                run_id=run_id,
                                timestamp=timestamp,
                                dataset_name=dataset_name,
                                domain=domain,
                                education_level=education_level,
                                api_name=api_name,
                                model=model,
                                label=label,
                                run_index=run_index,
                                question_number=question_number,
                                question_text=question_text,
                                solver_prompt=solver_prompt,
                                reference_prompt=reference_prompt,
                                evaluation_prompt=evaluation_prompt,
                                model_solution=model_solution,
                                reference_solution=reference_solution,
                                evaluation_data=evaluation_data,
                                elapsed=elapsed,
                                solver_usage=solver_usage,
                                reference_usage=reference_usage,
                                judge_usage=judge_usage,
                                solver_result=solver_result,
                                reference_result=reference_result,
                                judge_result=judge_result,
                                problem_pdf_sha256=problem_pdf_sha256,
                                solution_pdf_sha256=solution_pdf_sha256,
                                error=str(e),
                            )

                        all_results.append(result)
                        self.save_json(all_results, incremental_json_path)
                        self.save_csv(all_results, incremental_csv_path)

                        display(
                            f"{label} | Run {run_index} | Ejercicio {question_number} | "
                            f"Score={result['score']} | Error={result['technical_error']} | "
                            f"Tiempo={result['execution_time_seconds']:.2f}s"
                        )

            if not all_results:
                raise RuntimeError(
                    "No se genero ningun resultado experimental. La ejecucion se detiene "
                    "antes del analisis estadistico para evitar tablas sin columnas."
                )

            outputs = self.generate_output_documents(
                results=all_results,
                output_dir=OUTPUT_DIR,
                run_id=run_id,
                base_name=base_name,
                timestamp=timestamp,
                dataset_name=dataset_name,
            )

            display("")
            display("=" * 80)
            display("DOCUMENTOS GENERADOS")
            display("=" * 80)
            for key, path in outputs.items():
                display(f"{key.upper()}: {path}")

            display("Experimento finalizado.")

        except Exception as e:
            logging.exception("Global error")
            error_message = str(e)
            display(f"ERROR GLOBAL: {error_message}")
        finally:
            finished_callback(error_message)

    def generate_common_references(self, problem_text: str, solution_text: str, question_ids: List[str], display) -> Dict[str, Dict[str, Any]]:
        references: Dict[str, Dict[str, Any]] = {}

        total_questions = len(question_ids)
        for position, question_number in enumerate(question_ids, start=1):
            display(f"Referencia comun {position}/{total_questions} ({question_number}): extrayendo enunciado...")
            question_prompt = build_question_prompt(question_number, problem_text)
            question_result = self.safe_call(
                self.responses_backend.extract_question,
                source=problem_text,
                question_number=question_number,
                model=JUDGE_MODEL,
                prompt=question_prompt,
            )

            display(f"Referencia comun {position}/{total_questions} ({question_number}): extrayendo solucion oficial...")
            reference_prompt = build_reference_prompt(question_number, solution_text)
            reference_result = self.safe_call(
                self.responses_backend.extract_reference_solution,
                source=solution_text,
                question_number=question_number,
                model=JUDGE_MODEL,
                prompt=reference_prompt,
            )

            references[str(question_number)] = {
                "question_number": question_number,
                "question_text": question_result.text,
                "reference_solution": reference_result.text,
                "question_prompt": question_prompt,
                "reference_prompt": reference_prompt,
                "extractor_model": JUDGE_MODEL,
                "question_response_id": question_result.response_id,
                "reference_response_id": reference_result.response_id,
            }

        return references

    def build_experiments(self, problem_file_id: str, solution_file_id: str, problem_text: str, solution_text: str):
        experiments = []

        for condition in EXPERIMENT_MATRIX:
            if not condition.get("enabled", True):
                continue

            api_name = condition["api"]
            model = condition["model"]
            label = f"{api_name}_{slugify(model)}"

            if api_name == "assistants":
                backend = self.assistants_backend
                experiments.append({
                    "label": label,
                    "api": "Assistants API",
                    "api_key": "assistants",
                    "backend": backend,
                    "model": model,
                    "problem_source": problem_file_id,
                    "solution_source": solution_file_id,
                    "problem_text_for_prompt": None,
                    "solution_text_for_prompt": None,
                })
            elif api_name == "responses":
                backend = self.responses_backend
                experiments.append({
                    "label": label,
                    "api": "Responses API",
                    "api_key": "responses",
                    "backend": backend,
                    "model": model,
                    "problem_source": problem_text,
                    "solution_source": solution_text,
                    "problem_text_for_prompt": problem_text,
                    "solution_text_for_prompt": solution_text,
                })
            else:
                raise ValueError(f"API no reconocida: {api_name}")

        return experiments

    def evaluate_with_common_judge(self, evaluation_prompt: str) -> BackendResult:
        return self.judge_backend.call_response(
            model=JUDGE_MODEL,
            prompt=evaluation_prompt,
            reasoning_effort=REASONING_EFFORT_JUDGE,
        )

    def usage_with_cost(self, model: str, backend_result: BackendResult) -> Dict[str, Any]:
        return {
            "input_tokens": backend_result.input_tokens,
            "output_tokens": backend_result.output_tokens,
            "total_tokens": backend_result.total_tokens,
            "estimated_cost": estimate_cost(model, backend_result.input_tokens, backend_result.output_tokens),
        }

    def build_result_row(
        self,
        run_id: str,
        timestamp: str,
        dataset_name: str,
        domain: str,
        education_level: str,
        api_name: str,
        model: str,
        label: str,
        run_index: int,
        question_number: int,
        question_text: str,
        solver_prompt: str,
        reference_prompt: str,
        evaluation_prompt: str,
        model_solution: str,
        reference_solution: str,
        evaluation_data: Dict[str, Any],
        elapsed: float,
        solver_usage: Dict[str, Any],
        reference_usage: Dict[str, Any],
        judge_usage: Dict[str, Any],
        solver_result: BackendResult,
        reference_result: BackendResult,
        judge_result: BackendResult,
        problem_pdf_sha256: Optional[str],
        solution_pdf_sha256: Optional[str],
        error: Optional[str],
    ) -> Dict[str, Any]:
        total_input_tokens = int(solver_usage["input_tokens"] + reference_usage["input_tokens"] + judge_usage["input_tokens"])
        total_output_tokens = int(solver_usage["output_tokens"] + reference_usage["output_tokens"] + judge_usage["output_tokens"])
        total_tokens = int(solver_usage["total_tokens"] + reference_usage["total_tokens"] + judge_usage["total_tokens"])
        total_cost = float(solver_usage["estimated_cost"] + reference_usage["estimated_cost"] + judge_usage["estimated_cost"])

        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "experiment_hash": EXPERIMENT_HASH,
            "dataset_name": dataset_name,
            "domain": domain,
            "education_level": education_level,
            "experiment_label": label,
            "api": api_name,
            "model": model,
            "judge_model": JUDGE_MODEL,
            "run": run_index,
            "question_number": question_number,
            "question_text": question_text,
            "reference_source": "common_precomputed_reference",
            "score": int(evaluation_data.get("score", 0)),
            "completeness": evaluation_data.get("completeness"),
            "symbolic_correctness": evaluation_data.get("symbolic_correctness"),
            "pedagogical_quality": evaluation_data.get("pedagogical_quality"),
            "faithfulness_to_problem": evaluation_data.get("faithfulness_to_problem"),
            "subpart_accuracy": evaluation_data.get("subpart_accuracy"),
            "error_type": evaluation_data.get("error_type", "none"),
            "evaluation_comment": evaluation_data.get("reason", ""),
            "raw_evaluation": evaluation_data.get("raw_evaluation", ""),
            "model_solution": model_solution,
            "reference_solution": reference_solution,
            "solver_prompt": solver_prompt,
            "reference_prompt": reference_prompt,
            "evaluation_prompt": evaluation_prompt,
            "execution_time_seconds": float(elapsed),
            "solver_input_tokens": solver_usage["input_tokens"],
            "solver_output_tokens": solver_usage["output_tokens"],
            "solver_total_tokens": solver_usage["total_tokens"],
            "reference_input_tokens": reference_usage["input_tokens"],
            "reference_output_tokens": reference_usage["output_tokens"],
            "reference_total_tokens": reference_usage["total_tokens"],
            "judge_input_tokens": judge_usage["input_tokens"],
            "judge_output_tokens": judge_usage["output_tokens"],
            "judge_total_tokens": judge_usage["total_tokens"],
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_cost,
            "problem_pdf_sha256": problem_pdf_sha256,
            "solution_pdf_sha256": solution_pdf_sha256,
            "solver_endpoint": solver_result.endpoint,
            "solver_response_id": solver_result.response_id,
            "solver_response_model": solver_result.response_model,
            "solver_response_created": solver_result.response_created,
            "solver_system_fingerprint": solver_result.system_fingerprint,
            "solver_request_parameters": solver_result.request_parameters,
            "solver_retried_without_reasoning": solver_result.retried_without_reasoning,
            "solver_request_started_at_utc": solver_result.request_started_at_utc,
            "solver_request_finished_at_utc": solver_result.request_finished_at_utc,
            "solver_assistant_id": solver_result.assistant_id,
            "solver_thread_id": solver_result.thread_id,
            "solver_api_run_id": solver_result.run_id,
            "reference_endpoint": reference_result.endpoint,
            "reference_response_id": reference_result.response_id,
            "reference_response_model": reference_result.response_model,
            "reference_response_created": reference_result.response_created,
            "reference_system_fingerprint": reference_result.system_fingerprint,
            "reference_request_parameters": reference_result.request_parameters,
            "reference_retried_without_reasoning": reference_result.retried_without_reasoning,
            "reference_request_started_at_utc": reference_result.request_started_at_utc,
            "reference_request_finished_at_utc": reference_result.request_finished_at_utc,
            "reference_assistant_id": reference_result.assistant_id,
            "reference_thread_id": reference_result.thread_id,
            "reference_api_run_id": reference_result.run_id,
            "judge_endpoint": judge_result.endpoint,
            "judge_response_id": judge_result.response_id,
            "judge_response_model": judge_result.response_model,
            "judge_response_created": judge_result.response_created,
            "judge_system_fingerprint": judge_result.system_fingerprint,
            "judge_request_parameters": judge_result.request_parameters,
            "judge_retried_without_reasoning": judge_result.retried_without_reasoning,
            "judge_request_started_at_utc": judge_result.request_started_at_utc,
            "judge_request_finished_at_utc": judge_result.request_finished_at_utc,
            "technical_error": error is not None,
            "error": error,
        }

    def save_json(self, results: List[Dict[str, Any]], path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def save_csv(self, results: List[Dict[str, Any]], path: str):
        pd.DataFrame(results).to_csv(path, index=False, encoding="utf-8")

    def generate_output_documents(
        self,
        results: List[Dict[str, Any]],
        output_dir: str,
        run_id: str,
        base_name: str,
        timestamp: str,
        dataset_name: str,
    ) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        if not results:
            raise ValueError(
                "No hay resultados que exportar. Revisa la deteccion de ejercicios y los logs."
            )

        json_path = os.path.join(output_dir, f"resultados_{run_id}.json")
        csv_path = os.path.join(output_dir, f"resultados_{run_id}.csv")
        xlsx_path = os.path.join(output_dir, f"resultados_{run_id}.xlsx")
        txt_path = os.path.join(output_dir, f"informe_{run_id}.txt")
        tex_path = os.path.join(output_dir, f"paper_resultados_{run_id}.tex")
        config_path = os.path.join(output_dir, f"config_{run_id}.json")

        df = pd.DataFrame(results)

        stats_by_condition = self.compute_descriptive_stats(df, ["experiment_label", "api", "model", "domain"])
        stats_by_run = self.compute_descriptive_stats(df, ["experiment_label", "run"])
        stats_by_question = self.compute_descriptive_stats(df, ["question_number", "experiment_label"])
        stats_by_domain = self.compute_descriptive_stats(df, ["domain", "experiment_label"])
        model_comparisons = self.compute_pairwise_comparisons(df)
        factorial_analysis = self.compute_factorial_summary(df)
        tests = self.compute_preregistered_tests(df)
        educational_metrics = self.compute_educational_metrics(df)
        cost_efficiency = self.compute_cost_efficiency(df)
        error_taxonomy = self.compute_error_taxonomy(df)

        self.save_json(results, json_path)
        df.to_csv(csv_path, index=False, encoding="utf-8")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_config": EXPERIMENT_CONFIG,
                    "experiment_hash": EXPERIMENT_HASH,
                    "environment": get_environment_metadata(),
                    "dataset_name": dataset_name,
                    "base_name": base_name,
                    "timestamp": timestamp,
                    "generated_at_utc": utc_now_iso(),
                    "problem_pdf_sha256": (results[0].get("problem_pdf_sha256") if results else None),
                    "solution_pdf_sha256": (results[0].get("solution_pdf_sha256") if results else None),
                    "api_endpoints": {
                        "responses": "client.responses.create",
                        "assistants": "client.beta.assistants / threads / runs",
                    },
                    "inference_parameters": {
                        "temperature": "provider_default_not_explicitly_set",
                        "top_p": "provider_default_not_explicitly_set",
                        "seed": "not_set_or_not_exposed",
                        "text_verbosity": TEXT_VERBOSITY,
                        "reasoning_effort_solver": REASONING_EFFORT_SOLVER,
                        "reasoning_effort_reference": REASONING_EFFORT_REFERENCE,
                        "reasoning_effort_judge": REASONING_EFFORT_JUDGE,
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="resultados_detallados", index=False)
            stats_by_condition.to_excel(writer, sheet_name="stats_condicion", index=False)
            stats_by_run.to_excel(writer, sheet_name="stats_run", index=False)
            stats_by_question.to_excel(writer, sheet_name="stats_ejercicio", index=False)
            stats_by_domain.to_excel(writer, sheet_name="stats_dominio", index=False)
            model_comparisons.to_excel(writer, sheet_name="comparaciones", index=False)
            factorial_analysis.to_excel(writer, sheet_name="factorial", index=False)
            tests.to_excel(writer, sheet_name="tests", index=False)
            educational_metrics.to_excel(writer, sheet_name="metricas_educativas", index=False)
            cost_efficiency.to_excel(writer, sheet_name="coste_eficiencia", index=False)
            error_taxonomy.to_excel(writer, sheet_name="taxonomia_errores", index=False)

        self.save_txt_report(
            txt_path,
            df,
            stats_by_condition,
            stats_by_run,
            stats_by_question,
            stats_by_domain,
            model_comparisons,
            factorial_analysis,
            tests,
            educational_metrics,
            cost_efficiency,
            error_taxonomy,
        )

        self.save_latex_report(
            tex_path,
            stats_by_condition,
            stats_by_run,
            stats_by_question,
            stats_by_domain,
            model_comparisons,
            factorial_analysis,
            tests,
            educational_metrics,
            cost_efficiency,
            error_taxonomy,
        )

        return {
            "json": json_path,
            "csv": csv_path,
            "xlsx": xlsx_path,
            "txt": txt_path,
            "tex": tex_path,
            "config": config_path,
            "log": LOG_PATH,
        }

    def compute_descriptive_stats(self, df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        rows = []
        if df.empty:
            return pd.DataFrame()

        for keys, group in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)

            scores = group["score"].astype(float).tolist()
            times = group["execution_time_seconds"].astype(float).tolist()
            costs = group["estimated_cost_usd"].astype(float).tolist() if "estimated_cost_usd" in group else []
            tokens = group["total_tokens"].astype(float).tolist() if "total_tokens" in group else []
            ci_low, ci_high = safe_ci95(scores)

            row = {col: key for col, key in zip(group_cols, keys)}
            row.update({
                "n": len(group),
                "mean": float(np.mean(scores)) if scores else 0.0,
                "median": float(np.median(scores)) if scores else 0.0,
                "std": safe_std(scores),
                "min": float(np.min(scores)) if scores else 0.0,
                "max": float(np.max(scores)) if scores else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "ge80_percent": float((group["score"].astype(float) >= 80).mean() * 100),
                "perfect_percent": float((group["score"].astype(float) == 100).mean() * 100),
                "zero_percent": float((group["score"].astype(float) == 0).mean() * 100),
                "mean_time_seconds": float(np.mean(times)) if times else 0.0,
                "median_time_seconds": float(np.median(times)) if times else 0.0,
                "p95_time_seconds": safe_percentile(times, 95),
                "technical_errors": int(group["technical_error"].sum()) if "technical_error" in group else 0,
                "technical_error_percent": float(group["technical_error"].mean() * 100) if "technical_error" in group else 0.0,
                "mean_tokens": float(np.mean(tokens)) if tokens else 0.0,
                "mean_cost_usd": float(np.mean(costs)) if costs else 0.0,
                "total_cost_usd": float(np.sum(costs)) if costs else 0.0,
            })

            if "question_number" in group_cols:
                row["unstable"] = row["std"] > UNSTABLE_STD_THRESHOLD

            rows.append(row)

        return pd.DataFrame(rows)

    def compute_educational_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        metric_cols = [
            "completeness",
            "symbolic_correctness",
            "pedagogical_quality",
            "faithfulness_to_problem",
            "subpart_accuracy",
        ]
        rows = []
        for label, group in df.groupby("experiment_label"):
            row = {"experiment_label": label, "n": len(group)}
            for col in metric_cols:
                if col in group:
                    values = pd.to_numeric(group[col], errors="coerce").dropna()
                    row[f"{col}_mean"] = float(values.mean()) if len(values) else None
                    row[f"{col}_median"] = float(values.median()) if len(values) else None
                    row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            rows.append(row)
        return pd.DataFrame(rows)

    def compute_cost_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for label, group in df.groupby("experiment_label"):
            cost = float(group["estimated_cost_usd"].sum()) if "estimated_cost_usd" in group else 0.0
            total_tokens = int(group["total_tokens"].sum()) if "total_tokens" in group else 0
            mean_score = float(group["score"].mean())
            ge80_count = int((group["score"] >= 80).sum())
            perfect_count = int((group["score"] == 100).sum())
            rows.append({
                "experiment_label": label,
                "n": len(group),
                "total_cost_usd": cost,
                "mean_cost_usd": float(group["estimated_cost_usd"].mean()) if "estimated_cost_usd" in group else 0.0,
                "total_tokens": total_tokens,
                "mean_tokens": float(group["total_tokens"].mean()) if "total_tokens" in group else 0.0,
                "mean_score": mean_score,
                "score_per_usd": mean_score / cost if cost > 0 else None,
                "cost_per_ge80": cost / ge80_count if ge80_count > 0 else None,
                "cost_per_perfect": cost / perfect_count if perfect_count > 0 else None,
                "mean_latency": float(group["execution_time_seconds"].mean()),
                "p95_latency": safe_percentile(group["execution_time_seconds"].astype(float).tolist(), 95),
            })
        return pd.DataFrame(rows)

    def compute_error_taxonomy(self, df: pd.DataFrame) -> pd.DataFrame:
        if "error_type" not in df:
            return pd.DataFrame()
        rows = []
        total_by_label = df.groupby("experiment_label").size().to_dict()
        for (label, error_type), group in df.groupby(["experiment_label", "error_type"], dropna=False):
            total = total_by_label.get(label, len(group))
            rows.append({
                "experiment_label": label,
                "error_type": error_type,
                "count": len(group),
                "percent": 100 * len(group) / total if total else 0.0,
            })
        return pd.DataFrame(rows)

    def compute_factorial_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        if df.empty:
            return pd.DataFrame()

        temp = df.copy()
        temp["api_factor"] = temp["api"].str.replace(" API", "", regex=False)
        temp["model_factor"] = temp["model"]

        for factor in ["api_factor", "model_factor"]:
            for level, group in temp.groupby(factor):
                scores = group["score"].astype(float).tolist()
                ci_low, ci_high = safe_ci95(scores)
                rows.append({
                    "factor": factor,
                    "level": level,
                    "n": len(group),
                    "mean": float(np.mean(scores)),
                    "median": float(np.median(scores)),
                    "std": safe_std(scores),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "ge80_percent": float((group["score"] >= 80).mean() * 100),
                    "perfect_percent": float((group["score"] == 100).mean() * 100),
                })

        return pd.DataFrame(rows)

    def compute_pairwise_comparisons(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        labels = sorted(df["experiment_label"].dropna().unique().tolist())

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a_label = labels[i]
                b_label = labels[j]
                a = df[df["experiment_label"] == a_label]["score"].astype(float).tolist()
                b = df[df["experiment_label"] == b_label]["score"].astype(float).tolist()
                rows.append(self.compare_two_independent(a_label, b_label, a, b))

        out = pd.DataFrame(rows)
        if not out.empty:
            for col in ["welch_p", "mannwhitney_p", "levene_p"]:
                out[f"{col}_bonferroni"] = np.minimum(out[col].astype(float) * len(out), 1.0)
        return out

    def compare_two_independent(self, a_label: str, b_label: str, a: List[float], b: List[float]) -> Dict[str, Any]:
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        mean_diff = float(np.mean(b_arr) - np.mean(a_arr))

        try:
            welch = stats.ttest_ind(a_arr, b_arr, equal_var=False, nan_policy="omit")
            welch_stat, welch_p = float(welch.statistic), float(welch.pvalue)
        except Exception:
            welch_stat, welch_p = np.nan, np.nan

        try:
            mw = stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")
            mw_stat, mw_p = float(mw.statistic), float(mw.pvalue)
        except Exception:
            mw_stat, mw_p = np.nan, np.nan

        try:
            lev = stats.levene(a_arr, b_arr, center="median")
            lev_stat, lev_p = float(lev.statistic), float(lev.pvalue)
        except Exception:
            lev_stat, lev_p = np.nan, np.nan

        pooled_std = math.sqrt(((len(a_arr) - 1) * np.var(a_arr, ddof=1) + (len(b_arr) - 1) * np.var(b_arr, ddof=1)) / (len(a_arr) + len(b_arr) - 2)) if len(a_arr) > 1 and len(b_arr) > 1 else 0.0
        cohen_d = mean_diff / pooled_std if pooled_std > 0 else np.nan

        return {
            "condition_a": a_label,
            "condition_b": b_label,
            "n_a": len(a_arr),
            "n_b": len(b_arr),
            "mean_a": float(np.mean(a_arr)) if len(a_arr) else np.nan,
            "mean_b": float(np.mean(b_arr)) if len(b_arr) else np.nan,
            "difference_b_minus_a": mean_diff,
            "welch_t": welch_stat,
            "welch_p": welch_p,
            "mannwhitney_u": mw_stat,
            "mannwhitney_p": mw_p,
            "levene_stat": lev_stat,
            "levene_p": lev_p,
            "cohen_d": cohen_d,
        }

    def compute_preregistered_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        labels = sorted(df["experiment_label"].dropna().unique().tolist())

        if len(labels) < 2:
            return pd.DataFrame()

        # Contrastes independientes globales entre todas las condiciones.
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a_label, b_label = labels[i], labels[j]
                a = df[df["experiment_label"] == a_label]["score"].astype(float).tolist()
                b = df[df["experiment_label"] == b_label]["score"].astype(float).tolist()
                row = self.compare_two_independent(a_label, b_label, a, b)
                row["test_family"] = "independent_global"
                rows.append(row)

        # Contrastes apareados por ejercicio cuando hay dos condiciones comparables.
        pivot = df.groupby(["question_number", "experiment_label"])["score"].mean().unstack()
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a_label, b_label = labels[i], labels[j]
                if a_label not in pivot.columns or b_label not in pivot.columns:
                    continue
                paired = pivot[[a_label, b_label]].dropna()
                if len(paired) < 2:
                    continue
                a = paired[a_label].astype(float).values
                b = paired[b_label].astype(float).values
                diff = b - a

                try:
                    t_res = stats.ttest_rel(a, b, nan_policy="omit")
                    paired_t, paired_p = float(t_res.statistic), float(t_res.pvalue)
                except Exception:
                    paired_t, paired_p = np.nan, np.nan

                try:
                    w_res = stats.wilcoxon(a, b)
                    wilcoxon_stat, wilcoxon_p = float(w_res.statistic), float(w_res.pvalue)
                except Exception:
                    wilcoxon_stat, wilcoxon_p = np.nan, np.nan

                ci_low, ci_high = safe_ci95(diff.astype(float).tolist())

                rows.append({
                    "test_family": "paired_by_question",
                    "condition_a": a_label,
                    "condition_b": b_label,
                    "n_pairs": len(paired),
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "difference_b_minus_a": float(np.mean(diff)),
                    "diff_ci95_low": ci_low,
                    "diff_ci95_high": ci_high,
                    "paired_t": paired_t,
                    "paired_t_p": paired_p,
                    "wilcoxon_stat": wilcoxon_stat,
                    "wilcoxon_p": wilcoxon_p,
                    "wins_b": int((diff > 0).sum()),
                    "wins_a": int((diff < 0).sum()),
                    "ties": int((diff == 0).sum()),
                })

        out = pd.DataFrame(rows)
        if not out.empty:
            p_cols = [c for c in out.columns if c.endswith("_p") or c in ["welch_p", "mannwhitney_p", "levene_p"]]
            for col in p_cols:
                vals = pd.to_numeric(out[col], errors="coerce")
                mask = vals.notna()
                corrected = pd.Series(np.nan, index=out.index)
                corrected.loc[mask] = np.minimum(vals.loc[mask] * mask.sum(), 1.0)
                out[f"{col}_bonferroni"] = corrected

        return out

    def save_txt_report(self, path: str, *tables):
        names = [
            "RESULTADOS DETALLADOS",
            "ESTADISTICOS POR CONDICION",
            "ESTADISTICOS POR RUN",
            "ESTADISTICOS POR EJERCICIO",
            "ESTADISTICOS POR DOMINIO",
            "COMPARACIONES PAREADAS/INDEPENDIENTES",
            "ANALISIS FACTORIAL",
            "TESTS PRERREGISTRADOS",
            "METRICAS EDUCATIVAS",
            "COSTE Y EFICIENCIA",
            "TAXONOMIA DE ERRORES",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("INFORME DE BENCHMARK METODOLOGICO\n")
            f.write("=" * 80 + "\n")
            f.write(f"Fecha: {datetime.now().isoformat()}\n")
            f.write(f"Experiment hash: {EXPERIMENT_HASH}\n")
            f.write(f"Judge model: {JUDGE_MODEL}\n")
            f.write(f"Log: {LOG_PATH}\n\n")
            f.write("CONFIGURACION\n")
            f.write(json.dumps(EXPERIMENT_CONFIG, ensure_ascii=False, indent=2))
            f.write("\n\n")

            for name, table in zip(names, tables):
                f.write("\n" + name + "\n")
                f.write("-" * 80 + "\n")
                if isinstance(table, pd.DataFrame):
                    f.write(table.to_string(index=False))
                else:
                    f.write(str(table))
                f.write("\n")

    def save_latex_report(self, path: str, *tables):
        latex_tables = []
        captions = [
            "Resultados detallados",
            "Estadisticos por condicion experimental",
            "Estadisticos por run",
            "Estadisticos por ejercicio",
            "Estadisticos por dominio",
            "Comparaciones entre condiciones",
            "Resumen factorial API por modelo",
            "Contrastes prerregistrados",
            "Metricas educativas",
            "Coste y eficiencia",
            "Taxonomia de errores",
        ]
        labels = [
            "tab:resultados_detallados",
            "tab:stats_condicion",
            "tab:stats_run",
            "tab:stats_ejercicio",
            "tab:stats_dominio",
            "tab:comparaciones",
            "tab:factorial",
            "tab:tests",
            "tab:metricas_educativas",
            "tab:coste_eficiencia",
            "tab:taxonomia_errores",
        ]

        # Evitar una tabla LaTeX gigantesca de resultados detallados.
        for i, table in enumerate(tables):
            if not isinstance(table, pd.DataFrame) or table.empty:
                continue
            table_to_export = table.copy()
            if i == 0 and len(table_to_export) > 30:
                table_to_export = table_to_export.head(30)
            latex_tables.append(
                table_to_export.to_latex(
                    index=False,
                    longtable=True,
                    escape=False,
                    caption=captions[i],
                    label=labels[i],
                )
            )

        latex_document = rf"""
\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{geometry}}
\usepackage{{float}}
\geometry{{margin=2.5cm}}
\title{{Resultados experimentales del benchmark matematico}}
\author{{}}
\date{{{datetime.now().strftime('%d/%m/%Y')}}}

\begin{{document}}
\maketitle

\section{{Metodologia experimental}}

El experimento se configura como un benchmark factorial API $\times$ modelo. Cada condicion combina una API de ejecucion con un modelo de lenguaje. Cada ejercicio se ejecuta {N_RUNS} veces por condicion. La evaluacion se realiza mediante un protocolo LLM-as-a-Judge con un juez comun externo, fijo para todas las ramas, definido como \texttt{{{JUDGE_MODEL}}}. Antes de ejecutar las condiciones experimentales se genera una unica pareja de enunciado y solucion de referencia por ejercicio, reutilizada sin cambios en todas las APIs, modelos y repeticiones. La puntuacion principal se expresa en una escala 0--100 y se complementa con metricas educativas de completitud, correccion simbolica, calidad pedagogica, fidelidad al enunciado, exactitud por apartados y tipo de error.

El estudio prerregistra los siguientes estadisticos: media, mediana, desviacion tipica, intervalo de confianza al 95\%, porcentaje de respuestas con puntuacion mayor o igual a 80, porcentaje de respuestas perfectas, porcentaje de ceros, tiempo medio, latencia p95, tokens, coste estimado, errores tecnicos, prueba t pareada, Wilcoxon, Welch, Levene y correccion de Bonferroni.

\section{{Resultados}}

{chr(10).join(latex_tables)}

\end{{document}}
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(latex_document)


# ============================================================
# BACKEND ASSISTANTS API
# ============================================================

class AssistantsMathBackend:
    def __init__(self, client: OpenAI):
        self.client = client

    def upload_file(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            uploaded = self.client.files.create(file=f, purpose="assistants")
        logging.info("Uploaded file %s as %s", file_path, uploaded.id)
        time.sleep(60)
        return uploaded.id

    def create_assistant(self, model: str, role: str):
        if role == "solver":
            instructions = (
                "Eres un experto en matematicas. Trabaja con rigor, claridad y precision. "
                "Usa exclusivamente el documento adjunto para leer el enunciado. "
                "No consultes la solucion oficial. No inventes datos."
            )
            tools = [{"type": "file_search"}]
        elif role == "reference_extractor":
            instructions = (
                "Eres un extractor de soluciones oficiales. Usa exclusivamente el documento adjunto. "
                "No resuelvas el ejercicio por tu cuenta. Extrae o reconstruye fielmente la solucion de referencia. "
                "Si no encuentras la solucion, dilo claramente."
            )
            tools = [{"type": "file_search"}]
        elif role == "question_extractor":
            instructions = (
                "Eres un extractor de enunciados matematicos. Usa exclusivamente el documento adjunto. "
                "No resuelvas el ejercicio. Extrae o reconstruye fielmente el enunciado completo, "
                "incluidos todos los datos, condiciones y apartados."
            )
            tools = [{"type": "file_search"}]
        else:
            raise ValueError(f"Rol no reconocido: {role}")

        return self.client.beta.assistants.create(
            name=f"math_{role}_{slugify(model)}",
            instructions=instructions,
            tools=tools,
            model=model,
        )

    def solve_problem(self, source: str, question_number: int, model: str, prompt: Optional[str] = None) -> BackendResult:
        assistant = self.create_assistant(model, "solver")
        thread = self.client.beta.threads.create()
        prompt = prompt or build_solver_prompt(question_number, None)
        self.send_message(thread.id, prompt, source)
        return self.run_and_get_response(thread.id, assistant.id, requested_model=model)

    def extract_question(self, source: str, question_number: int, model: str, prompt: Optional[str] = None) -> BackendResult:
        assistant = self.create_assistant(model, "question_extractor")
        thread = self.client.beta.threads.create()
        prompt = prompt or build_question_prompt(question_number, None)
        self.send_message(thread.id, prompt, source)
        return self.run_and_get_response(thread.id, assistant.id, requested_model=model)

    def extract_reference_solution(self, source: str, question_number: int, model: str, prompt: Optional[str] = None) -> BackendResult:
        assistant = self.create_assistant(model, "reference_extractor")
        thread = self.client.beta.threads.create()
        prompt = prompt or build_reference_prompt(question_number, None)
        self.send_message(thread.id, prompt, source)
        return self.run_and_get_response(thread.id, assistant.id, requested_model=model)

    def send_message(self, thread_id: str, content: str, file_id: Optional[str] = None):
        kwargs = {"thread_id": thread_id, "role": "user", "content": content}
        if file_id is not None:
            kwargs["attachments"] = [{"file_id": file_id, "tools": [{"type": "file_search"}]}]
        self.client.beta.threads.messages.create(**kwargs)

    def run_and_get_response(self, thread_id: str, assistant_id: str, requested_model: str) -> BackendResult:
        started_at = utc_now_iso()
        run = self.client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)

        while True:
            time.sleep(POLL_SECONDS)
            run_status = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            logging.info("Assistants status: %s", run_status.status)

            if run_status.status == "completed":
                messages = self.client.beta.threads.messages.list(thread_id=thread_id)
                if not messages.data:
                    raise RuntimeError("El assistant no devolvio mensajes.")
                message = messages.data[0]
                return BackendResult(
                    text=message.content[0].text.value,
                    endpoint="Assistants API",
                    response_id=getattr(message, "id", None),
                    response_model=getattr(run_status, "model", requested_model),
                    response_created=getattr(message, "created_at", None),
                    system_fingerprint=getattr(run_status, "system_fingerprint", None),
                    request_parameters={
                        "model_requested": requested_model,
                        "temperature": "provider_default_not_explicitly_set",
                        "top_p": "provider_default_not_explicitly_set",
                        "seed": "not_set_or_not_exposed",
                    },
                    request_started_at_utc=started_at,
                    request_finished_at_utc=utc_now_iso(),
                    assistant_id=assistant_id,
                    thread_id=thread_id,
                    run_id=getattr(run_status, "id", getattr(run, "id", None)),
                )

            if run_status.status in ["queued", "in_progress"]:
                continue

            if run_status.status in ["failed", "cancelled", "expired"]:
                last_error = getattr(run_status, "last_error", None)
                raise RuntimeError(f"Run Assistants terminado con estado {run_status.status}. Last error: {last_error}")

            raise RuntimeError(f"Estado Assistants inesperado: {run_status.status}")


# ============================================================
# BACKEND RESPONSES API
# ============================================================

class ResponsesMathBackend:
    def __init__(self, client: OpenAI):
        self.client = client

    def get_num_questions(self, problem_text: str) -> int:
        prompt = f"""
Lee el siguiente documento de examen e indica cuantos ejercicios o preguntas hay.

DOCUMENTO:
{problem_text}

Responde solo con un numero entero.
""".strip()
        result = self.call_response(MODEL_FOR_QUESTION_COUNT(), prompt, reasoning_effort="low")
        match = re.search(r"\d+", result.text)
        if not match:
            raise ValueError(f"No se pudo detectar el numero de ejercicios. Respuesta: {result.text}")
        return int(match.group())

    def solve_problem(self, source: str, question_number: int, model: str, prompt: Optional[str] = None) -> BackendResult:
        prompt = prompt or build_solver_prompt(question_number, source)
        return self.call_response(model=model, prompt=prompt, reasoning_effort=REASONING_EFFORT_SOLVER)

    def extract_question(self, source: str, question_number: int, model: str, prompt: Optional[str] = None) -> BackendResult:
        prompt = prompt or build_question_prompt(question_number, source)
        return self.call_response(model=model, prompt=prompt, reasoning_effort=REASONING_EFFORT_REFERENCE)

    def extract_reference_solution(self, source: str, question_number: int, model: str, prompt: Optional[str] = None) -> BackendResult:
        prompt = prompt or build_reference_prompt(question_number, source)
        return self.call_response(model=model, prompt=prompt, reasoning_effort=REASONING_EFFORT_REFERENCE)

    def call_response(self, model: str, prompt: str, reasoning_effort: str = "low") -> BackendResult:
        kwargs = {
            "model": model,
            "input": prompt,
            "text": {"verbosity": TEXT_VERBOSITY},
        }

        # Algunos modelos antiguos pueden no aceptar reasoning. Si falla, se reintenta sin reasoning.
        if reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        started_at = utc_now_iso()
        retried_without_reasoning = False
        request_parameters = {
            "model_requested": model,
            "reasoning_effort_requested": reasoning_effort,
            "text_verbosity": TEXT_VERBOSITY,
            "temperature": "provider_default_not_explicitly_set",
            "top_p": "provider_default_not_explicitly_set",
            "seed": "not_set_or_not_exposed",
        }

        try:
            response = self.client.responses.create(**kwargs)
        except Exception as first_error:
            if "reasoning" not in kwargs:
                raise
            logging.warning("Retrying Responses call without reasoning for model %s: %s", model, first_error)
            kwargs.pop("reasoning", None)
            retried_without_reasoning = True
            response = self.client.responses.create(**kwargs)

        usage = get_usage_dict(response)
        text = response.output_text.strip()
        return BackendResult(
            text=text,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            estimated_cost=estimate_cost(model, usage["input_tokens"], usage["output_tokens"]),
            endpoint="Responses API",
            response_id=getattr(response, "id", None),
            response_model=getattr(response, "model", model),
            response_created=getattr(response, "created_at", getattr(response, "created", None)),
            system_fingerprint=getattr(response, "system_fingerprint", None),
            request_parameters=request_parameters,
            retried_without_reasoning=retried_without_reasoning,
            request_started_at_utc=started_at,
            request_finished_at_utc=utc_now_iso(),
        )



def MODEL_FOR_QUESTION_COUNT() -> str:
    # Modelo estable para tareas simples auxiliares.
    return JUDGE_MODEL


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    gui = GUIManager()
    gui.run()
