#!/usr/bin/env python3
"""Prepara el repositorio EBAU-LLM de forma NO DESTRUCTIVA e idempotente.

Esta version sustituye al reorganizador anterior. No mueve ni borra los resultados,
por lo que funciona tambien si la primera ejecucion quedo a medias.

Uso (desde cualquier ubicacion):
    python PREPARAR_REPOSITORIO_V2.py

El script debe estar en la raiz del repositorio. Crea una capa cientifica de
organizacion, copia solo los scripts pequenos a code/, genera inventarios, hashes,
mapa de ubicaciones y un informe de seguridad. Los datos grandes permanecen donde
esten para evitar errores de rutas largas en Windows.
"""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import re
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent

ANALYSIS_FILES = {
    "01_auditoria_datos.py", "02_dataset_maestro.py", "03_estadistica_descriptiva.py",
    "04_inferencia.py", "05_figuras_publicacion.py", "06_tablas_publicacion.py",
    "common.py", "run_pipeline.py"
}
EXPERIMENT_NAMES = {
    "gpt_POO_paper_v8_4_detector_ebau.py",
    "gpt_poo_paper_v8_4_detector_ebau.py",
}
TEXT_EXTS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".html"}
SECRET_PATTERNS = [
    ("openai_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("assigned_api_key", re.compile(r"OPENAI_API_KEY\s*=\s*['\"][^'\"]+['\"]", re.I)),
]

# Posibles ubicaciones, incluidas las creadas por una ejecucion parcial de la v1.
LOCATION_GROUPS = {
    "EBAU source documents": [
        "Documentos EBAU articulo",
        "data/source_documents/EBAU_PDF",
    ],
    "Experimental raw outputs": [
        "Resultados experimento",
        "data/raw_outputs",
    ],
    "Human-validation interface": [
        "Evaluacion_HTML_2023",
        "validation/html_interface",
    ],
    "Human-validation data and scripts": [
        "script_y_excel_validacion_humana",
        "validation/human",
    ],
    "Original analytical pipeline": [
        "pipeline",
        "code/analysis",
        "code/experiment",
        "docs/pipeline_original",
    ],
}


def ensure_dirs() -> None:
    for rel in (
        "code/analysis", "code/experiment", "code/validation",
        "metadata", "docs", "licenses", "supplementary",
        "results", "validation"
    ):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def safe_copy(source: Path, destination: Path) -> str:
    """Copia un archivo pequeno sin sobrescribir contenido distinto."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if sha256(source) == sha256(destination):
                return "already_identical"
        except OSError:
            pass
        candidate = destination.with_name(destination.stem + "__from_original" + destination.suffix)
        counter = 2
        while candidate.exists():
            candidate = destination.with_name(destination.stem + f"__from_original_{counter}" + destination.suffix)
            counter += 1
        destination = candidate
    shutil.copy2(source, destination)
    return destination.relative_to(ROOT).as_posix()


def copy_code_layer() -> list[tuple[str, str, str]]:
    actions: list[tuple[str, str, str]] = []
    candidates = []
    for rel in ("pipeline", "code/analysis", "code/experiment", "script_y_excel_validacion_humana", "validation/human"):
        base = ROOT / rel
        if base.exists():
            candidates.extend(p for p in base.rglob("*.py") if p.is_file())

    seen_hashes: set[str] = set()
    for source in sorted(candidates, key=lambda p: p.as_posix().lower()):
        try:
            digest = sha256(source)
        except OSError as exc:
            actions.append((source.relative_to(ROOT).as_posix(), "not_copied", f"read_error: {exc}"))
            continue
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        if source.name in ANALYSIS_FILES:
            destination = ROOT / "code/analysis" / source.name
        elif source.name in EXPERIMENT_NAMES or "detector" in source.name.lower():
            destination = ROOT / "code/experiment" / source.name
        elif "valid" in source.name.lower() or "human" in source.name.lower():
            destination = ROOT / "code/validation" / source.name
        else:
            destination = ROOT / "code/analysis" / source.name
        try:
            result = safe_copy(source, destination)
            actions.append((source.relative_to(ROOT).as_posix(), "copied_or_present", result))
        except OSError as exc:
            actions.append((source.relative_to(ROOT).as_posix(), "not_copied", f"copy_error: {exc}"))
    return actions


def file_count_and_size(base: Path) -> tuple[int, int]:
    count = 0
    total = 0
    if not base.exists():
        return count, total
    for p in base.rglob("*"):
        if p.is_file():
            count += 1
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return count, total


def write_location_map() -> None:
    rows = []
    for logical_group, relative_locations in LOCATION_GROUPS.items():
        for rel in relative_locations:
            path = ROOT / rel
            count, size = file_count_and_size(path)
            rows.append((logical_group, rel, path.exists(), count, size))
    out = ROOT / "metadata/LOCATION_MAP.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["logical_group", "relative_location", "exists", "file_count", "bytes"])
        writer.writerows(rows)


def infer_pdf_metadata(name: str) -> tuple[str, str, str]:
    normalized = name.lower()
    replacements = str.maketrans("áéíóúüñ", "aeiouun")
    normalized = normalized.translate(replacements)
    year = next((year for year in ("2023", "2024", "2025") if year in normalized), "")
    document_type = "solucion" if "soluc" in normalized else ("enunciado" if "enuncia" in normalized else "no_identificado")
    region_aliases = [
        ("andalucia", "Andalucia"), ("aragon", "Aragon"), ("asturias", "Asturias"),
        ("baleares", "Baleares"), ("canarias", "Canarias"), ("cantabria", "Cantabria"),
        ("cataluna", "Cataluna"), ("cleon", "Castilla y Leon"),
        ("cvalenciana", "Comunidad Valenciana"), ("extremadura", "Extremadura"),
        ("galicia", "Galicia"), ("lamancha", "Castilla-La Mancha"), ("madrid", "Madrid"),
        ("murcia", "Murcia"), ("navarra", "Navarra"), ("pais vasco", "Pais Vasco"),
        ("rioja", "La Rioja"),
    ]
    region = next((label for token, label in region_aliases if token in normalized), "")
    return region, year, document_type


def all_files() -> list[Path]:
    excluded_names = {"FILE_MANIFEST.csv", "SHA256SUMS.txt", "SECURITY_SCAN.csv"}
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or ".git" in p.parts or p.name in excluded_names:
            continue
        files.append(p)
    return sorted(files, key=lambda p: p.as_posix().lower())


def write_inventories() -> tuple[int, int]:
    rows = []
    errors = []
    for p in all_files():
        rel = p.relative_to(ROOT).as_posix()
        try:
            size = p.stat().st_size
            digest = sha256(p)
            rows.append((rel, size, digest))
        except OSError as exc:
            errors.append((rel, repr(exc)))

    with (ROOT / "metadata/FILE_MANIFEST.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["relative_path", "bytes", "sha256"])
        writer.writerows(rows)

    with (ROOT / "SHA256SUMS.txt").open("w", encoding="utf-8") as fh:
        for rel, _size, digest in rows:
            fh.write(f"{digest}  {rel}\n")

    with (ROOT / "metadata/MANIFEST_ERRORS.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["relative_path", "error"])
        writer.writerows(errors)

    # Recoge PDF en ambas ubicaciones posibles y elimina duplicados por hash.
    pdf_rows = []
    seen = set()
    for relbase in ("Documentos EBAU articulo", "data/source_documents/EBAU_PDF"):
        base = ROOT / relbase
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.pdf"), key=lambda x: x.name.lower()):
            try:
                digest = sha256(p)
                if digest in seen:
                    continue
                seen.add(digest)
                region, year, document_type = infer_pdf_metadata(p.name)
                pdf_rows.append((p.relative_to(ROOT).as_posix(), p.name, region, year, document_type, p.stat().st_size, digest, "", "Third-party public-sector document; see licenses/THIRD_PARTY_EBAU_NOTICE.md"))
            except OSError as exc:
                pdf_rows.append((p.relative_to(ROOT).as_posix(), p.name, "", "", "read_error", "", "", "", repr(exc)))

    with (ROOT / "metadata/EBAU_INVENTORY.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["relative_path", "filename", "region_inferred", "year_inferred", "document_type_inferred", "bytes", "sha256", "official_source_url", "rights_statement"])
        writer.writerows(pdf_rows)
    return len(rows), len(errors)


def security_scan() -> list[tuple[str, int, str]]:
    findings = []
    for p in all_files():
        if p.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append((p.relative_to(ROOT).as_posix(), line, label))
    with (ROOT / "metadata/SECURITY_SCAN.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "line", "finding_type"])
        writer.writerows(findings)
    return findings


def write_run_report(actions, file_count, errors, findings) -> None:
    report = ROOT / "metadata/PREPARATION_REPORT.md"
    lines = [
        "# Repository preparation report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This run was non-destructive: no experimental data were moved or deleted.",
        "",
        f"- Files inventoried successfully: {file_count}",
        f"- Manifest read errors: {errors}",
        f"- Potential secret findings: {len(findings)}",
        f"- Code-layer actions recorded: {len(actions)}",
        "",
        "## Code-layer actions",
        "",
        "| Source | Status | Destination/detail |",
        "|---|---|---|",
    ]
    for source, status, detail in actions:
        lines.append(f"| `{source}` | {status} | `{detail}` |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print("Preparando la capa cientifica sin mover ni borrar archivos...")
    ensure_dirs()
    actions = copy_code_layer()
    write_location_map()
    print("Generando inventarios y hashes; puede tardar varios minutos...")
    file_count, errors = write_inventories()
    findings = security_scan()
    write_run_report(actions, file_count, errors, findings)
    print("\nProceso completado.")
    print(f"Archivos inventariados: {file_count}")
    print(f"Errores de lectura: {errors}")
    print(f"Posibles secretos: {len(findings)}")
    print("Revise metadata/PREPARATION_REPORT.md y metadata/SECURITY_SCAN.csv.")
    if findings:
        print("ATENCION: no haga push hasta revisar y revocar cualquier clave real detectada.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
