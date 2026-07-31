# Guía de reproducibilidad

## A. Reproducir los análisis sin volver a consultar la API

1. Instalar Python 3.12.
2. Desde la raíz, crear un entorno: `python -m venv .venv`.
3. Activarlo e instalar: `pip install -r requirements.txt`.
4. Revisar las rutas de entrada definidas en `code/analysis/common.py` y en el README original conservado dentro de `code/analysis/`.
5. Ejecutar, en orden:
   - `python code/analysis/01_auditoria_datos.py`
   - `python code/analysis/02_dataset_maestro.py`
   - `python code/analysis/03_estadistica_descriptiva.py`
   - `python code/analysis/04_inferencia.py`
   - `python code/analysis/05_figuras_publicacion.py`
   - `python code/analysis/06_tablas_publicacion.py`

También puede utilizarse `python code/analysis/run_pipeline.py` si las rutas configuradas coinciden con esta estructura.

## B. Repetir las llamadas a la API

1. Definir `OPENAI_API_KEY` en el entorno.
2. Revisar el script de `code/experiment/`.
3. Confirmar modelos, endpoints, costes y parámetros disponibles en la fecha de réplica.
4. Ejecutar sobre una copia de trabajo; no sobrescribir `data/raw_outputs/`.

## C. Verificación

Compare las salidas regeneradas con `results/analysis_outputs/` y use `SHA256SUMS.txt` para comprobar que los artefactos preservados no han cambiado.

## Alcance

La regeneración de tablas y análisis desde los JSON preservados es reproducción computacional. Una nueva consulta a modelos comerciales es una replicación y no puede garantizar resultados byte a byte idénticos.
