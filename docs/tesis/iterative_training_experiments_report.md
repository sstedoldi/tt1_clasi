# Reporte de entrenamientos iterativos (HS04)

## 1) Alcance
Este documento resume cómo se entrenaron y dónde se validan los resultados de:

- Baselines clásicos: **Doc2Vec (D2V)** y **FastText (FT)**.
- DistilBERT en GCP: **FE**, **PFT**, **FFT** y **FFT11**.

## 2) Baselines: configuración de entrenamiento

Notebook fuente: `01-HSrecomm_Baselines_HS04_iter_train.ipynb`.

### Flujo general
- Se lee dataset raw con columnas `HS06` y `GOODS_DESCRIPTION`.
- Se crea target `HS04` desde prefijo de 4 dígitos de `HS06`.
- Se crean variantes de texto:
  - `GOODS_DESCRIPTION` (RAW)
  - `PREPRO_DESCRIPTION` (PREPRO)
  - `PREPRO_DESCRIPTION + NGRAM_DESCRIPTION` (PREPRO+NGRAM)

### Iteraciones y semillas
- Se define `iterations = 10`.
- Se genera una lista de 10 semillas aleatorias (`seeds`) y se entrena una corrida por semilla.
- Validación por muestreo bootstrap con `test_fraction = 0.05`.

### Modelos entrenados
Para cada semilla y tipo de texto:
- **Doc2Vec**: entrenado y evaluado en top-1..top-5.
- **FastText**: entrenado y evaluado en top-1..top-5.

### Salida de resultados
Por cada combinación (modelo + tipo de texto) se guarda:
- CSV con métricas por iteración en `results/baselines/*.csv`.
- Joblib con dataframes de scoring en `results/baselines/*.joblib`.

## 3) DistilBERT: configuración de entrenamiento

Código fuente principal: `gcp/gcp_task.py` + YAMLs de job en `gcp/job_*.yml`.

### Flujo general en `gcp_task.py`
- Carga del dataset desde GCS (`--data_path`) o local.
- Limpieza básica: `dropna`, `drop_duplicates`.
- Target fijo `HS04 = HS06[:4]`.
- Input textual fijo en RAW: `text_col='GOODS_DESCRIPTION'`.
- Entrenamiento iterativo vía `distilbert_utils.iterative_training(...)`.

### Tipos de entrenamiento
Mapeo de `train_type`:
- `fe`: `fine_tune=False`, `n_finetune_layers=0` (Feature Encoder fijo).
- `fft`: `fine_tune=True`, `n_finetune_layers=0` (full fine-tuning).
- `pft`: `fine_tune=True`, `n_finetune_layers=2` (fine-tuning parcial).

### Jobs GCP y parámetros
- `job_fe.yml`: 5 iteraciones, hasta 20 épocas, batch 512, lr 5e-4, imagen v10.
- `job_pft.yml`: 5 iteraciones, hasta 12 épocas, batch 128, lr 1e-4, imagen v10.
- `job_fft.yml`: 5 iteraciones, hasta 7 épocas, batch 128, lr 5e-5, imagen v10.
- `job_fft11.yml`: 5 iteraciones, hasta 7 épocas, batch 128, lr 5e-5, imagen v11 y `--no-bootstrap`.

### Salida de resultados DistilBERT
Por tipo de entrenamiento, en `results/distilbert/<tipo>/`:
- `metrics_*.csv`: métricas finales top-1..top-5 por iteración.
- `history_all_iters_*.csv`: curva de entrenamiento por época e iteración (`train_loss`, `val_loss`, `val_topk`).
- `training_execution.log` / `experiments_..._training_execution.log`: trazas de ejecución.
- `training_summary.png` / `experiments_..._training_summary.png`: resumen visual del training.

## 4) Formato de resultados y cómo validarlos

### Baselines (`results/baselines`)
- Cada CSV tiene 10 filas (una por iteración) y columnas:
  - `top_1_acc`, `top_2_acc`, `top_3_acc`, `top_4_acc`, `top_5_acc`.
- El nombre de archivo codifica:
  - modelo (`D2V` o `FT`),
  - variante de texto (`GOODS_DESCRIPTION`, `PREPRO_DESCRIPTION`, `PREPRO_DESCRIPTION_NGRAM_DESCRIPTION`),
  - seed final en el nombre (familia de corrida).

### DistilBERT (`results/distilbert`)
- `metrics_*.csv`: 5 filas (una por iteración) con columnas:
  - `model`, `top_1_acc`, ..., `top_5_acc`.
- `history_all_iters_*.csv`: múltiples filas por `iteration` y `epoch`, con:
  - `train_loss`, `val_loss`,
  - `train_top1_acc`..`train_top5_acc`,
  - `val_top1_acc`..`val_top5_acc`.

## 5) Notebook de análisis creado

Se creó el notebook:

- `05-HSrecomm_Iterative_Results_Analysis.ipynb`

Incluye:
1. Carga y normalización de resultados de baselines y DistilBERT.
2. Gráficos de performance de baselines.
3. Gráficos de performance de DistilBERT + curvas de loss/accuracy por época.
4. Comparaciones entre baselines.
5. Comparaciones entre variantes DistilBERT.
6. Comparación mejor baseline vs mejor DistilBERT.
7. Exportación de tablas resumen en `results/iterative_analysis/`.

## 6) Archivos exportados por el notebook

- `results/iterative_analysis/baseline_summary_stats.csv`
- `results/iterative_analysis/distilbert_summary_stats.csv`
- `results/iterative_analysis/distilbert_best_epochs_by_iteration.csv`
- `results/iterative_analysis/best_baseline_vs_best_distilbert.csv`

