# Clasificación arancelaria con procesamiento de lenguaje natural

Repositorio de la tesis de la **Maestría en Explotación de Datos y Descubrimiento de Conocimiento (UBA)** de Santiago S. Tedoldi, dirigida por el Dr. Bruno Bianchi.

El proyecto estudia la recomendación asistida de partidas arancelarias **HS04** a partir de descripciones comerciales en texto libre. Compara baselines basados en Doc2Vec y FastText con distintas estrategias de ajuste de DistilBERT, y complementa la evaluación Top-K con análisis de error, explicabilidad y scoring de confianza.

> La herramienta está planteada como apoyo a la decisión: propone un conjunto acotado de códigos candidatos y señales de confianza para priorizar la revisión humana. No reemplaza el criterio de un especialista ni las reglas legales de clasificación.

## Informe de tesis

La versión más reciente incluida en el repositorio es [Tesis_tedoldi_p1.pdf](docs/tesis/Tesis_tedoldi_p1.pdf), fechada el **10 de julio de 2026**. El documento describe el contexto, marco teórico, diseño experimental, resultados, análisis del error, limitaciones y líneas futuras.

## Problema y datos

El caso de estudio utiliza aproximadamente 500 mil pares `(GOODS_DESCRIPTION, HS06)` en inglés provenientes del proyecto BACUDA de la Organización Mundial de Aduanas. A partir de HS06 se deriva la etiqueta HS04.

La tarea tiene las siguientes características:

- clasificación multiclase sobre **1.133 partidas HS04** observadas;
- fuerte desbalance entre clases;
- descripciones breves, heterogéneas, ruidosas y a menudo ambiguas;
- evaluación mediante accuracy acumulada Top-1 a Top-5, coherente con un sistema recomendador.

Los datos crudos no se versionan en Git. Los notebooks esperan archivos bajo `data/`; es necesario obtenerlos por la vía autorizada y respetar las rutas indicadas en cada cuaderno.

## Metodología

1. **EDA y representación:** perfilado del corpus, análisis del desbalance y de la tokenización, embeddings contextuales y proyecciones PCA/t-SNE.
2. **Baselines:** Doc2Vec y FastText sobre texto crudo (`RAW`), texto normalizado (`PREPRO`) y texto normalizado con n-gramas (`PREPRO+NGRAM`). Los resultados se estiman sobre 10 iteraciones.
3. **DistilBERT:** comparación entre encoder fijo (`FE`), fine-tuning parcial de dos capas (`PFT`) y fine-tuning completo (`FFT`/`FFT11`). Los experimentos iterativos se ejecutaron en Vertex AI con GPU NVIDIA L4.
4. **Análisis del error:** agregación por HS04, variables estructurales y diagnósticas, modelos explicativos y análisis SHAP.
5. **Uso y extensiones:** inferencia por lote, scoring de confianza y evaluación exploratoria de un esquema DistilBERT + LLM.

## Resultados principales

Promedios de los experimentos iterativos versionados en `results/iterative_analysis/`:

| Modelo | Variante | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Doc2Vec | PREPRO+NGRAM | 60,84 % | 68,48 % | 71,49 % | 73,15 % | 74,31 % |
| FastText | PREPRO | 57,66 % | 70,56 % | 76,37 % | 79,80 % | 82,05 % |
| DistilBERT | FFT11 | **66,37 %** | **75,53 %** | **79,76 %** | **82,22 %** | **83,88 %** |

El entrenamiento final FFT, con semilla 32 y validación del 1 %, obtuvo **66,72 % Top-1** y **83,83 % Top-5**. Sus métricas y configuración están en `results/distilbert/fft_final/`.

El análisis concluye que:

- DistilBERT con fine-tuning completo es la alternativa más efectiva y estable;
- el preprocesamiento no afecta por igual a todas las familias: Doc2Vec se beneficia de n-gramas y FastText del texto normalizado;
- las variables estructurales explican poco del rendimiento por clase (`R² = 0,043`), mientras que las señales de probabilidad e incertidumbre elevan el poder explicativo (`R² = 0,326`);
- el desempeño no es homogéneo entre partidas y existen errores de alta confianza, por lo que las probabilidades no deben interpretarse como certeza ni usarse sin controles.

La prueba con un LLM se conserva como exploración sobre una muestra pequeña (267 casos); no mejora el Top-1 del modelo final y no constituye el resultado principal de la tesis.

## Estructura del repositorio

```text
.
├── 00-HSrecomm_EDA.ipynb
├── 01-HSrecomm_Baselines_HS04_*.ipynb
├── 02-HSrecomm_DistiltBERT_HS04_train.ipynb
├── 03-HSrecomm_DistiltBERT_HS04_viz.ipynb
├── 04-HSrecomm_iter_results_analysis.ipynb
├── 05-HSrecomm_DistiltBERT_HS04_test&use.ipynb
├── 06-HSrecomm_DistiltBERT_HS04_error.ipynb
├── 07-HSrecomm_DistiltBERT_HS04_scoring.ipynb
├── 10-HSrecomm_Sampling_val.ipynb
├── 11-HSrecomm_DistiltBERT_HS04_genai.ipynb
├── data/                       # datos locales; ignorados por Git
├── deployment/                 # prototipos históricos de inferencia
├── docs/                       # tesis, diseños experimentales y bibliografía
├── eda/                        # perfiles y visualizaciones exploratorias
├── gcp/                        # entrenamiento en Vertex AI y configuración de jobs
├── results/                    # métricas, logs, figuras y predicciones versionadas
├── doc2vec_utils.py
├── fasttext_utils.py
└── hs04_distilbert_inference.py
```

`0-old/`, `models/`, `data/`, los entornos virtuales y los pesos binarios se excluyen mediante `.gitignore`.

## Instalación

Se recomienda Python 3.10–3.12. Algunas librerías científicas o de GPU pueden no disponer todavía de ruedas para versiones más nuevas.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux o WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

La instalación estándar de `torch` desde PyPI es suficiente para CPU. Para usar una GPU NVIDIA, instale primero la distribución de PyTorch compatible con la versión de CUDA del equipo siguiendo la documentación oficial y luego ejecute `pip install -r requirements.txt`.

Los jobs de Vertex AI utilizan un entorno más acotado, definido por separado en `gcp/requirements.txt` y la imagen base indicada en `gcp/Dockerfile`.

## Flujo sugerido

Para reproducir el recorrido completo, abra Jupyter Lab y ejecute los cuadernos según el prefijo numérico:

```powershell
jupyter lab
```

- `00`: exploración y visualización;
- `01`: baselines e iteraciones;
- `02`–`03`: entrenamiento y visualización de DistilBERT;
- `04`: comparación de experimentos iterativos;
- `05`: prueba e inferencia;
- `06`: análisis del error;
- `07`: scoring de confianza;
- `10`: auditoría del muestreo;
- `11`: evaluación exploratoria con GenAI.

Los notebooks contienen resultados persistidos, pero varios entrenamientos requieren datos locales y hardware acelerado. Las credenciales de Google Cloud y OpenAI solo son necesarias para los cuadernos que consumen esos servicios; no deben guardarse en el repositorio.

## Inferencia con el modelo final

La interfaz reusable se encuentra en `hs04_distilbert_inference.py`:

```python
from hs04_distilbert_inference import HS04DistilBERTPredictor

predictor = HS04DistilBERTPredictor()
result = predictor.predict("air cleaner filter assembly for diesel engine", top_k=5)
print(result["predictions"])
```

Para ejecutar este ejemplo debe existir el archivo no versionado:

```text
results/distilbert/fft_final/final_model/pytorch_model.bin
```

La configuración y el diccionario de etiquetas sí están versionados. El tokenizador y el encoder base `distilbert-base-uncased` se descargan desde Hugging Face en el primer uso, salvo que ya estén disponibles en caché.

## Artefactos relevantes

- `results/iterative_analysis/`: resúmenes comparativos de baselines y DistilBERT.
- `results/distilbert/fft_final/final_metrics.json`: métricas del entrenamiento final.
- `results/distilbert/fft_final/final_val_predictions.csv`: predicciones de validación.
- `docs/tesis/baselines_experimental_design.md`: diseño de baselines.
- `docs/tesis/distilbert_experimental_design.md`: diseño de DistilBERT.
- `docs/tesis/distilbert_scoring_analysis.md`: análisis del scoring.
- `docs/tesis/iterative_training_experiments_report.md`: reporte de experimentos iterativos.

## Limitaciones y trabajo futuro

El estudio se limita a datos en inglés, una fuente no identificada por país y etiquetas HS04. Antes de cualquier uso operativo se requiere validación externa, calibración, monitoreo y revisión normativa. Las líneas futuras incluyen HS06/NCM/SIM, textos normativos y notas explicativas, escenarios multilingües, recuperación semántica y RAG, re-ranking experto, explicabilidad interna del Transformer y validación productiva del scoring.

---

Última actualización de la documentación: **agosto de 2026**.
