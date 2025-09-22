# Clasificación Arancelaria con NLP

Este repositorio recopila el trabajo de taller de tesis de la maestría de **Maestría en Explotación de Datos y Generación del Conocimiento (UBA)** orientado a la clasificación arancelaria de mercaderías a partir de descripciones comerciales en texto libre. El objetivo inicial fue validar la factibilidad de entrenar modelos basados en *transformers* para recomendar códigos de la nomenclatura armonizada (HS) a nivel de partida (HS04), a pesar de contar con un dataset altamente desbalanceado y con miles de clases.

El proyecto evoluciona hacia la tesis de la **Maestría en Explotación de Datos y Generación del Conocimiento (UBA)**, donde se ampliará la batería de modelos, el análisis de errores y la capacidad de explicar las predicciones.

## Contenido del repositorio

- **`data/`**: fuentes principales de datos, incluyendo el corpus de descripciones comerciales, la nomenclatura HS y derivados utilizados para EDA y entrenamiento. Contiene particiones predefinidas (`train_data/`, `test_data/`) y tablas auxiliares con features agregados.
- **`docs/`**: material de referencia (papers, presentaciones y notas) utilizado para contextualizar el problema y documentar avances académicos.
- **`eda/`**: artefactos y perfiles exploratorios generados durante el análisis de datos.
- **`sample_data/`**: muestras reducidas para comprender la estructura de los códigos HS y ensayar pipelines en un entorno liviano.
- **`deployment/`**: prototipos de inferencia, incluyendo un predictor Doc2Vec temprano (`HscodePredict.py`) y notebooks de despliegue.
- **`results/`**: visualizaciones interactivas (PCA, t-SNE) y métricas de entrenamiento para las diferentes variantes del modelo DistilBERT fine-tuned a HS04.
- **Notebooks (`*.ipynb`)**: guían todo el flujo de trabajo, desde el EDA (`HSrecomm_EDA.ipynb`) hasta el entrenamiento, evaluación de errores y experimentos de transferencia.

## Dataset

El corpus principal está compuesto por ~500k tuplas `(GOODS_DESCRIPTION, HS06)`. A partir de este insumo se derivan los códigos HS02 y HS04, y se incorporan descripciones oficiales de la nomenclatura HS06 para análisis de similitud y enriquecimiento semántico.

Durante el EDA se generaron variables agregadas (longitud, indicadores de sub-tokenización) que permiten caracterizar la calidad de las descripciones y cuantificar el desbalance extremo entre códigos.

## Metodología de trabajo

1. **Profiling y EDA**: depuración básica, análisis de frecuencias por nivel HS, ingeniería de variables de longitud y tokenización, y evaluación de similitud entre descripciones comerciales y nomenclaturas oficiales mediante DistilBERT pre-entrenado.
2. **Representaciones**: generación de *embeddings* con `distilbert-base-uncased` y reducción de dimensionalidad (PCA, t-SNE, UMAP) para identificar clústeres temáticos y evaluar la separación de clases.
3. **Arquitectura encoder-classifier**: se construyó una clase PyTorch que combina el encoder DistilBERT y un clasificador feed-forward adaptable al número de clases objetivo. Se exploraron tres configuraciones: transfer learning congelando el encoder, fine-tuning parcial (capas superiores) y fine-tuning total.
4. **Evaluación**: se reportaron métricas *Top-N accuracy* en el set de validación, alcanzando un 63.9 % Top-1 y 82.3 % Top-5 con el modelo ajustado de extremo a extremo.
5. **Visualización post-ajuste**: comparación entre embeddings originales y fine-tuned para observar mejoras en la separabilidad de capítulos HS a través de proyecciones 2D/3D.
6. **Error Analysis preliminar**: revisión cualitativa de predicciones con baja confianza y casos representativos para motivar el desarrollo de un metamodelo de calidad.

## Configuración del entorno

1. Crear un entorno virtual (Python 3.10+ recomendado) e instalar dependencias:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   > Para GPU NVIDIA es necesario instalar PyTorch desde el índice oficial de CUDA 12.1 (ver comentarios en `requirements.txt`).

2. Descargar o ubicar los conjuntos de datos en `data/`. Los notebooks asumen rutas relativas y utilizan las particiones predefinidas en `train_data/` y `test_data/`.

3. Lanzar Jupyter Lab/Notebook para ejecutar los cuadernos en el orden sugerido: EDA → entrenamiento → evaluación → despliegue.

## Resultados actuales

- **Top-N accuracy (HS04)**

  | Configuración | Top-1 | Top-3 | Top-5 |
  | ------------- | ----- | ----- | ----- |
  | Transfer learning (encoder congelado) | 50.5 % | 66.1 % | 72.2 % |
  | Fine-tuning 2 capas superiores | 63.3 % | 77.2 % | 81.7 % |
  | Fine-tuning total | **63.9 %** | **77.9 %** | **82.3 %** |

- **Insights clave**: las descripciones de vehículos (HS02 87) forman clústeres claros, mientras que categorías químicas y textiles requieren modelos más expresivos o datos adicionales. La similitud promedio entre descripciones comerciales y nomenclaturas oficiales ronda 0.84, lo que respalda la eficacia del enfoque basado en embeddings contextuales.

## Ruta hacia la tesis de maestría

### Extensiones inmediatas

- Implementar **modelos baselines** (Doc2Vec, FastText) para cuantificar el valor agregado del fine-tuning de DistilBERT y documentar la comparativa de desempeño y costo computacional.
- Analizar la **entropía y dispersión** de las probabilidades de salida del clasificador para caracterizar la confianza del modelo y detectar predicciones ambiguas.

### Líneas de investigación futura

- Profundizar en el **análisis de errores**: estudiar sistemáticamente los casos donde el modelo falla (por capítulo, longitud de texto, similitud semántica) e identificar patrones que guíen mejoras de datos o arquitectura.
- Entrenar un **metamodelo de control de calidad** que, a partir de las probabilidades, la longitud del texto y medidas de similitud, determine cuándo una predicción es confiable o debe escalarse a revisión humana.
- Evaluar modelos pre-entrenados en **español** y ampliar el dataset con descripciones multilingües para soportar escenarios reales en Argentina (NCM/SIM).
- Explorar soluciones **híbridas ML + LLMs**, donde un LLM actúe como verificador o generador de explicaciones, integrando interpretabilidad y asistencia interactiva.
- Escalar la investigación a niveles HS06 y regionales, incluyendo estrategias de **transfer learning extendido** (más epochs, curriculum learning) y técnicas de manejo de desbalance extremo.

## Cómo contribuir

1. Registrar los experimentos (hiperparámetros, semillas y métricas) en los cuadernos o en archivos bajo `results/`.
2. Incluir referencias bibliográficas y fuentes externas en `docs/` para mantener el contexto académico del proyecto.
3. Documentar scripts y notebooks con celdas explicativas orientadas a lectores futuros de la tesis.

## Referencias

La bibliografía principal se encuentra listada en el documento del trabajo de especialización e incluye investigaciones sobre clasificación arancelaria, técnicas de embeddings y modelos *transformer*. Para citas completas, consultar el apartado "Bibliografía" dentro de la documentación del proyecto.

---

> Última actualización: septiembre de 2025. Este repositorio continuará evolucionando conforme se iteren las etapas de la tesis de maestría.
