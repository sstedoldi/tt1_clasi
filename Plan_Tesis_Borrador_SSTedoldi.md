# Plan de tesis (borrador)

- **Título:** Clasificación arancelaria con procesamiento de lenguaje natural  
- **Tesista:** Santiago Sebastián Tedoldi  
- **Director:** Dr. Bruno Bianchi  

## 1. Resumen
Este proyecto aborda la **recomendación de posiciones arancelarias HS** a partir de descripciones de mercaderías en lenguaje natural. Es un problema **multiclase** con **miles de etiquetas** y **fuerte desbalanceo**. Se compararán modelos *transformer* de última generación contra *baselines* robustos de representación distribuida.  
La **métrica principal** será *top-N accuracy* (N ∈ {1,…,5}), complementada por métricas por-clase para capturar el desbalance.

## 2. Problema y motivación
La clasificación arancelaria impacta en recaudación, tiempos de liberación y riesgos de fraude. En la práctica, las descripciones comerciales suelen ser **breves, ambiguas y heterogéneas**, dificultando la tarea. El objetivo es **asistir la clasificación** a nivel HS04, priorizando precisión en Top-N y explicabilidad básica.

## 3. Antecedentes (TT1)
Se entrenó un clasificador con **DistilBERT** (transfer learning, *fine-tuning* parcial y total) para **HS04**, obteniendo aprox. **Top-1 ≈ 64 %** y **Top-5 ≈ 82 %**. Se observaron **clústeres semánticos** claros (p.ej., vehículos) y **mayor dificultad** en química/textiles, coherente con la ambigüedad de descripciones.

## 4. Objetivos de esta tesis
**Objetivo general.** Evaluar comparativamente representaciones de texto y estrategias de entrenamiento para **recomendar códigos HS04** con alta *top-N accuracy* y análisis de error accionable.  

**Objetivos específicos:**
1. **Preprocesamiento**: normalización, *stopwords*, *n-gramas*; medir cuándo ayudan/perjudican la clasificación.  
2. **Baselines fuertes:** entrenar y evaluar **Doc2Vec** y **FastText** para cuantificar el valor incremental de *transformers*.  
3. **Comparativa:** protocolo único de *splits*, semillas y métricas (macro/weighted).  
4. **Análisis del error:** identificar patrones de confusión por capítulo/partida y casos “difíciles”.  
5. **Documentación:** *reporting* técnico y buenas prácticas.

## 5. Datos
- **Datos:** descripciones comerciales (texto) + etiquetas **HS04**. 
- **Idioma:** 100 % descripciones en inglés.
- **Fuente:** capacitación internacional en el programa BACUDA de la Organización Mundial de Aduanas (WCO).
- **Tamaño:** 500 mil casos.

## 6. Metodología
- **Preprocesamiento (ablación on/off):** limpieza básica, minúsculas, normalización, *stopwords*, *n-gramas* (cuando aplique).  
- **Modelado:**
  - **Encoders de texto:**
    - (i) **Doc2Vec**  
    - (ii) **FastText**  
    - (iii) **DistilBERT** con *fine-tuning*
  - **Clasificador (para todas las variantes):** MLP **FCN** con dos capas lineales (in_dim → 1024 → n_classes), **ReLU + BatchNorm + Dropout=0.3**.
  - **Entrenamiento y selección de modelo:** *stratified split* train/val/test; semillas fijas; *early-stopping* y *weight decay*.
- **Métricas y análisis del error:**
  - **Top-N accuracy** (N=1..5); **macro-F1**, **balanced accuracy**.  
  - **Matriz de confusión** por secciones/capítulos y **curvas de cobertura-precisión** por umbral de confianza.  
  - **Estudio de patrones** de mayores pérdidas y error.

## 7. Plan de trabajo y cronograma
- **Dic-2025:** Preprocesamiento (ablaciones) + *baselines* (**Doc2Vec/FastText**) entrenados y evaluados.  
- **Feb-2026:** *Fine-tuning* DistilBERT + **comparativa integral** (tablas y *plots*); **análisis del error**.  
- **Mar-2026:** **Redacción y documentación** (metodología, resultados, limitaciones, reproducibilidad).  
- **Abr-2026:** **Defensa y presentación** (presentación con *insights* principales y la documentación).

## 8. Riesgos y mitigación
- **Sobreajuste:** *dropout*, *early-stopping*, *weight decay*.  
- **Calidad/dominio de datos:** tendrá en cuenta durante el análisis del error.  
- **Desbalance de clases:** *class weights*, muestreo estratificado y reporte macro.

## 9. Resultados esperados y contribuciones
- **Cuantificación** del valor incremental de *transformers* vs *baselines* en HS04.  
- **Mejor encoder** seleccionado, con/sin preprocesamiento, con **justificación empírica**.  
- Nuevo **Baseline sólido** para la comunidad.  
- **Entendimiento del error** cometido por el nuevo baseline.

## 10. Aspectos éticos y de datos
Se garantizará **cumplimiento normativo**, **protección de datos** y uso de información **no identificable**.

## 11. Bibliografía (inicial)
- Bojanowski, P. G. (2017). Enriching Word Vectors with Subword Information.
- Commission, E. (2025). Council decision on the position to be taken on behalf of the European Union in the World Customs Organization Council in relation to a WCO Article 16 Recommendation amending the Harmonised System. Brussels: COM(2025) 235 final - 2025/0114(NLE).
- Devlin, J. C. (2019). Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Harsani, P. A. (2020). A Study using Machine Learning with NGram Model in Harmonized System Classification.
- Ignacio Marra de Artiñano, F. R. (2023). Automatic Product Classification in International Trade: Machine Learning and Large Language Models.
- Lau, J. H. (2016). An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation.
- Lee, E. K. (2021). Classification of Goods Using Text Descriptions With Sentences Retrieval.
- Lee, E. K. (2023). Explainable Product Classification for Customs.
- Sanh, V. D. (2020). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
- UNCTAD. (2025). Global trade update. 
- VUCE. (29 de 06 de 2025). Obtenido de Ventanilla �nica del Comercio Exterior - Argentina: https://www.vuce.gob.ar/
- Wikipedia. (28 de 06 de 2025). Harmonized System. Obtenido de Wikipedia: https://en.wikipedia.org/wiki/Harmonized_System
- Xie, Y. S. (2022). Text classification in shipping industry using unsupervised models and Transformer based supervised models.