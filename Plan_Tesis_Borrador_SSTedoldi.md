
# Plan de tesis (borrador)

**Título (minúsculas):** clasificación arancelaria con nlp: modelos transformer y metamodelos de calidad para hs04/hs06  
**Tesista:** Santiago Sebastián Tedoldi  
**Director:** Dr. Bruno Bianchi  

## 1. Resumen
Este proyecto aborda la recomendación de posiciones arancelarias HS a partir de descripciones de mercaderías en lenguaje natural, un problema con miles de clases y fuerte desbalanceo. Se propone un enfoque orientado a producción que combine modelos *transformer* para clasificación con un **meta-modelo de calidad** para decidir cuándo confiar en la predicción (o escalar a revisión experta) y un **componente LLM** acotado para verificación/explicación. Se medirán *top-N accuracy*, cobertura vs. precisión (calibración), y costos/latencias.
  
## 2. Problema y motivación
La clasificación arancelaria afecta la recaudación, tiempos de liberación y riesgos de fraude. En contextos reales, las descripciones son breves, ambiguas y heterogéneas. El objetivo es elevar la calidad de la recomendación y entregar señales de confianza y explicabilidad útiles para decisión humana.

## 3. Antecedentes (TT1)
Se entrenó un clasificador basado en *DistilBERT* con *fine-tuning* parcial y total para HS04, logrando aproximadamente **Top-1 ~64%** y **Top-5 ~82%**. Se observaron clústeres semánticos claros en algunas secciones (p. ej., vehículos) y mayor dificultad en química/textiles.

## 4. Objetivos de esta tesis
1. **Baselines fuertes** con Doc2Vec y FastText para cuantificar el valor incremental de *transformers* considerando costo/latencia.  
2. **Meta-modelo de calidad** (features de confianza: entropía/dispersiones de probabilidad, longitud del texto, similitud semántica con catálogos HS) para calibrar cobertura vs. exactitud.  
3. **ML + LLM acotado** como verificador/explicador en casos difíciles, evaluando impacto en utilidad percibida y costo.  
4. **Ablaciones y reproducibilidad**: semillas, *splits* fijos, *error slicing*, reporte por familias HS, paquete reproducible.

## 5. Metodología
- **Datos**: descripciones (texto) + etiquetas HS06; derivación HS04/HS02; uso de textos oficiales HS para enriquecimiento semántico.  
- **Modelado**: (i) Doc2Vec/FastText → clasificadores lineales/árboles; (ii) *Transformer* (e.g., DistilBERT multilingüe) con *fine-tuning*; (iii) **meta-modelo** entrenado sobre *out-of-fold predictions* y métricas de confianza.  
- **LLM**: verificación/explicación *on-demand* solo para casos de baja confianza (prompts acotados y *caching*).  
- **Métricas**: Top-1/Top-5, precisión-cobertura (calibración), latencia, costo; análisis por *slices* (longitud, familia HS).

## 6. Plan de trabajo y cronograma
- **Q4‑2025**: Baselines (Doc2Vec/FastText), taxonomía de errores, suite de métricas con costo/latencia.  
- **Q1‑2026**: Meta-modelo de calidad y calibración; *error slicing* y análisis de cobertura.  
- **Q2‑2026**: Módulo LLM verificador/explicador; mini estudio de usabilidad (entrevista/encuesta breve).  
- **Q3‑2026**: Consolidación, ablation studies finales, *package* reproducible y redacción final.

## 7. Riesgos y mitigación
- **Drift / sesgo de etiquetas** → particiones estratificadas, validación temporal, *slices*.  
- **Costo/latencia LLM** → prompts mínimos, *caching*, limitar a baja confianza.  
- **Sobreajuste** → semillas fijas, *CV*, ablations, regularización.

## 8. Resultados esperados y contribuciones
- Mejora de *top‑N* y decisiones con confianza calibrada.  
- Procedimiento reproducible para Aduanas con guía de costos y umbrales de cobertura.  
- Componentes explicables que integren ML clásico, *transformers* y LLM de manera pragmática.

## 9. Aspectos éticos y de datos
Cumplimiento de marcos regulatorios y protección de datos; uso de información no identificable; resguardo de accesos y *logs*.

## 10. Bibliografía (inicial)
[Agregar referencias: informes WCO/IDB; artículos sobre HS classification; calibración y *conformal prediction*; Doc2Vec/FastText; *transformers* multilingües.]
