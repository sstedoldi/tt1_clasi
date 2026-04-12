# DistilBERT HS04: confidence scoring, dual target scales, and selective rejection

Fuentes principales: [07-HSrecomm_DistiltBERT_HS04_scoring.ipynb](../../07-HSrecomm_DistiltBERT_HS04_scoring.ipynb), [results/distilbert/fft_final/final_val_predictions.csv](../../results/distilbert/fft_final/final_val_predictions.csv), [results/distilbert/fft_final/final_metrics.json](../../results/distilbert/fft_final/final_metrics.json), [results/distilbert/fft_final/final_model/training_config.json](../../results/distilbert/fft_final/final_model/training_config.json), [hs04_distilbert_inference.py](../../hs04_distilbert_inference.py) y [data/hs06_full_eng.csv](../../data/hs06_full_eng.csv).

## Objetivo

Este notebook no vuelve a entrenar el clasificador principal `DistilBERT`, sino que construye una capa adicional de scoring sobre las predicciones del modelo final `FFT`. La idea es estimar, para cada observacion individual, cuan confiable es la recomendacion emitida por el clasificador y si conviene aceptarla automaticamente o derivarla a revision humana.

La nueva version del analisis extiende la prueba de concepto original en cuatro direcciones:

- compara dos escalas ordinales alternativas para el target del meta-modelo;
- incorpora features semanticas production-safe calculadas sobre `Top1` a `Top5`;
- controla el riesgo de overfitting mediante filtrado de features dentro de cada fold;
- y evalua los resultados con validacion cruzada repetida, en lugar de depender de una sola particion hold-out interna.

## Contexto del experimento

El punto de partida sigue siendo el modelo final entrenado en `results/distilbert/fft_final/final_model`, correspondiente a una corrida unica de `full fine-tuning` sobre `distilbert-base-uncased`, con:

- `target_col = HS04`
- `text_col = GOODS_DESCRIPTION`
- `max_length = 300`
- `batch_size = 128`
- `lr = 5e-5`
- `seed = 32`
- hold-out final del `1 %`

Sobre ese hold-out final se dispone de `2.677` observaciones y el notebook recupera embeddings `[CLS]` de dimension `768` para todas las descripciones. Tambien se embeben `942` textos legales HS04 construidos a partir de `data/hs06_full_eng.csv`.

Las metricas directas del clasificador principal, ya reportadas en `final_metrics.json`, son:

| Metrica | Valor |
| --- | ---: |
| Top-1 accuracy | 66,72 % |
| Top-2 accuracy | 75,46 % |
| Top-3 accuracy | 79,68 % |
| Top-4 accuracy | 82,03 % |
| Top-5 accuracy | 83,83 % |

Estas metricas siguen describiendo la calidad promedio del ranking del clasificador, pero no resuelven por si solas el problema operativo de decidir cuando conviene confiar en una recomendacion concreta.

## Enfoque metodologico

### 1. Embeddings del hold-out y de la nomenclatura legal

El notebook reutiliza el encoder del modelo final para obtener embeddings de:

- `2.677` descripciones del hold-out;
- `942` textos legales HS04.

Esto permite construir senales adicionales de confianza sin volver a entrenar el clasificador base.

### 2. Similaridad semantica production-safe

La version anterior utilizaba una similitud coseno entre la descripcion observada y el texto legal del `True Label`. Esa variable es util para diagnostico, pero no es production-safe porque la etiqueta verdadera no esta disponible al momento de inferencia.

Por eso, en la nueva version se distinguen dos tipos de variables:

- `cosine_sim_desc_vs_true_hs_text`: solo diagnostica, excluida del meta-modelo;
- `cosine_sim_desc_vs_top1_hs_text` a `cosine_sim_desc_vs_top5_hs_text`: features candidatas para produccion.

Con esto, la capa de scoring se apoya exclusivamente en informacion disponible al momento de usar el sistema.

### 3. Feature engineering del meta-modelo

El conjunto de variables del meta-modelo combina dos familias:

1. Variables de probabilidad:

- `Proba Top1` a `Proba Top5`
- `prob_std`
- `prob_range`
- `prob_margin_1_2`
- `prob_margin_1_3`
- `prob_entropy_norm`
- `prob_weighted_rank_score`

2. Variables de similitud semantica con la nomenclatura legal:

- `cosine_sim_desc_vs_top1_hs_text` a `cosine_sim_desc_vs_top5_hs_text`
- `cosine_pred_mean`
- `cosine_pred_std`
- `cosine_pred_range`
- `cosine_margin_1_2`
- `cosine_weighted_by_proba`
- `joint_top1_signal`

Este conjunto busca capturar simultaneamente:

- cuan concentrada o dispersa esta la distribucion de probabilidad;
- cuan alineada semanticamente esta la descripcion con las alternativas sugeridas;
- y cuan consistente es la primera recomendacion en terminos probabilisticos y semanticos.

### 4. Dos targets ordinales alternativos

El notebook compara dos definiciones del target a predecir:

1. Escala `10_to_0`

- `10` si la clase verdadera coincide con `Top1`
- `9` si coincide con `Top2`
- `8` si coincide con `Top3`
- `7` si coincide con `Top4`
- `6` si coincide con `Top5`
- `0` si no aparece en el top-5

2. Escala `5_to_0`

- `5` si la clase verdadera coincide con `Top1`
- `4` si coincide con `Top2`
- `3` si coincide con `Top3`
- `2` si coincide con `Top4`
- `1` si coincide con `Top5`
- `0` si no aparece en el top-5

La distribucion observada es:

| Posicion correcta | Casos | Score `10_to_0` | Score `5_to_0` |
| --- | ---: | ---: | ---: |
| fuera del Top-5 | 433 | 0 | 0 |
| Top-5 | 48 | 6 | 1 |
| Top-4 | 63 | 7 | 2 |
| Top-3 | 113 | 8 | 3 |
| Top-2 | 234 | 9 | 4 |
| Top-1 | 1.786 | 10 | 5 |

La escala `10_to_0` es mas dispersa y deja huecos entre el score `0` y los aciertos top-k, mientras que `5_to_0` comprime esa informacion en una escala mas compacta.

### 5. Control de overfitting y validacion cruzada repetida

Dado que la muestra de evaluacion es acotada, el notebook no usa una unica particion `80/20`. En su lugar, utiliza:

- `RepeatedStratifiedKFold`
- `5` folds
- `10` repeticiones

En total, cada combinacion de target y modelo se evalua sobre `50` particiones.

Ademas, dentro de cada fold de entrenamiento:

- se eliminan columnas constantes;
- se imputan faltantes con la mediana del fold;
- se filtran variables altamente correlacionadas;
- y se conservan las mejores `10` features segun mutual information.

Esto reduce el riesgo de overfitting y hace que la comparacion entre modelos sea metodologicamente mas robusta.

### 6. Modelos tabulares comparados

Para cada una de las dos escalas se ajustan tres meta-modelos de regresion:

- `HistGradientBoostingRegressor`
- `RandomForestRegressor`
- `XGBoostRegressor`

El problema no es clasificar codigos HS, sino anticipar la calidad esperada de la recomendacion emitida por el clasificador principal.

## Resultados

### 1. Estabilidad en la seleccion de features

La seleccion de features fue notablemente estable entre folds y entre ambas escalas. Las variables elegidas con mayor frecuencia fueron:

- `Proba Top2`
- `Proba Top3`
- `Proba Top5`
- `cosine_sim_desc_vs_top1_hs_text`
- `cosine_weighted_by_proba`
- `joint_top1_signal`

En ambas escalas, estas features aparecen en el `100 %` de los folds. Tambien aparecen con mucha frecuencia:

- `Proba Top4` (`98 %`)
- `cosine_pred_mean` (`82 %` a `84 %`)
- `prob_range` (`80 %`)

Esto sugiere que la capa de scoring aprende de dos fuentes complementarias:

- la estructura probabilistica del ranking top-k;
- y la alineacion semantica entre la descripcion y la nomenclatura legal de los codigos sugeridos.

### 2. Desempeno de regresion por modelo y por escala

Los resultados promedio de la validacion cruzada repetida fueron los siguientes:

| Escala | Modelo | MAE medio | RMSE medio | R² medio |
| --- | --- | ---: | ---: | ---: |
| `10_to_0` | RandomForest | 1,954 | 3,034 | 0,302 |
| `10_to_0` | XGBoost | 1,966 | 3,087 | 0,277 |
| `10_to_0` | HistGradientBoosting | 2,012 | 3,209 | 0,219 |
| `5_to_0` | RandomForest | 1,035 | 1,527 | 0,340 |
| `5_to_0` | XGBoost | 1,039 | 1,553 | 0,317 |
| `5_to_0` | HistGradientBoosting | 1,058 | 1,614 | 0,263 |

En ambas escalas:

- `RandomForest` fue el mejor meta-modelo;
- `XGBoost` quedo muy cerca, pero por debajo;
- `HistGradientBoosting` fue sistematicamente el mas debil.

Si se mira solo el ajuste de regresion, la escala `5_to_0` parece superior, ya que logra:

- menor `MAE`,
- menor `RMSE`,
- y mayor `R²`.

Sin embargo, como el objetivo final no es un score continuo en si mismo, sino separar mejor casos aceptables de casos que conviene rechazar, esa conclusion debe complementarse con una mirada operativa.

### 3. Politica de rechazo selectivo

Para cada modelo y para cada escala se evaluaron umbrales sobre el score predicho. El mejor umbral fue muy estable:

- `> 7.5` para la escala `10_to_0`
- `> 3.5` para la escala `5_to_0`

Los mejores resultados por modelo fueron:

| Escala | Modelo | Threshold | Coverage | Precision among accepted | Precision lift | Good-case recall | Bad-case filter |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `10_to_0` | RandomForest | 7,5 | 68,58 % | 93,82 % | +9,99 pp | 76,74 % | 73,71 % |
| `10_to_0` | XGBoost | 7,5 | 70,72 % | 92,86 % | +9,03 pp | 78,33 % | 68,72 % |
| `10_to_0` | HistGradientBoosting | 7,5 | 70,70 % | 92,39 % | +8,57 pp | 77,92 % | 66,69 % |
| `5_to_0` | RandomForest | 3,5 | 69,07 % | 93,54 % | +9,72 pp | 77,06 % | 72,33 % |
| `5_to_0` | XGBoost | 3,5 | 70,34 % | 93,02 % | +9,20 pp | 78,04 % | 69,60 % |
| `5_to_0` | HistGradientBoosting | 3,5 | 70,35 % | 92,64 % | +8,81 pp | 77,73 % | 67,89 % |

Las diferencias entre escalas son pequenas, pero consistentes:

- la escala `10_to_0` logra un poco mas de precision lift;
- y tambien un poco mas de filtrado de casos malos;
- mientras que `5_to_0` ofrece cobertura y recall casi equivalentes, con mejor ajuste de regresion.

### 4. Recomendacion general del notebook

La tabla final del notebook selecciona como mejor combinacion global a:

- escala `10_to_0`
- modelo `RandomForest`
- umbral recomendado `> 7.5`

Esta recomendacion se apoya en que, frente a la alternativa `5_to_0`, el mejor `RandomForest` de `10_to_0` obtiene:

- mejor `threshold_utility_score` (`68,10` vs `67,85`);
- mayor precision sobre los aceptados (`93,82 %` vs `93,54 %`);
- mayor lift sobre baseline (`+9,99` pp vs `+9,72` pp);
- y mejor capacidad de filtrar casos malos (`73,71 %` vs `72,33 %`).

La ventaja no es grande, por lo que no debe leerse como una diferencia concluyente, sino como una evidencia moderada a favor de la escala mas dispersa cuando el objetivo es la separacion operativa entre aceptar y rechazar.

## Interpretacion sustantiva

El hallazgo principal de la nueva version es que el output del clasificador principal contiene informacion suficiente como para construir una capa adicional de decision, y que esa informacion no se limita a las probabilidades top-k. La similitud semantica con los codigos propuestos aparece de manera estable como una de las senales mas utiles del meta-modelo.

Esto refuerza la idea de un sistema de clasificacion selectiva:

- el clasificador principal genera un ranking de codigos HS04;
- el meta-modelo estima la calidad esperada de esa recomendacion;
- y una politica de threshold decide si conviene aceptar automaticamente o derivar a revision humana.

En un entorno aduanero, esta arquitectura puede ser mas valiosa que una clasificacion "siempre obligatoria", porque redistribuye el riesgo: automatiza los casos mas claros y preserva la revision experta para los mas ambiguos.

## Limitaciones y cuidados de interpretacion

### 1. El scoring sigue siendo una capa post-hoc

El notebook no modifica el `DistilBERT` final ni mejora su entrenamiento. Lo que hace es modelar ex post la calidad esperada de una prediccion ya emitida.

### 2. La validacion sigue ocurriendo dentro del hold-out final del clasificador

La nueva version mejora mucho la robustez interna al usar validacion cruzada repetida, pero sigue trabajando sobre el mismo archivo `final_val_predictions.csv`. Por eso, los resultados son mas fuertes que los de una sola particion, aunque todavia deberian presentarse como evidencia pre-productiva y no como validacion externa definitiva.

### 3. El target es ordinal, no probabilistico

Tanto `10_to_0` como `5_to_0` son escalas ordinales de utilidad operativa. No equivalen a probabilidades calibradas de acierto, sino a una medida sintetica de la posicion de la clase correcta dentro del ranking top-k.

### 4. La comparacion entre escalas depende del criterio de negocio

La escala `5_to_0` ajusta mejor como regresion, pero la escala `10_to_0` separa ligeramente mejor los casos aceptables de los rechazables. La eleccion final depende de si se prioriza:

- interpretabilidad y compactacion del target, o
- capacidad de screening para una politica de abstencion.

## Texto sugerido para la tesis

Con el objetivo de complementar la evaluacion del modelo final DistilBERT, se desarrollo un notebook de scoring orientado a estimar la confianza de cada recomendacion individual. A diferencia de una evaluacion limitada a metricas top-k, este enfoque busca responder no solo que tan bien rankea el clasificador las clases correctas en promedio, sino tambien con que nivel de certeza conviene aceptar automaticamente una prediccion concreta. Para ello se partio del archivo de predicciones del hold-out final (`n = 2.677`) y se construyo una capa auxiliar de features basada en dos familias de variables: por un lado, probabilidades top-k y medidas de dispersion de la salida del clasificador; por otro, medidas de similitud coseno entre la descripcion comercial y los textos legales asociados a los codigos HS04 recomendados (`Top1` a `Top5`). Esta ultima extension permitio reemplazar una variable diagnostica basada en la etiqueta verdadera por senales production-safe disponibles al momento de inferencia.

Sobre esa base se compararon dos definiciones ordinales alternativas del target de scoring: una escala `10-to-0`, que asigna valor `10` si la clase correcta coincide con la primera recomendacion y desciende hasta `6` si aparece en la quinta posicion, y una escala mas compacta `5-to-0`, que resume la misma informacion entre `5` y `1`, con `0` para los casos fuera del top-5. Para modelar ambas escalas se evaluaron tres meta-modelos tabulares (`HistGradientBoostingRegressor`, `RandomForestRegressor` y `XGBoostRegressor`) bajo un esquema de validacion cruzada repetida de `5` folds y `10` repeticiones, incorporando ademas un filtrado de features dentro de cada fold de entrenamiento para reducir riesgo de overfitting. Los resultados muestran que `RandomForest` fue el mejor meta-modelo en ambas escalas, y que la escala `5-to-0` ofrece un ajuste ligeramente mejor como problema de regresion. Sin embargo, cuando el criterio pasa a ser operativo, es decir, cuando se evalua la capacidad para separar casos aceptables de casos que conviene rechazar, la escala `10-to-0` obtiene una leve ventaja. En particular, la combinacion `10-to-0 + RandomForest`, con threshold `> 7,5`, alcanza `93,82 %` de precision sobre los casos aceptados, un lift de `9,99` puntos porcentuales respecto del baseline y una tasa de filtrado de casos malos de `73,71 %`. En consecuencia, el analisis sugiere que el modelo final no solo produce rankings utiles, sino que expone senales internas suficientemente ricas como para habilitar una politica de clasificacion asistida con abstencion selectiva en los casos ambiguos.

## Conclusions in English

The comparison shows that `RandomForest` is the strongest meta-model under both scoring schemes. It consistently outperforms `HistGradientBoosting`, while `XGBoost` is competitive but does not surpass `RandomForest` in the current setup.

Although the `5-to-0` target yields slightly better regression fit in its own scale, the `10-to-0` target produces slightly better operational separation. In particular, the best `10-to-0` configuration (`RandomForest`, threshold `> 7.5`) achieves the highest threshold utility score, the highest precision lift over baseline, and the strongest bad-case filtering. This suggests that the wider scale may be more useful when the goal is to separate high-confidence predictions from cases that should be rejected.

The gap between both scales is not large, so the choice depends on the final production objective. If interpretability and a more compact target are preferred, the `5-to-0` scale remains a solid option. However, if the main priority is to maximize screening power for acceptance versus rejection, the current results slightly favor the `10-to-0` scale.

Feature selection is also quite stable across folds. The most consistently selected variables are not only the probability scores, but also the legal-text similarity features, especially `cosine_sim_desc_vs_top1_hs_text`, `cosine_weighted_by_proba`, and `joint_top1_signal`. This indicates that the meta-model is learning from both confidence dispersion and semantic alignment with the predicted HS candidates, which is a positive sign for robustness.

The selected thresholds are also very stable across models: `7.5` for the `10-to-0` scale and `3.5` for the `5-to-0` scale. This stability suggests that the acceptance rule is not arbitrary and that the separation boundary is structurally supported by the data.

## Proximos pasos recomendados

Si se quisiera profundizar esta linea en la tesis o en una implementacion posterior, los siguientes pasos serian los mas naturales:

1. validar la politica de scoring sobre un conjunto completamente externo;
2. complementar la comparacion con curvas cobertura-precision y coverage-risk;
3. explorar calibracion probabilistica o clasificacion ordinal explicita del meta-modelo;
4. analizar las matrices de confusion y los casos rechazados para entender mejor en que regiones del score se concentran los errores de negocio.
