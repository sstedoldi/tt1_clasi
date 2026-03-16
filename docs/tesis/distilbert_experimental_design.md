# DistilBERT HS04: diseno experimental, detalle metodologico y resultados

Fuentes principales: [gcp/gcp_task.py](../../gcp/gcp_task.py), [gcp/distilbert_utils.py](../../gcp/distilbert_utils.py), [gcp/job_fe.yml](../../gcp/job_fe.yml), [gcp/job_pft.yml](../../gcp/job_pft.yml), [gcp/job_fft.yml](../../gcp/job_fft.yml), [gcp/job_fft11.yml](../../gcp/job_fft11.yml), [gcp/job_final.yml](../../gcp/job_final.yml) y [results/distilbert](../../results/distilbert/).  
Notebooks de apoyo: [04-HSrecomm_iter_results_analysis.ipynb](../../04-HSrecomm_iter_results_analysis.ipynb), [03-HSrecomm_DistiltBERT_HS04_viz.ipynb](../../03-HSrecomm_DistiltBERT_HS04_viz.ipynb) y [07-HSrecomm_DistiltBERT_HS04_scoring.ipynb](../../07-HSrecomm_DistiltBERT_HS04_scoring.ipynb).

## Diseno experimental de los modelos DistilBERT

El bloque experimental con `DistilBERT` tuvo un objetivo mas acotado que el de los baselines clasicos. En lugar de comparar familias de embeddings y variantes de texto, aqui se fijo una unica representacion de entrada, `GOODS_DESCRIPTION` en crudo, y se estudio el efecto de distintos grados de ajuste fino sobre un encoder preentrenado. La comparacion, por lo tanto, no se organiza alrededor del preprocesamiento textual sino del regimen de entrenamiento del modelo: `FE` (`fixed encoder`), `PFT` (`partial fine-tuning`) y `FFT` (`full fine-tuning`), a lo que luego se suma una corrida `FFT11` como variante refinada del esquema de entrenamiento completo y una corrida `final` orientada a dejar un modelo definitivo.

El corpus de partida es el mismo que el usado en el resto de la tesis. Desde `raw_data_HScodes_desc.txt` se leen pares `(HS06, GOODS_DESCRIPTION)`, se eliminan faltantes y duplicados exactos, y luego se construye la variable objetivo `HS04` tomando los primeros cuatro digitos de `HS06`. El dataset efectivo queda asi en 267.780 observaciones unicas y 1.133 clases `HS04`, lo que mantiene el caracter multiclase, desbalanceado y de cola larga ya observado en los baselines.

Todos los jobs comparten la misma arquitectura base. Se utiliza `DistilBertTokenizerFast` y el encoder preentrenado `distilbert-base-uncased`, con `max_length=300`. Sobre el embedding del primer token se agrega una cabeza de clasificacion compuesta por una capa lineal `768 -> 1024`, activacion `ReLU`, `BatchNorm1d`, `Dropout(0.3)` y una capa final `1024 -> 1133`. La funcion de perdida es `CrossEntropyLoss` y el optimizador es `Adam`.

El protocolo de validacion iterativa se implementa en `iterative_training`. En cada regimen se ejecutan 5 iteraciones con semillas diferentes. En `FE`, `PFT` y `FFT` la validacion usa `test_fraction=0.05` con `bootstrap=True`, es decir, un 5 % del corpus muestreado con reemplazo. Esto implica una advertencia metodologica importante: la muestra de validacion puede repetir observaciones y no coincide con una particion hold-out clasica. En `FFT11`, en cambio, se mantiene el 5 % pero se explicita `--no-bootstrap`, por lo que la validacion pasa a ser un muestreo sin reemplazo y resulta mas interpretable como estimacion de generalizacion.

En todos los jobs se usa `early stopping` sobre `val_loss` con `patience=3` y `warmup_epochs=1`. Ademas, todos corren sobre la misma clase de infraestructura en Vertex AI: `g2-standard-4` con una `NVIDIA L4`, `num_workers=2` y estrategia `FLEX_START`. Esto vuelve razonable atribuir las diferencias de desempeno principalmente al regimen de ajuste fino y no al hardware.

## Detalle por job de entrenamiento

### `job_fe.yml`

Este job implementa el escenario mas conservador. En codigo equivale a `fine_tune=False` y `n_finetune_layers=0`, por lo que el encoder `DistilBERT` queda completamente congelado y solo se entrena la cabeza de clasificacion. Es, en los hechos, una prueba de cuanta informacion clasificatoria puede extraerse de los embeddings preentrenados sin adaptar el lenguaje del modelo al dominio arancelario.

- Imagen: `hs-classifier:v10`
- Regimen: `FE`
- Iteraciones: `5`
- `max_epochs`: `20`
- `batch_size`: `512`
- `lr`: `5e-4`
- Validacion: `5 %` con bootstrap
- Semillas observadas en los artefactos: `31187689`, `491185480`, `737166323`, `95156473`, `954562093`

El comportamiento por epoca muestra una convergencia lenta y relativamente estable. Los mejores puntos de validacion aparecen recien al final del horizonte de entrenamiento, entre las epocas 19 y 20, lo que es consistente con un esquema donde solo aprende la capa superior y el encoder no puede reorganizar sus representaciones internas.

### `job_pft.yml`

Este job implementa el caso intermedio. En codigo corresponde a `fine_tune=True` y `n_finetune_layers=2`, es decir, se descongelan unicamente los dos ultimos bloques transformer de `DistilBERT`. La idea experimental es permitir cierta adaptacion al dominio sin asumir el costo completo, ni el riesgo de sobreajuste, de un fine-tuning total.

- Imagen: `hs-classifier:v10`
- Regimen: `PFT`
- Iteraciones: `5`
- `max_epochs`: `12`
- `batch_size`: `128`
- `lr`: `1e-4`
- Validacion: `5 %` con bootstrap
- Semillas observadas: `140871553`, `320099395`, `420029892`, `550264748`, `748447230`

En `PFT` la mejora respecto de `FE` es inmediata. Los mejores valores de validacion aparecen temprano, entre las epocas 4 y 5, aunque el `early stopping` deja corridas efectivas de 7 u 8 epocas. Esto sugiere que una porcion limitada del encoder alcanza para capturar regularidades del dominio que la cabeza sola no logra absorber.

### `job_fft.yml`

Este job implementa el ajuste fino completo de `DistilBERT`. En codigo se expresa como `fine_tune=True` y `n_finetune_layers=0`, donde el cero no significa congelar sino exactamente lo contrario: liberar todos los parametros del encoder. Conceptualmente, este es el escenario donde el modelo tiene mayor capacidad para reacomodar sus representaciones semanticas al problema HS04.

- Imagen: `hs-classifier:v10`
- Regimen: `FFT`
- Iteraciones: `5`
- `max_epochs`: `7`
- `batch_size`: `128`
- `lr`: `5e-5`
- Validacion: `5 %` con bootstrap
- Semillas observadas: `189334375`, `371945249`, `447461326`, `550941617`, `721389304`

El resultado mas interesante de este job no es solo el mejor nivel de accuracy, sino tambien la forma de la curva. En las cinco iteraciones el mejor `val_loss` se alcanza sistematicamente en la epoca 4, y luego aparecen pequenas oscilaciones o deterioros. Esto sugiere que, una vez habilitado el ajuste fino total, el modelo absorbe muy rapido la senal util del dominio.

### `job_fft11.yml`

Este job conserva la logica de `FFT` pero introduce dos cambios operativos relevantes. Primero, usa la imagen `v11`, que refleja una version posterior del pipeline. Segundo, desactiva el bootstrap mediante `--no-bootstrap`. Experimentalmente, `FFT11` funciona como una repeticion controlada del mejor regimen anterior bajo una validacion mas limpia.

- Imagen: `hs-classifier:v11`
- Regimen: `FFT`
- Iteraciones: `5`
- `max_epochs`: `7`
- `batch_size`: `128`
- `lr`: `5e-5`
- Validacion: `5 %` sin reemplazo
- Semillas observadas: `125430804`, `131345983`, `691504352`, `724791594`, `742268813`

El log preservado en [results/distilbert/fft11/experiments_job_fft_v11_training_execution.log](../../results/distilbert/fft11/experiments_job_fft_v11_training_execution.log) muestra un costo medio de aproximadamente 54,2 minutos por epoca y una duracion total de 31,8 horas para las cinco iteraciones. Igual que en `FFT`, el mejor `val_loss` aparece en la epoca 4 en las cinco corridas. La principal diferencia es la reduccion de la varianza entre iteraciones.

### `job_final.yml`

Este job ya no es una bateria iterativa sino una corrida unica para consolidar un modelo final. Mantiene el regimen `FFT`, usa la imagen `v11-final`, y define `--final`, con lo que pasa a la funcion `training` en lugar de `iterative_training`.

- Imagen: `hs-classifier:v11-final`
- Regimen: `FFT`
- Corrida unica
- `max_epochs`: `5`
- `batch_size`: `128`
- `lr`: `5e-5`
- Validacion: `1 %` sin reemplazo
- `seed`: `32`

El modelo final se guarda en [results/distilbert/fft_final/final_model](../../results/distilbert/fft_final/final_model/). El log preservado en [results/distilbert/fft_final/training_execution.log](../../results/distilbert/fft_final/training_execution.log) marca unas 4,53 horas totales, con 53,99 minutos promedio por epoca. Su principal uso en la tesis deberia ser como modelo de referencia para despliegue y para notebooks posteriores de scoring, no como comparacion estricta contra las corridas iterativas, porque cambia el tamano y la composicion del conjunto de validacion.

### `job_final_test.yml`

Este job es un `smoke test` del flujo final. Replica el esquema de `job_final.yml`, pero reduce `max_epochs` a `1` y cambia `job_dir` a `final_train_smoketest_v11`. No aporta evidencia metodologica nueva; sirve para verificar que el pipeline, la imagen y la escritura de artefactos funcionen antes de lanzar la corrida definitiva.

## Resultados de los modelos DistilBERT

La Tabla 1 resume el desempeno medio de las 5 iteraciones para cada regimen iterativo.

| Regimen | Top-1 | Top-3 | Top-5 | Desvio Top-1 | Mejor epoca tipica |
| --- | ---: | ---: | ---: | ---: | ---: |
| FE | 52,67 % | 67,93 % | 73,71 % | 0,43 pp | 19-20 |
| PFT | 65,07 % | 79,01 % | 83,30 % | 0,28 pp | 4-5 |
| FFT | 66,27 % | 79,77 % | 83,92 % | 0,48 pp | 4 |
| FFT11 | **66,37 %** | **79,76 %** | **83,88 %** | **0,15 pp** | 4 |

El salto entre `FE` y `PFT` es el hallazgo principal. Congelar completamente el encoder deja al modelo apenas por encima de algunos baselines clasicos, mientras que habilitar el ajuste de los dos ultimos bloques agrega alrededor de 12,4 puntos porcentuales en `Top-1` y cerca de 9,6 puntos en `Top-5`. Esto indica que el conocimiento preentrenado de `DistilBERT` es util como punto de partida, pero insuficiente por si solo para resolver la tarea arancelaria con buen nivel de precision.

Entre `PFT` y `FFT` la mejora adicional existe, pero ya es mucho mas acotada. El fine-tuning completo agrega aproximadamente 1,2 puntos en `Top-1` y 0,6 puntos en `Top-5`. En otras palabras, una vez que el modelo puede adaptar parte del encoder al dominio, la ganancia marginal de descongelarlo por completo es real pero menor.

`FFT11` confirma el rendimiento de `FFT` y, sobre todo, mejora su estabilidad. El promedio `Top-1` sube levemente de 66,27 % a 66,37 %, pero el cambio mas relevante es que la desviacion estandar cae de 0,48 a 0,15 puntos. Esa reduccion de dispersion es consistente con el cambio a validacion sin bootstrap y vuelve a `FFT11` la variante mas defendible como mejor modelo experimental.

## Comparacion con los baselines

Si se compara `FFT11` con el mejor baseline en `Top-1`, que fue `Doc2Vec + PREPRO + NGRAM`, la mejora es de 5,53 puntos porcentuales: 66,37 % contra 60,84 %. Si, en cambio, la comparacion se hace contra el mejor baseline en `Top-5`, que fue `FastText + PREPRO`, la ganancia es mas moderada pero sigue siendo positiva: 83,88 % contra 82,05 %, es decir, +1,83 puntos.

Desde la perspectiva de tesis, este contraste es importante porque muestra dos cosas a la vez. Primero, `DistilBERT` no solo mejora la primera sugerencia, sino tambien la calidad de la lista corta de candidatos. Segundo, la mayor ventaja aparece en `Top-1`, que es precisamente la metrica donde un encoder contextual deberia mostrar mejor su capacidad para discriminar entre descripciones lexicalmente proximas.

## El modelo final y su lectura correcta

La corrida final con `seed=32` arroja las siguientes metricas sobre su hold-out del 1 %: `Top-1 = 66,72 %`, `Top-2 = 75,46 %`, `Top-3 = 79,68 %`, `Top-4 = 82,03 %` y `Top-5 = 83,83 %`.

Estas cifras son coherentes con `FFT11`, pero no deberian leerse como una prueba concluyente de mejora adicional. El motivo es metodologico: cambia el protocolo de evaluacion. El archivo [results/distilbert/fft_final/final_val_predictions.csv](../../results/distilbert/fft_final/final_val_predictions.csv) contiene 2.677 observaciones, equivalentes al 1 % del dataset, y cubre 507 clases efectivas. Es, por lo tanto, una validacion mas chica y con menor cobertura de clases que las corridas iterativas del 5 %. En la tesis conviene presentarla como corrida de consolidacion del modelo, no como el benchmark principal.

## Observaciones sobre los artefactos disponibles

En la copia local del repositorio, los logs `training_execution.log` de `FE`, `PFT` y `FFT` estan vacios, mientras que los CSV de metricas e historial si estan completos. Por eso el detalle fino de esas corridas debe reconstruirse desde [results/distilbert/fe/history_all_iters_fe.csv](../../results/distilbert/fe/history_all_iters_fe.csv), [results/distilbert/pft/history_all_iters_pft.csv](../../results/distilbert/pft/history_all_iters_pft.csv) y [results/distilbert/fft/history_all_iters_fft.csv](../../results/distilbert/fft/history_all_iters_fft.csv). En cambio, `FFT11` y `final` si conservan log utilizable.

## Imagenes recomendadas para la tesis

### Material ya producido y reutilizable

1. [results/distilbert/fe/training_summary.png](../../results/distilbert/fe/training_summary.png), [results/distilbert/pft/training_summary.png](../../results/distilbert/pft/training_summary.png), [results/distilbert/fft/training_summary.png](../../results/distilbert/fft/training_summary.png) y [results/distilbert/fft11/experiments_job_fft_v11_training_summary.png](../../results/distilbert/fft11/experiments_job_fft_v11_training_summary.png). Son las figuras mas faciles de incorporar para mostrar la evolucion de `val_loss` y `val_top5_acc` por iteracion.
2. [results/emb_goods_desc_finetuned_pca_top10_2d.html](../../results/emb_goods_desc_finetuned_pca_top10_2d.html) y [results/emb_goods_desc_finetuned_tsne_top10_2d.html](../../results/emb_goods_desc_finetuned_tsne_top10_2d.html). Sirven para ilustrar la estructura del espacio embebido aprendido por el modelo ajustado.
3. [results/emb_hs06_nomen_finetuned_pca_nomen_all_2d.html](../../results/emb_hs06_nomen_finetuned_pca_nomen_all_2d.html) y [results/emb_hs06_nomen_finetuned_tsne_nomen_all_2d.html](../../results/emb_hs06_nomen_finetuned_tsne_nomen_all_2d.html). Son utiles si queres relacionar embeddings de descripciones con textos legales o nomenclatura.

### Figuras que conviene agregar

1. **Barras comparativas de Top-1, Top-3 y Top-5 con barras de error**. Esta deberia ser la figura principal de resultados. Resume de inmediato que `FE` queda claramente rezagado, que `PFT` produce el salto grande y que `FFT11` es el mejor esquema.
2. **Curvas de aprendizaje superpuestas por regimen** usando los historiales de validacion agregados. Conviene mostrar al menos `val_loss` y `val_top5_acc`. La idea es hacer visible que `FE` converge tarde, mientras que `PFT`, `FFT` y `FFT11` alcanzan su mejor zona muy temprano.
3. **Scatter o slope chart de mejora respecto del mejor baseline**. Permite mostrar el delta de `DistilBERT` sobre `Doc2Vec + PREPRO + NGRAM` en `Top-1` y sobre `FastText + PREPRO` en `Top-5`, sin mezclar comparaciones injustas.
4. **Grafico de dispersion entre iteraciones** para `Top-1` y `Top-5`. Un boxplot o strip plot deja ver que `FFT11` no solo gana en media, sino tambien en estabilidad.

Si hubiera que priorizar solo dos imagenes, las mas defendibles son: (i) la comparativa de `Top-k` con barras de error; y (ii) una curva de aprendizaje agregada donde se vea que el mejor punto de `FFT` y `FFT11` aparece sistematicamente en la epoca 4.
