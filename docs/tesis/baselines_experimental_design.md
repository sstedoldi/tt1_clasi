# Baselines HS04: diseño experimental, detalle metodológico y resultados

Fuente principal: [01-HSrecomm_Baselines_HS04_iter_train.ipynb](../../01-HSrecomm_Baselines_HS04_iter_train.ipynb).  
Artefactos de resultados: [results/baselines](../../results/baselines/).

## Diseño experimental de los baselines

El diseño experimental de los baselines buscó aislar dos fuentes de variación: por un lado, la familia de embeddings utilizada para representar el texto; por otro, el grado de intervención aplicado sobre la descripción comercial antes del entrenamiento. En consecuencia, se compararon dos modelos base, `Doc2Vec` y `FastText`, y en cada uno de ellos se evaluaron tres variantes de texto de entrada: `GOODS_DESCRIPTION` en crudo, `PREPRO_DESCRIPTION` luego de normalización y remoción de ruido, y `PREPRO_DESCRIPTION + NGRAM_DESCRIPTION` para incorporar combinaciones locales de términos.

El corpus inicial contiene 500.000 registros `(HS06, GOODS_DESCRIPTION)`. Tras remover 232.220 duplicados exactos, el conjunto de trabajo quedó conformado por 267.780 observaciones únicas. Sobre ese universo se derivó la variable objetivo `HS04`, obteniendo 1.133 clases efectivas; a su vez, el corpus conserva 4.190 códigos HS06 y 96 capítulos HS02. Este punto es metodológicamente relevante porque el problema no es una clasificación balanceada ni de pocas clases, sino una tarea multiclase con fuerte desbalance y una cola larga de partidas con muy baja frecuencia.

La evaluación se realizó mediante 10 iteraciones con muestreo bootstrap. En cada iteración se tomó un 5 % del corpus para validación, equivalente a 13.389 observaciones, mediante muestreo con reemplazo. Luego, el conjunto de entrenamiento se construyó excluyendo del corpus original los índices seleccionados para validación. De este modo, el procedimiento introduce variación entre iteraciones y permite observar no solo la media de desempeño, sino también su estabilidad. Las diez semillas utilizadas fueron: `226944881`, `768593320`, `366261559`, `531210901`, `715275432`, `908272322`, `173892995`, `625333247`, `370247453` y `503070489`.

Las métricas reportadas fueron `Top-1` a `Top-5 accuracy`. La elección de métricas acumuladas de tipo top-k es consistente con el objetivo práctico del problema: asistir la clasificación arancelaria proponiendo un conjunto acotado de códigos candidatos, más que forzar una única predicción exacta. En términos operativos, un sistema de recomendación aduanera es más útil cuando logra ubicar el código correcto dentro de las primeras alternativas sugeridas, aun cuando no siempre ocupe la primera posición.

Cabe notar, además, que los archivos CSV exportados bajo `results/baselines` quedaron nombrados con la última semilla de cada bloque (`503070489`), aunque cada archivo resume las 10 iteraciones de la configuración correspondiente. Por lo tanto, la lectura correcta de esos artefactos es agregada por configuración y no como una corrida individual.

## Detalle metodológico de Doc2Vec

Para `Doc2Vec` se utilizó la implementación de `gensim`, bajo una configuración de tipo `PV-DBOW` (`dm=0`) con `vector_size=254`, `window=5`, `min_count=1`, `sample=1e-4`, `alpha=0.025`, `min_alpha=0.001` y `epochs=50`. La construcción del vocabulario se realizó en cada iteración únicamente con el conjunto de entrenamiento, y el modelo se entrenó utilizando todos los núcleos disponibles (`workers=os.cpu_count()`).

Un aspecto importante de esta implementación es que cada texto de entrenamiento se representó como un `TaggedDocument`, pero utilizando el código `HS04` como tag del documento. En otras palabras, el espacio latente no quedó indexado por identificadores únicos de observación, sino directamente por las clases objetivo. Esta decisión convierte al baseline en una variante de recomendación por similitud contra vectores asociados a códigos HS04, más que en un `Doc2Vec` clásico seguido por un clasificador supervisado independiente.

La inferencia se realizó con `infer_vector` sobre cada descripción del conjunto de validación y, a partir de ese embedding inferido, se recuperaron los cinco vectores más similares dentro de `model.dv`. Como las etiquetas almacenadas en ese espacio son directamente los códigos `HS04`, la salida del modelo ya queda expresada como un ranking de partidas candidatas. Finalmente, la evaluación acumuló aciertos para `Top-1` a `Top-5`.

Desde el punto de vista experimental, esta formulación tiene dos consecuencias. La primera es que `Doc2Vec` actúa como un recomendador de códigos por proximidad semántica en el espacio latente. La segunda es que el preprocesamiento puede tener un efecto dual: si elimina ruido y refuerza regularidades útiles, mejora el embedding; pero si remueve demasiada especificidad léxica, puede deteriorar la discriminación entre partidas cercanas.

## Detalle metodológico de FastText

Para `FastText` se implementó un wrapper específico (`FastTextDocVec`) sobre la versión de `gensim`, con el objetivo de aproximar una interfaz comparable a la usada en `Doc2Vec`. La configuración utilizada fue `dim=254`, `window=5`, `min_count=1`, `epochs=50`, `sg=1`, `min_n=3` y `max_n=6`. En consecuencia, el modelo operó bajo una lógica `skip-gram` con información de subpalabras de longitud 3 a 6 caracteres.

La mecánica de representación difiere de `Doc2Vec`. Una vez entrenado el modelo de palabras, el embedding de cada documento se obtuvo como el promedio de los embeddings de sus palabras, seguido de normalización L2. Esos vectores documentales quedaron almacenados para todas las observaciones de entrenamiento. Durante la inferencia, el texto de validación se proyecta al mismo espacio y luego se calcula la similitud coseno respecto de todos los embeddings documentales del entrenamiento.

La recomendación final no surge directamente de los documentos más cercanos, sino de una agregación por etiqueta: el modelo toma los 50 vecinos más próximos, suma sus puntajes de similitud por código `HS04` y devuelve el ranking de clases con mayor score acumulado. Por lo tanto, `FastText` se comporta aquí como un esquema híbrido entre embeddings subléxicos y recuperación por vecinos más cercanos con votación ponderada.

Esta formulación es especialmente pertinente para descripciones comerciales breves y ruidosas. Dado que `FastText` trabaja con n-gramas de caracteres, puede capturar regularidades morfológicas, abreviaturas, variantes ortográficas y fragmentos parcialmente informativos que suelen aparecer en catálogos o descripciones de comercio exterior. Al mismo tiempo, el costo computacional de comparar contra todos los documentos de entrenamiento es mayor que en la versión usada de `Doc2Vec`, lo cual aparece reflejado en los tiempos totales de corrida.

## Resultados de los baselines

La Tabla 1 resume los resultados medios de las 10 iteraciones, junto con la desviación estándar entre corridas y el tiempo total informado en el notebook para cada bloque experimental.

| Configuración | Top-1 | Top-3 | Top-5 | Tiempo total |
| --- | ---: | ---: | ---: | ---: |
| Doc2Vec + raw | 51,20 % +- 0,44 | 64,86 % +- 0,35 | 69,37 % +- 0,38 | 49 min |
| Doc2Vec + prepro | 47,45 % +- 0,53 | 59,68 % +- 0,48 | 64,09 % +- 0,50 | 74 min |
| Doc2Vec + prepro + n-gram | **60,84 % +- 0,69** | 71,49 % +- 0,47 | 74,31 % +- 0,42 | 204 min |
| FastText + raw | 53,70 % +- 0,46 | 71,88 % +- 0,46 | 77,73 % +- 0,31 | 234 min |
| FastText + prepro | 57,66 % +- 0,50 | **76,37 % +- 0,40** | **82,05 % +- 0,36** | 302 min |
| FastText + prepro + n-gram | 57,84 % +- 0,46 | 76,28 % +- 0,39 | 81,97 % +- 0,34 | 530 min |

Los resultados muestran un patrón claro. En `Doc2Vec`, la versión con texto preprocesado sin n-gramas empeora respecto del texto crudo, lo que sugiere que la normalización por sí sola elimina parte de la señal discriminativa útil para separar partidas HS04. Sin embargo, cuando a ese preprocesamiento se le agregan n-gramas, el desempeño mejora de manera marcada y alcanza el mejor `Top-1` de todos los baselines: 60,84 %. Esto indica que, para este modelo, la composición local de términos ayuda a reconstruir parte de la especificidad perdida durante la limpieza textual.

En `FastText` se observa una dinámica distinta. El pasaje de texto crudo a texto preprocesado mejora de forma consistente todas las métricas, con una ganancia de aproximadamente 4 puntos porcentuales en `Top-1` y más de 4 puntos en `Top-5`. En cambio, agregar n-gramas explícitos sobre el texto ya preprocesado no aporta una mejora sustantiva: `Top-1` sube apenas marginalmente y `Top-3`/`Top-5` quedan levemente por debajo. Esto es coherente con la lógica del propio modelo, que ya incorpora información subléxica mediante n-gramas de caracteres; por lo tanto, añadir n-gramas de palabras puede introducir redundancia y costo computacional sin traducirse en una mejora clara.

Desde una lectura comparativa, `Doc2Vec + prepro + n-gram` ofrece la mejor primera sugerencia individual, mientras que `FastText + prepro` ofrece la mejor lista corta de candidatos. Esta diferencia es conceptualmente importante para la tesis. Si el objetivo operativo es maximizar la probabilidad de que el código correcto aparezca en la primera recomendación, la mejor variante baseline es `Doc2Vec + prepro + n-gram`. En cambio, si se privilegia una herramienta de asistencia humana donde el clasificador propone varias alternativas plausibles para revisión experta, `FastText + prepro` resulta más competitivo, ya que domina en `Top-3` y `Top-5`.

También conviene destacar la relación entre calidad y costo. La familia `FastText` supera sistemáticamente a `Doc2Vec` en `Top-3` y `Top-5`, pero lo hace con tiempos de entrenamiento bastante mayores en esta implementación. El caso más extremo es `FastText + prepro + n-gram`, que demanda 530 minutos y no mejora de manera relevante respecto de `FastText + prepro`, que requiere 302 minutos. En términos de eficiencia experimental, esto refuerza la idea de que la variante `FastText + prepro` es la mejor opción dentro de su familia, mientras que `Doc2Vec + prepro + n-gram` concentra la mejor relación entre `Top-1` y costo dentro de la suya.

## Imágenes recomendadas para la sección de resultados

En este notebook no quedaron figuras embebidas en formato imagen; las salidas relevantes son tablas y resúmenes HTML. Por eso, para la escritura de tesis conviene distinguir entre material reutilizable ya producido y figuras nuevas fáciles de derivar de los CSV exportados.

### Material ya producido y reutilizable

1. La tabla de transformación de texto mostrada en el notebook, donde se observan juntas `GOODS_DESCRIPTION`, `PREPRO_DESCRIPTION` y `NGRAM_DESCRIPTION`, sirve para ilustrar de manera concreta qué cambia entre las tres variantes textuales.
2. Las tablas de métricas por iteración y sus resúmenes descriptivos pueden reutilizarse como base para una tabla consolidada de resultados, sin necesidad de volver a ejecutar entrenamiento.

### Figuras recomendadas para agregar

1. **Curvas Top-k por configuración**. Un gráfico de líneas con `k=1..5` en el eje horizontal y `accuracy` en el vertical, incluyendo las seis configuraciones. Esta figura debería ser la principal, porque permite ver de inmediato que `Doc2Vec + prepro + n-gram` lidera en `Top-1`, mientras que `FastText + prepro` domina cuando se amplía el número de recomendaciones.
2. **Boxplot de estabilidad entre iteraciones**. Un boxplot para `Top-1` y otro para `Top-5`, con una caja por configuración. Esto permite mostrar que las seis variantes son relativamente estables, pero que las diferencias de media son sistemáticas y no producto de una sola semilla afortunada.
3. **Gráfico de dispersión accuracy vs. tiempo de entrenamiento**. En el eje X, tiempo total por bloque; en el eje Y, `Top-1` o `Top-5`. Esta visualización es especialmente valiosa para una tesis aplicada porque muestra el trade-off entre calidad predictiva y costo computacional.
4. **Barras de mejora relativa respecto del texto crudo**. Un gráfico por familia (`Doc2Vec`, `FastText`) con la variación en puntos porcentuales frente a la configuración `raw`. Esta figura ayuda a discutir, de manera muy clara, cuándo el preprocesamiento agrega valor y cuándo no.

Si hubiera que priorizar solo dos figuras, las más defendibles son: (i) la curva comparativa `Top-k`, porque resume el comportamiento de recomendación; y (ii) el scatter de `accuracy` vs. tiempo, porque conecta el resultado técnico con una decisión de ingeniería y despliegue.
