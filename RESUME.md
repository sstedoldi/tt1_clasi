# Clasificación arancelaria con procesamiento de lenguaje natural

## Resumen ejecutivo para evaluación académica

**Tesista:** Ing. Santiago S. Tedoldi  
**Director:** Dr. Bruno Bianchi  
**Maestría en Explotación de Datos y Descubrimiento de Conocimiento - Universidad de Buenos Aires**

## Propósito del trabajo

La tesis estudia cómo asistir la clasificación arancelaria de mercaderías mediante procesamiento de lenguaje natural y aprendizaje automático. A partir de una descripción comercial en texto libre, el sistema recomienda un ranking de códigos HS04 que podrían corresponder al producto.

La clasificación arancelaria tiene consecuencias tributarias, regulatorias, estadísticas y operativas. También es una tarea difícil: las descripciones suelen ser breves, incompletas, ruidosas o ambiguas, mientras que el universo de códigos posibles es amplio y contiene categorías semánticamente cercanas.

El trabajo no propone reemplazar al especialista. Su objetivo es reducir el espacio de búsqueda, ordenar alternativas y aportar señales de confianza que permitan concentrar la revisión humana en los casos más complejos.

## Dimensión del problema

El caso de estudio utiliza aproximadamente 500 mil registros en inglés provenientes del proyecto BACUDA de la Organización Mundial de Aduanas. Cada registro vincula una descripción comercial con un código del Sistema Armonizado.

La variable objetivo se construyó a nivel HS04, con las siguientes características:

- 1.133 códigos diferentes observados;
- fuerte desbalance entre códigos frecuentes y poco representados;
- textos cortos, heterogéneos y con vocabulario técnico;
- diferencias entre códigos que pueden depender de detalles sobre material, función, uso o tipo de producto;
- evaluación mediante accuracy acumulada Top-1 a Top-5, adecuada para un sistema de recomendación.

## Alcance experimental

El trabajo desarrolla un ciclo completo de investigación aplicada:

1. Análisis exploratorio, perfilado del corpus y estudio del desbalance.
2. Representación de textos y visualización mediante PCA y t-SNE.
3. Construcción de modelos de referencia con Doc2Vec y FastText.
4. Evaluación de texto crudo, texto normalizado y texto enriquecido con n-gramas.
5. Diseño de un clasificador basado en DistilBERT para 1.133 códigos HS04.
6. Comparación entre encoder fijo, fine-tuning parcial y fine-tuning completo.
7. Ejecución iterativa con múltiples muestras y semillas para medir estabilidad.
8. Entrenamiento de un modelo final y desarrollo de una interfaz de inferencia.
9. Análisis sistemático del error agregado por código.
10. Construcción de modelos explicativos y evaluación de señales de incertidumbre.
11. Exploración complementaria de un esquema híbrido entre DistilBERT y un modelo generativo.

## Modelos de referencia

Se entrenaron modelos basados en Doc2Vec y FastText para establecer referencias competitivas frente al modelo Transformer. Cada familia se evaluó sobre tres variantes del texto:

- descripción sin procesamiento;
- descripción normalizada y depurada;
- descripción procesada con incorporación de n-gramas.

Las configuraciones se evaluaron durante diez iteraciones. Esto permitió estudiar tanto el desempeño promedio como su variabilidad.

Los resultados muestran que el preprocesamiento no afecta de igual manera a todas las técnicas. Doc2Vec alcanza su mejor resultado al incorporar n-gramas, mientras que FastText obtiene su mejor desempeño con el texto normalizado. Esto demuestra que las decisiones de preparación del texto deben validarse para cada familia de modelos y no aplicarse como una receta universal.

## Modelo basado en DistilBERT

El modelo principal combina el encoder preentrenado `distilbert-base-uncased` con una red de clasificación adaptada a los 1.133 códigos posibles. Se compararon cuatro regímenes de entrenamiento:

- encoder congelado, entrenando únicamente el clasificador;
- fine-tuning parcial de las capas superiores;
- fine-tuning completo del encoder;
- una variante refinada de fine-tuning completo con validación sin bootstrap.

Los experimentos se ejecutaron en Google Cloud Vertex AI con GPU NVIDIA L4. El pipeline incorporó múltiples semillas, early stopping, warm-up, persistencia de configuraciones, métricas y logs de entrenamiento.

El fine-tuning completo produjo el mejor resultado y confirmó que no es suficiente utilizar la representación lingüística preentrenada como un componente fijo: el encoder debe adaptar sus representaciones al dominio arancelario.

## Resultados principales

Los promedios de los experimentos iterativos fueron:

| Modelo | Configuración | Top-1 | Top-3 | Top-5 |
| --- | --- | ---: | ---: | ---: |
| Doc2Vec | Texto normalizado + n-gramas | 60,84 % | 71,49 % | 74,31 % |
| FastText | Texto normalizado | 57,66 % | 76,37 % | 82,05 % |
| DistilBERT | Fine-tuning completo FFT11 | **66,37 %** | **79,76 %** | **83,88 %** |

El entrenamiento final obtuvo 66,72 % de accuracy Top-1 y 83,83 % Top-5.

En términos prácticos, el modelo final recomienda el código correcto como primera alternativa en aproximadamente dos de cada tres casos. En más de ocho de cada diez casos, el código correcto aparece dentro de las primeras cinco recomendaciones.

DistilBERT supera claramente a los modelos de referencia en Top-1 y conserva una ventaja en Top-5. El resultado muestra el valor de las representaciones contextuales para distinguir productos cuyos códigos dependen de relaciones finas entre términos, materiales, funciones y usos.

## Análisis del error

La evaluación no se limita a métricas globales. Se construyó una unidad de análisis agregada por código HS04 para estudiar por qué algunos códigos se recomiendan correctamente con mayor frecuencia que otros.

Para cada código se combinaron variables relacionadas con:

- cantidad y distribución de ejemplos de entrenamiento;
- longitud y variabilidad de las descripciones;
- complejidad de la tokenización;
- desempeño medio del clasificador;
- probabilidades asignadas a los primeros candidatos;
- márgenes entre recomendaciones;
- concentración, dispersión y entropía de las probabilidades.

El análisis muestra que el rendimiento es heterogéneo. Algunos códigos alcanzan resultados casi perfectos, mientras que otros presentan dificultades persistentes. La cantidad de ejemplos influye, pero no explica por sí sola el error. También importan la diversidad de las descripciones, su calidad informativa y la cercanía semántica entre códigos competidores.

Se identificaron errores provocados por descripciones insuficientes y también errores de alta confianza en casos conceptualmente fronterizos. Por lo tanto, una probabilidad elevada no debe interpretarse automáticamente como una recomendación correcta.

## Anexo: valoración y confianza de las recomendaciones

La valoración de la confianza de cada recomendación se presenta como un **anexo exploratorio y queda fuera de los objetivos y del cuerpo principal de la tesis**. Su inclusión busca mostrar una posible continuidad aplicada a partir de los hallazgos del análisis del error, sin considerarla parte de la validación central del clasificador.

Se entrenaron modelos auxiliares para determinar qué variables permiten explicar el desempeño medio por código.

El modelo que utiliza solamente características estructurales de los datos obtuvo un poder explicativo bajo, con un R² promedio de 0,043. Al incorporar probabilidades, márgenes y medidas de incertidumbre, el R² aumentó a 0,326 y disminuyeron los errores de estimación.

El análisis SHAP mostró que aproximadamente el 80 % de la importancia del modelo diagnóstico se concentra en variables derivadas de las probabilidades. La forma de la distribución predictiva contiene, por lo tanto, información útil para anticipar la calidad de una recomendación.

Este resultado preliminar permite proyectar un sistema que no solo sugiera códigos, sino que también ayude a decidir cuándo una recomendación puede aceptarse con mayor confianza y cuándo debe escalarse a revisión especializada. No obstante, el scoring propuesto requiere mayor validación antes de incorporarse al alcance principal o utilizarse en un contexto operativo.

## Contribuciones

Las contribuciones principales de la tesis son:

- formulación y evaluación de un problema arancelario de gran escala con más de mil códigos;
- comparación controlada entre embeddings tradicionales y un modelo Transformer;
- medición del efecto diferencial del preprocesamiento textual;
- evaluación del grado de fine-tuning necesario para adaptar DistilBERT al dominio;
- diseño iterativo para estudiar estabilidad y variabilidad;
- demostración de una mejora mediante representaciones contextuales;
- análisis del error por código, más allá de las métricas globales;
- construcción de pipelines reproducibles y una interfaz reutilizable de inferencia;
- discusión del modelo como asistente sujeto a revisión humana.

De manera complementaria, el anexo aporta una primera exploración de señales de incertidumbre para valorar recomendaciones, sin formar parte de las contribuciones exigidas por los objetivos principales.

## Limitaciones y continuidad

El estudio utiliza datos en inglés provenientes de una fuente específica y se restringe al nivel HS04. Sus resultados no deben trasladarse directamente a códigos HS06, NCM o SIM, ni a operaciones argentinas, sin validación adicional.

El modelo tampoco incorpora todavía todas las fuentes jurídicas y técnicas empleadas por un especialista, como notas legales, reglas de interpretación y antecedentes administrativos. Algunas clasificaciones no pueden resolverse a partir de la descripción comercial disponible, independientemente de la capacidad del algoritmo.

Las líneas futuras incluyen clasificación más desagregada, datos multilingües, incorporación de textos normativos, recuperación semántica, RAG, re-ranking experto, calibración de probabilidades, explicabilidad interna del Transformer y validación productiva del scoring de confianza.

## Valor global del trabajo

La tesis aporta evidencia de que la clasificación arancelaria asistida mediante procesamiento de lenguaje natural es técnicamente viable y puede alcanzar un rendimiento útil como sistema de recomendación.

Su aporte no se limita a seleccionar un modelo con buenas métricas. El trabajo aborda de manera integrada la calidad de los datos, las representaciones lingüísticas, el diseño experimental, la estabilidad, el costo computacional, el análisis del error, la incertidumbre y las condiciones necesarias para un uso responsable.

El resultado constituye una base sólida para desarrollar asistentes capaces de reducir el esfuerzo de búsqueda, presentar alternativas priorizadas y concentrar la intervención humana en los casos más ambiguos o riesgosos, manteniendo siempre el carácter especializado y normativo de la decisión final.
