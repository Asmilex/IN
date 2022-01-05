# Exámenes Inteligencia de Negocio

## 27 de enero de 2014

1. **Preguntas:**
   1. **Aunque por abuso del lenguaje hablamos de KDD y de minería de datos como sinónimos, indica las diferencias entre los términos.**
   2. **Identifica y describe brevemente las etapas de KDD.**

Knowledge discovery in databases es el preoceso mediante el cual se identifican patrones válidos, novedosos, potencialmente útiles y principalmente entendibles en una base de datos. La minería de datos es una parte de este proceso.

Las etapas que incluye son:
1. Selección de datos: determinar la fuente de información.
2. Almacenamiento de datos: se diseña un esquema para unificar de forma operativa la fase 1.
3. Limpieza de los datos: mejora la calidad de éstos y los resultados de la minería
4. Preprocesamiento: selección, limpieza y transformación de los datos a usar.
5. Data mining: uso de algoritmos para extraer conocimiento y/o patrones como parte del proceso del KDD. Es la parte más importante.
6. Integración y evaluación: verificar cómo de buenos son los modelos que hemos conseguido.
7. Difusión: divulgar sobre lo que hemos aprendido. Hacer gráficos y esas cosas.

2. **Enumera y describe dos problemas abordados en minería de datos. Pon un ejemplo de aplicación real y menciona un algoritmo clásico para ellos.**

Hoy en día el uso de minería de datos está muy extendido. Por mencionar algunos ejemplos, se encuentra el análisis de datos producidos por experimentos científicos (medicina, biología, etc...), procesamiento de datos de redes sociales...

Por ejemplo, se puede el algoritmo K-Means para estudiar cómo se agrupan los usuarios de una red social. Estudiando las relaciones entre personas y sus intereses, se puede saber cómo actúa un grupo de personas no conocido a priori.

3. **¿Por qué es interesante realizar selección de atributos antes de construir un clasificador? Enumera brevemente dos justificaciones.**

A la hora de construir modelos, más atributos no siempre es mejor. Muchas veces, una mayor cantidad de atributos supone un mayor tiempo de entrenamiento del modelo. Además, podría suponer overfitting.

Una de las partes más importantes de la ciencia de datos es tener datos de calidad. Aplicando técnicas como Análisis de Componentes Principales o Análisis Factorial, se podrían extraer variables ocultas o latentes con el fin de construir un modelo más robusto. Otro aspecto importante sería eliminar aquellos atributos que tenga un número excesivamente alto de ruido u outliers, ya que podrían estar sesgando nuestra información.

4. **Describir el significado de las medidas de soporte y confianza en reglas de asociación.**

Dada una regla `X => Y`, se define el soporte como la probabilidad de que X e Y se encuentren en una transacción; y la confianza es la probabilidad de que Y se encuentre en una transacción habiéndose dado X (la probabilidad condicionada).

En otras palabras, el soporte es la evidencia de cómo de frecuente es un item en nuestros datos, mientras que la confianza es el porcentaje de veces que nuestro condicional se evalúa como cierto.

5. **Suponed un conjunto de datos de clasificación que tiene 4 atributos de entrada, 500 ejemplos y 3 clases. Tres de los atributos de entrada son numéricos en [1.0, 5.0], y el cuarto es categórico con 4 valores diferentes. ¿Qué técnicas de preprocesamiento aplicarías para emplear técnicas de vecino más cercano?**

Sabemos que las técnicas de vecino más cercano no se llevan bien con atributos discretos. Al utilizar distancias, es recomendable que los atributos sean continuos.

Supondré que la parte de limpieza ya ha sido realizada, y nos encontramos ante atributos de calidad (es decir, NaNs tratados, posibles outliers eliminados, ruido minimizado...). Antes de aplicar preprocesamiento, es necesario que pasemos el atributo categórico a una variable numérica. En este punto, tomaría una métrica de evaluación del clasificador.

Una vez comprobado su rendimiento, procederíamos al preprocesamiento.

Primero, normalizaría los atributos continuos. En principio, la primera idea que se me ocurre es utilizar la Z-Score, puesto que respeta la media y desviación típica. Una vez hecho esto, evaluaría el modelo de nuevo, tomando nota de los cambios en el resultado.

Tras esto, toca estudiar el atributo discreto. ¿Qué ocurre si lo eliminamos? ¿Mejora la clasificación? Si no es así, ¿cómo podemos tratarlo en la clasificación? Primero, cambiaría los parámetros del clasificador, como la distancia usada. Además, una de las ideas que trataría de aplicar sería usar alguna medida de distancia robusta a datos mixtos, como la Gower.


## 17 de enero de 2020

**1. ¿Por qué (y en qué situaciones) es interesante aplicar un preprocesamiento basado en filtros o un preprocesamiento basado en ensemble? Explicar brevemente ambos, pros y contras.**

Aplicamos filtros para eliminar el ruido de nuestro conjunto. El ruido es una parte inevitable del dataset en la mayoría de los casos, así que debemos desarrollar estrategias para combatirlo. De otra forma, podrían sesgar negativamente nuestro modelo e impedir un desarrollo correcto.

[Enlace con la respuesta. Es de alguien de la UGR](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fsci2s.ugr.es%2Fsites%2Fdefault%2Ffiles%2Ffiles%2Fpublications%2Fbooks%2Fslides%2FCap5%2520-%2520Dealing%2520with%2520Noisy%2520Data.pptx&wdOrigin=BROWSELINK)

El **filtrado de ruido** son mecanismos de procesamiento que intectan detectar (y eliminar, generalmente) instancias ruidas en el set de entrenamiento. De esta forma, se reduce el tamaño del set de entrrenamiento.

Una de las ventajas de separar la fase de aprendizaje y la de filtrado de ruido es que, de esta forma, las instancias ruidosas no afectan al diseño del clasificador.

(¿Creo que se refiere a esto?)
El **preprocesamiento basado en ensemble** es un tipo de filtrado de ruido. Utiliza un conjunto de algoritmos de aprendizaje para crear clasificadores en diferentes subconjuntos del set de entrenamiento. Se utilizan como filtro de ruidos. Generalmente, se propuso C4.5 y KNN con k = 1.

Aunque en general el proceso es beneficioso, existen diferentes problemas. Son dignos de mención la dificultad para distinguir qué es ruido de lo que no es, y la posible eliminación de outliers.

- Los ejemplos borderline suponen un gran problema. Algunas de las técnicas de eliminación de ruido podrían quitar ejemplos que resulten beneficiosos al clasificador, pues podrían aportar información sobre las fronteras.
- No siempre es posible distinguir outliers y ruido. Se debe proceder con cuidado, pues los outliers a veces son necesarios para el aprendizaje.


**2. Preguntas:**
    **1. Explicad las etapas de un modelo aprendizaje de análisis de sentimientos.**
    **2. ¿Qué aporta el machine learning en el análisis de sentimientos**

(Esto creo que no lo hemos dado, me lo salto)

**3. Supongamos un problema con clases no balanceadas, 3/4 clase A y 1/4 clase B. Se aplica preprocesamiento (SMOTE) y un clasificador Random Forest y el clasificador en un particionamiento 5fcv obtiene una media de 75% en clasificación. Explicad qué otras características puede tener el problema que justifiquen su mal comportamiento. Enumerarlas y justificarlas.**

Ese 75% nos hace sospechar que únicamente se está aprendiendo la clase A.

Cuando se aplica SMOTE, debemos tener cuidado de que lo que se genera tiene sentido. El algoritmo, aunque es una buena técnica, no es infalible. Se podrían estar generando instancias en sitios donde no tiene sentido, y el clasificador no consigue aprender bien la clase B.

Para solucionarlo, debemos plantearnos por qué sale ese 75%. ¿Cuál es la tasa de acierto en la clase A? ¿Y en el clase B? ¿Ha ocurrido esto debido a que SMOTE no se ha comportado correctamente?

También debemos preguntarnos si es correcto usar un Random Forest. ¿Quizás otros clasificadores arrojen un mejor resultado? ¿O es posible que debamos ajustar los hiperparámetros?

Se podría investigar también otro tipo de preprocesamiento. Por ejemplo, eliminar ruido o quitar outliers.


**4. (1,5 ptos.) Disponemos de la siguiente base de datos conteniendo 4 transacciones**
**TID artículos comprados**
**- t1 K,A,D,B**
**- t2 D,A,C,E,B**
**- t3 C,A,B,E**
**- t4 B,A,D**

**Suponiendo los umbrales mínimos de soporte y confianza al 50% y 90% respectivamente, se pide obtener todos los conjuntos frecuentes fijado dicho soporte usando el algoritmo Apriori y las reglas asociadas al nivel indicado de confianza.**