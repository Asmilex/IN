# Exámenes Inteligencia de Negocio

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