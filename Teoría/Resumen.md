# Resumen Inteligencia de Negocio

- [4. Predicción: clasificación](#4-predicción-clasificación)
  - [Generalidades](#generalidades)
  - [Algunos clasificadores sencillitos](#algunos-clasificadores-sencillitos)
    - [ZeroR](#zeror)
    - [OneR](#oner)
    - [K-Nearest Neighbours](#k-nearest-neighbours)
  - [Evaluación de clasificadores](#evaluación-de-clasificadores)
  - [Árboles y reglas de asociación](#árboles-y-reglas-de-asociación)
  - [Clasificadores basados en reglas](#clasificadores-basados-en-reglas)
  - [Clasificadores basados en métodos bayesianos](#clasificadores-basados-en-métodos-bayesianos)
  - [Clasificadores basados en instancias](#clasificadores-basados-en-instancias)
  - [Clasificadores basados en redes neuronales](#clasificadores-basados-en-redes-neuronales)
  - [Clasificadores basados en máquinas de soporte vectorial](#clasificadores-basados-en-máquinas-de-soporte-vectorial)
  - [Multiclasificadores](#multiclasificadores)
- [5. Preprocesamiento de datos](#5-preprocesamiento-de-datos)
  - [Introducción](#introducción)
  - [Integración, limpieza y transformación](#integración-limpieza-y-transformación)
  - [Datos imperfectos](#datos-imperfectos)
  - [Reducción de datos](#reducción-de-datos)
- [Tema 6: Clustering](#tema-6-clustering)
  - [K-Means](#k-means)
  - [Mean Shift](#mean-shift)
  - [DBSCAN](#dbscan)
  - [BIRCH](#birch)
  - [Medidas](#medidas)
  - [Métodos aglomerativos](#métodos-aglomerativos)
  - [Métodos divisivos](#métodos-divisivos)
- [Tema 7: Patrones frecuentes y reglas de asociación](#tema-7-patrones-frecuentes-y-reglas-de-asociación)
  - [Algoritmo APRIORI](#algoritmo-apriori)
  - [Medidas de interés](#medidas-de-interés)
- [Tema 8: Deep Learning](#tema-8-deep-learning)
  - [La nueva forma de entrenar RNA multicapa: aprendizaje profundo](#la-nueva-forma-de-entrenar-rna-multicapa-aprendizaje-profundo)
  - [CNN: Convolucional Neural Network](#cnn-convolucional-neural-network)
- [Tema 9: Problemas regulares](#tema-9-problemas-regulares)
- [9. Problemas singulares](#9-problemas-singulares)

## 4. Predicción: clasificación

### Generalidades

Tipos de variables habituales:
- Numérica (ordinal)
- Categórica o nominal (no hay orden)

Tipos de aprendizaje automático:
|                | **Supervisado** |  **No supervisado**  |
|:---------------|:---------------:|:--------------------:|
| **Categórico** |  Clasificación  | Reglas de asociación |
| **Continuo**   |    Regresión    |      Clustering      |

Etapas del proceso de clasificación:
Ejemplos de entrenamiento -> sistema de clasificación -> ejemplos de prueba. Extraer estadísticas sobre el modelo en estos ejemplos de prueba.

Matriz de confusión:
| Predicho \ Real | **Positivo** | **Negativo** |
|:----------------|:------------:|:------------:|
| **Positivo**    |      VP      |      FN      |
| **Negativo**    |      FP      |      VN      |

- Acierto = $\frac{VP + VN}{VP + VN + FP + FN}$; precisión o exactitud.
- Error = $1 - acierto$
- Velocidad: tiempo necesario para la construcción y el uso del modelo.
- Robustez: capacidad para tratar con valores desconocidos.
- Escalabilidad: aumento del tiempo necesario con el tamaño de la base datos.
- Interpretabilidad: comprensibilidad del modelo obtenido.
- Complejidad del modelo: tamaño del árbol, número de reglas...

### Algunos clasificadores sencillitos

#### ZeroR
- Todas las instancias se clasifican en la clase mayoritaria.

#### OneR

- Clasificador formado por reglas con una única variable en el antecedente.
- Reglas del tipo `si variable == valor => clase = categoría`

#### K-Nearest Neighbours
- Almacenas una tabla con los ejemplos disponibles, junto a la clase asociada a cada uno de ellos
- Cuando quieras clasificar un ejemplo, seleccionas los k vecinos más cercanos a él, y se clasifica como la clase que más aparece entre esos vecinos.


### Evaluación de clasificadores

- El objetivo es hacer una estimación honesta del modelo generado.
- Utilizar como bondad la tasa de acierto sobre el set de entrenamiento no te vale para nada. Suele ser demasiado optimista.
- Técnicas de evaluación:
  - **Hold-out**:
    - Divides la base de datos en set de entrenamiento y set de pruebas.
    - Elementos del training obtenidos normalmente por muestreo sin reemplazamiento. Los del test son aquellos que no han aparecido en el del training.
    - Usado en bases de datos grandes
  - **Validación cruzada** (Cross-validation):
    - Divides la base de datos en $k$ subconjuntos $\text{Folds} = \{S_1, ..., S_k\}$.
    - Sacas $k$ clasificadores, utilizando como set de entrenamiento $\text{Folds} - S_i$, y como test $S_i$.
    - Devuelves como tasa de acierto el promedio obtenido en las $k$ iteraciones.
    - Se suele usar en bases de datos de tamaño moderado.
  - **Leaving one out**:
    - Cross validation pero $k = \text{número de registros}$. Es decir, utilizas todas las instancias menos una. Y testeas sobre esa.
    - Tardas un huevo y medio en ejecutarlo, así que te vale solo para bases de datos chiquitillas.
  - **Bootstrap**:
    - Basado en muestreo con reposición. A partir de una base de datos de tamaño $n$, seleccionas $n$ instancias al azar, y se utilizan como set de entrenamiento. Como es con reposición, la probabilidad de que no se seleccione un ejemplo es $(1 - \frac{1}{n})^n \rightarrow e^{-1} \approx 0.368$.
    - Para calcular el error sobre el set de test, corriges el error para no ser tan pesimista. Te sale $\text{error} = 0.632 \cdot error_{test} + 0.368 \cdot error_{training}$

### Árboles y reglas de asociación

- Clasificador que en función de un conjunto de atributos permite determinar a qué clase pertenece el objeto de estudio.
- **Estructura**:
  - Nodo: una prueba realizada sobre los atributos
  - Hojas: clase de la variable objeto de la clasificación.
- **Construcción**:
   1. Al principio, todos los ejemplos de entrenamiento están en el nodo raíz.
   2. Se van dividiendo recursivamente en base a los atributos seleccionados.
      - Si los atributos son categóricos, hace falta discretizarlos.
      - Se seleccionan en base a una heurística.
   3. Se poda el árbol para quitar ramas que describen ruido o datos anómalos.
   4. Condiciones para terminar:
      - Todos los ejemplos pertenecen a la misma clase.
      - No quedan más atributos para seguir particionando => utilizar el voto de la mayoría.
      - No quedan ejemplos.
- ¿Cómo seleccionar cuál es el **mejor atributo**?
  - Primero, definimos la entropía de una variable, porque no somos unos psicópatas como el que hizo los apuntes:
    - La entropía de una variable X, $H(X) = -\sum_{i}{p(x_i) \cdot log_{2}(p(x_i))}$. Además, consideramos $X^{*}$ el atributo a seleccionar.
    - **La entropía mide la aleatoriedad, sorpresa o incertidumbre al predecir una cierta clase**.
  - Probablemente este tocho de fórmulas no sirva para nada, así que puedes ignorarlo dentro de lo razonable. Eso sí, apréndete los nombres y la idea detrás de cada medida.
  - La ganancia de información, **InfoGain**, viene dado por $X^{*} = \max_{X}{H(C) - H(C \vert X)}$. Es usado por ID3.
    - Suponiendo que hay dos clases, $P$, $N$, que tienen $p$, $n$ elementos cada una en el set considerado, entonces, la cantidad de información necesaria para decidir si un ejemplo pertenece a $P$ o $N$ viene dado por la siguiente expresión: $H(C) = H(p, n) = - \frac{p}{p + n} \cdot log_2(\frac{p}{p + n}) - \frac{n}{p + n} \cdot log_2(\frac{n}{p + n})$.
    - Consideremos que, utilizando un atributo $A$, el conjunto se puede dividir en $v$ conjuntos diferentes. Si ahora hay $p_i$ de la clase $P$, y $n_i$ de la clase $N$, entonces $H(C \vert A) = \sum_{i = 1}^{v}{\frac{p_i + n_i}{p+n}H(p_i, n_i)}$.
    - La información que se podría ganar con una rama que considere $A$ es $Gain(A) = H(p, n) - H(C \vert A)$.
  - Modificando un poco lo anterior para tener en cuenta el ratio, obtenemos **GainRatio**, definido como $X^{*} = \max_{X}{\frac{H(C) - H(C \vert X)}{H(X)}}$. Lo introdujo C4.5 (*que no es el hermano del creador de la música de Minecraft*).
  - Por último, CART usó el **índice Gini**. Sabiendo que $G = 1 - \sum_{i = 1}^{n}{p_i^2}$, (donde $p_i$ es la frecuencia relativa de la clase $i$ en el conjunto de datos), entonces $X^{*} = \max_{X}{G(C) - G(C \vert X)}$.
    - Si se divide el conjunto de datos en $T_1$, $T_2$ de tamaños $N_1$, $N_2$, entonces $gini_split(T) = \frac{N_1}{N}gini(T1) + \frac{N_2}{N}gini(T_2)$.
- **Ventajas e inconvenientes** de los árboles de decisión:
  - **Ventajas**:
    - Fáciles de utilizar y eficientes
    - Generan reglas que son fáciles de interpretrar
    - Escalan mejor que otro tipo de técnicas
    - Tratan bien los datos con ruido
  - **Inconvenientes**:
    - No manejan bien los atributos continuos
    - Tratan de dividir el dominio de los atributos en regiones rectangulares. No todos los problemas son de ese tipo.
    - Tienen dificultad para trabajar con valores perdidos.
    - Pueden tener problemas de sobreaprendizaje.
    - No detectan correlaciones entre atributos.
- **Algoritmos basados en árboles de decisión**:
  - **ID3**:
    - Utiliza conceptos de teoría de la información
    - Reduce el número de comparaciones
    - Utiliza la ganancia de la información (**InfoGain**). En cada paso, intenta maximizarla.
    - El espacio de hipótesis es completo; la función objetivo está incluida en él.
    - No es capaz de determinar todos los árboles compatibles con los ejemplos de entrenamiento, ni puede proponer ejemplos que reduzcan el espacio de búsqueda.
    - No hay vuelta atrás => **se queda estancado en óptimos locales**.
    - **Permite ruido** en los ejemplos de entrenamiento.
    - **Por la ganancia de información, tiene tendencia a elegir atributos con muchos valores**.
    - **Se prefieren árboles cortos** frente a largos, con los atributos que producen mayor ganancia cerca de la raíz.
    - **Refinamientos posibles** (*aka problemas que tiene*):
      - Cuándo parar en la construcción del árbol
      - Atributos continuos
      - Ejemplos incompletos
  - **C4.5**:
    - Mejora a ID3 en los siguientes aspectos:
      - **Datos perdidos**: se ignoran en la construcción del árbol. Para clasificar uno con valores perdidos, se predice en base a los otros atributos del registro.
      - **Datos continuos**: Se divide en rangos en base a los valores que toma en el conjunto de entrenamiento.
      - **Soluciones para el sobreaprendizaje**:
        - Prepoda: se decide cuándo dejar de subdividir el árbol. No se divide un nodo cuando se tiene poca confianza en él
        - Postpoda: se construye el árbol y después se poda. Estrategias:
          - Reemplazamiento de subárboles; se reemplaza un subárbol por una hoja si al hacerlo el error es similar al original
          - Elevación de subárboles: reemplaza un subárbol por otro si se utiliza más.
    - Permite generar reglas a partir del árbol.
    - Selección de atributos: **se utiliza GainRatio**.
    - No maneja bien clases desbalanceadas

### Clasificadores basados en reglas

- Una regla está formada por:
  - Antecedente: contiene un predicado que se evalúa como verdadero o falso
  - Consecuente: la clase que se predice.
- **Los árboles de decisión y las reglas no son equivalentes**. Los árboles tienen orden implícito. Las reglas no. El árbol se crea mirando todas las clases. Cuando se genera una regla, solo se examina la clase
- Cómo generar reglas de decisión:
  - Mediante **árboles de decisión**:
    - No se realiza un particionamiento exhaustivo del espacio de soluciones
  - Mediante **cobertura**:
    - Intentan generar reglas que cubran exactamente 1 clase.
    - Suelen generar la mejor regla posible optimizando la probabilidad de clasificación deseada.
    - Ejemplo: **OneR**
    - Ejemplo: **PRISM**. Genera reglas para cada clase mirando el conjunto de entrenamiento, y añadiendo reglas que describan todos los ejemplos de dicha clase.
      - No garantiza obtener el conjunto óptimo de reglas, al ser un greedy.
      - Sufre de sobreajuste.
      - Es un poco truño a día de hoy.

### Clasificadores basados en métodos bayesianos

- Representan incertidumbre asociada a los procesos de forma natural.
- Utilizan el teorema de bayes de la probabilidad condicionada, y la hipótesis de *Máxima A Posteriori* (MAP).
- Ejemplo típico: **Naive Bayes**
  - **Supone que todos los atributos son independientes conocida la variable clase**.
  - Simplifica la hipótesis MAP.
  - Consigue buenos resultados a pesar de esa suposición.
  - Variables discretas => estimación por máxima verosimilitud. Variables continuas => se asume una distribución normal.
- **Ventajas e inconvenientes**:
  - **Ventajas:**
    - Fácil de implementar
    - Buenos resultados en gran parte de los casos
  - **Inconvenientes:**
    - La suposición de independencia supone una falta de precisión. Casi siempre existe una cierta correlación. Para solucionarlo, se utilizan reyes de creencia bayesianas. Combinan razonamiento bayesiano con relaciones causales entre atributos.

### Clasificadores basados en instancias

- Basados en aprendizaje por analogía.
- **Paradigma perezoso**: no se construye modelo, sino que se utiliza la propia base de datos. Se trabaja cuando se llega un nuevo ejemplo a clasificar.
- Suelen estar basados en KNN.
  - Se suele normalizar.
  - Se suelen utilizar las distancias euclídeas, Manhattan y Minkowski
  - Para los missing values, se asigna la máxima componente posible (i.e., si la normalización es en $[0, 1]$ entonces le asignas $1$).
  - Para no considerar todas las variables igual de importantes, se pueden asignar pesos.
  - Matices:
    - Robusto frente a ruido (siempre que $k \gt 1$).
    - Eficaz, pues utiliza funciones lineales locales para aproximar la función objetivo.
    - Válido para clasificación y predicción numérica.
    - Ineficiente en memoria as fuck.
    - La distancia entre vecinos podría estar dominada por variables irrelevantes
    - Complejidad de $O(\text{complejidad de la distancia} \cdot n^2)$

### Clasificadores basados en redes neuronales

- Ventajas y desventajas:
  - **Ventajas**:
    - Gran tasa de acierto en la predicción
    - Más robustas que los árboles de decisión por los pesos
    - Robustez ante la presencia de errores (ruido, outliers...)
    - Gran capacidad de salida (nominal, numérica, vectorial...)
    - Eficiencia (== rapidez) en la evaluación de nuevos casos
    - Mejoran su rendimiento mediante aprendizaje y éste puede continuar después de que se haya aplicado al conjunto de entrenamiento.
  - **Desventajas**:
    - Mucho tiempo de entrenamiento
    - Gran parte del entrenamiento es ensayo y error
    - Poca interpretabilidad del modelo. Se suele decir que es una caja negra (pero hoy en día no es cierto esto).
    - Difícil incorporar conocimiento del dominio.
    - Atributos de entrada deben ser numéricos
    - Generar reglas a partir del modelo no es inmediato
    - Pueden tener problemas de sobreaprendizaje.
- ¿Cuándo utlizarlas?
  - La dimensión de la entrada es alta
  - La salida es un valor discreto o vector de valores
  - Posibilidad de datos con ruido
  - La forma de la función objetiva es desconocida
  - La interpretabilidad de los datos no es importante
- Matices:
  - Todas las variables numéricas se normalizan en $[-1, 1]$
  - Algoritmo de backpropagation:
    - Basado en la técnica del gradiente descendente
    - No permite conexiones hacia atrás (retroalimentación) en la red
- Si quieres aprender en condiciones, te vas al canal de [DotCSV](https://www.youtube.com/watch?v=MRIv2IwFTPg&list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ) y te pones a mirar sus vídeos, que están muchísimo mejor que estos apuntes. Las redes neuronales cambian rapidísimamente, y no merece la pena aprenderse este tipo de cosas así como así. Sí, te estoy diciendo que ignores estos apuntes.


### Clasificadores basados en máquinas de soporte vectorial

- Una SVM es un modelo de aprendizaje que se fundamente en la teoría de *aprendizaje estadístico*. La idea básica es encontrar un hiperplano canónico que maximice el margen del conjunto de datos de entrenamiento. Esto nos garantiza una buena capacidad de generación.
- Requiere que los datos sean linealmente separables.
- Explotan la información que proporciona el producto escalar entre los datos disponibles.
- El problema es encontrar la función $\Phi$ que los separa. Por ello, se utilizan las funciones kernel, que es un producto interno de dos elementos en algún espacio de características inducido.
- Se utilizan los multiplicadores de Lagrange habitualmente.
- Fue diseñado para resolver problemas de clasificación binaria. Para escalarlo a problemas multiclase, se realizan otras técnicas que veremos en un segundín.

### Multiclasificadores

- La idea es inducir $n$ clasificadores en uno solo. Se utlizará una salida que es combinación de lo que proporciona cada uno.
- Pueden estar basados en diferentes técnicas.
- Se pueden aplicar sobre el mismo clasificador o con diferentes.
- Formas de proporcionarse información los unos a los otros:
  - **Bagging**, o bootstrap aggregating:
    - Cada clasificador se induce independientemente de los demás. Incluye una fase de diversificación sobre los datos.
    - Funciona bien para algoritmos de aprendizaje inestables; es decir, aquellos que con un pequeño cambio, producen resultados muy diferentes.
    - Ejemplo: **random forest**.
  - **Boosting**:
    - **Muestreo ponderado**: En lugar de hacer un muestreo aleatorio de los datos de entrenamiento, se ponderan las muestrar para concentrar el aprendizaje en los ejemplos más difíciles. Los ejemplos más cercanos a la frontera son más difíciles de clasificar, por lo que recibirán un peso mayor.
    - **Votos ponderados**:
      - En lugar de combinar los clasificadores con el mismo peso en el voto, se utiliza un voto ponderado.
      - Esta es la regla de combinación para el conjunto de clasificadores débiles.
      - En conjunción con la estrategia de muestro anterior, esto produce un clasificador más fuerte.
    - **Los modelos no se construyen independientemente**. La salida del clasificador $i$-ésimo afecta al $(i+1)$-ésimo. Se hace énfasis en los que se han clasificado erróneamente en los anteriores.
    - Ejemplo: adaptive boosting
    - [Una página boniquilla para aprender las diferencias. Se entiende mejor así](https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/)
- **Binarización**:
  - Estrategia divide y vencerás. Se descomponen los problemas en varios, y después se reconstruye todo.
  - Tipos:
    - **One on one** (OVO 🦉) :
      - Uno pa uno sin camisa entre las posibles clases.
      - Se le llama pairwise learning, round robin...
      - Fase de agregación: se proponen diferentes vías para agrupar los clasificadores
    - **One for all** (OVA *como las de los animes*):
      - Venir a por mí con la cara destapad[a](https://www.youtube.com/watch?v=8hJ-okFUtT4): se enfrenta una clase ante el resto. Se separa ésta de todas las demás. Una vez por cada clase.
    - Ventajas:
      - Clasificadores más pequeños
      - Fronteras de decisión más simples.


## 5. Preprocesamiento de datos

### Introducción

- Preprocesamiento: tareas para disponer de datos de calidad previstos al uso de algoritmos de extracción de conocimiento
- Importancia:
  - Los datos pueden ser impuros, lo que conduciría a modelos inútiles. Pueden estar incompletos, tener ruido, o ser inconsistentes.
  - El preprocesamiento puede generar conjuntos de datos más pequeños que el original, lo que mejora la eficiencia.
  - Genera datos de calidad. Ya' know, trash in, trash out. Luego veremos técnicas que se usan para solucionarlo.
- Suele llevar la mayor parte del tiempo del proceso de ciencia de datos.
- La **preparación de datos** es el conjunto de técnicas que inicializan los datos adecuadamente para que sirvan de entrada a los algoritmos. Compuesta por:
  - Limpieza
  - Normalización
  - Transformación
  - Integración
  - Missing values imputation
  - Identificación de ruido
- La **reducción de datos** son las técnicas que se utilizan para simplificar los datasets. Incluye
  - Selección de características
  - Selección de instancias
  - Discretización
- Un orden sensato de preprocesamiento es
  1. Ruido
  2. Valores perdidos
  3. Selección de instancias

### Integración, limpieza y transformación
- **Integración de los datos**:
  - Obtiene los datos de diferentes fuentes de información.
  - Resuelve problemas de representación y codificación
  - Integra los datos desde diferentes tablas para crear información homogénea.
  - Cosas a tener en cuenta:
    - Hay que asegurar que las entidades se emparejan correctamente.
    - Detectar duplicados e inconsistencias
    - Cazar redundancia producida por las diferentes fuentes
    - Cuidado con los conflictos entre las bases de datos diferentes. Puede deberse a escalas diferentes, o incluso que no tengan sentido ninguno
  - El análasis de correlaciones resulta útil en este caso.
- **Limpieza**:
  - Objetivos:
    - Resolver inconsistencias
    - Rellenar/imputar valores perdidos == missing values
    - Suavizar el ruido
    - Identificar outliers y tratarlos.
  - Algunos algoritmos lo tienen incorporado. Pero ojito con fiarte.
- **Transformación**:
  - Aplicarle funciones para que se vuelvan más fáciles de tratar. Ejemplos: quitar calles y poner códigos postales; agregar atributos, normalizar.
  - Tipos de normalización:
    - min-max: normalizar a un intervalo determinado
    - Zero mean (aka Z-score): normalizar en función de la media y la desviación estándar. Útil cuando tienes demasiados outliers.
    - Por escala decimal.

### Datos imperfectos

- **Valores perdidos** (aka missing values). Para tratarlos, tienes varias opciones:
  - Ignorar la tupla
  - Rellenar manualmente los datos (buena suerte con eso)
  - Utilizar una constante global para todos. No sirve de mucho
  - Rellenar utilizando la media o mediana
  - Rellenar con el valor más probable. Para eso, hace falta inferirlo, lo cual requiere algún modelo. Es la mejor opción. Entre algunos de los métodos que se utilizan, están
    - KNN (KNNI)
    - KNN con pesos (WKNNI)
    - Basada en clustering (KMI)
    - Basada en SVM (SVMI)
    - Event covering (EC), singular value descomposition (SVDI), local least squares imputation (LLSI)
- **Ruido**: datos que podrían no tener sentido ninguno. Muy dependientes del problema.
  - Tipos:
    - Class noise:
      - Ejemplos contradictorios
      - Mal etiquetados
    - Attribute noise:
      - Valores erróneos
      - Missing values
      - Valores que te dan igual (don't care)
  - También se pueden considerar los **ejemplos borderline** (límite) a aquellos que podrían ser o no ruidos, y los **ruidosos** (noisy) a los que seguro lo son
  - Técnicas para eliminarlos o suavizarlos:
    - **Ensemble filter** (EF):
      - Para cada algoritmo del aprendizaje, se utiliza una validación cruzada k-fold para etiquetar cada ejemplo como correcto o mal etiquetado.
      - Se usa un esquema de votación para determinar el conjunto final de ejemplos ruidosos:
        - Votación por consenso: ejemplo mal clasificado por todos
        - Votación por mayoría
    - **Cross validated committees filter** (CVCF). Similar a EF, pero con dos ideas principales:
      - El mismo algoritmo de aprendizaje (C4.5) se utiliza para crear clasificadores en varios subconjuntos de entrenamiento. Es especialmente importante utilizar árboles de decisión, porque funcionan bien como filtro para el ruido.
      - Cada clasificador construido con la validación cruzada k-fold se utiliza para etiquetar todos los ejemplos de entrenamiento (no solo el de validación) como correctos o mal etiquetado.
    - **Iterative partitioning filter** (IPF):
      - Elimina los datos ruidosos en múltiples iteraciones de CVCF. Para cuando se alcanza un cierto criterio. Generalmente, cuando el porcentaje de ejemplos ruidos está por debajo de un umbral.
- Detección de **datos anómalos**:
  - Valor erróneo != anómalo (outlier).
  - Outliers: valores correctos, pero estadísticamente raros.
    - Son un problema para los algoritmos que aplican pesos.
    - Técnicas de detección:
      - Fijar una distancia y ver los que están a tomar por saco.
      - Clustering parcial: mirar quién se queda fuera de los clusters al intentar agrupar los ejemplos.
    - ¿Y qué hacemos con ellos?
      - Ignorarlos
      - Filtrar o reemplazar la columna. Un poco extremo
      - Filtrar o reemplazar la fila. Podría sesgar el modelo. Cuidadito.
      - Reemplazar el valor por un nulo. No está mal si el algoritmo trabaja bien con missing values
      - Discretizar: los valores discretos no se ven tan afectados por las anomalías, al tratarse de categorías.
  - Se pueden eliminar los datos erróneos mediante técnicas de suavizado:
    - **Binning**: se consultan los vecinos. Los valores se distribuyen en cajas o intervalos (bins). Tiene un par de variantes:
      - Binning uniforme en intervalos (equiwidth)
      - Bininng uniforme en el contenido (equidepth):
        - Suavizar por la media o mediana
        - Suavizar por las fronteras
    - **Regresión**: aplicar técnicas de regresión

### Reducción de datos

- Seleccionar datos relevantes para facilitar la minería. Vamos a ver ahora qué se puede hacer:
- **Selección de características** (feature selection):
  - Encontrar un subconjunto de variables que optimice la probabilidad de clasificar.
    - Más atributos $\nRightarrow$ más éxito en la clasificación
    - Reducir la dimensión del problema reduce la complejidad y el tiempo de ejecución
    - Con menos variables, la capacidad de generalizar aumenta
    - Los valores para ciertos atributos pueden ser costosos de obtener
    - Resultados más simples hacen que sea más fácil entender el modelo
  - Es un problema de búsqueda (del subconjunto de atributos óptimo)
    - Los algoritmos tienen una estrategia de búsqueda y una función objetivo que evalúa el subconjunto.
    - $2^\text{número de atributos}$ resultados posibles.
    - Funciones objetivo:
      - **Envolventes** (wrappers): consiste en aplicar la técnica de aprendizaje que hemos escogido y ver cómo rinden.
      - **Filtros** (filters): evalúa los subconjuntos basándose en la información que contienen. Medidas filtro:
        - Medidas de separabilidad: usan distancia entre las clases
        - Correlaciones
        - Basadas en teoría de la información. Difíciles de calcular; se suelen usar heurísticas
        - Medidas de consistencia: intentan encontrar el número mínimo de características que puedan separar las clases de la misma forma que lo hace el conjunto completo de variables.
  - **Ventajas**:
    - Envolventes:
      - Exactitud: más exactos que los de filtro
      - Capacidad para generalizar: poseen capacidad para evitar el sobreajuste debido a las técnicas de validación cruzada
    - Filtro:
      - Rápidos. Suelen limitarse a cálculos de frecuencias
      - Generalidad: al evaluar propiedades intrínsecas de los datos y no su interacción con el clasificador, te vale para cualquiera.
  - **Desventajas**:
    - Envolventes:
      - Muy costosos: para cada evaluación hay que aprender un modelo y validarlo. A ver qué algoritmo usas fiera.
      - Pérdida de generalidad: la solución está sesgada por el clasificador que uses.
    - Filtros:
      - Tendencia a meter muchas variables.
  - Según la salida del algoritmo, se identifican algunos tipos:
    - Algoritmos **subconjunto de atributos**: devuelven un subconjunto optimizado según algún criterio de evaluación
    - Algoritmos de **ranking**: devuelven una llista de atributos ordenados según algún criterio de evaluación.
  - Esto no sé exactamente dónde va (porque los apuntes se entienden regular nada más):
    - **Selección hacia delante**: empiezas con el vacío, y empiezas a meter atributos.
      - Funciona mejor cuando hay el óptimo tiene pocas variables
      - Incapaz de eliminar
    - **Selección hacia atrás**: empiezas con el total, y te lías a quitar atributos.
      - Funciona mejor cuando el óptimo tiene muchas variables
      - Tienes que reevaluar la utilidad de algunos atributos previamente descartados
    - **Selección l-más r-menos**: generalización de forward y backward (Palante patrás)
    - **Selección bidireccional**: implementación paralela de foward y backward. Hay que asegurar que los atributos eliminados por uno no son metidos por el otro.
    - **Selección flotante**: extensión de l-más y r-menos que evita fijar l, r a priori. Hay dos métodos: uno comienza por el vacío, y otro por el total.
  - Tipos de algoritmos:
    - **Exhaustivos**: garantizan el óptimo, pero el número de evaluaciones se te va de las manos (exponencial)
      - Branch and bound, beam search
    - **Heurísticos**: añaden o eliminan variables al subconjunto candidato de forma secuencial. Se quedan pillados en óptimos locales
      - Selección hacia delante, selección hacia atrás, selección l-más r-menos, búsqueda bidireccional, selección secuencial flotante
    - **Estocásticos**: usan aleatoriedad para salir de óptimos locales:
      - Ascensión de colinas con reinicios, enfriamiento estocástico, algoritmos genéticos, enfriamiento simulado
      - Están de SPM socio.
- **Selección de instancias**:
  - Elige ejemplos que sean relevantes. Descarta la basurilla. Menos datos, más exactitud (=> generaliza mejor) y modelos más simples. La misma pesca, vamos
  - Tipos:
    - **Muestreo** (con y sin reposición)
    - **Selección de prototipos o aprendizaje basado en instancias**:
      - Dirección de búsqueda: incremental, decremental, por lotes, mezclada y fija
      - Tipo de selección: condensación, edición, híbrido
      - Tipo de evaluación: filtrada o envolvente
    - **Aprendizaje activo**
  - Otra cosa que no sé realmente dónde va porque esto es un desastre. Algoritmos de selección de instancias creo:
    - **Condensed nearest neighbour** (CNN). Algoritmo clásico de condensación:
      - Incremental
      - Inserta solo las instancias mal clasificadas a partir de una selección aleatoria de una instancia de cada clase
      - Dependiente del orden de presentación
      - Tiende a retener puntos pertenecientes al borde
    - **Edited Nearest Neighbour** (ENN):
      - Por lotes
      - Se eliminan aquellas instancias que se clasifican erróneamente usando sus k vecinos más cercanos
      - Suaviza fronteras, pero retiene el resto de puntos (muchos redundantes)
    - **AIIKNN**: ENN iterativo con k = 3, 5, 7
  - Eficiencia: el orden de los algoritmos es superior a $O(n^2)$. Suele rondar $O(n^3)$.
  - Los principales problemas a afrontar son: eficiencia, recursos, generalización, representación.
  - Para grandes bases de datos, puedes usar estrategias de estratificación con los algoritmos de selección de instancias.
  - **Conjuntos de datos no balanceados**: algunos datasets tienen problemas con el recuento de las clases. Por ejemplo, que una clase tenga el 99% de las instancias, y otra el 1%.
  - Para procesarlas:
    - Técnicas de reducción para balancear las clases, reduciendo las mayoritarias
    - Realizar oversampling (añadir instancias de las clases menos representativas).
      - Un método bueno es **SMOTE**.
- **Discretización**:
  - Características:
    - Muy útiles.
    - Representan información más concisa, son fáciles de entender, más cercanos a la representación del conocimiento
    - Puede hacerse antes de la obtención de conocimiento o durante esa misma etapa.
    - Algunos algoritmos solo admiten valores discretos.
  - Dependiente de las necesidades:
    - **Supervisados vs no supervisados**: consideran o no el atributo objetivo
      - No supervisados:
        - Discretización de igual amplitud. Pueden producir desequilibrios
        - Igual frecuencia
          - Evita desequilibrios, y te da puntos de corte más intuitivos
          - Pero cuidado. Deberías crear cajas para valores especiales.
        - Clustering
      - Supervisados:
        - Basados en entropía
          - Entropy MDLP; minimum description length principle. Encontrar el coste de comunicación entre un emisor y un receptor. Una partición inducida por un punto de corte es aceptada si y solo si el coste del mensaje requerido para enviar antes de particionar es mayor que el requerido después de particionar
        - Métodos chi cuadrado
    - **Dinámicos vs estáticos**: mientras se construye el modelo
    - **Locales vs globales**: centrados en una subregión o considerando todo el espacio
    - **Top down vs bottom up**: empiezan con una lista vacía o llena de puntos de corte
    - **Directos vs incrementales**: usan o no un proceso de optimización posterior
  - Y cuál de todas las formas es mejor?
    - [a](https://twitter.com/misterjagger_/status/1397647916722962437)
    - Puedes evaluarlo teniendo en cuenta el número de intervalos, número de inconsistencias causadas, tasa de acierto predictivo, tamaño del modelo generado...


## Tema 6: Clustering

Se enmarca en el contexto del aprendizaje no supervisado con variables que toman valores continuos.

El clustering se puede ver como una tarea de preprocesado antes de aplicar alguna técnica de descubrimiento de conocimiento o como una técnica de descubrimiento del conocimiento en sí para obtener información sobre la distribución de los datos.

* Medidas de distancia:
  * Un único atributo numérico $A$: $d(x,y)=A(X)-Y(X)$
  * Varios atributos numéricos: distancia euclídea.
  * Atributos nominales: 1 si los valores son diferentes y 0 si son el mismo.
* Las medidas son sensibles al rango de valores que toman las variables $\Rightarrow$ hay que **normalizar**.

### K-Means

Necesita como argumento el número de clusters. Pilla k centroides aleatorios y asigna cada punto al centroide más cercano, recalcula el centro del cluster y vuelve a asignar los puntos al centroide más cercano. Así sucesivamente hasta que no haya cambios en los clusters.

* Eficiente: $O(tkn)$ donde $t\equiv\text{ numero de iteraciones }, n\equiv\text{ numero de objetos }, k\equiv\text{ numero de clusters }$.
* Puede finalizar en un óptimo local, lo cual se puede solucionar reinicializando la semilla aleatoria o usando técnicas de búsqueda más potentes.
* Sólo fufa cuando el concepto de medida es definible.
* Hay que fijar el número de clusters.
  * Iterar distintos valores y elegir la mejor solución.
* Débil ante ruido y outliers.
* Sólo genera clusters convexos.

### Mean Shift

Fija un radio (*bandwidth*) y va desplazando centroides a regiones más densas. El radio se puede estimar por KNN. Los clusters que genera son convexos

### DBSCAN

* Acepta como parámetros un radio `eps` y el tamaño mínimo `minPts`.
* A partir de un punto busca otros puntos que pillen dentro del radio y si hay más de `minPts` lo añade al cluster, así hasta que no se alcancen más puntos. Los puntos que no entran en ningún cluster los etiqueta como ruido.
* `eps` se puede estimar con k-distancia.
* Puede encontrar cluster con distintas formas y es robusto a outliers.
* En las prácticas no vale pa na.

### BIRCH

Agrupa conforme se reciben objetos (clustering incremental). $CF=\{N, LS, SS\}$:

* $N$ Number of objects.
* $LS$: Linear Sum.
* $SS$: Squared Sum.

Cuando llega un objeto va descendiendo por el árbol escogiendo el CF más cercano en cada nodo, y al llegar a una hoja si se puede meter en un CF se mete y si no se crea un CF nuevo si hay menos de $L$. En caso de haber $L$ se divide la hoja en dos tomando los dos CF más lejanos de la hoja anterior.

### Medidas

* **Silhouette**: Mide cómo de similares son los objetos de un cluster en comparación con los de otros. Se calculan coeficientes $s(i)$ por cada objeto, mejor cuanto más cercano a 1, si se acerca a 0 significa que está en la frontera de dos clusters. La media de todos los $s(i)$ es el coeficiente *silhouette*.
* **Calinski-Harabasz**: Razón entre la dispersión intra-clusters y la dispersión interclusters. Cuanto mayor sea el valor de este coeficiente, mejor.

### Métodos aglomerativos

En cada paso se fusionan los clusters más cercanos.

* **Enlace simple:** Se minimiza la distancia mínima entre elementos de cada grupo.
* **Enlace completo:** Se minimiza la distancia máxima entre elementos de cada grupo.
* **Varianza mínima (Ward):** Fusiona el par de clusters que genera un agrupamiento con mínima varianza.
* **Distancia entre centroides.**

### Métodos divisivos

Partiendo de un sólo cluster, se subdivide hasta que se alcanza un criterio de parada o cada cluter contiene un solo objeto. Variantes:

* **Unidimensional**: Partición en base a una variable.
* **Multidimensional**: Se consideran todas las variables.

## Tema 7: Patrones frecuentes y reglas de asociación

Se enmarca en el contexto del aprendizaje no supervisado con variables discretas. Idea:

```
Antecedente => Consecuente [soporte, confianza]
```

### Algoritmo APRIORI

* No hay que fijar atributos, se generan de forma automática
* Variedades para tratar todo tipo de datos
* Especificar mínimo soporte y máximo número de reglas.
* **Principio a priori**: Cualquier subconjunto de un conjunto frecuente es frecuente.
* **Principio de poda en Apriori:** Si un conjunto no es frecuente, no hay necesidad de generar sus superconjuntos.

Parámetros **soporte** y **confianza**:

* Si el soporte mínimo **s** es alto habrá pocas reglas que ocurren con frecuencia, si es bajo habrá muchas reglas que ocurren raramente.
* Si la confianza mínima **c** es alta habrá pocas reglas 'casi ciertas lógicamente' y si es baja habrá muchas reglas muy inciertas.
* Valores típicos: soporte entre 2-10% y confianza entre 70-90%.

### Medidas de interés

* $soporte = P(antecedente\cap consecuente)$
* $confianza = \text{Probablilidad condicionada } = \frac{soporte}{P(antecedente)} = P(\text{consecuente} \vert \text{antecedente})$

El problema de la confianza es que se calcula en base únicamente de los atributos que aparecen en la regla, no se tiene en cuenta el total de datos.

* Interés, correlación, empuje, *lift*:
  $$
  lift(A\Rightarrow B) = \dfrac{P(B/A)}{P(B)}=\dfrac{P(A\cap B)}{P(A)P(B)}
  $$
  Si A y B son independientes entonces $P(A\cap B)=P(A)P(B)$ y $lift=1$.

  Si $lift>1$ A y B están positivamente correlacionados. Si $lift < 1$ entonces A y B están negativamente correlacionados.

## Tema 8: Deep Learning

Utilizar una red neuronal con varias capas de nodos entre entrada y salida. Estas capas hacen una identificación de características y procesamiento en una serie de etapas. Para cada ejemplo de entrenamiento se propaga la entrada por la red para obtener una salida, que se compara con la salida esperada y se retropropaga el error para ir ajustando suavemente los pesos desde la última hasta la primera capa.

### La nueva forma de entrenar RNA multicapa: aprendizaje profundo

Las capas sin salida se entrenan para ser un codificador automático (auto-encoder), aprendiendo buenas características que describen lo que viene de la capa anterior. Un auto-encoder está entrenado con un algoritmo de ajuste de peso para reproducir la entrada. Como hay menos unidades que entradas, están obligadas a convertirse en buenos detectores de características.

Las técnicas tradicionales utilizaban un simple detector de características antes de utilizar un clasificador. En deep learning se usa una serie de capas ('layers'): input layer, hidden layers y output layers.

### CNN: Convolucional Neural Network

No existe forma humana de explicar esto peor de lo que está en las diapositivas así que utiliza la [página esta](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2).

Problema: Necesita una gran cantidad de datos. Solución: preprocesado o 'data augmentation', que consiste en replicar instancias del conjunto de entrenamiento con alguna transformación como traslaciones, rotaciones, simetrías... Esto favorece la robustez del modelo.

## Tema 9: Problemas regulares

Clases desbalanceadas: Supone un problema a la hora de la correcta identificación de los conceptos a aprender. Las características intrínsecas de los datos son fuente de diversos problemas, como el solapamiento o los valores perdidos. La clase mayoritaria solapa la minoritaria, dejando fronteras ambiguas.

El uso de métricas como la precisión (accuracy) conduce a conclusiones erróneas, se sobreajusta la clase mayoritaria.

|         | Predicción + | Predicción - |
| ------- | ------------ | ------------ |
| Class + | TP           | FN           |
| Class - | FP           | TN           |

* Positive True Ratio: TPR (sensitivity) $a^+=\frac{TP}{TP+FN}$ Los que aciertas de la clase positiva.
* Negative True Ratio: TNR (specificity) $a^-=\frac{TN}{TN+FP}$ Los que aciertas de la clase negativa.
* True ratio (G-mean): Media geométrica de $a^+$ y $a^-$: $GM=\sqrt{a^+\cdot a^-}$
* F1-score es la media armónica de precisión y recall:
  * Precisión: $PPV=\frac{TP}{TP+FP}$
  * Recall: sensitivity (TPR)
  * $F_1 = 2 \dfrac{precision\cdot recall}{precision+recall}$
* F-Measure en general: $F_\beta=(1+\beta^2)\dfrac{precision\cdot recall}{\beta^2\cdot precision+recall}$
* $AUC=\dfrac{1+TPR-FPR}2$









## 9. Problemas singulares

- **Resampling**:
  - **CNN**: Selecciona aleatoriamente los ejemplos de la clase mayoritaria que no pueden clasificarse correctamente.
  - **Tomek links**: quitar ejemplos borderline y ruido de la clase mayoritaria.
- **Oversampling**:
  - **Random oversampling**: tiene el efecto de hacer que la región de decisión de la clase minoritaria sea muy específica. En un árbol de decisión, produce overfitting.
  - **SMOTE**: generaliza la región de decisión de la clase minoritaria. Presta atención a los ejemplos de la clase, sin producir overfitting. Cuidado, que actúa un poco a lo loco.
    - SMOTE + Tomek: *Instead of removing only the majority class examples that form Tomek links, examples from both classes are removed*.
    - SMOTE + ENN: removes any example whose class label differs from the class of at least two of their neighbors. Quita más que con Tomek. Quita de ambas clases.