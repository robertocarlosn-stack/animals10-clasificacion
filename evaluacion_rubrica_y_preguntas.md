# Evaluación del proyecto y preparación de la defensa

Esta revisión usa [`conf/items.jpeg`](conf/items.jpeg) para los ítems y puntos, [`conf/Rubrica.jpeg`](conf/Rubrica.jpeg) para la escala global de implementación y defensa, y la [propuesta](conf/propuesta_clasificacion_animals10.pdf) para los renglones que no se ven completos en la captura de ítems. La valoración es una **auditoría de evidencias**, no una nota oficial. La columna «Puntuación obtenida» de la captura está en cero porque no muestra una evaluación del proyecto entregado.

## Lectura de la rúbrica global

La captura distingue dos dimensiones. Con **implementación completa**, una defensa excelente corresponde al **100 %**, una defensa regular al **75 %** y una defensa insuficiente al **25 %**. Con **implementación parcial**, una defensa excelente corresponde al **50 %**, una regular al **25 %** y una insuficiente al **0 %**. Por eso conviene poder explicar las decisiones y las reservas del estudio, además de mostrar que el código corre.

## Revisión de los ítems (10 puntos máximos)

| Ítem | Máx. | Evidencia observada | Juicio y atención en la defensa |
| --- | ---: | --- | --- |
| Introducción | 0,10 | Inicio del [notebook](clasificacion_animals10.ipynb): problema, pregunta y objetivos. | Cubierto. Explicar por qué importa comparar modelos sobre representaciones visuales. |
| EDA | 0,60 | Etapa 2: inventario, balance, dimensiones, canales, muestras, incidencias y duplicados. [`resumen_etapa2.json`](results/resumen_etapa2.json). | Cubierto. Justificar exclusión de 2 imágenes y agrupación antes de partir. |
| Feature Extraction | 0,25 | Etapa 3: MobileNetV2/ImageNet congelada, 224 × 224 y 1 280 componentes. | Cubierto. Distinguir extracción de entrenamiento de la red. |
| Feature Selection | 0,25 | Nueva nota explícita en etapa 3: se omite el filtro porque la rúbrica permite omitirlo en imágenes o transfer learning. | **Omisión justificada**. PCA/t-SNE son visualización y no entran al clasificador. |
| Optimización de hiperparámetros | 3,50 | Etapa 4: `GridSearchCV`, 5 folds × 2 repeticiones, búsqueda de `C`, pesos, kernel/gamma, gráficas y curvas. [`resultados_busqueda.csv`](results/resultados_busqueda.csv). | Cubierto con la reducción permitida por costo. Explicar que la búsqueda usó 5 235 imágenes y que sus mejores scores son optimistas por selección. |
| Comparación estadística de 2 técnicas | 3,50 | Etapa 5: mismos 10 folds para SVM y logística, puntuaciones pareadas, Shapiro, t corregida, intervalo, efecto y Wilcoxon de sensibilidad. [`resumen_inferencia.json`](results/resumen_inferencia.json). | Cubierto. Es la parte de mayor riesgo oral: defender la dependencia entre folds y la diferencia entre significancia estadística y relevancia práctica. |
| Pipelines | 0,50 | Etapa 4: `Pipeline(StandardScaler, clasificador)` dentro de cada fold. | Cubierto. El extractor congelado se ejecuta antes; el escalador **sí** se ajusta solo con train en cada fold. |
| t-SNE | 1,00 | Etapa 6: 1 500 puntos de desarrollo, colores por clase, PCA 50 previo y t-SNE 2D. [Figura](figures/figura_tsne_desarrollo.png). | Cubierto. No interpretar distancias globales ni t-SNE como medida de accuracy. |
| Idioma | 0,10 | Narrativa, ejes, documentación y conclusiones en español. | Cubierto en términos generales. Algunos nombres técnicos y citas permanecen en inglés; revisar ortografía en la presentación oral. |
| Referencias | 0,10 | Última sección del notebook: referencias numeradas con fuente, año y páginas cuando corresponde, más registro de asistencia de IA. | Cubierto. Verificar la preferencia exacta del docente para recursos web y citas en la defensa. |
| Conclusiones | 0,10 | Etapa 7 y [`explicaciones.md`](explicaciones.md): respuesta a H0, test, clases difíciles y límites. | Cubierto. Evitar afirmar una ventaja práctica mínima que el intervalo aún no demuestra. |

**Balance:** los once ítems, que suman 10 puntos máximos, tienen evidencia en el notebook; esto respalda una implementación sustancialmente completa. No asigno una nota final de 10/10: depende de que el docente acepte las justificaciones y de la defensa. No se verificó una ejecución desde cero en Google Colab: el notebook tiene **27 celdas de código ejecutadas localmente, sin salidas de error**, y el README explica la preparación de datos. La ruta manual de descarga/montaje en Colab debe ensayarse antes de la entrega si se evaluará allí.

**Riesgos concretos de descuento:** la defensa puede quedar en nivel regular si no se explica la dependencia de los folds o el margen práctico; una ejecución nueva en Colab puede requerir ajuste de rutas o paquetes; la búsqueda reducida a 2 × 5 y a 5 235 imágenes debe justificarse con el costo medido; y los posibles errores de etiqueta del test exigen una interpretación cautelosa. Son límites declarados, no fallos observados en la ejecución local.

### Cifras que conviene recordar

| Dato | Valor |
| --- | ---: |
| Imágenes inspeccionadas / válidas | 26 179 / 26 177 |
| Desarrollo / test | 20 941 / 5 236 |
| Búsqueda / comparación / desarrollo no usado en búsqueda | 5 235 / 10 471 / 5 235 |
| CV de búsqueda y comparación | 5 folds × 2 repeticiones |
| Macro F1 media en comparación: logística / SVM | 0,9556 / 0,9620 |
| Diferencia media SVM − logística | +0,00642 |
| Prueba t corregida, p bilateral | 0,01075 |
| Intervalo corregido aproximado del 95 % | [+0,00189; +0,01095] |
| Margen de relevancia práctica predefinido | 0,005 |
| Test de la SVM: accuracy / macro F1 / weighted F1 | 0,9712 / 0,9684 / 0,9711 |
| F1 de la clase más difícil, vaca | 0,9290 |

## Preguntas probables y respuestas sugeridas

### Problema, datos y representación

**1. ¿Cuál es exactamente la pregunta de investigación?**

Si SVM y regresión logística presentan una diferencia de macro F1 al clasificar las diez clases de Animals-10 usando los mismos embeddings de MobileNetV2. La comparación formal se hizo en folds pareados de desarrollo; el test se reservó para la SVM elegida.

**2. ¿Por qué usar macro F1 en vez de solo accuracy?**

Las clases tienen tamaños distintos. Macro F1 calcula F1 por clase y luego promedia dando el mismo peso a cada una; accuracy puede verse dominada por clases más numerosas. Reporté también accuracy y weighted F1 como contexto.

**3. ¿Qué hiciste en el EDA y qué encontraste?**

Revisé conteos por clase, resoluciones, canales y ejemplos visuales, más calidad y duplicados. De 26 179 archivos excluí dos incidencias; quedaron 26 177 imágenes válidas. El inventario y las incidencias se resumen en la etapa 2 y en `results/`.

**4. ¿Por qué no hiciste un split aleatorio simple?**

Imágenes idénticas o casi idénticas podrían quedar a ambos lados y elevar artificialmente el rendimiento. Primero agrupé con hash exacto y hash perceptual (`dHash`, umbral Hamming 3); después estratifiqué sin separar grupos. Eso reduce fuga detectable, aunque no garantiza detectar toda relación entre imágenes.

**5. ¿Qué significa «MobileNetV2 congelada»?**

Se usan pesos preentrenados en ImageNet como extractor, sin actualizar sus parámetros ni utilizar su capa final de clasificación. Con `include_top=False` y `pooling='avg'` cada imagen produce 1 280 características. Los modelos entrenados son la regresión logística y la SVM sobre esos vectores.

**6. ¿Por qué no aplicaste feature selection?**

La rúbrica permite omitirla en imágenes o transfer learning. Conservé los 1 280 componentes para ambos clasificadores y evité agregar otro filtro e hiperparámetros. PCA y t-SNE se usan solo para explorar, nunca para entrenar o predecir.

### Optimización y pipelines

**7. ¿Cómo evitaste fuga durante la validación?**

El `StandardScaler` está dentro de cada `Pipeline`; `GridSearchCV` lo ajusta con el train de cada fold y transforma su validación con esos parámetros. Los embeddings se extrajeron con una red congelada, sin ajustar el extractor sobre validación ni test.

**8. ¿Por qué 5 folds y 2 repeticiones, cuando la rúbrica recomienda 10 × 10?**

La misma rúbrica permite reducir a 2 × 5 si el costo es elevado. SVM RBF sobre miles de vectores de 1 280 dimensiones era costosa en CPU. Mantuve los mismos diez splits para ambos modelos y grupos indivisibles. Menos repeticiones implican una estimación menos estable; lo reconozco como límite.

**9. ¿Por qué separaste 5 235 imágenes para búsqueda y 10 471 para comparación?**

Porque elegir el mejor hiperparámetro y probar una hipótesis sobre los mismos scores produciría sesgo de selección. La búsqueda fijó configuraciones en una muestra; la comparación usó una parte independiente de desarrollo. Las 5 235 imágenes restantes se incorporaron al ajuste final.

**10. ¿Qué configuraciones ganaron y cuánto costaron?**

Logística: `C=0,1`, sin `class_weight`, macro F1 media de búsqueda 0,9518 y 0,39 s por ajuste de fold. SVM: RBF, `C=10`, `gamma='scale'`, sin `class_weight`, macro F1 0,9541 y 2,96 s. Esas medias de búsqueda son descriptivas, no la prueba de hipótesis.

**11. ¿Qué significan `C` y `gamma`?**

`C` regula la penalización de errores: valores mayores permiten un ajuste más estricto a los datos de entrenamiento. En SVM RBF, `gamma` controla la escala de influencia de cada ejemplo; una elección inadecuada puede dar un rendimiento muy bajo. El gráfico de hiperparámetros muestra ese contraste.

**12. ¿Qué te dijeron las curvas de aprendizaje?**

La validación de logística se estabilizó cerca de 0,953. La de SVM subió al aumentar la muestra de entrenamiento, aproximadamente de 0,936 a 0,957. Esto sugiere beneficio de más datos en ese rango; no es garantía de mejora indefinida ni sustituye la comparación independiente.

### Comparación estadística: preguntas prioritarias

**13. ¿Cuáles fueron H0, H1 y el nivel de significancia?**

H0: la diferencia media esperada de macro F1 entre SVM y logística es cero. H1: es distinta de cero. Definí la diferencia como SVM menos logística y fijé una prueba bilateral con `α=0,05`.

**14. ¿Por qué la comparación es pareada?**

Ambos modelos se evaluaron sobre **exactamente los mismos** diez folds de comparación. Cada diferencia usa las dos puntuaciones del mismo fold; así se reduce la variación atribuible al split.

**15. ¿Por qué no usaste la t pareada ordinaria como prueba principal?**

Los folds repetidos comparten observaciones y sus diferencias no son independientes. La t ordinaria subestimaría la incertidumbre. Usé la corrección de Nadeau y Bengio como prueba principal y conservé la t ordinaria y Wilcoxon solo como sensibilidad.

**16. ¿Cuál fue la evidencia numérica a favor de SVM?**

En la parte de comparación, las medias fueron 0,9556 (logística) y 0,9620 (SVM). SVM ganó los diez pares. La diferencia media fue +0,00642 macro F1; la t corregida dio `p=0,01075` e intervalo aproximado del 95 % [+0,00189; +0,01095].

**17. ¿Cómo interpretas `p=0,01075`?**

Bajo H0 y los supuestos de la prueba corregida, una diferencia al menos tan extrema como la observada sería poco compatible con igualdad media. Como `p<0,05`, rechazo H0. **No** significa que la probabilidad de H0 sea 1,075 %.

**18. ¿La ventaja es prácticamente importante?**

La media, 0,00642, supera el margen fijado de 0,005, pero el intervalo corregido comienza en 0,00189. Por ello hay evidencia estadística de diferencia bajo este diseño, mientras que una mejora mínima de medio punto porcentual no queda establecida con la misma seguridad.

**19. ¿Qué aportó Shapiro-Wilk?**

Sobre las diez diferencias dio `p=0,399`: no detectó una desviación clara de normalidad. Con solo diez observaciones su poder es limitado; ese resultado **no prueba** normalidad.

**20. ¿Por qué muestras Wilcoxon si no lo usas como decisión final?**

Como análisis de sensibilidad solicitado por la propuesta. Su `p=0,00195` apunta en la misma dirección, pero también supone pares independientes; por la dependencia de los folds repetidos, la conclusión principal descansa en la t corregida.

**21. ¿Qué significa el tamaño de efecto `d_z=1,896`?**

Es la media de las diferencias dividida por su desviación estándar muestral, un resumen descriptivo de los diez pares. Por la correlación entre folds no lo presento como efecto independiente ni como garantía de relevancia externa.

### Test, visualizaciones y límites

**22. ¿Cuándo se abrió el test?**

Después de fijar los hiperparámetros y elegir la SVM mediante la comparación de desarrollo. Se ajustó la SVM con las 20 941 imágenes de desarrollo y se evaluó una vez en las 5 236 de test.

**23. ¿Por qué no comparaste ambos modelos en test?**

La prueba formal ya se hizo en desarrollo independiente de la búsqueda. El test tenía la función de estimar el rendimiento final del modelo seleccionado, no de tomar otra decisión entre algoritmos. Ambos modelos sí se reentrenaron con todo desarrollo; solo la SVM se evaluó en test.

**24. ¿Cómo interpretas que el macro F1 de test, 0,9684, sea mayor que el de comparación, 0,9620?**

El ajuste final usa más imágenes y el test tiene otra composición. Ambas circunstancias podrían contribuir; el experimento no permite atribuir causalmente la diferencia a una sola. No se deben comparar esos dos números como si fueran folds del mismo procedimiento.

**25. ¿Qué clase fue la más difícil y por qué?**

Vaca tuvo la menor F1 (0,9290) y sensibilidad 0,9115. De sus 373 ejemplos, 11 se predijeron como caballo y 11 como oveja. Fondo rural, pose y presencia de varios animales son posibles causas visuales; habría que comprobarlas sistemáticamente, no afirmarlas solo por la matriz.

**26. ¿Qué demuestra t-SNE?**

Es una visualización exploratoria de 1 500 embeddings de desarrollo, 150 por clase. Se aplicó PCA a 50 dimensiones y después t-SNE (`perplexity=30`, 1 000 iteraciones, semilla 42). Permite inspeccionar agrupamientos y solapamientos; sus distancias globales no son exactas y no mide rendimiento predictivo.

**27. ¿La brecha de puntuaciones de los ejemplos equivale a confianza probabilística?**

No. Es la diferencia entre las dos puntuaciones de decisión más altas de SVM, sin calibración probabilística. Se usa para seleccionar ejemplos llamativos, no para afirmar una probabilidad de acierto.

**28. ¿Qué haces con posibles etiquetas erróneas en test?**

Las señalo como casos para revisión humana. No las cambio después de mirar resultados porque eso alteraría la evaluación reservada. Si se corrige el dataset, se debe documentar una nueva versión y definir otro protocolo de evaluación.

**29. ¿Qué no puedes concluir de este estudio?**

No demuestra que SVM gane en cualquier conjunto de animales, cámara o dominio; no prueba una mejora práctica mínima de 0,005; no verifica que todas las etiquetas sean correctas ni que el agrupamiento encuentre todo cuasiduplicado. Tampoco se entrenó de extremo a extremo MobileNetV2.

**30. ¿Cómo reproducirías el trabajo?**

Descargar Animals-10, colocar las diez carpetas en `data/`, instalar `requirements.txt` con Python 3.12, ejecutar el notebook desde la raíz y revisar `artifacts/`. `results/` guarda una copia ligera de los splits, configuraciones, scores y métricas. Las semillas, versiones y huellas se registran; la primera creación del extractor necesita pesos de ImageNet.

## Guion de respuesta en 60 segundos

«Comparé regresión logística y SVM sobre embeddings congelados de MobileNetV2 para clasificar Animals-10. Limpié dos incidencias de 26 179 archivos y separé grupos de imágenes similares antes del split, dejando 20 941 para desarrollo y 5 236 para test. Busqué parámetros con cinco folds repetidos dos veces, y comparé los modelos en otra parte de desarrollo con los mismos diez folds. La SVM obtuvo una macro F1 media de 0,9620 frente a 0,9556; la diferencia corregida fue +0,00642, con `p=0,01075`. Hay evidencia estadística a favor de SVM, aunque la magnitud práctica mínima sigue incierta. Tras fijarla, dio macro F1 0,9684 en test. La clase más difícil fue vaca y la generalización a imágenes externas queda por estudiar».

## Antes de la defensa

1. Ensayar una ejecución desde cero en Colab si esa plataforma será revisada; el control actual es una ejecución local completa.
2. Abrir las [curvas de aprendizaje](figures/figura_curvas_aprendizaje.png), el [gráfico de diferencias](figures/figura_diferencias_comparacion.png), la [matriz de confusión](figures/figura_matriz_confusion_test.png) y la [dispersión t-SNE](figures/figura_tsne_desarrollo.png) para explicar cada eje sin leer el notebook.
3. Practicar especialmente las preguntas 8–9 y 13–20: optimización y comparación suman **7 de 10 puntos** en los ítems.
