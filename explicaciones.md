# Explicaciones del proyecto Animals-10

Este documento acompaña a `clasificacion_animals10.ipynb`. Registra el propósito de cada etapa, las decisiones metodológicas y la forma de comprobar sus resultados. El trabajo sigue la propuesta de `conf/propuesta_clasificacion_animals10.pdf`.

## Plan de etapas

| Etapa | Alcance | Estado |
| --- | --- | --- |
| 1. Preparación | Introducción, rutas, semillas y versiones | Implementada |
| 2. Datos | EDA, calidad, duplicados y partición | Implementada |
| 3. Representaciones | Preprocesamiento y MobileNetV2 | Implementada |
| 4. Modelado | Pipelines, optimización y curvas | Implementada |
| 5. Inferencia | Comparación estadística pareada | Implementada |
| 6. Evaluación | Test final, t-SNE y errores | Implementada |
| 7. Cierre | Conclusiones, referencias y revisión | Implementada |

## Etapa 1. Preparación

### Objetivo

Dejar un punto de partida ejecutable para el estudio y fijar las convenciones que usarán las etapas posteriores. El notebook presenta la pregunta de investigación y el orden experimental, registra el entorno y comprueba que existan las diez carpetas de clases.

### Decisiones

- Se usa `SEED = 42` para las operaciones aleatorias de Python y NumPy. En etapas posteriores se pasará esta semilla explícitamente a la partición, la validación cruzada y las bibliotecas que la requieran. La semilla facilita la reproducción, aunque algunas operaciones en GPU pueden seguir siendo no deterministas.
- El notebook toma `Path.cwd()` como raíz del proyecto. Debe ejecutarse desde la carpeta que contiene `data/` y `conf/`; en Colab hay que situarse primero allí. Las imágenes locales evitan descargar de nuevo el dataset o utilizar un token.
- Las diez carpetas usan sus nombres originales en italiano como identificadores. La traducción al español se reserva para las explicaciones y las figuras. El orden de clases queda fijado en `CLASS_NAMES`.
- Se muestran versiones de las bibliotecas relevantes para registrar el entorno. Una biblioteca marcada como `no instalado` señala una dependencia que deberá resolverse antes de la etapa que la utilice. En particular, TensorFlow se necesitará para MobileNetV2 en la etapa 3.
- `artifacts/` queda definida como ruta para salidas posteriores; esta etapa no crea archivos allí.

### Ejecución y comprobación

1. Abre `clasificacion_animals10.ipynb` en Jupyter o Google Colab.
2. Sitúa el directorio de trabajo en la raíz del proyecto y verifica que `data/` contenga las diez carpetas de animales.
3. Ejecuta las dos celdas de código. La primera informa la semilla, las rutas y las versiones. La segunda confirma las diez clases y muestra su traducción.

Si faltan carpetas, la segunda celda se detiene con un mensaje que indica la ruta esperada. Todavía no se inspeccionan imágenes, no se separan datos y no se entrenan modelos.

## Etapa 2. Datos, calidad y partición

### Objetivo

Caracterizar las imágenes disponibles, registrar los archivos que no se pueden usar y crear una partición reproducible que evite que duplicados o imágenes muy similares aparezcan a la vez en desarrollo y test.

### Inventario y análisis exploratorio

Se recorren únicamente las diez carpetas de clases. Se admiten archivos JPG, JPEG, PNG, WebP y BMP. Cada imagen se decodifica completamente para detectar errores; se registran su ruta relativa, clase, dimensiones, relación ancho/alto, formato, modo de color, medias RGB y una medida de intensidad. Los archivos con extensiones no admitidas, errores de lectura o miniaturas casi uniformes (desviación estándar menor que 2) se anotan en `artifacts/incidencias_eda.csv` y se excluyen de la partición.

La inspección visual de los grupos con etiquetas distintas detectó una imagen de caballo en la carpeta `mucca`. Su ruta se documenta en `MANUAL_EXCLUSIONS` dentro del notebook y se registra como etiqueta incorrecta en el archivo de incidencias. También se observó una imagen completamente negra en `farfalla`; la regla de miniaturas casi uniformes permite excluirla de forma reproducible. Estas correcciones se hacen antes de crear la partición.

El notebook presenta una tabla de balance con porcentajes, estadísticas de resolución, recuentos de formato y modo, histogramas de ancho y relación de aspecto, distribución RGB aproximada y un panel con dos ejemplos reproducibles por clase. La distribución de color se calcula con miniaturas de 32 × 32 píxeles para reducir el costo; no debe interpretarse como un histograma exacto de todos los píxeles originales.

### Duplicados y prevención de fuga

SHA-256 identifica archivos idénticos. Un *difference hash* (dHash) de 64 bits agrupa además candidatos visualmente similares si difieren en tres bits o menos. Se usa un árbol BK para buscar vecinos y una estructura de unión para formar grupos transitivos. Esta regla es conservadora respecto a la fuga entre particiones, pero un hash perceptual puede unir imágenes distintas por accidente. El notebook informa el tamaño de los grupos y muestra ejemplos de grupos que contienen más de una etiqueta; esos casos requieren revisión visual antes de usar los resultados para afirmar que las etiquetas son correctas.

La agrupación ocurre **antes** del split. `StratifiedGroupKFold` genera cinco candidatos con semilla 42 y grupos indivisibles. Se elige el candidato cuyo tamaño de test y distribución por clase se aproximan más al objetivo del 20 %. La estratificación con grupos es aproximada: la tabla final muestra el porcentaje real de test en cada clase. Se verifica que desarrollo y test no compartan ningún grupo y que las diez clases estén presentes en test.

### Resultados observados en los datos locales

Se inspeccionaron 26 179 archivos de imagen. Dos se excluyeron por los motivos descritos arriba, de modo que quedaron **26 177 imágenes válidas**. La clase más numerosa es `cane` (4 863) y la menos numerosa es `elefante` (1 446); esto justifica usar macro F1 como métrica principal. La mediana de resolución es 300 × 225 píxeles. Hay 26 127 imágenes JPEG y 50 PNG entre los archivos válidos.

No se encontraron archivos idénticos por SHA-256. La regla dHash formó **1 127 grupos** con más de una imagen, que reúnen 2 513 imágenes. Después de excluir los dos archivos problemáticos, no quedaron grupos con etiquetas distintas. La partición contiene **20 941 imágenes de desarrollo** y **5 236 de test** (20,00 %). Las diez clases están presentes en ambos conjuntos y ningún grupo cruza entre ellos. Estos números corresponden al estado actual de los datos locales; si cambian los archivos, hay que regenerar los artefactos y actualizar esta sección.

### Artefactos

| Archivo | Contenido |
| --- | --- |
| `artifacts/inventario_imagenes.csv` | Metadatos de las imágenes válidas, hashes, grupo y partición |
| `artifacts/incidencias_eda.csv` | Archivos excluidos y motivo |
| `artifacts/particion.csv` | Rutas relativas, etiquetas, grupo y conjunto asignado |
| `artifacts/resumen_etapa2.json` | Semilla, umbral de similitud y recuentos finales |

Las etapas siguientes leerán `artifacts/particion.csv` en lugar de repetir o cambiar la partición. El test solo se utilizará en la evaluación final. La descripción del conjunto completo en esta etapa no implica seleccionar modelos con el test.

### Ejecución y comprobación

Ejecuta el notebook desde la primera celda. La inspección de todas las imágenes puede tardar varios minutos. Comprueba las incidencias y, sobre todo, los grupos con etiquetas distintas que aparecen al final del análisis de duplicados. Si se modifica la regla de agrupación o se corrigen archivos, vuelve a ejecutar esta etapa completa para regenerar los cuatro artefactos antes de continuar.

## Etapa 3. Representaciones visuales

### Objetivo

Convertir cada imagen válida de **desarrollo** en un vector de 1 280 características que pueda utilizarse en la validación cruzada de los modelos clásicos. La red no se ajusta al dataset: se utiliza MobileNetV2 preentrenada en ImageNet, sin su clasificador final y con pooling global promedio. El conjunto de test no se carga en esta etapa.

### Preprocesamiento y modelo

La lectura aplica la orientación EXIF, convierte a RGB y redimensiona a 224 × 224 píxeles. Después se llama a `tf.keras.applications.mobilenet_v2.preprocess_input`, que lleva los valores de píxeles de `[0, 255]` a `[-1, 1]`. El redimensionamiento directo puede deformar la relación de aspecto de algunas imágenes; se mantiene aquí para seguir la propuesta y asegurar una entrada uniforme. No se aplica aumento de datos porque no se está entrenando la red.

El extractor utiliza `include_top=False`, `weights='imagenet'` y `pooling='avg'`; se fija `trainable=False`. Una prueba de ocho imágenes comprueba la forma de entrada `(8, 224, 224, 3)`, la forma de salida `(8, 1280)` y la ausencia de valores no finitos. La documentación oficial de [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) y de su [preprocesamiento](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/preprocess_input) respalda estos parámetros.

### Extracción, caché y aislamiento de test

Las imágenes de desarrollo se recorren en el orden de `artifacts/particion.csv` y se procesan en lotes de 32. Los vectores `float32` se escriben progresivamente en `artifacts/embeddings_desarrollo.npy`. `artifacts/etiquetas_desarrollo.csv` conserva las rutas y etiquetas en el mismo orden. Un archivo de progreso permite reanudar la extracción después de una interrupción; al terminar, el archivo de metadatos guarda la huella SHA-256 del manifiesto, la versión de TensorFlow y la configuración del extractor. La caché solo se reutiliza cuando estos datos y la alineación de etiquetas coinciden.

Los embeddings de test se calculan en la etapa 6 con el mismo extractor y preprocesamiento, después de concluir la selección del modelo. El escalado de características se ajusta **dentro de cada fold** mediante un pipeline de scikit-learn en la etapa 4.

### Entorno y ejecución

En el entorno local de Windows se creó `.venv` y se instaló `tensorflow-cpu` 2.21.0 para comprobar el código. Puede activarse con `.venv\Scripts\activate` antes de abrir Jupyter. En Google Colab, ejecuta el notebook desde la raíz del proyecto con acceso a las imágenes y una instalación de TensorFlow; la primera creación de MobileNetV2 descargará los pesos de ImageNet. La ejecución con CPU puede tardar varios minutos. Si se interrumpe, vuelve a ejecutar las celdas: el archivo de progreso permite continuar desde el último lote confirmado.

### Resultados observados

La extracción local se completó para las **20 941 imágenes de desarrollo**. `artifacts/embeddings_desarrollo.npy` tiene forma `(20941, 1280)`, tipo `float32` y ocupa aproximadamente 107 MB. Todas sus entradas son finitas; la norma L2 mediana de los vectores es 21,654. Las 20 941 filas de `artifacts/etiquetas_desarrollo.csv` coinciden en orden y ruta con el manifiesto de la etapa 2. La huella SHA-256 registrada en los metadatos coincide con el manifiesto actual y no se generó un archivo de embeddings de test.

### Gráficos de análisis

`artifacts/figura_embeddings_desarrollo.png` reúne una dispersión PCA de hasta 200 embeddings por clase y un diagrama de cajas de la norma L2 por clase. Los datos proyectados quedan en `artifacts/proyeccion_pca_desarrollo.csv`. Las dos primeras componentes explican aproximadamente **9,99 %** de la varianza de esta muestra, de modo que la superposición o separación en el plano debe interpretarse con cautela. En la figura se distinguen parcialmente algunos animales, pero no permite medir por sí sola la calidad de clasificación. El t-SNE previsto para la etapa 6 ofrecerá otra vista exploratoria.

## Etapa 4. Pipelines, búsqueda y curvas

### Objetivo y separación de funciones

Esta etapa selecciona hiperparámetros sin consultar test y deja una parte independiente de desarrollo para la comparación estadística de la etapa 5. Los 20 941 embeddings de desarrollo se dividen aproximadamente por mitad mediante `StratifiedGroupKFold` con semilla fija. Una mitad se reserva para **comparación**. De la otra mitad se toma otra mitad, también estratificada por grupos, para **búsqueda**: alrededor de 5 000 imágenes. Las restantes quedan marcadas `search_unused`; se incluyeron en el reentrenamiento final, pero no intervinieron en la búsqueda ni en la prueba estadística. El archivo `artifacts/roles_desarrollo.csv` registra estas asignaciones.

Dentro de búsqueda se generan diez particiones: cinco folds estratificados por grupos con dos semillas distintas. Es el equivalente práctico de repetir la validación estratificada, conservando los grupos indivisibles. Se usa la **misma lista de folds** para regresión logística y SVM. Un `Pipeline` encapsula `StandardScaler` y el clasificador, de modo que las medias y desviaciones del escalado se aprenden solo con el entrenamiento de cada fold. [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) y [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) documentan este comportamiento.

### Espacio de búsqueda

| Modelo | Parámetros explorados | Configuraciones |
| --- | --- | ---: |
| Regresión logística | `C` = 0,1; 1; 10 y `class_weight` = `None` o `balanced`; L2 con `lbfgs` | 6 |
| SVM lineal (`LinearSVC`) | `C` = 0,1 o 1 y `class_weight` = `None` o `balanced` | 4 |
| SVM RBF | `C` = 1 o 10; `gamma` = `scale` o 0,01; `class_weight=None` | 4 |

La búsqueda utiliza macro F1 como criterio de selección y registra también accuracy y weighted F1. Es una cuadrícula intencionalmente pequeña porque cada configuración se ajusta en diez folds; permite estudiar regularización, ponderación de clases y no linealidad sin multiplicar innecesariamente el costo de SVM. Los tiempos y la variación de las puntuaciones se conservan en `artifacts/resultados_busqueda.csv`, incluidos los valores por fold. `artifacts/configuracion_modelos.json` guarda los mejores parámetros y el diseño experimental.

La primera ejecución con `SVC(kernel='linear')` en los diez folds resultó demasiado costosa en CPU y se detuvo antes de producir resultados. La rama lineal se cambió por `LinearSVC`, que sigue siendo una SVM lineal. Una segunda ejecución con unas 10 000 imágenes de búsqueda también se detuvo por costo. Por ello se limitó la búsqueda a unas 5 000 imágenes, conservando los diez folds y la cuadrícula. Un perfil de proceso mostró que `LinearSVC` con `dual=False` seguía siendo lento; una prueba local con `dual=True`, `tol=1e-2` y `max_iter=2000` terminó sin advertencias de convergencia, por lo que se adoptó esa configuración. La rama RBF permanece en `SVC`. Estos cambios se realizaron antes de consultar la parte de comparación y test; la limitación es que los hiperparámetros elegidos con una muestra menor pueden no ser óptimos para el reentrenamiento final con todo desarrollo.

### Gráficas y curvas de aprendizaje

El notebook grafica macro F1 frente a `C` para ambos modelos y distingue `kernel` y `gamma` en SVM. La dispersión entre folds es descriptiva; los folds repetidos comparten observaciones y no representan diez experimentos independientes. Las curvas de aprendizaje usan tres folds de búsqueda y cuatro tamaños crecientes para mostrar puntuaciones de entrenamiento y validación. Sus datos se guardan en `artifacts/curvas_aprendizaje.csv`. La [función `learning_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html) vuelve a ajustar el pipeline en cada tamaño y fold.

Se añade una dispersión de **macro F1 medio frente al tiempo medio de ajuste** para cada candidato. Incluye una vista completa y una ampliación de las configuraciones competitivas, ya que algunas opciones de `gamma` tienen puntuaciones muy bajas y comprimen la escala general. Esta comparación ayuda a valorar el costo de las mejoras observadas. El notebook guarda las figuras en `artifacts/figura_hiperparametros.png`, `artifacts/figura_rendimiento_tiempo.png` y `artifacts/figura_curvas_aprendizaje.png`.

### Lectura de resultados

La búsqueda reporta la mejor configuración de cada modelo, su macro F1 media, desviación y tiempo. Las curvas ayudan a evaluar sobreajuste o beneficio potencial de más datos. **La búsqueda no constituye una prueba de hipótesis**: elegir el máximo de CV introduce optimismo en esas puntuaciones. En la etapa 5 se miden los dos modelos ya fijados sobre la parte de comparación, que no intervino en la selección.

### Resultados observados en búsqueda

La parte de búsqueda contiene **5 235 imágenes**, la parte de comparación reservada **10 471** y otras **5 235** quedaron fuera de la búsqueda. No hay grupos compartidos entre las tres funciones. Se evaluaron las 6 configuraciones de regresión logística y las 8 de SVM en diez folds cada una.

La mejor regresión logística usa `C=0,1` y `class_weight=None`: macro F1 medio **0,9518** (desviación entre folds **0,0051**) y tiempo medio de ajuste **0,39 s** por fold. La mejor SVM usa kernel RBF, `C=10`, `gamma='scale'` y `class_weight=None`: macro F1 medio **0,9541** (desviación **0,0043**) y tiempo medio **2,96 s** por fold. La diferencia descriptiva entre las medias seleccionadas es aproximadamente **0,0023** a favor de SVM; todavía no se interpreta como evidencia estadística porque procede de la búsqueda.

Las opciones RBF con `gamma=0,01` rindieron mucho peor, visible en el panel completo de dispersión; el panel ampliado permite comparar las configuraciones cercanas a macro F1 de 0,95. Las curvas muestran puntuación de entrenamiento cercana a 1 en ambos modelos. La validación de regresión logística se estabiliza alrededor de 0,953, mientras la de SVM sube aproximadamente de 0,936 a 0,957 al aumentar el tamaño de entrenamiento dentro de la muestra de búsqueda. Esto sugería que SVM podría aprovechar más datos; la evaluación independiente de la etapa 5 resolvió la comparación formal.

## Etapa 5. Comparación estadística pareada

### Objetivo y diseño

Esta etapa responde a la pregunta de investigación sin volver a seleccionar hiperparámetros. Los parámetros fijados en la etapa 4 se aplican a las **10 471 imágenes de comparación**, que no participaron en la búsqueda. Se generan cinco folds estratificados por grupos y se repiten con otra semilla. Regresión logística y SVM usan exactamente los mismos diez pares de índices de entrenamiento y validación. El escalado permanece dentro de cada pipeline; no se carga el test final.

Para cada fold se guardan macro F1, accuracy, weighted F1, tamaños de las particiones y tiempos de ajuste en `artifacts/scores_comparacion.csv`. La diferencia se define como `macro F1(SVM) - macro F1(logística)`: positiva favorece a SVM. La hipótesis nula es una diferencia media de cero; la alternativa es una diferencia distinta de cero, con `alpha = 0,05` y prueba bilateral.

### Inferencia y límites

Las diez puntuaciones no son independientes: los folds repetidos comparten imágenes. Por eso la prueba principal es el **t-test corregido de Nadeau y Bengio**. Su error estándar es `sqrt((1/10 + n_valid/n_train) × varianza muestral de las diferencias)`, usando la razón media de tamaños de los diez folds. Se reportan el valor p bilateral, un intervalo aproximado del 95 % y la dirección de la diferencia. La [explicación oficial de scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html) muestra cómo la prueba t ordinaria puede subestimar la varianza al tratar los folds como independientes.

Se incluyen Shapiro-Wilk, t pareado ordinario y Wilcoxon como diagnósticos y análisis de sensibilidad; sus valores p no reemplazan la prueba corregida porque también tratan los pares como independientes. El tamaño de efecto `d_z = media / desviación estándar` es descriptivo. Se fija un margen práctico de **±0,005 macro F1** (medio punto porcentual) para discutir la magnitud, sin alterar la decisión estadística. Un valor p mayor o igual que 0,05 significa que no se rechaza H0; no demuestra equivalencia entre modelos. Con diez folds y corrección aproximada, la incertidumbre sigue siendo relevante.

### Gráficos

- `artifacts/figura_comparacion_pareada.png`: puntuaciones unidas por fold y dispersión frente a la diagonal de igualdad.
- `artifacts/figura_diferencias_comparacion.png`: diferencias fold por fold, media e intervalo corregido, cero y margen práctico.
- `artifacts/figura_qq_diferencias.png`: gráfico Q-Q de diferencias y valor p de Shapiro.

Los valores numéricos de la inferencia se guardan también en `artifacts/resumen_inferencia.json`. La lectura de estas figuras debe apoyarse en los valores y el intervalo, no solo en la posición visual de los puntos.

### Resultados observados e interpretación

En los diez folds de comparación, la regresión logística obtuvo macro F1 medio **0,9556** y SVM **0,9620**. SVM superó a regresión logística en los diez pares. La diferencia media `SVM - logística` fue **+0,00642** (0,642 puntos porcentuales). El t-test corregido produjo **t = 3,205**, **p bilateral = 0,01075** e intervalo aproximado del 95 % **[+0,00189; +0,01095]**. Con `alpha = 0,05`, se **rechaza H0**: los datos de comparación ofrecen evidencia de una diferencia media a favor de SVM bajo este procedimiento.

El valor p de Shapiro-Wilk fue **0,399**: con solo diez diferencias, no se detectó una desviación clara de normalidad, pero esto tampoco prueba que la distribución sea normal. El tamaño de efecto descriptivo `d_z` fue **1,896**. La media supera el margen práctico de 0,005, aunque el intervalo corregido incluye diferencias menores que ese margen. Por ello la superioridad estadística es más clara que la magnitud práctica mínima. Los valores p del t pareado ordinario (**0,00020**) y de Wilcoxon (**0,00195**) se conservan solo como sensibilidad; ignorarían la dependencia de los folds si se usaran como prueba principal.

La comparación se hizo sobre aproximadamente la mitad del conjunto de desarrollo y con un extractor congelado. Sus resultados orientan la selección de SVM como modelo principal para la etapa 6. El intervalo y el valor p corregidos son aproximaciones para este diseño de CV repetida y no sustituyen evidencia de generalización a otro conjunto de datos.

La etapa 6 reentrena el modelo principal con todo desarrollo, procesa el test reservado una sola vez y presenta métricas por clase, matriz de confusión, t-SNE y análisis de errores.

## Etapa 6. Evaluación final y análisis visual

### Procedimiento

Una vez elegida la SVM RBF en la etapa 5, se extrajeron embeddings de las **5 236 imágenes de test** con el mismo MobileNetV2 congelado y el mismo preprocesamiento. El archivo `artifacts/embeddings_test.npy` tiene forma `(5236, 1280)`. Un metadato liga la caché al manifiesto y a la configuración del extractor. Se comprobó que ningún grupo del test aparece en desarrollo.

Se ajustó el pipeline `StandardScaler` + `SVC` RBF (`C=10`, `gamma='scale'`, sin ponderación de clases) con **todo el desarrollo, 20 941 imágenes**. El test se predijo una vez con esos parámetros fijos. `artifacts/predicciones_test.csv` guarda la etiqueta real, la predicha y una brecha entre las dos puntuaciones de decisión más altas; esta brecha no es una probabilidad calibrada. El pipeline ajustado se guardó en `artifacts/modelo_final_svm.joblib` y los parámetros y huellas de datos en `artifacts/predicciones_test_meta.json`.

Se reentrenó también la regresión logística (`C=0,1`, sin ponderación de clases) con las 20 941 imágenes de desarrollo, según la propuesta. Su pipeline se guardó en `artifacts/modelo_final_logistico.joblib` y sus metadatos en `artifacts/modelo_final_logistico_meta.json`. No se hicieron predicciones de test con ese modelo; la comparación entre algoritmos se resolvió en la etapa 5, y el test se reservó para el modelo seleccionado.

### Métricas

En test se obtuvo **accuracy = 0,9712**, **macro F1 = 0,9684** y **weighted F1 = 0,9711**. Corresponden a 5 236 imágenes; son una estimación final para este conjunto, no una segunda comparación estadística con regresión logística. `artifacts/metricas_test.json` conserva el reporte completo por clase. La matriz de confusión normalizada por clase real está en `artifacts/figura_matriz_confusion_test.png` y el gráfico de precisión, sensibilidad y F1 por clase en `artifacts/figura_metricas_por_clase.png`.

La clase con menor F1 fue **vaca: 0,9290**, con sensibilidad 0,9115. Entre sus 373 imágenes, 11 se predijeron como caballo y 11 como oveja. Le siguió oveja con F1 0,9504. Las F1 más altas fueron araña (0,9860) y gallina (0,9839). La matriz permite distinguir errores absolutos de proporciones: las clases tienen soportes desiguales, por lo que macro F1 resume el rendimiento dando el mismo peso a cada clase.

### Dispersión t-SNE y ejemplos

`artifacts/figura_tsne_desarrollo.png` representa 150 embeddings por clase de **desarrollo** (1 500 puntos), reducidos primero con PCA a 50 componentes y después con t-SNE a dos dimensiones. t-SNE utilizó `perplexity=30`, `learning_rate='auto'`, inicialización PCA, `max_iter=1000` y semilla 42. Los puntos se agrupan en buena medida por clase, con zonas de superposición, especialmente entre animales visualmente parecidos. La figura es exploratoria: las distancias y tamaños de grupos en t-SNE no son métricas de rendimiento ni permiten deducir por sí solos la precisión del clasificador. Los datos de la proyección están en `artifacts/proyeccion_tsne_desarrollo.csv`. Se usó desarrollo para esta visualización, sin adaptar el modelo a test.

`artifacts/figura_ejemplos_test.png` muestra cuatro aciertos y cuatro errores seleccionados por brecha de decisión alta. Los aciertos seleccionados son perros claros. En dos errores destacados, la imagen etiquetada como elefante parece mostrar un animal marino y la etiquetada como ardilla parece mostrar conejos; **son posibles errores de etiqueta que requieren revisión humana del archivo original**. Una imagen real de vacas se predijo como gallina. Estos ejemplos no son una muestra aleatoria de los errores y no deben usarse para estimar su frecuencia. `artifacts/pares_confusion_test.csv` contiene todos los pares de clases confundidos, ordenados por frecuencia.

## Etapa 7. Conclusiones, referencias y revisión

### Conclusión sobre la pregunta de investigación

En la comparación pareada independiente de desarrollo, la SVM superó a la regresión logística en los diez folds y mostró una diferencia media de **+0,00642 macro F1**. La prueba corregida dio **p = 0,01075**; a nivel 0,05 se rechaza la igualdad media bajo ese procedimiento. El intervalo aproximado del 95 % fue **[+0,00189; +0,01095]**. Aunque la media supera el margen práctico fijado de 0,005, el intervalo incluye efectos menores; por tanto, la magnitud práctica mínima no queda establecida con la misma claridad. Tras fijar la SVM, su evaluación final alcanzó macro F1 **0,9684** en test. No se utilizó test para seleccionar hiperparámetros ni para la prueba pareada.

### Alcance y mejoras posibles

El análisis se limita a Animals-10 y a embeddings de una MobileNetV2 congelada. La búsqueda se hizo con una parte relativamente pequeña de desarrollo por costo de CPU. La partición por grupos reduce fuga por duplicados y cuasiduplicados detectados, pero no garantiza independencia respecto de imágenes relacionadas que el criterio de agrupación no identifique. Los posibles errores de etiqueta observados en test merecen revisión con un protocolo independiente antes de modificar el conjunto o repetir la evaluación. Para extender el estudio se podrían evaluar otro extractor, fine-tuning, intervalos por remuestreo de grupos y datos externos; cada nueva comparación requeriría un conjunto de evaluación sin uso previo.

### Reproducibilidad y entregables

El notebook conserva salidas de todas las etapas y gráficos ejecutados. `README.md` describe instalación y organización de datos; `requirements.txt` fija las versiones probadas; `.gitignore` excluye imágenes, token, entorno y cachés; `LICENSE` cubre el código y la documentación propios, no el dataset ni los pesos preentrenados. `figures/` ofrece una selección pequeña de gráficos; las tablas, embeddings y el modelo ajustado se regeneran en `artifacts/`. La semilla principal es 42; pequeñas variaciones numéricas son posibles entre equipos o aceleradores.

### Referencias

1. A. Corrado, [“Animals-10”](https://www.kaggle.com/datasets/alessiocorrado99/animals10), Kaggle.
2. M. Sandler et al., [“MobileNetV2: Inverted Residuals and Linear Bottlenecks”](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html), *CVPR*, 2018.
3. J. Deng et al., [“ImageNet: A Large-Scale Hierarchical Image Database”](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf), *CVPR*, 2009.
4. F. Pedregosa et al., [“Scikit-learn: Machine Learning in Python”](https://jmlr.org/papers/v12/pedregosa11a.html), *JMLR*, vol. 12, pp. 2825–2830, 2011.
5. C. Nadeau y Y. Bengio, [“Inference for the Generalization Error”](https://link.springer.com/article/10.1023/A:1024068626366), *Machine Learning*, vol. 52, pp. 239–281, 2003.
6. L. van der Maaten y G. Hinton, [“Visualizing Data using t-SNE”](https://jmlr.org/papers/v9/vandermaaten08a.html), *JMLR*, vol. 9, pp. 2579–2605, 2008.

### Indicaciones de IA registradas

- Preparación y documentación: “construir un notebook etapa por etapa, pedir autorización para iniciar cada etapa, generar explicaciones en archivo MD”.
- Visualización: “Quisiera que se generen graficos de analisis, dispersión comparacion para: Etapa 3 y 4 segun corresponda”.
- Continuación: “Una vez que termine ejecutar todas las erapas faltantes”.

Estas instrucciones guiaron la estructura y los gráficos; las métricas citadas proceden de las ejecuciones guardadas en el notebook.
