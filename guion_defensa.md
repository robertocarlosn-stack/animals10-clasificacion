# Guion breve para la defensa

## 1. Pregunta y datos

La pregunta es si una SVM y una regresión logística difieren en macro F1 al clasificar Animals-10 sobre representaciones de MobileNetV2. Se inspeccionaron 26 179 imágenes, se excluyeron dos por calidad y se separaron 20 941 para desarrollo y 5 236 para test. Las particiones respetan grupos de imágenes duplicadas o muy parecidas detectadas.

## 2. Representación y modelos

MobileNetV2, preentrenada en ImageNet y congelada, transforma cada imagen de 224 × 224 en un vector de 1280 valores. Cada modelo se ajusta dentro de un pipeline con `StandardScaler`. La búsqueda usó 5 235 imágenes y cinco folds estratificados por grupos repetidos dos veces. La mejor regresión logística tuvo `C=0,1`; la mejor SVM tuvo kernel RBF, `C=10` y `gamma='scale'`.

## 3. Evidencia estadística

Los dos modelos se compararon en los mismos diez folds de una parte independiente de 10 471 imágenes. La macro F1 media fue **0,9556** para regresión logística y **0,9620** para SVM. La diferencia SVM menos logística fue **+0,00642**. Por la dependencia entre folds repetidos, la prueba principal fue una t corregida: **p bilateral = 0,01075**, intervalo aproximado del 95 % **[+0,00189; +0,01095]**. Se rechaza la hipótesis de igualdad media a nivel 0,05. El intervalo incluye diferencias menores que el margen práctico predefinido de 0,005, así que la magnitud práctica sigue siendo incierta.

## 4. Evaluación final y errores

Se reentrenaron ambos modelos con todo desarrollo. Solo la SVM seleccionada se evaluó en test: **accuracy 0,9712**, **macro F1 0,9684** y **weighted F1 0,9711**. Vaca fue la clase más difícil, con F1 **0,9290**, y se confundió sobre todo con caballo y oveja. La matriz de confusión muestra los conteos y proporciones; t-SNE es una vista exploratoria de embeddings de desarrollo. Algunos ejemplos con errores de alta puntuación sugieren posibles etiquetas incorrectas y requieren revisión manual.

## 5. Límites y respuesta final

La evidencia favorece a SVM en esta comparación sobre Animals-10. Los resultados no garantizan la misma ventaja en imágenes externas. La búsqueda usó una submuestra por costo, el extractor permaneció congelado y la agrupación no detecta necesariamente todas las relaciones entre imágenes. Una continuación razonable sería revisar etiquetas dudosas, probar otro extractor o fine-tuning y validar en una fuente externa sin reutilizar el test actual para nuevas decisiones.

## Preguntas previsibles

- **¿Por qué macro F1?** Da el mismo peso a las diez clases pese a soportes desiguales.
- **¿Por qué separar la búsqueda de la comparación?** Evita usar las mismas puntuaciones optimizadas para declarar una diferencia estadística.
- **¿Por qué una prueba t corregida?** Los folds repetidos comparten observaciones; una t ordinaria trataría sus puntuaciones como independientes.
- **¿Por qué no evaluar también la regresión logística en test?** La comparación formal ya se hizo en desarrollo y la SVM se eligió antes de abrir test. El test sirve para una evaluación final del modelo principal.
