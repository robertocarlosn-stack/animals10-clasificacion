# Clasificación de Animals-10 con MobileNetV2 y modelos supervisados

Proyecto de clasificación de imágenes de diez animales. Un extractor MobileNetV2 preentrenado y congelado transforma cada imagen en un vector de 1280 características. Sobre esos vectores se comparan regresión logística y máquinas de vectores de soporte (SVM). La pregunta principal es si la diferencia de macro F1 entre ambos modelos se sostiene en una comparación estadística pareada.

## Archivos principales

- [`clasificacion_animals10.ipynb`](clasificacion_animals10.ipynb): análisis completo, código ejecutado y gráficos.
- [`explicaciones.md`](explicaciones.md): decisiones y resultados de cada etapa.
- [`guion_defensa.md`](guion_defensa.md): relato breve y respuestas a preguntas previsibles.
- [`requirements.txt`](requirements.txt): dependencias del entorno probado con Python 3.12.
- `figures/`: selección de gráficos para consultar sin abrir el notebook.
- `artifacts/`: cachés, tablas, modelo y resultados generados al ejecutar; esta carpeta no se versiona.

## Datos y entorno

Descarga [Animals-10 de Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10) y coloca las imágenes en `data/`, con las carpetas originales `cane`, `cavallo`, `elefante`, `farfalla`, `gallina`, `gatto`, `mucca`, `pecora`, `ragno` y `scoiattolo`. La carpeta `data/` y cualquier token de descarga están excluidos del repositorio.

En PowerShell, desde la raíz del proyecto:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install notebook
.venv\Scripts\python.exe -m ipykernel install --user --name animals10 --display-name "Python (Animals-10)"
.venv\Scripts\python.exe -m jupyter notebook clasificacion_animals10.ipynb
```

También puedes abrir el archivo en VS Code y seleccionar el kernel `Python (Animals-10)`. Ejecuta las celdas en orden desde el directorio raíz: el notebook usa `Path.cwd()` para localizar `data/` y `artifacts/`. En Google Colab, sitúa primero el directorio de trabajo en una copia del proyecto que incluya `data/`, instala las dependencias necesarias y ejecuta las celdas en orden. La primera ejecución descarga los pesos públicos de MobileNetV2; se requiere conexión a Internet para ese paso.

## Diseño experimental

Se inspeccionaron 26 179 archivos y se excluyeron dos por calidad. Se separaron 20 941 imágenes para desarrollo y 5 236 para test mediante grupos de imágenes idénticas o muy parecidas, evitando que un grupo aparezca en ambos lados. La búsqueda de hiperparámetros utilizó 5 235 imágenes de desarrollo y cinco folds estratificados por grupo repetidos dos veces. La comparación estadística utilizó otras 10 471 imágenes de desarrollo y los mismos diez folds para los dos modelos. Las 5 235 imágenes restantes se reservaron para el ajuste final. El escalado de características se aprende dentro de cada fold.

La SVM RBF seleccionada usa `C=10`, `gamma='scale'` y `class_weight=None`. La regresión logística seleccionada usa `C=0.1` y `class_weight=None`. En la comparación independiente, la macro F1 media fue 0,9620 para SVM y 0,9556 para regresión logística; la diferencia media fue +0,00642. La prueba t corregida para validación cruzada repetida dio `p=0,01075` (bilateral). El análisis y sus límites se explican en [`explicaciones.md`](explicaciones.md).

Ambos modelos se reentrenaron con las 20 941 imágenes de desarrollo. La SVM seleccionada obtuvo **accuracy 0,9712** y **macro F1 0,9684** en las 5 236 imágenes de test; la regresión logística no se evaluó en test. La clase con menor F1 fue vaca (0,9290). Puedes consultar la [matriz de confusión](figures/figura_matriz_confusion_test.png), la [dispersión t-SNE](figures/figura_tsne_desarrollo.png), la [comparación pareada](figures/figura_comparacion_pareada.png) y los [ejemplos de test](figures/figura_ejemplos_test.png). La carpeta `figures/` contiene los once gráficos finales.

## Reproducibilidad y límites

Las particiones, semillas, parámetros y versiones quedan registrados en el notebook y en los metadatos de `artifacts/`. La caché de embeddings permite reanudar una extracción interrumpida. El test se procesa después de fijar el modelo ganador. La ejecución completa en CPU puede tardar bastante, especialmente al extraer características de las 26 177 imágenes. Los resultados corresponden a este conjunto y a MobileNetV2 congelado; no prueban generalización a otras fuentes de imágenes.

El código y la documentación del proyecto se distribuyen bajo la [licencia MIT](LICENSE). Las imágenes Animals-10 y los pesos ImageNet mantienen sus propias condiciones de uso y no se incluyen en esta licencia.
