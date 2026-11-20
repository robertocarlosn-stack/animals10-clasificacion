"""Copia resultados ligeros de una ejecución completa al snapshot versionado."""

from pathlib import Path
from shutil import copy2


FILES = (
    "resumen_etapa2.json",
    "incidencias_eda.csv",
    "particion.csv",
    "roles_desarrollo.csv",
    "configuracion_modelos.json",
    "resultados_busqueda.csv",
    "curvas_aprendizaje.csv",
    "scores_comparacion.csv",
    "resumen_inferencia.json",
    "metricas_test.json",
    "matriz_confusion_conteos.csv",
    "matriz_confusion_normalizada.csv",
    "pares_confusion_test.csv",
    "predicciones_test.csv",
    "predicciones_test_meta.json",
    "embeddings_desarrollo_meta.json",
    "embeddings_test_meta.json",
    "modelo_final_logistico_meta.json",
    "proyeccion_pca_desarrollo.csv",
    "proyeccion_tsne_desarrollo.csv",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "artifacts"
    target = root / "results"
    missing = [name for name in FILES if not (source / name).is_file()]
    if missing:
        raise SystemExit("Faltan resultados de la ejecución: " + ", ".join(missing))
    target.mkdir(exist_ok=True)
    for name in FILES:
        copy2(source / name, target / name)
    print(f"Se actualizaron {len(FILES)} archivos en {target}.")


if __name__ == "__main__":
    main()
