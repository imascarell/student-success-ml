#classification.py

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report


def construir_pipelines_clasificacion(preprocesador) -> dict:

    """
    Esta función construye los pipelines de clasificación: logística, random forest y gradient boosting.
    Args:
        preprocesador: el ColumnTransformer preprocesador. Aplica escalado a features numéricas y one-hot encoding a features categóricas.
    Returns:
        pipelines (dict): diccionario con nombre del modelo y pipeline.
    """

    pipelines = {

        #reg logística
        "Logistica": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", LogisticRegression(max_iter=1000, random_state=42))
        ]),

        #random forest
        "Random Forest": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", RandomForestClassifier(n_estimators=200, random_state=42))
        ]),

        #descenso gradiente
        "Gradient Boosting": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", GradientBoostingClassifier(n_estimators=200, random_state=42))
        ])
    }

    return pipelines


def construir_baseline_clasificacion() -> DummyClassifier:

    """
    Esta función construye el baseline de clase mayoritaria.
    Args:
        None.
    Returns:
        baseline (DummyClassifier): el clasificador baseline.
    """

    baseline = DummyClassifier(strategy="most_frequent")

    return baseline


def evaluar_clasificacion(y_real, y_pred) -> tuple:

    """
    Esta función evalúa las predicciones de clasificación.
    Args:
        y_real: las etiquetas reales.
        y_pred: las etiquetas predichas.
    Returns:
        metricas (dict): diccionario con accuracy, macro_f1 y balanced_accuracy.
        reporte (str): el classification report como texto.
    """

    metricas = {
        "accuracy": accuracy_score(y_real, y_pred),
        "macro_f1": f1_score(y_real, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred)
    }

    reporte = classification_report(y_real, y_pred)

    return metricas, reporte
